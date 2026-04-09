#!/usr/bin/env python3
"""Adaption AI — End-to-end pipeline: upload → adapt → evaluate → download.

Usage:
    export ADAPTION_API_KEY="pt_live_..."
    python e2e_pipeline.py training_data.csv --prompt instruction --completion response

Optional flags:
    --hallucination-mitigation   Enable web-search grounding
    --reasoning-traces           Add chain-of-thought to completions
    --length LEVEL               minimal | concise | detailed | extensive
    --safety CATS                Comma-separated safety categories
    --max-rows N                 Subsample cap
    --estimate-only              Print cost estimate and exit
    --timeout SECS               Wait timeout (default 3600)
"""

import argparse
import os
import sys
import time

from adaption import Adaption, DatasetTimeout


def parse_args():
    p = argparse.ArgumentParser(description="Adaption AI end-to-end pipeline")
    p.add_argument("file", help="Path to local dataset file (.csv/.json/.jsonl/.parquet)")
    p.add_argument("--name", help="Custom dataset name")
    p.add_argument("--prompt", required=True, help="Prompt column name")
    p.add_argument("--completion", help="Completion column name")
    p.add_argument("--chat", help="Chat column name (alternative to prompt+completion)")
    p.add_argument("--context", nargs="*", help="Context column names")
    p.add_argument("--hallucination-mitigation", action="store_true")
    p.add_argument("--reasoning-traces", action="store_true")
    p.add_argument("--deduplication", action="store_true")
    p.add_argument("--preference-pairs", action="store_true")
    p.add_argument("--prompt-rephrase", action="store_true")
    p.add_argument("--length", choices=["minimal", "concise", "detailed", "extensive"])
    p.add_argument("--safety", help="Comma-separated safety categories")
    p.add_argument("--max-rows", type=int)
    p.add_argument("--estimate-only", action="store_true")
    p.add_argument("--timeout", type=int, default=3600)
    return p.parse_args()


def main():
    args = parse_args()
    client = Adaption(api_key=os.environ.get("ADAPTION_API_KEY"))

    # 1. Upload
    print(f"Uploading {args.file}...")
    upload_kwargs = {"name": args.name} if args.name else {}
    result = client.datasets.upload_file(args.file, **upload_kwargs)
    dataset_id = result.dataset_id
    print(f"Dataset created: {dataset_id}")

    # 2. Wait for file processing
    print("Waiting for ingestion...")
    while True:
        st = client.datasets.get_status(dataset_id)
        if st.row_count is not None:
            print(f"Ingested {st.row_count} rows")
            break
        time.sleep(2)

    # 3. Build run kwargs
    col_map = {"prompt": args.prompt}
    if args.completion:
        col_map["completion"] = args.completion
    if args.chat:
        col_map["chat"] = args.chat
    if args.context:
        col_map["context"] = args.context

    run_kwargs = {"column_mapping": col_map}

    # Brand controls
    bc = {}
    if args.hallucination_mitigation:
        bc["hallucination_mitigation"] = True
    if args.length:
        bc["length"] = args.length
    if args.safety:
        bc["safety_categories"] = [s.strip() for s in args.safety.split(",")]
    if bc:
        run_kwargs["brand_controls"] = bc

    # Recipe specification
    recipes = {}
    if args.reasoning_traces:
        recipes["reasoning_traces"] = True
    if args.deduplication:
        recipes["deduplication"] = True
    if args.preference_pairs:
        recipes["preference_pairs"] = True
    if args.prompt_rephrase:
        recipes["prompt_rephrase"] = True
    if recipes:
        run_kwargs["recipe_specification"] = {"recipes": recipes}

    # Job specification
    if args.max_rows:
        run_kwargs["job_specification"] = {"max_rows": args.max_rows}

    # 4. Estimate or run
    if args.estimate_only:
        est = client.datasets.run(dataset_id, estimate=True, **run_kwargs)
        print(f"Estimated credits: {est.estimated_credits_consumed}")
        print(f"Estimated minutes: {est.estimated_minutes}")
        return

    run = client.datasets.run(dataset_id, **run_kwargs)
    print(f"Run started: {run.run_id}")
    print(f"  Est. credits: {run.estimated_credits_consumed}")
    print(f"  Est. minutes: {run.estimated_minutes}")

    # 5. Wait for completion
    try:
        final = client.datasets.wait_for_completion(dataset_id, timeout=args.timeout)
        print(f"Finished: {final.status}")
        if final.error:
            print(f"Error: {final.error.message}", file=sys.stderr)
            sys.exit(1)
    except DatasetTimeout:
        print(f"Timed out after {args.timeout}s", file=sys.stderr)
        sys.exit(1)

    # 6. Evaluate
    print("Checking evaluation...")
    for _ in range(60):
        ev = client.datasets.get_evaluation(dataset_id)
        if ev.status in ("succeeded", "failed", "skipped"):
            break
        time.sleep(5)
    if ev.status == "succeeded" and ev.quality:
        print(f"Quality — before: {ev.quality.score_before}, after: {ev.quality.score_after}")
        print(f"Improvement: {ev.quality.improvement_percent}%")
    else:
        print(f"Evaluation status: {ev.status}")

    # 7. Download
    url = client.datasets.download(dataset_id)
    print(f"Download URL: {url}")


if __name__ == "__main__":
    main()