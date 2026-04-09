#!/usr/bin/env python3
"""Adaption AI — Async pipeline example.

Demonstrates the AsyncAdaption client for non-blocking usage
in async applications (FastAPI, notebooks with nest_asyncio, etc.).
"""

import asyncio
import os

from adaption import AsyncAdaption, DatasetTimeout


async def main():
    client = AsyncAdaption(api_key=os.environ.get("ADAPTION_API_KEY"))

    # Upload
    result = await client.datasets.upload_file("training_data.csv")
    dataset_id = result.dataset_id
    print(f"Dataset: {dataset_id}")

    # Wait for ingestion
    while True:
        st = await client.datasets.get_status(dataset_id)
        if st.row_count is not None:
            break
        await asyncio.sleep(2)

    # Estimate
    est = await client.datasets.run(
        dataset_id,
        column_mapping={"prompt": "instruction", "completion": "response"},
        brand_controls={"hallucination_mitigation": True, "length": "concise"},
        recipe_specification={"recipes": {"reasoning_traces": True}},
        estimate=True,
    )
    print(f"Estimate: {est.estimated_credits_consumed} credits")

    # Run
    run = await client.datasets.run(
        dataset_id,
        column_mapping={"prompt": "instruction", "completion": "response"},
        brand_controls={"hallucination_mitigation": True, "length": "concise"},
        recipe_specification={"recipes": {"reasoning_traces": True}},
    )
    print(f"Run: {run.run_id}")

    # Wait
    try:
        final = await client.datasets.wait_for_completion(dataset_id, timeout=3600)
        print(f"Status: {final.status}")
    except DatasetTimeout:
        print("Timed out")
        return

    # Evaluate
    ev = await client.datasets.get_evaluation(dataset_id)
    if ev.status == "succeeded" and ev.quality:
        print(f"Score: {ev.quality.score_before} → {ev.quality.score_after}")

    # Download
    url = await client.datasets.download(dataset_id)
    print(f"Download: {url}")

    # List all datasets
    async for ds in client.datasets.list():
        print(f"  {ds.dataset_id} — {ds.status}")


if __name__ == "__main__":
    asyncio.run(main())