#!/usr/bin/env python3
"""Reference solution for the adaption-ai augmentation pipeline task.
The deterministic grader should score this ~1.0 (skillgrade --validate)."""
import os

from adaption import Adaption

client = Adaption(api_key=os.environ["ADAPTION_API_KEY"])

# 1. Ingest: upload the local CSV
uploaded = client.datasets.upload_file("data.csv")
dataset_id = uploaded.dataset_id

# 2. Estimate cost before running the real job
quote = client.datasets.run(
    dataset_id,
    column_mapping={"prompt": "instruction", "completion": "response"},
    estimate=True,
)
print(quote.estimated_credits_consumed, quote.estimated_minutes)

# 3. Adapt: hallucination mitigation + recipes + idempotency for safe retries
run = client.datasets.run(
    dataset_id,
    column_mapping={"prompt": "instruction", "completion": "response"},
    brand_controls={"hallucination_mitigation": True},
    recipe_specification={
        "recipes": {"reasoning_traces": True, "deduplication": True}
    },
    job_specification={"max_rows": 500, "idempotency_key": "job-2026-001"},
)

# 4. Wait: poll with exponential backoff and a timeout
client.datasets.wait_for_completion(run.run_id, timeout=600)

# 5. Evaluate: fetch quality metrics
ev = client.datasets.get_evaluation(run.run_id)
print(ev.quality.score_before, ev.quality.score_after, ev.quality.improvement_percent)

# 6. Export: presigned URL to the augmented dataset
url = client.datasets.download(run.run_id)
print(url)
