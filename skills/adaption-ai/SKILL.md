---
name: adaption-ai
description: Adaption AI SDK for synthetic data augmentation and dataset adaptation. Use when building data pipelines with the Adaption Python SDK, uploading datasets (local files, Hugging Face, Kaggle), running augmentation/adaptation jobs, configuring brand controls (hallucination mitigation, safety categories, length), recipe specifications (reasoning traces, deduplication, preference pairs, prompt rephrase), evaluating dataset quality, downloading results, or any workflow involving `pip install adaption`, `from adaption import Adaption`, Adaptive Data, or the adaptionlabs.ai API. Also trigger when the user mentions synthetic data generation for fine-tuning, dataset augmentation pipelines, DPO preference pair generation, or grounding-based hallucination reduction on training data.
---

# Adaption AI SDK Skill

Build dataset augmentation pipelines with [Adaption's Adaptive Data](https://docs.adaptionlabs.ai/) platform. This skill covers the full lifecycle: ingest → adapt → wait → evaluate → export.

## Quick Reference

```
pip install adaption
```

```python
from adaption import Adaption
client = Adaption(api_key="pt_live_...")  # or set ADAPTION_API_KEY env var
```

**Async client**: `from adaption import AsyncAdaption`

## Core Lifecycle

1. **Ingest** — Upload local file, import from HuggingFace, or import from Kaggle
2. **Adapt** — Start an augmentation run with column mapping + optional controls
3. **Wait** — Poll for completion with exponential backoff
4. **Evaluate** — Fetch quality metrics (score_before/after, improvement %)
5. **Export** — Download augmented dataset via presigned URL

## Ingest Methods

### Local file upload
Supported formats: `.csv`, `.json`, `.jsonl`, `.parquet`
```python
result = client.datasets.upload_file("training_data.csv")
result = client.datasets.upload_file("data.csv", name="my-dataset")  # custom name
dataset_id = result.dataset_id
```

### Hugging Face import (async on server — poll before running)
```python
resp = client.datasets.create_from_huggingface(
    url="https://huggingface.co/datasets/org/repo",
    files=["train.csv"],
)
```

### Kaggle import (requires Kaggle API creds registered in Adaption settings)
```python
resp = client.datasets.create_from_kaggle(
    url="https://www.kaggle.com/datasets/org/dataset-name",
    files=["data.csv"],
)
```

## Running Adaptation Jobs

### `datasets.run()` — the central method

**Required**: `dataset_id`, `column_mapping` with at least `"prompt"` key.

```python
run = client.datasets.run(
    dataset_id,
    column_mapping={
        "prompt": "instruction",       # required
        "completion": "response",      # optional
        # "chat": "conversation",      # optional — alternative to prompt+completion
        # "context": ["source", "ref"],# optional — list of context columns
    },
    # --- Optional controls (see references/ for details) ---
    brand_controls={...},
    recipe_specification={...},
    job_specification={...},
    estimate=True,  # dry-run: get cost quote without starting
)
```

**Response fields**: `run.run_id`, `run.estimated_credits_consumed`, `run.estimated_minutes`

### Brand Controls (`brand_controls`)
| Key | Type | Purpose |
|-----|------|---------|
| `hallucination_mitigation` | `bool` | Web-search grounding to reduce fabricated content |
| `length` | `"minimal"│"concise"│"detailed"│"extensive"` | Target verbosity |
| `safety_categories` | `list[str]` | Content categories to enforce (e.g. `["harassment","hate"]`) |

### Recipe Specification (`recipe_specification`)
```python
recipe_specification={
    "recipes": {
        "reasoning_traces": True,          # chain-of-thought in completions
        "deduplication": True,             # remove near-duplicates
        "preference_pairs": True,          # DPO chosen/rejected pairs
        "prompt_rephrase": True,           # rephrase prompts for variety
        "prompt_metadata_injection": True, # inject context into prompts
    },
    "version": "...",  # optional schema version
}
```

### Job Specification (`job_specification`)
```python
job_specification={
    "max_rows": 500,           # subsample for pilots
    "idempotency_key": "...",  # safe retries
}
```

## Wait, Evaluate, Export

```python
from adaption import DatasetTimeout

# Wait (exponential backoff 2s→30s, default timeout 1hr)
try:
    status = client.datasets.wait_for_completion(dataset_id, timeout=600)
except DatasetTimeout as e:
    print(f"Still running after {e.timeout}s")

# Evaluate quality
ev = client.datasets.get_evaluation(dataset_id)
# ev.status: pending | running | succeeded | failed | skipped
# ev.quality.score_before, ev.quality.score_after, ev.quality.improvement_percent

# Download
url = client.datasets.download(dataset_id)  # presigned S3 URL
```

## When to read more

| Need | Read |
|------|------|
| Full run parameters, all API endpoints, HTTP/curl examples | `references/api-reference.md` |
| Hallucination mitigation, safety, length, reasoning traces guides | `references/guides.md` |
| End-to-end scripts (upload→adapt→evaluate→download) | `scripts/e2e_pipeline.py` |
| Async patterns | `scripts/async_pipeline.py` |

## Key Patterns

- **Always estimate first** on large datasets: `estimate=True` before real run
- **Poll after HF/Kaggle import**: ingestion is async; wait for `row_count is not None`
- **Evaluation lags adaptation**: run may show `succeeded` before eval finishes — poll `get_evaluation` separately
- **Combine controls freely**: `brand_controls` + `recipe_specification` + `job_specification` all compose on one `datasets.run()` call

## Anti-Patterns to Avoid

- **Running a real job without estimating first** — on large datasets, always call with `estimate=True` and check `estimated_credits_consumed` before spending budget
- **Reading results right after HF/Kaggle import** — ingestion is async; wait until `row_count is not None` before running a job, or you'll process an empty dataset
- **Treating run completion as evaluation completion** — the run can show `succeeded` while the evaluation is still `running`; poll `get_evaluation` separately
- **Retrying without an `idempotency_key`** — naive retries on network errors create duplicate jobs; set `job_specification.idempotency_key`
- **Missing the `prompt` key in `column_mapping`** — it is required; jobs fail with confusing errors when the mapping is incomplete

## When to Use / Not Use

**Use this skill when:**

- Generating synthetic training data or augmenting existing datasets for fine-tuning
- Building DPO preference pairs, reasoning traces, or grounded (hallucination-mitigated) training data
- Working with the Adaption Python SDK (`pip install adaption`) or the adaptionlabs.ai API

**Do NOT use this skill when:**

- The data only needs simple cleaning/formatting — a local pipeline is cheaper
- The work must stay fully on-premises with no external platform dependency
- You are not using Adaption's platform — use the generic dataset pipelines of the framework you're already on