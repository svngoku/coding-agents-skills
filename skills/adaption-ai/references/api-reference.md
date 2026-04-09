# Adaption AI — API Reference

> Source: https://docs.adaptionlabs.ai/api
> SDK version: 0.2.0 (`pip install adaption`)
> Base URL: `https://api.adaptionlabs.ai`
> Auth header: `Authorization: Bearer $ADAPTION_API_KEY`

## Table of Contents
1. [Datasets — Create](#datasets-create)
2. [Datasets — Get](#datasets-get)
3. [Datasets — List](#datasets-list)
4. [Datasets — Get Status](#datasets-get-status)
5. [Datasets — Download](#datasets-download)
6. [Datasets — Publish](#datasets-publish)
7. [Datasets — Run](#datasets-run)
8. [Datasets — Get Evaluation](#datasets-get-evaluation)
9. [Upload — Initiate / Complete / Complete By ID](#upload-subresource)

---

## Datasets — Create
`POST /api/v1/datasets`

Unified ingest. Discriminated by `source.type`: `"file"`, `"huggingface"`, or `"kaggle"`.

### File source
```json
{
  "source": {
    "type": "file",
    "name": "my-training-data",
    "file_format": "csv"  // csv | json | jsonl | parquet
  }
}
```
Returns `dataset_id`, `status`, and `upload_instructions` (presigned S3 PUT URL + `s3_key`).

### HuggingFace source
```json
{
  "source": {
    "type": "huggingface",
    "url": "https://huggingface.co/datasets/org/repo",
    "files": ["train.csv"]
  }
}
```

### Kaggle source
```json
{
  "source": {
    "type": "kaggle",
    "url": "https://www.kaggle.com/datasets/org/name",
    "files": ["data.csv"]
  }
}
```
Kaggle creds must be registered at Adaption API keys settings first.

**Python convenience methods** (handle presigned upload automatically):
```python
client.datasets.upload_file("data.csv")
client.datasets.upload_file("data.csv", name="custom-name")
client.datasets.create_from_huggingface(url=..., files=[...])
client.datasets.create_from_kaggle(url=..., files=[...])
```

---

## Datasets — Get
`GET /api/v1/datasets/{dataset_id}`

Returns full `Dataset` record including `evaluation_summary` when available.
```python
ds = client.datasets.get(dataset_id)
ds.evaluation_summary.score_after  # if evaluation finished
```

---

## Datasets — List
`GET /api/v1/datasets`

Auto-paginated iterator. Optional filters: `status`, `limit`.
```python
for dataset in client.datasets.list(status="succeeded", limit=10):
    print(dataset.dataset_id, dataset.name)
```

---

## Datasets — Get Status
`GET /api/v1/datasets/{dataset_id}/status`

Ingestion/run progress. Fields: `status`, `row_count`, `error`.
```python
st = client.datasets.get_status(dataset_id)
```

---

## Datasets — Download
`GET /api/v1/datasets/{dataset_id}/download`

Returns presigned S3 download URL.
```python
url = client.datasets.download(dataset_id)
```

---

## Datasets — Publish
`POST /api/v1/datasets/{dataset_id}/publish`

Publish a dataset to an external platform.

---

## Datasets — Run
`POST /api/v1/datasets/{dataset_id}/run`

Central method. Validates column mapping, reserves credits, starts augmentation pipeline.

### Full parameter schema

**column_mapping** (required for real runs):
- `prompt`: str — required, prompt/instruction column
- `completion`: str — optional, response column
- `chat`: str — optional, conversation column (alternative to prompt+completion)
- `context`: list[str] — optional, context columns

**brand_controls** (optional):
- `hallucination_mitigation`: bool — web-search grounding
- `length`: `"minimal"` | `"concise"` | `"detailed"` | `"extensive"`
- `safety_categories`: list[str] — e.g. `["harassment", "hate"]`

**recipe_specification** (optional):
- `recipes.reasoning_traces`: bool — chain-of-thought
- `recipes.deduplication`: bool — remove near-duplicates
- `recipes.preference_pairs`: bool — DPO chosen/rejected
- `recipes.prompt_rephrase`: bool — rephrase for variety
- `recipes.prompt_metadata_injection`: bool — inject context
- `version`: str — recipe schema version

**job_specification** (optional):
- `max_rows`: int (min 1) — subsample cap
- `idempotency_key`: str — safe retries

**estimate**: bool — when `True`, returns cost quote without starting run.

### Response
- `run_id`: str (null for estimate-only)
- `estimated_credits_consumed`: float
- `estimated_minutes`: float
- `estimate`: bool

---

## Datasets — Get Evaluation
`GET /api/v1/datasets/{dataset_id}/evaluation`

Returns evaluation pipeline status and quality metrics.

```python
ev = client.datasets.get_evaluation(dataset_id)
# ev.status: "pending" | "running" | "succeeded" | "failed" | "skipped"
# ev.quality (when succeeded):
#   .score_before (0-10)
#   .score_after (0-10)
#   .improvement_percent
#   .percentile_after (when available)
#   letter grades
```

---

## Upload Subresource

### Initiate: `POST /api/v1/datasets/upload/initiate`
### Complete: `POST /api/v1/datasets/upload/complete`
### Complete By ID: `POST /api/v1/datasets/{dataset_id}/upload/complete`

Low-level upload flow (the convenience method `upload_file` wraps all three).