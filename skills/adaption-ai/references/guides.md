# Adaption AI — Guides Reference

> Source: https://docs.adaptionlabs.ai/guides/

## Table of Contents
1. [Mitigating Hallucinations](#mitigating-hallucinations)
2. [Reasoning Traces](#reasoning-traces)
3. [Safety and Length Constraints](#safety-and-length-constraints)
4. [Processing Large Datasets](#processing-large-datasets)
5. [Evaluating Dataset Quality](#evaluating-dataset-quality)

---

## Mitigating Hallucinations

Enable web-search grounding via `brand_controls.hallucination_mitigation = True`.

**When to use**: Customer support, RAG workflows, compliance-sensitive text, any pipeline where invented details are costly to catch.

```python
run = client.datasets.run(
    dataset_id,
    column_mapping={"prompt": "instruction", "completion": "response"},
    brand_controls={"hallucination_mitigation": True},
)
```

Combines with other brand_controls (length, safety) on the same run. Use `estimate=True` first to compare cost impact.

---

## Reasoning Traces

Add chain-of-thought intermediate steps via `recipe_specification.recipes.reasoning_traces = True`.

**When to use**: Debugging adaptation quality, building CoT training data, auditability/compliance, distillation pipelines.

```python
run = client.datasets.run(
    dataset_id,
    column_mapping={"prompt": "instruction", "completion": "response"},
    recipe_specification={"recipes": {"reasoning_traces": True}},
)
```

### Other recipe toggles (compose freely):
- `deduplication` — remove near-duplicate rows
- `preference_pairs` — generate DPO chosen/rejected pairs
- `prompt_rephrase` — rephrase prompts for diversity
- `prompt_metadata_injection` — inject context into prompts

---

## Safety and Length Constraints

Specification-first: length and safety are structural objectives, not post-processing.

### Length
Values: `"minimal"`, `"concise"`, `"detailed"`, `"extensive"`

```python
brand_controls={"length": "concise"}
```

### Safety Categories
```python
brand_controls={"safety_categories": ["harassment", "hate"]}
```
Values come from the API schema. Completions violating selected categories are filtered.

### Combined
```python
brand_controls={
    "length": "detailed",
    "safety_categories": ["harassment", "hate"],
    "hallucination_mitigation": True,
}
```

---

## Processing Large Datasets

Use `job_specification.max_rows` to subsample before committing to a full run.

```python
# Estimate cost on subset
quote = client.datasets.run(
    dataset_id,
    column_mapping={"prompt": "instruction", "completion": "response"},
    job_specification={"max_rows": 500},
    estimate=True,
)

# Run on subset
run = client.datasets.run(
    dataset_id,
    column_mapping={"prompt": "instruction", "completion": "response"},
    job_specification={"max_rows": 500},
)
```

Omit `max_rows` for full dataset processing.

---

## Evaluating Dataset Quality

Evaluation runs on its own schedule after adaptation succeeds.

### Dedicated endpoint
```python
ev = client.datasets.get_evaluation(dataset_id)
# ev.status: pending | running | succeeded | failed | skipped
# ev.quality.score_before  (0-10)
# ev.quality.score_after   (0-10)
# ev.quality.improvement_percent
# ev.quality.percentile_after
```

### Via dataset record
```python
ds = client.datasets.get(dataset_id)
ds.evaluation_summary  # compact mirror of headline metrics
```

### Polling pattern
```python
import time
while True:
    ev = client.datasets.get_evaluation(dataset_id)
    if ev.status in ("succeeded", "failed", "skipped"):
        break
    time.sleep(5)
```

**Note**: `get_status` does NOT include evaluation — use `get` or `get_evaluation`.