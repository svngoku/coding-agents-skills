---
name: clairic
description: >
  Build typed extraction with the unofficial VianneyMI/clairic Python SDK for
  Cloudflare's Clairic model. Use for Clairic, AsyncClairic, Pydantic answer
  models, annotated field specs, null/confidence handling, estimates, or
  forecasts. Also trigger for Workers AI Clairic REST/bindings, SDK-versus-service
  differences, and offline MockTransport tests.
---

# Clairic — Unofficial Python SDK

Extract typed values and uncertainty metadata with `clairic`, the **unofficial**
Python SDK maintained at `VianneyMI/clairic`, not by Cloudflare. The hosted model
is Cloudflare Workers AI's `@cf/cloudflare/clairic`. This skill covers SDK
**0.1.0**, inspected at `d4b0f3390a05e2310f0fc033803e65658abfcfae` on
**2026-10-26**. The service is in preview; refresh its contract before rollout.

## Quick Reference

| Need | Read |
| --- | --- |
| All nine field types, specs, aliases, uncertainty, export/storage | [Typed fields and results](references/typed-fields-and-results.md) |
| Auth, sync/async clients, errors, timeouts, offline checks | [Clients, errors, and testing](references/clients-errors-and-testing.md) |
| Official REST/Workers binding, images/options, billing and limits | [Cloudflare API and limits](references/cloudflare-api-and-limits.md) |
| Pinned sources, ownership, known discrepancies, verification scope | [Sources and compatibility](references/sources-and-compatibility.md) |
| Executable synthetic SDK contract checks, no inference | [Offline checks](scripts/offline_checks.py) |

## Core Workflow

### 1. Choose the Correct Surface

- Use this SDK for **text extraction in Python 3.10+**, with Pydantic 2 models.
- Use the official REST API or Workers AI binding for images or
  `options.include_quantiles` / `options.return_state_text`. SDK 0.1.0 does
  **not** expose these arguments, even though its answer types can parse quantiles.
- A Workers binding is not a JavaScript edition of this SDK. The Python client
  has no chat, embeddings, streaming, batch, or general decision-question API.
- Extract evidence first. Do exact arithmetic, policy, authorization, and actions
  in ordinary code. Typed output is not proof that the values are correct.

### 2. Pin and Configure the SDK

```bash
python -m pip install "clairic==0.1.0"
```

`Clairic()` reads `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN`, or accepts
keyword-only `account_id=` / `api_token=`. It does not load `.env` automatically.
The official model page's `CLOUDFLARE_AUTH_TOKEN` sample variable is **not** an SDK
alias. Keep tokens server-side, out of source, browser code, and logs.

### 3. Declare Answer Models, Not Bare Value Models

```python
from datetime import date
from typing import Annotated

from pydantic import BaseModel, Field
from clairic import Clairic, DateAnswer, Number, NumberAnswer, Result, TextAnswer


class InvoiceExtraction(BaseModel):
    invoice_number: TextAnswer
    total: Annotated[NumberAnswer[float], Number(unit="USD", decimals=2)]
    due_date: DateAnswer = Field(description="The invoice payment due date")


def extract_invoice(text: str, reference_day: date) -> Result[InvoiceExtraction]:
    # Calling this function performs real, metered Cloudflare inference.
    with Clairic(timeout=30.0) as client:
        result = client.extract(
            text,
            InvoiceExtraction,
            reference_date=reference_day,
        )
    if result.truncated:
        raise ValueError("Incomplete extraction requires review")
    return result
```

The `model` argument to `extract()` is the **Pydantic class**, not a model ID.
Use `Field(description=...)` for field instructions, `Annotated` for matching
specs, and keyword arguments for all specs. `reference_date` anchors relative
dates; establish the intended date with the caller rather than silently guessing.

Do not use `Optional[TextAnswer]`, bare `str` / `float`, arbitrary nested models,
or `list[TextAnswer]` as extraction fields. Scalar answer `.value` is already
nullable. Use `TextListAnswer` or parameterized `ListAnswer` for lists.

### 4. Handle Absence, Uncertainty, and Incomplete Input Separately

For scalar answers, inspect:

- `.value`: the extracted value, possibly `None`.
- `.p_null`: probability that no appropriate value exists.
- `.confidence`: correctness conditional on a value existing.
- `.guess`: a candidate, **not** permission to replace a missing value.

`null_threshold` changes the service's absence decision, not a confidence
threshold. Never silently use `answer.value or answer.guess`: it mishandles both
missing values and valid zero values. Check `is None` explicitly.

Estimates and forecasts have distributions/intervals, not scalar confidence or
`p_null`. A p10–p90 interval is the model's central 80% interval, not guaranteed
real-world coverage. Calibrate acceptance/review rules on representative data;
keep arithmetic, payment approval, and other consequential actions outside the
model. A non-null amount alone is not an approved payment.

### 5. Export Values or Preserve the Full Result Deliberately

```python
class InvoiceValues(BaseModel):
    invoice_number: str | None
    total: float | None
    due_date: date | None


def export_invoice(result: Result[InvoiceExtraction]) -> InvoiceValues:
    if result.truncated:
        raise ValueError("Incomplete extraction requires review")
    return result.values_as(InvoiceValues)
```

`values_as()` uses matching **Python attribute names** and runs target Pydantic
validation. Nonnullable target fields can reject a missing value. Exporting
values discards uncertainty metadata; apply review policy before acting on them.

For storage, use `result.model_dump_json()` and
`Result[InvoiceExtraction].model_validate_json(stored_json)`. A raw Cloudflare
REST envelope is **not** a serialized SDK `Result`.

### 6. Reuse Clients and Make Reliability Policy Explicit

Use `with Clairic(...)` or `async with AsyncClairic(...)`; await the async
`extract()`. Reuse a client across related calls. The default HTTPX timeout is
30 seconds, not a separate total deadline. A supplied HTTP client owns its own
timeout and must be closed by its caller.

The SDK has no automatic retries. Handle configuration/model errors separately
from API/transport failures; cap any application retries and budget for repeat
inference. HTTP 403 and 429 use generic `APIError`, not special permission or
rate-limit exception classes. See the client reference for the full mapping.

### 7. Verify Offline Before Authorized Live Testing

```bash
python skills/clairic/scripts/offline_checks.py
python3 -m unittest discover -s skills/clairic/tests -v
./scripts/run-evals.sh clairic --validate
```

The first command requires the pinned SDK and uses only synthetic HTTPX
`MockTransport` responses with dummy credentials. The grader tests use the Python
standard library. The skillgrade reference-validation command is separate;
environment/tooling failures must be reported, not called passing evaluations.

Do not run upstream live tests to “check imports.” Even their invalid-token test
can send a network request without real credentials. Wrangler local development
can still invoke the hosted model and incur usage. Offline checks do not measure
model accuracy, calibration, real billing, or live permissions.

## Anti-Patterns to Avoid

- Calling this the official Cloudflare SDK, or inventing a JavaScript SDK/CLI.
- Passing an image, `stream=True`, `options=`, or a model ID to SDK `extract()`.
- Treating nullable values, guesses, confidence, and `p_null` as interchangeable.
- Using `null_threshold` as a confidence cutoff or hardcoding universal policy thresholds.
- Declaring an integer estimate without a whole-number `step`.
- Using a raw REST envelope as `Result[MyModel]` JSON or remapping aliases implicitly.
- Assuming one request means one billed pass, free retries, or guaranteed latency.
- Logging response/error bodies indiscriminately; they may contain extracted data.
- Promising self-hosted weights, privacy guarantees, or unsupported API compatibility.

## When to Use / Not Use

Use for Python Clairic extraction, field/schema design, uncertainty-aware result
handling, SDK troubleshooting, and selecting between the SDK and official API.

Do not use as a general chat/agent framework, a PDF/document-file loader, a
TypeSafe/Jev primitive adapter, an exact arithmetic engine, or an authorization
system. Images require the official surface, not invented Python SDK support.
