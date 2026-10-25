# Generate a Review-Only Invoice Extractor

Create exactly these output artifacts in the workspace: `invoice.py`, `requirements.txt`,
and `design.md`. Target Python 3.10+ and the **unofficial** `clairic==0.1.0` Python SDK.
Do not install dependencies, use environment credentials, or make any network/inference calls.

## Required Code

Use canonical unaliased imports from `clairic`, `pydantic`, `typing`, `datetime`, and optionally
`math`; `from __future__ import annotations` is allowed. Pin `clairic==0.1.0` in
`requirements.txt` (comments allowed; no additional direct dependencies needed).

In `invoice.py`, declare this Pydantic schema, using public SDK imports:

- `class InvoiceExtraction(BaseModel)`
- `invoice_number: TextAnswer`
- `total: Annotated[NumberAnswer[float], Number(unit="USD", decimals=2)]`
- `due_date: DateAnswer` (optionally use `Field(description="...")` for date interpretation).

Add a synchronous `extract_invoice(text)` function. Keep this live-only helper structurally
auditable: its body is a single `with Clairic(timeout=30.0) as client:` block, directly returning
`client.extract(text, InvoiceExtraction, reference_date=date(2026, 10, 26))`.
The constructor defaults read `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN`; do not embed
credentials, override the API root, or add unsupported `images`, `options`, or `stream` arguments.
Do not call this helper. No constructor, extraction, or credential access may occur at module
import. Apart from imports/docstrings, put only the schema and the two required function
definitions at module scope, with imports before the schema and the schema before helpers.
Do not add a CLI, decorators, argument defaults, or executable examples. If annotating function
signatures, use `str`, `Result[InvoiceExtraction]`, and `float | None` as appropriate.

Add a separate, pure `accept_total(result) -> float | None` function. Return `None` if **any** of
these conditions holds:

1. `result.truncated` is true.
2. `result.data.total.value` is `None`.
3. The total's `p_null >= 0.5`.
4. The total's `confidence < 0.9`.
5. The value is non-finite or negative.

Otherwise return the original total `.value`; accept zero, `p_null < 0.5`, and `confidence == 0.9`.
Never substitute `guess` or a fabricated zero. Assume a well-formed SDK result with float total
values and finite probability metadata. These are **synthetic evaluation thresholds**, not a
universally calibrated deployment policy or permission to pay an invoice.

### Accepted Pure-Helper Subset

The grader never imports candidate `invoice.py`. It extracts only `accept_total` and checks
a deliberately restricted AST subset before testing it against synthetic objects. Keep that
helper within: simple local-name assignments, `if`/`else`, `return`, Boolean operations,
comparisons, `not`/numeric unary signs, numeric/None/Boolean constants, and optional docstrings.
Read only `result`, local variables, and the attributes `truncated`, `data`, `total`, `value`,
`p_null`, and `confidence`. The only callable is `isfinite(value)` (from `math`) or
`math.isfinite(value)` (with `import math`). Do not use imports inside the helper, loops,
comprehensions, subscripts, arithmetic, mutation, nested definitions, dynamic calls, dunder
access, decorators, or argument defaults. Annotations are not executed. This is an intentionally
narrow accepted solution profile, not a general Python sandbox.

## Required Design Note

In `design.md`, concisely explain:

- The unofficial Python SDK versus Cloudflare's hosted `@cf/cloudflare/clairic` service;
  SDK 0.1.0 is text-only and does not expose image/options/stream parameters.
- Absence probability (`p_null`) versus confidence conditional on a value existing;
  a high-confidence guess is not an authoritative missing value.
- Why these particular cutoffs are synthetic, and why real thresholds need calibration,
  business validation and review, especially for missing/uncertain/truncated results.
- Authorized live calls transmit data, require account-scoped credentials and are metered;
  retries may repeat billing, preview limits can change, and field splitting into multiple
  passes can multiply input-token use. SDK timeouts are not a separate total wall-clock cap.
- Typed extraction is not arithmetic verification or payment authorization: ordinary code
  owns calculations and business rules; consequential payment decisions require independent
  checks and appropriate approval.
- Exactly what offline AST/synthetic tests establish, versus what requires a separately
  authorized live check (accuracy/calibration, token permissions, limits, latency, billing).
  Do not claim a live test, local model weights, automatic retries, or production readiness.
