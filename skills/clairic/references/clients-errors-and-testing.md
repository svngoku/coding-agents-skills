# Clients, Errors, and Testing

Contract: unofficial Python `clairic==0.1.0`, checked 2026-10-26.
Authority: the pinned client, exceptions, and offline upstream tests linked below.
These SDK details are not guarantees about other Cloudflare clients.

## Authentication and Construction

Both `Clairic` and `AsyncClairic` accept keyword-only:

| Parameter | Default / meaning |
| --- | --- |
| `account_id` | `CLOUDFLARE_ACCOUNT_ID` fallback |
| `api_token` | `CLOUDFLARE_API_TOKEN` fallback |
| `base_url` | `https://api.cloudflare.com/client/v4`, API root, not full run URL |
| `timeout` | `30.0`, float or `httpx.Timeout` |
| `http_client` | Optional matching HTTPX sync/async client |

Missing credentials raise `ClairicConfigurationError`. Falsy explicit credential
strings also fall back to the environment. `.env` loading is not automatic.
The official model sample's `CLOUDFLARE_AUTH_TOKEN` name must be mapped explicitly
if used with this wrapper.

Real calls send bearer credentials to `base_url`. Override it only for a trusted
HTTPS API root/proxy; this SDK has no destination allowlist. Never log bearer
headers, credentials, or sensitive response/error bodies indiscriminately.

## Extraction and Resource Ownership

```python
from datetime import date
from pydantic import BaseModel
from clairic import AsyncClairic, Clairic, Result, TextAnswer


class Extraction(BaseModel):
    reference: TextAnswer


def extract_text(text: str) -> Result[Extraction]:
    # Real, metered call when this function is invoked.
    with Clairic(timeout=30.0) as client:
        return client.extract(text, Extraction, reference_date=date(2026, 10, 26))


async def extract_text_async(text: str) -> Result[Extraction]:
    async with AsyncClairic(timeout=30.0) as client:
        return await client.extract(
            text, Extraction, reference_date=date(2026, 10, 26)
        )
```

`extract(state, model, *, reference_date=None)` takes nonempty text and an
extraction **class**. No image/options/model-ID override or stream/batch method
is exposed. The client makes one POST to
`/accounts/{account_id}/ai/run/@cf/cloudflare/clairic`, with body `model="clairic"`,
`state`, compiled `fields`, and optional ISO `reference_date`.

Use `close()` / `aclose()` or context managers. An SDK-created HTTP client is
closed by the SDK; an injected client remains caller-owned. With injection,
configure timeout on the HTTPX client: the SDK's `timeout=` is ignored.
HTTPX timeout settings do not establish a separate total wall-clock deadline.
Bound concurrency explicitly for async workloads.

## Exceptions and Truncation

| Type | Actual SDK behavior |
| --- | --- |
| `ClairicConfigurationError` | Missing credentials at construction |
| `ModelDefinitionError` | Unsupported extraction declaration before POST |
| `AuthenticationError(APIError)` | HTTP 401 with a parseable envelope |
| `InvalidRequestError(APIError)` | HTTP 400 / 422 |
| `ContextWindowExceededError(APIError)` | HTTP 413 |
| `APIError` | Other HTTP/envelope failures, including 403/429/5xx; malformed/incompatible responses |
| `APIConnectionError` | HTTPX transport failure |
| `APITimeoutError(APIConnectionError)` | HTTPX timeout |
| `ClairicTruncationWarning` | Returned `truncated=True`; flag remains on result |

All SDK exceptions derive from `ClairicError`. `APIError` has `message`,
`status_code`, optional `code`, and optional `request_id`. Transport errors do
not promise those API metadata attributes. A successful HTTP status with
`success: false` is still an error. Malformed response parsing can happen before
status-specific mapping; error messages may contain returned values.

Treat truncation as a review condition. Where incomplete extraction must fail,
use a scoped warning policy:

```python
import warnings
from clairic import ClairicTruncationWarning


def strict_extract(client, text, extraction_model):
    with warnings.catch_warnings():
        warnings.simplefilter("error", ClairicTruncationWarning)
        return client.extract(text, extraction_model)
```

There are **no automatic retries**. Do not retry invalid models/auth blindly.
For application-owned transient retries, cap attempts, use backoff/jitter, honor
available server guidance, and budget for repeated inference. Do not assume
timeouts mean the service did not execute or that retries are idempotent/free.

## Executable Offline Checks

From the repository root, use an isolated environment if needed:

```bash
python3 -m venv .venv-clairic
.venv-clairic/bin/python -m pip install "clairic==0.1.0"
.venv-clairic/bin/python skills/clairic/scripts/offline_checks.py
python3 -m unittest discover -s skills/clairic/tests -v
```

Keep the environment out of version control. Dependency installation contacts
the package registry; the checks themselves must not call Cloudflare.

[The check script](../scripts/offline_checks.py) injects HTTPX `MockTransport`,
uses explicit dummy credentials, and verifies synthetic request/response behavior.
This is the basic public-API pattern:

```python
import httpx
from pydantic import BaseModel
from clairic import Clairic, TextAnswer


class Extraction(BaseModel):
    label: TextAnswer


def respond(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={
        "success": True, "errors": [], "messages": [],
        "result": {
            "model": "clairic", "truncated": False,
            "answers": {"label": {
                "type": "text", "value": "synthetic", "guess": "synthetic",
                "p_null": 0.01, "confidence": 0.99,
            }},
            "usage": {
                "state_tokens": 5, "image_tokens": 0, "schema_tokens": 10,
                "input_tokens": 15, "output_tokens": 0, "passes": 1,
            },
        },
    })


with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
    with Clairic(
        account_id="offline-account",
        api_token="offline-not-a-real-token",
        http_client=transport,
    ) as client:
        result = client.extract("synthetic", Extraction)
        assert result.data.label.value == "synthetic"
    assert not transport.is_closed  # injected transport is caller-owned
```

Dummy strings are test fixtures, not real credentials. Do not replace the mock
with a default transport in an offline test. Assert predictions only against
chosen fixtures, never fixed confidence scores for future live calls.

## Evaluation Boundaries

The skillgrade task checks generated SDK integration code and a synthetic review
policy; its standard-library grader is separate from these installed-SDK checks.
Run `./scripts/run-evals.sh clairic --validate` after harness changes. A passing
deterministic reference check is not a passing LLM rubric or agent benchmark.

If skillgrade's local provider cannot copy/execute the reference or the rubric
provider lacks credentials, report the full validation as blocked. Do not weaken
the grader or claim missing rubric results passed. The executable reference and
grader can also be run directly in a temporary workspace for an offline check.

Validation snapshot **2026-10-26**: the direct reference scored **1.0**. Full
validation with skillgrade **0.5.0** was **blocked**, not passed: the local
provider ran `bash reference-invoice` without copying that script into its
workspace; the LLM rubric also reported missing `GEMINI_API_KEY`. The harness
and grader were not weakened to hide these failures.

From the repository root, reproduce the direct deterministic check:

```python
from pathlib import Path
import subprocess
import sys
import tempfile

skill = Path.cwd() / "skills" / "clairic"
with tempfile.TemporaryDirectory(prefix="clairic-reference-") as workspace:
    subprocess.run(
        [str(skill / "solutions" / "reference-invoice")],
        cwd=workspace, check=True,
    )
    subprocess.run(
        [sys.executable, "-I", "-S", str(skill / "graders" / "check.py")],
        cwd=workspace, check=True,
    )
```

Upstream defaults exclude tests marked `live`. Do not select `-m live` in offline
validation: an invalid-token live test can still send a request without real
credentials. The upstream schema-drift script is a public-document network
check, not offline testing and not inference.

Mocks establish request wiring, SDK parsing, lifecycle, and selected local
validation. They do **not** establish extraction accuracy, actual service
permissions, billing, limits, latency, or confidence calibration.

## Sources

- [Pinned client](https://github.com/VianneyMI/clairic/blob/d4b0f3390a05e2310f0fc033803e65658abfcfae/src/clairic/_client.py)
- [Exceptions](https://github.com/VianneyMI/clairic/blob/d4b0f3390a05e2310f0fc033803e65658abfcfae/src/clairic/errors.py)
- [Offline upstream tests](https://github.com/VianneyMI/clairic/blob/d4b0f3390a05e2310f0fc033803e65658abfcfae/tests/test_client.py)
