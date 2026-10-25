# Operations, Safety, and Documentation Drift

Snapshot: 2026-10-25. Refresh provider documentation before production decisions.
This reference is not a security certification, legal opinion, or replacement SDK.

## Credentials and submitted data

- Keep `TYPESAFE_API_KEY` in approved server-side secret management.
- Do not put keys in request state, source, fixtures, shell history, or logs.
- Do not print settings files or secret-bearing objects to prove a key exists.
  Verify presence/authentication without exposing the value.
- Browser opt-in such as `dangerouslyAllowBrowser` does not protect an API key.
- SDK debug logs can contain request/response bodies even when authorization
  headers are redacted. Minimize data before constructing those bodies.
- Do not automatically submit private session archives, repositories, customer
  records, or unnecessary personal information.
- Treat model output as untrusted data for policy purposes. Semantic injection
  detection does not replace hard authorization and tool boundaries.

If an accidental secret disclosure is found, stop reproducing it, report only
the affected location/type, and follow the user's approved incident process.
Do not rotate keys or change configuration without authorization.

Sources: [Python advanced usage](https://docs.typesafe.ai/sdk/python/usage/advanced.md),
[JavaScript SDK](https://docs.typesafe.ai/sdk/javascript.md).

## Models, cost, and context

The researched models page lists `jev-1.14.0` and `jev-1.13.0`.
At the snapshot, both `jev-latest` and `jev-preview` resolve to 1.14.0.
An alias is not a stable evaluation target. Record the returned version and pin
a supported version after tuning thresholds.

Snapshot pricing is $0.042 per million input tokens with free output tokens.
Verify current pricing before budgeting; old cookbook output is not a quote.

Default context is documented as 64k per request and 32k per question.
`TypeSafe-Danger-Zone-Long-Context: true` increases both to 262k for the documented
1.13/1.14 models, with slower and less accurate behavior. Do not enable it by
default to avoid reducing state.

No separate question-count limit is documented, but context/request/rate limits
still apply. Extra questions consume tokens. Rate limits are organization-specific
requests/sec and tokens/sec. English is documented as strongest; evaluate other
languages independently.

Sources: [models](https://docs.typesafe.ai/models.md),
[request rules](https://docs.typesafe.ai/api/errors.md).

## Request and JSON rules

The researched rules include:

- 32 MiB compressed **and** decompressed request limits.
- UTF-8 without BOM, maximum nesting depth 64, finite JSON numbers.
- Preserve large exact identifiers as strings rather than risky numeric values.
- One compression layer: gzip/x-gzip or zstd; zstd window at most 8 MiB.
- Unsupported or multiple encodings produce 415; malformed compression produces 400.

Do not confuse JSON schema acceptance with semantic suitability. Minimize state
before considering long-context overrides or compression.

## Errors and retries

| Condition | Documented HTTP status | Response |
| --- | --- | --- |
| Request/context rules, including `max_tokens_exceeded` | 400 | Reduce/fix the actual request |
| Authentication | 401 | Check approved key configuration without printing it |
| Request too large | 413 | Reduce payload; inspect expanded assets |
| Unsupported encoding | 415 | Fix compression/gateway compatibility |
| Schema validation | 422 | Fix interface-specific question/request shape |
| Rate limiting | 429 | Honor bounded retry policy and budgets |
| Overload | 529 | Use bounded retry/backoff; report sustained failure |

Do not repeatedly resend an invalid schema or rejected credential unchanged.
Never bypass tool permissions or increase autonomy automatically.

Both official SDKs document two retries after the initial attempt for connection/
timeout failures and 408/429/500–599 responses. Initial backoff is 500 ms, doubles
up to 5 s, subtracts jitter up to 25%, and honors retry headers.

| Surface | Timeout/retry distinction |
| --- | --- |
| Python | Seconds; default HTTP-operation timeout 10 s; total retry budget 30 s |
| JavaScript | Milliseconds; default 10,000 ms per attempt; no equivalent total budget |
| JavaScript retry headers | Delay capped at 60,000 ms before fallback behavior |
| Local MCP | Its own code/defaults; do not assume either SDK policy |

SDKs can zstd-compress requests of at least 1 KiB when runtime support exists.
Disable compression if an approved gateway cannot handle it.
Cancellation and exhausted retries are not successful judgments.

Preserve safe error class/status/request ID for diagnostics. Do not dump raw
request/response bodies or repeatedly log potentially sensitive error details.

Sources: [errors](https://docs.typesafe.ai/api/errors.md),
[Python retries](https://docs.typesafe.ai/sdk/python/api/retries.md),
[JavaScript retry policy](https://docs.typesafe.ai/sdk/javascript/api/interfaces/RetryPolicy.md).

## Privacy and legal boundaries

**No training is not zero retention.** First-party agreements/help describe
processing for service delivery, billing, abuse prevention, legal obligations,
and telemetry. No single fixed default retention duration is established by
the reviewed pages. Enterprise zero-data-retention requires agreement.

Do not promise data residency, a BAA, independently verified compliance, or
access to an auth-gated report based on public marketing pages. The SOC 2 Type II
report itself was not reviewed in this research.

Provider restrictions on distillation/imitation deserve legal/provider
clarification before downstream-learning uses. Do not resolve that legal question
through a model probability or assume a cookbook overrides an agreement.

Sources: [MCA](https://typesafe.ai/legal/mca),
[DPA](https://typesafe.ai/legal/data-processing),
[privacy](https://typesafe.ai/legal/privacy-policy),
[retention help](https://help.typesafe.ai/articles/5974925716-how-is-submitted-data-used-or-retained).

## Known documentation contradictions and gaps

Prefer the chosen interface's current schema/types for transport details, the
dedicated primitive/confidence specification for semantics, and measured behavior
for completion claims. Record unresolved contradictions rather than hiding them.

| Observed drift | Safe handling |
| --- | --- |
| Older text-only descriptions vs newer experimental vision docs | Isolate explicit images/Box opt-in |
| HTTP Score permits one level; JavaScript requires two | Use 2–10 non-null levels portably |
| Advanced/JS nullable Score criteria vs HTTP schema | Avoid null level descriptions in portable examples |
| Box confidence prose vs nullable schema/SDK | Handle null |
| OpenAPI 16x16 and Python 64x64 image examples vs minimum 280 per side | Validate real image constraints |
| Quickstart Node 20 vs SDK Node 22+ | Follow current SDK requirement |
| Python basic guide has a future 2026-11-14 timestamp | Do not infer release status from that timestamp |
| Cookbook claims Choice-tail sensitivity absent from confidence formula | Prefer dedicated formula; retain full distribution |
| Hierarchy prose batching vs separate concurrent requests in code | Measure actual scheduling |
| Historical cached benchmark/cost claims | Label historical; do not claim live verification |
| Old `/migrating-to-v1.md` link returns 404 | Use live index, SDK changelog/types; do not retain it as a working guide |
| One-option Choice vs confidence formula undefined at n=1 | Avoid relying on unspecified confidence behavior |

Use the [source ledger](sources.md) to refresh the relevant pages. Do not
automatically reinstall an upstream skill over local customizations.
