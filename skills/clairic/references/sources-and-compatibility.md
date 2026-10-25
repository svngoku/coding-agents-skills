# Sources and Compatibility

Research snapshot: **2026-10-26**. The exact user-provided
[SDK repository](https://github.com/VianneyMI/clairic) and
[Cloudflare model page](https://developers.cloudflare.com/workers-ai/models/clairic/)
were both accessible (HTTP 200). Linked source, tests, schemas, package metadata,
and platform documentation were cross-checked, not just README examples.

## Ownership and Version

- `VianneyMI/clairic` is an **unofficial Python SDK**, not affiliated with,
  endorsed by, or supported by Cloudflare.
- SDK **0.1.0**, tag `v0.1.0`, inspected commit
  `d4b0f3390a05e2310f0fc033803e65658abfcfae`, released 2026-10-25.
- Python **3.10+**; dependencies `httpx>=0.27`, `pydantic>=2.7`,
  `typing-extensions>=4.12`. The repository does not publish a JavaScript SDK
  or a CLI entrypoint.
- The SDK is MIT-licensed. Its NOTICE separately credits Cloudflare documentation
  under CC BY 4.0. Neither the SDK license nor this repository's license grants
  rights to model weights or overrides the hosted service's terms.
- Official hosted model: `@cf/cloudflare/clairic`, **preview**. Open weights were
  described as planned, not as already released.

## Source Inventory

All sources below were checked on 2026-10-26. SDK source links are pinned;
official platform pages and PyPI metadata are moving sources.

| Source | What it establishes |
| --- | --- |
| [SDK release](https://github.com/VianneyMI/clairic/releases/tag/v0.1.0) | Version, release date, commit |
| [PyPI metadata](https://pypi.org/pypi/clairic/json) | Distribution, dependencies, release artifact hashes |
| [Package metadata](https://github.com/VianneyMI/clairic/blob/d4b0f3390a05e2310f0fc033803e65658abfcfae/pyproject.toml) | Python floor, dependencies, offline/live test defaults |
| [Exports](https://github.com/VianneyMI/clairic/blob/d4b0f3390a05e2310f0fc033803e65658abfcfae/src/clairic/__init__.py) | Public imports; private compiler helpers are not public API |
| [Client](https://github.com/VianneyMI/clairic/blob/d4b0f3390a05e2310f0fc033803e65658abfcfae/src/clairic/_client.py) | Auth, HTTP protocol, errors, timeout and ownership |
| [Compiler](https://github.com/VianneyMI/clairic/blob/d4b0f3390a05e2310f0fc033803e65658abfcfae/src/clairic/_compile.py) | Model declarations, aliases, local validation |
| [Specs](https://github.com/VianneyMI/clairic/blob/d4b0f3390a05e2310f0fc033803e65658abfcfae/src/clairic/specs.py) | Keyword-only field options, bounds and limits |
| [Answers](https://github.com/VianneyMI/clairic/blob/d4b0f3390a05e2310f0fc033803e65658abfcfae/src/clairic/answers.py) | Scalar/list/distribution shapes and semantics |
| [Result](https://github.com/VianneyMI/clairic/blob/d4b0f3390a05e2310f0fc033803e65658abfcfae/src/clairic/result.py) | Usage, values export, JSON round trips |
| [Exceptions](https://github.com/VianneyMI/clairic/blob/d4b0f3390a05e2310f0fc033803e65658abfcfae/src/clairic/errors.py) | Error hierarchy and truncation warning |
| [Client tests](https://github.com/VianneyMI/clairic/blob/d4b0f3390a05e2310f0fc033803e65658abfcfae/tests/test_client.py) | MockTransport, status mapping, transport ownership |
| [Compiler tests](https://github.com/VianneyMI/clairic/blob/d4b0f3390a05e2310f0fc033803e65658abfcfae/tests/test_compile.py) | All field mappings and declaration failures |
| [Spec tests](https://github.com/VianneyMI/clairic/blob/d4b0f3390a05e2310f0fc033803e65658abfcfae/tests/test_specs.py) | Option validation and observed API differences |
| [Answer tests](https://github.com/VianneyMI/clairic/blob/d4b0f3390a05e2310f0fc033803e65658abfcfae/tests/test_answers.py) | Recorded result types, nulls, spans, quantiles |
| [Result tests](https://github.com/VianneyMI/clairic/blob/d4b0f3390a05e2310f0fc033803e65658abfcfae/tests/test_result.py) | Typed storage/export behavior |
| [Live tests](https://github.com/VianneyMI/clairic/blob/d4b0f3390a05e2310f0fc033803e65658abfcfae/tests/live/test_live.py) | Network-dependent tests, including an invalid-token request |
| [SDK license](https://github.com/VianneyMI/clairic/blob/d4b0f3390a05e2310f0fc033803e65658abfcfae/LICENSE) and [NOTICE](https://github.com/VianneyMI/clairic/blob/d4b0f3390a05e2310f0fc033803e65658abfcfae/NOTICE) | Ownership and licensing distinctions |
| [Official model page](https://developers.cloudflare.com/workers-ai/models/clairic/) | Model ID, preview status, model-level behavior |
| [Input schema](https://developers.cloudflare.com/workers-ai/models/clairic/schema-input.json) and [output schema](https://developers.cloudflare.com/workers-ai/models/clairic/schema-output.json) | Raw service request/result, images and options |
| [Extraction concepts](https://developers.cloudflare.com/workers-ai/concepts/extraction-models/) | Evidence, absence, confidence, computation/forecast caveats |
| [Bindings](https://developers.cloudflare.com/workers-ai/configuration/bindings/) and [REST setup](https://developers.cloudflare.com/workers-ai/get-started/rest-api/) | Official platform invocation and response envelopes |
| [Run endpoint](https://developers.cloudflare.com/api/resources/ai/methods/run/) | Bearer auth and Workers AI permission documentation |
| [Pricing](https://developers.cloudflare.com/workers-ai/platform/pricing/) and [limits](https://developers.cloudflare.com/workers-ai/platform/limits/) | Metering, free allowance, preview/local-mode caveats |
| [Launch changelog](https://developers.cloudflare.com/changelog/post/2026-10-19-clairic-workers-ai/) and [Cloudflare article](https://blog.cloudflare.com/clairic-extraction-model/) | Supplementary examples, limitations, planned weights |

## Discrepancies to Preserve

1. **Source signatures beat shorthand tables.** Specs are keyword-only. The
   README's positional-looking option tables are not callable syntax.
2. **Credential names differ.** Official sample `CLOUDFLARE_AUTH_TOKEN` is not
   read automatically by the SDK; it reads `CLOUDFLARE_API_TOKEN`.
3. **Service support is not wrapper support.** Images, output options, and
   returned `state_text` are not exposed by SDK 0.1.0. Quantile result fields
   do not imply a quantile request parameter.
4. **One call is not one pass.** A client docstring's single-pass/typical-latency
   wording does not override official multi-pass metering or establish an SLA.
5. **Published schemas are not exhaustive SDK contracts.** Usage and some
   answer shapes have stricter SDK requirements. Raw schemas are permissive in
   places; some observed upstream acceptance exceeds their descriptions.
6. **Include the input model key.** Launch examples sometimes omit it; the
   inspected input schema marks it required and the SDK includes it.
7. **Example predictions differ.** README, official examples, and recorded
   fixtures show different values/percentiles. They are examples, not fixed
   outputs to assert against live calls.
8. **No automatic dotenv, retries, batching, streaming, or token preflight.**
   Do not turn convenience examples or generic Workers AI features into SDK APIs.

## Verification Scope

Research used static source/test inspection and compared published versus
vendored input/output JSON schemas (equal at the snapshot). Upstream reports of
passing live tests/API probes are upstream claims, not independent execution.

The bundled script and grader tests provide reproducible **offline** checks.
They do not prove hosted extraction quality, actual quotas, cost, calibration,
credentials, or model availability. Full skillgrade validation and its LLM rubric
are separate from direct deterministic reference validation. Report missing
provider credentials/tooling failures honestly.

Refresh this ledger when upgrading the SDK or relying on changed preview
features. Recheck the compiler, client, types, schemas, pricing, and tests
together; do not update a version string alone.
