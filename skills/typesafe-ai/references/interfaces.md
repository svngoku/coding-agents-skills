# Interfaces and Contracts

Snapshot: 2026-10-25. These examples are documentation, not evidence that a local
SDK or network integration was executed. Re-read the chosen interface's docs and
installed types before implementation. Model examples intentionally pin the
then-current `jev-1.14.0`; refresh availability and re-evaluate thresholds on upgrades.

## Choose the actual surface

| Surface | Request/response distinction |
| --- | --- |
| HTTP | Explicit required `model`, JSON state/questions; structured API response |
| Python | `system_one`; constructor or request model; typed answers; seconds |
| JavaScript | `systemOne`; `defaultModel` or request model; milliseconds |
| Local MCP | Discover its schema; wrapper defaults and text envelopes can differ |

The official indexed docs do not define an MCP `judge` contract. The example
wrapper inspected during this research supports Noul/Choice/Score, requires
instructions, and defaults an omitted model. This is a local observation, not a
guarantee about another installation. It does not expose public API assets/Box.

Load the real tool if deferred. Inspect its required fields and return format.
Do not guess namespaces, send SDK options to it, or modify the server to make an
unsupported example work.

## HTTP

`POST https://api.typesafe.ai/v1/systemone`, bearer authentication, JSON UTF-8.
Keep the key server-side and outside request bodies, source files, and logs.

```json
{
  "model": "jev-1.14.0",
  "state": {"message": "Please refund the duplicate charge."},
  "questions": {
    "billing": {
      "type": "noul",
      "instructions": "Is `message` about a payment or billing issue?"
    }
  }
}
```

Required top-level fields are `model`, `state`, and nonempty `questions`.
Optional `assets` belongs to explicitly opted-in image workflows.
State accepts a string, object, or array, not a top-level scalar number, boolean,
or null.

| Primitive | Current HTTP shape |
| --- | --- |
| Choice | `type: "choice"` and criteria map; instructions optional |
| Score | `type: "score"` and ordered criteria array; instructions optional |
| Noul | `type: "noul"`; optional instructions and `{true, false}` criteria |
| Box | `type: "box"`; optional instructions; no criteria |

Even where instructions are optional in the schema, provide complete model-visible
meaning. The local MCP's required instructions are not an HTTP-wide requirement.

HTTP Choice permits 1–255 options. HTTP Score permits 1–10 levels, while JavaScript
requires at least two; nullable Score criteria also differ between sources.
Use 2–10 non-null descriptions for portable integrations.

A synthetic response:

```json
{
  "model": "jev-1.14.0",
  "answers": {"billing": {"type": "noul", "noul": 0.98}},
  "usage": {"input_tokens": 120, "output_tokens": 12}
}
```

HTTP Score legend/probability keys are strings. HTTP usage counts are required
integers; SDK types can allow null counts. Preserve those distinctions.

Sources: [System One API](https://docs.typesafe.ai/api/system-one.md),
[OpenAPI](https://api.typesafe.ai/openapi.json).

## Python

Package: `typesafe-sdk`; documented Python requirement: **3.10+**.
The reviewed changelog's latest entry is 0.7.5, dated 2026-10-14.

Prerequisite for optional live use: the official SDK installed in the user's
chosen environment and `TYPESAFE_API_KEY` supplied through approved secret
management. Do not read or print the key to verify its presence.

```python
from typesafe_sdk import Noul, TypeSafeClient

with TypeSafeClient(model="jev-1.14.0") as client:
    result = client.system_one(
        state={"message": "Please refund the duplicate charge."},
        questions={
            "billing": Noul(
                instructions="Is `message` about a payment or billing issue?"
            )
        },
    )
    probability = result.answers["billing"].noul
```

The documented method accepts state and questions, with keyword options including
`model`, `assets`, `retry`, `timeout`, `extra_headers`, `extra_body`, and
`response_model`. `AsyncTypeSafeClient` provides the awaited equivalent.

- Reuse clients and close them with context managers.
- Closing also closes a supplied HTTP client; decide who owns it.
- Access `.answers` or filtered `.choices`, `.scores`, `.nouls`, `.boxes`.
- Python converts Score level keys to integers.
- Preserve `request_id` and `raw_http_response` when troubleshooting safely.
- Timeouts are in seconds, unlike JavaScript's milliseconds.
- `extra_body` is a last-write-wins shallow merge that can override core fields.
  It does not make arbitrary parameters supported API features.
- Custom Pydantic response models are supported; they do not change server semantics.

Sources: [Python SDK](https://docs.typesafe.ai/sdk/python.md),
[sync client](https://docs.typesafe.ai/sdk/python/api/clients/sync.md),
[async client](https://docs.typesafe.ai/sdk/python/api/clients/async.md),
[responses](https://docs.typesafe.ai/sdk/python/api/types/responses.md).

## JavaScript and TypeScript

Package: `@typesafe-ai/sdk`; documented version: 0.7.0; **Node 22+**.
Prefer the SDK requirement over the older quickstart's conflicting Node 20 claim.
Browser usage is disabled by default. `dangerouslyAllowBrowser` is not a production
credential-management solution.

```typescript
import { TypeSafeClient, noul } from "@typesafe-ai/sdk";

const client = new TypeSafeClient({ defaultModel: "jev-1.14.0" });
const result = await client.systemOne({
  state: { message: "Please refund the duplicate charge." },
  questions: {
    billing: noul("Is `message` about a payment or billing issue?"),
  },
});
const probability = result.answers.billing.noul;
```

Request options are separate:

```typescript
client.systemOne(
  { state, questions, model, assets },
  { headers, retry, timeout, signal },
);
```

The second snippet illustrates parameter positions, not a self-contained program.
Do not copy undefined example variables as a working integration.

- Helpers are `choice(instructions, criteria)`, `score(instructions, criteria)`,
  `noul(instructions, criteria?)`, and experimental `box(instructions?)`.
- Type inference preserves answer keys and Choice labels.
- `.withResponse()` returns `{data, response, requestId}`.
- `.asResponse()` gives the raw response; do not consume its body twice.
- Invalid requests reject the returned promise; constructors can throw.
- Timeout values are milliseconds.
- `client.models.list()` returns an array; Python
  `client.models.list().models` and HTTP `{models: [...]}` differ.

Sources: [JavaScript SDK](https://docs.typesafe.ai/sdk/javascript.md),
[client](https://docs.typesafe.ai/sdk/javascript/api/classes/TypeSafeClient.md),
[API promise](https://docs.typesafe.ai/sdk/javascript/api/classes/APIPromise.md).

## Unwrap and validate deliberately

The semantic API does not generate free-form JSON prose. An MCP transport may
nevertheless serialize the API object inside text content. Distinguish:

1. Transport/tool failure, including an MCP `isError` result.
2. The wrapper's text or content envelope.
3. The parsed API result's `model`, `answers`, and `usage`.
4. Each expected answer's primitive-specific fields.

Reject missing/mismatched answers before downstream execution. Check a Choice
against known labels and keep its distribution. Keep Score legends and direction.
Accept uncertainty as valid data rather than treating it as malformed output.

Use official SDK validation for production. The skill's offline checker tests
only its synthetic text-example profile and does not implement the complete
HTTP, SDK, MCP transport, or experimental image contracts.
