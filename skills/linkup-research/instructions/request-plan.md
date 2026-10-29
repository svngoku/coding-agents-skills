# Bounded Asynchronous Research Plan

Plan a primary-source comparison of **2024 cloud revenue growth for Microsoft, Amazon, and
Google**. The user explicitly selects `mode: research`, the bounded recommended
`reasoningDepth: M`, and a sourced answer. Submit once; persist the task id and poll the
same task. A local wait timeout must not spend money by creating another task.

## Artifact Contract

Create **`request-plan.json`** and **`design.md`** in the workspace root. This is offline
configuration generation, not executable client code. Do not call APIs, install packages,
read keys, import candidate code, or fabricate successful responses. URLs and entities below
are synthetic/public placeholders, not claims about a real organization.

Use a JSON object with `version: 1`. The plan grammar below is closed: use only the named
keys; do not put planner fields into REST bodies. All API calls use one local `transport`:

- `baseUrl`: `https://api.linkup.so` (request paths include `/v1`).
- `headers`: `Authorization: Bearer ${LINKUP_API_KEY}` and `Content-Type: application/json`.
  The key expression is an unevaluated runtime environment placeholder, never a real key.
- `timeoutSeconds`: a request timeout in the inclusive range 1–60 seconds.

Each POST request is `{method, path, body}`; do not add SDK or MCP fields. Local dot paths
below describe response lookup, not literal request fields. Body strings can vary naturally.
For this exercise, keep `q` at 30–4000 characters; this is a local task bound,
not a published API limit. Query quality is rubric-scored.
Inline JSON Schemas use `type`, `properties`, `required` (all listed property names), and
optional boolean `additionalProperties`; harmless `description`/`title` annotations are fine.
String fields may use `format: uri` where requested. Nullable strings use `["string","null"]`.
Arrays use `type: array` with `items`; do not replace row objects with top-level arrays.

The local `policy` always has `networkDuringPlanning: false`, `credentialSource: environment`,
`maxRetries: 0`, `sourcePolicy: preserve`, `onError: stop-and-report`, and `writeTargets`
containing exactly `request-plan.json` and `design.md` (order irrelevant). Additional policy
fields for this task are specified below. No actual files besides these two need be generated.

Write a short design note explaining routing, payload/response distinctions, source handling,
missing values, and the most important limitation of the plan. This note is assessed by rubric,
not keyword matching. JSON is limited to 64 KiB and the note to 16 KiB; both must be regular
UTF-8 files, not symlinks. Property ordering and insignificant whitespace do not matter.
Every deterministic group is essential: any failed group caps its score at 0.6.

## Local Polling Grammar

Use `submit` for the POST, not a synchronous request/result fiction. Add local `poll` with
exactly these fields:

| Field | Required behavior/value |
| --- | --- |
| `method` / `path` | `GET` and the same endpoint's `/${taskId}` path, including `/v1` |
| `taskIdFrom` | `submit.id` |
| `persistTaskIdBeforePolling` | `true`; save the returned id before waiting |
| `initialDelaySeconds`, `backoffMultiplier`, `maxDelaySeconds` | Use the task's schedule below |
| `maxRequestsPerSecond` | Greater than zero, no greater than 1 |
| `deadlineSeconds` | A finite integer from 30 to 1800 |
| `maxAttempts` | A finite integer from 1 to 100 |
| `states` | `pending: wait`, `processing: wait`, `completed: read-output`, `failed: report-error` |
| `onTimeout` / `onCancel` | `return-task-id` for both, without a new submission |
| `onUnknownStatus` | `stop-and-report` |
| `onAmbiguousSubmit` | `stop-and-reconcile` (no assumed server idempotency) |
| `resubmitWhileActive` | `false` |

Both the attempt bound and the wall-clock deadline apply; stop on whichever occurs first.
`taskId` is a local variable populated from the saved submission id, not an API body field.
The polling GET has no body or additional headers: it uses the shared Linkup transport.
No list endpoint or new POST substitutes for polling the saved task.

## Research-Specific Grammar

Top-level fields: `version`, `transport`, `submit`, `poll`, `result`, `policy`.

- Use Research REST endpoints, not SDK namespace methods. `submit.body` has exactly `q`,
  `mode`, `reasoningDepth`, and `outputType`. Use the explicit selections above.
- Begin polling after 2 seconds, multiply the delay by 2 after each nonterminal response,
  and cap it at 10 seconds. Use both a finite deadline and attempt bound.
- Local `result` has exactly `answerPath`, `sourcesPath`, `errorPath`, `preserveSources`.
  Map to the actual nested completion answer/sources and failure error fields.
  Preserve full source records with `preserveSources: true`.
- Add `maxSubmissions: 1` and `missingValuePolicy: report-unknown` to local `policy`.

Explain the synchronous submission envelope versus completed output, both terminal states,
timeout/cancellation resumability, and ambiguous submission failures. Distinguish the server
default depth from the explicitly recommended M choice. No automatic blind POST retries.
