# Closed-Beta Events Extraction Plan

The synthetic user **explicitly confirms their organization has closed-beta Extract access**.
Plan an extraction seeded at **`https://example.com/events`** for public events across its
listing pagination and event detail links. Capture event name, start date, venue, and event
URL; verify URLs. This access confirmation is an exercise assumption, not proof that an
arbitrary key is entitled. Do not submit anything or check actual account credentials.

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

## Extract-Specific Grammar

Top-level fields: `version`, `transport`, `submit`, `poll`, `result`, `policy`.

- Use the Extract REST endpoint. `submit.body` has exactly `q`, `url`, `schema`, `verifyUrls`.
  Use the supplied seed and enable URL verification.
- `schema` is a **single-row JSON object**, not an array wrapper. Its properties are
  `name` (string), `startDate` and `venue` (nullable strings), and `eventUrl` (URI string).
  Require all four fields.
- Poll every 30 seconds: `initialDelaySeconds: 30`, `backoffMultiplier: 1`,
  `maxDelaySeconds: 30`, with a bounded attempt count and deadline.
- Local `result` has exactly `resultUrlPath`, `rowsReturnedPath`, `creditsUsedPath`,
  and `download`. Map the first three to the actual completed-output fields, not inline rows.
- `download` has exactly:
  - `method: GET`, `urlFrom: output.resultUrl`, `headers: {}`; never forward the Linkup bearer
    credential to the result host.
  - `requireHttps: true`, `followRedirects: false`, `timeoutSeconds` in 1–60,
    `maxDownloads: 1`, and a positive integer `maxBytes` no greater than 50000000.
  - `format: ndjson`, `recordType: object`, `saveAs: events.ndjson`.
  - `persistBeforeExpiry: true`, `urlLifetimeHours` set to the documented lifetime,
    and `onExpiredUrl: report-expired-no-resubmit`.
- Add `maxSubmissions: 1`, `missingValuePolicy: null-not-invented`,
  `betaAccess: user-confirmed-synthetic`, and `onForbidden: stop-no-retry` to local `policy`.

Explain why a seeded listing differs from open-web Search, why GET listing is not discovery,
why a 403 must stop, how NDJSON is downloaded/persisted before expiry, and why the plan does
not promise exhaustive coverage or a fixed cost. File download/parsing is a future runtime
action in this declarative plan; do not create `events.ndjson` during this offline exercise.
