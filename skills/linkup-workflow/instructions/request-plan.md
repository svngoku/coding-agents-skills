# Two-Call Discovery-to-Fetch Workflow

Plan retrieval of the official public careers page for synthetic **Northstar Metrics**.
The URL is unknown. First discover one official careers URL with Search; only after the
source evidence identifies a single official URL, Fetch it. The hard budget is **two Linkup
calls**, including retries. Produce a source-preserving draft handoff; no CRM or email writes,
no scheduler, and no actual execution. A workflow is local orchestration, not a Linkup endpoint.

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

## Workflow-Specific Grammar

Top-level fields: `version`, `transport`, `steps`, `bindings`, `result`, `policy`.

- `steps` is an ordered array of exactly two entries:
  1. `{id: discover, request: ...}`. Use the Search REST endpoint with exactly
     `q`, `depth`, `outputType` in its body. A single standard discovery is sufficient;
     select raw search results rather than a sourced-answer or structured wrapper.
  2. `{id: fetch, dependsOn: [discover], request: ...}`. Use Fetch with exactly `url`
     and `renderJs`; the local URL template must be `${officialUrl}`, and rendering is on.
- `bindings` contains only `officialUrl`, an object with:
  `fromStep: discover`, `fromPath` set to the response key holding Search result records,
  `select: single-official-careers-url`, `requireOfficialDomainEvidence: true`,
  `requireHttps: true`, `onMissing: stop-no-fetch`, `onAmbiguous: stop-no-fetch`.
  `${officialUrl}` is local substitution after result validation, not an API field or a
  literal URL sent over the wire. Do not guess a hostname or use an aggregator as official.
- Local `result` has exactly `pageMarkdownFrom: fetch.markdown`,
  `pageUrlFrom: bindings.officialUrl`, `discoverySourcesFrom: discover.results`,
  `preserveSources: true`, and `handoff: draft-only`. Dot paths identify step responses
  and the local binding, not REST body fields.
- Add `maxRequests: 2`, `missingValuePolicy: report-unknown`, and `externalWrites: []`
  to local `policy`. Zero retries is deliberate: a third call exceeds this exercise's budget.

Explain why this is not a `/v1/workflow` call, an invented MCP tool, or a parallel Tasks batch.
Describe the dependency, missing/ambiguous URL stop, source preservation, and draft-only handoff.
