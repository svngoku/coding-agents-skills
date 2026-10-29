# Pricing Discovery and Retrieval Plan

Plan **one Search call** that discovers the official pricing page for synthetic **Northstar
Metrics**, then scrapes that discovered page and returns company, products, plan names,
published prices, and exact source URLs. Choose the depth that supports the sequential
discovery-then-scrape requirement; do not pretend a one-shot depth can follow discovered URLs.
No known pricing URL is supplied.

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

## Search-Specific Grammar

Top-level fields: `version`, `transport`, `request`, `result`, `policy`.

- Use the Search REST endpoint. `request.body` contains exactly `q`, `depth`, `outputType`,
  `structuredOutputSchema`, and `includeSources`. Ask for structured output with sources.
- `structuredOutputSchema` is a **JSON string**, not a nested body object. Its root object has
  `company` (string) and `plans` (array). Each plan row is an object with `product` and `name`
  (strings), `price` (nullable string), and `sourceUrl` (string with `format: uri`).
  Require every field at its respective object level.
- Local `result` has exactly `dataPath`, `sourcesPath`, and `preserveSources`. Set the first
  two to the actual top-level response keys for structured output with sources; set the
  last to `true` so full source records and URLs survive downstream processing.
- Add `maxRequests: 1` and `missingValuePolicy: null-not-invented` to local `policy`.

Explain why the chosen depth is needed, how the source wrapper differs from an input
outputType, and why absent pricing is null rather than inferred.
