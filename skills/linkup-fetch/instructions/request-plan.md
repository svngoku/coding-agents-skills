# Typed Careers Page Fetch Plan

Plan one Fetch of the supplied **`https://example.com/careers`**. This is a known public page,
not discovery or a site crawl. Treat it as potentially JavaScript-rendered. Extract a jobs
array, splitting one job into one record per listed city. Preserve page Markdown, current
raw content plus its media type, and source-page identity. Use the current raw-content names,
not the deprecated HTML-only compatibility names.

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

## Fetch-Specific Grammar

Top-level fields: `version`, `transport`, `request`, `result`, `policy`.

- `request.body` contains exactly `url`, `renderJs`, `schema`, `instructions`, and
  `includeRawContent`. Enable rendering and raw-content inclusion.
- Fetch `schema` is a **JSON object**, not Search's serialized `structuredOutputSchema`.
  The root has `jobs` (array); each row has `position` and `city` (strings).
  Require the listed fields at both object levels, but account for actual response omissions.
- Write meaningful `instructions` (30–4000 characters) describing one job per city and
  omission of unavailable values. Instructions are valid only alongside a schema.
- Local `result` has exactly `markdownPath`, `dataPath`, `faviconPath`, `rawContentPath`,
  `contentTypePath`, `pageUrlFrom`, and `missingFields`. Map the first five to current
  response keys, `pageUrlFrom` to `request.body.url`, and `missingFields` to `omit`.
- Add `maxRequests: 1` and `missingValuePolicy: omit-not-invented` to local `policy`.

Explain why Fetch, not Search/Extract, fits this page; why schema-required fields may still
be absent; and why this does not authenticate or follow links.
