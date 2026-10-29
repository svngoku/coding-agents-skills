# Linkup Deep Research

Adapted from Linkup's MIT source pack (2026-09-15 snapshot). Read the
[2026-10-29 compatibility corrections](sources-and-compatibility.md) before copying API examples.
Examples require an approved connection/budget; do not run paid calls for validation.

The Research endpoint is an autonomous agent that investigates the web iteratively for minutes, cross-checks claims across sources, and returns a synthesized, cited answer. Reach for it only when a single `deep` search can't cover the job — many entities, many facets, verification needed, or a report-style deliverable.

This uses the REST API, so it needs `LINKUP_API_KEY` in the environment:

```shell
test -n "$LINKUP_API_KEY" || echo "Missing LINKUP_API_KEY"
```

## Choose mode and depth explicitly

Always set `mode` and `reasoningDepth` for predictable latency, cost, and output shape.

- **mode** — `answer` (a precise question with one correct answer, self-verified), `investigate` (deep analysis of one defined entity), `research` (broad, multi-angle exploration across many entities). Rule of thumb: ends with "?" and expects one answer → `answer`; "the state of / analysis of" → `research`; one entity comprehensively → `investigate`.
- **reasoningDepth** — `S` (~2-5 min), `M` (~3-7 min, recommended for a bounded report; API default is L), `L` (~5-10 min, market maps / multi-company), `XL` (~10-20 min, only when the user wants exhaustive coverage).
- **outputType** — `sourcedAnswer` for reports; `structured` only with a `structuredOutputSchema`.

## Call it via the MCP tool or REST

If the **`linkup-research`** and **`linkup-get-research`** MCP tools are available, use them: `linkup-research` takes `query` (and optional `mode`) and returns a task `id`; poll `linkup-get-research` with that `id`. The MCP tools fix `outputType` to `sourcedAnswer` and the inspected source exposes `reasoningDepth`; discover the actual installed schema. Use REST for unsupported fields or `structured` output.

## Submit, then poll with backoff

`POST /v1/research` returns `{id, status: "pending"}` immediately. Poll `GET /v1/research/{id}` with exponential backoff (start 2s, cap 10s) until `status` is `completed` or `failed`. Never poll faster than once per second. Failed tasks are not charged.

```shell
curl -sS -X POST "https://api.linkup.so/v1/research" \
  -H "Authorization: Bearer $LINKUP_API_KEY" -H "Content-Type: application/json" \
  -d '{"q":"Compare 2024 cloud revenue growth for Microsoft, Amazon, and Google from primary sources.","mode":"answer","reasoningDepth":"M","outputType":"sourcedAnswer"}'

curl -sS "https://api.linkup.so/v1/research/RESEARCH_ID" -H "Authorization: Bearer $LINKUP_API_KEY"
```

## Write a well-scoped prompt

Specify the angles to cover, leads to pursue, facts to verify, entities to compare, constraints/exclusions, and the output structure. More precise input yields more thorough, aligned output.

## Guidance

- Research costs more and runs for minutes ($0.25–$2.50 by depth). Tell the user what you're about to run and why before starting.
- Ground every claim in the returned `sources`. When done, summarize in plain English: what was researched, the answer, key sources, and roughly how long it took.
- Retry only genuine transient failures with bounded backoff. A `429` can indicate insufficient credits; never resubmit an active research job.
- Running research for a list of entities (a batch of companies, a nightly job)? Submit them as `"research"` tasks through `/v1/tasks` and poll one list — see the `linkup-tasks` skill.

For the full decision matrix, mode examples, patterns, and polling details, read `linkup-specialized-endpoints.md` (Research section) in this skill's directory.
