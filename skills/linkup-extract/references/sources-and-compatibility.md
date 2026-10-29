# Linkup Sources and Compatibility

Checked **2026-10-29**. These are local adaptations, not an official Linkup
release or a guarantee of account access, tool availability, or model accuracy.

## Source and License

The documented distribution, [LinkupPlatform/skills](https://github.com/LinkupPlatform/skills/tree/dfc1d043546537fdd63ec7c39b5828979a5f69b0),
has no license file at the checked pin. The copied material instead comes from
the newer official **MIT-licensed** [linkup-for-agents source pack](https://github.com/LinkupPlatform/linkup-for-agents/tree/de40e04c39f6740bb4f70b00f2ef3ce04b6b88bc),
commit **`de40e04c39f6740bb4f70b00f2ef3ce04b6b88bc`**. Its
**Copyright (c) 2026 Linkup** and permission notice are preserved in `../LICENSE`.

All original reference material for this identity is retained. The original
skill body is adapted into `upstream-skill.md`; the local entry point follows
this collection's loader conventions. Filenames use kebab-case; broken bundled
paths, REST field/schema examples, date filters, deprecated Fetch fields, SDK
examples, depth selection, MCP fallbacks, and unbounded polling were corrected.
Workflow also bundles the query optimizer it references, quotes YAML query
scalars, and labels other skill identities as routing recommendations.
The newer source pack's sixth Tasks skill is **not** part of this five-skill import.

## Current REST Contracts

Base URL: `https://api.linkup.so`. Bearer credentials come from
`LINKUP_API_KEY` through approved secret management, never from generated files.

| Capability | Request | Response / boundary |
| --- | --- | --- |
| Search | `POST /v1/search`: `q`, `depth`, `outputType` | `searchResults` returns `{results}`; `sourcedAnswer` returns `{answer,sources}`. `structured` requires a root-object schema serialized as `structuredOutputSchema`; `includeSources:true` wraps output as `{data,sources}`. |
| Fetch | `POST /v1/fetch`: `url`, optional `renderJs`, `mode`, `schema`, `instructions`, `includeRawContent` | Markdown plus optional `data`; `instructions` requires object `schema` and is limited to 4,000 characters. Missing values can be omitted even when marked required. |
| Research | `POST /v1/research`: `q`, `outputType`, explicit `mode` and `reasoningDepth` recommended | Persist `id`; poll `GET /v1/research/{id}`. `pending`/`processing` precede `completed`/`failed`. Use `output` on completion and `error` on failure. |
| Extract | Closed-beta `POST /v1/extract`: `q`, known listing `url`, optional single-row object `schema`, `verifyUrls` | Poll `GET /v1/extract/{id}`. Completed `output` has `creditsUsed`, `rowsReturned`, and `resultUrl`; download and parse NDJSON, not inline rows. The download URL expires after 24 hours. |
| Workflow | An application plan using the supported endpoints | No `/v1/workflow` endpoint or Workflow MCP tool is established. CRM writes, email verification, and scheduling are separate, explicitly authorized handoffs. |

`structuredWithSources` is a response variant, **not** an input `outputType`.
Search's schema field and Fetch's object `schema` are not interchangeable.

## Retrieval, Jobs, and Surface Differences

- Search depths are `flash` (**Alpha**), `fast`, `standard`, `deep`. Flash/fast
  retrieve keyword-shaped queries without scraping or chaining; answer/structured
  outputs can still add LLM synthesis. Standard can scrape one already-supplied
  URL, not discover then scrape it in the same call. Use deep or separate stages
  for discovered-URL chains.
- Unlike the September snapshot's old advice, current Search/Research support
  `fromDate` and `toDate`. Use ordered `YYYY-MM-DD` values later than 1970-01-01
  for intentional index-date restrictions. Metadata dates may reflect publication
  or updates; also state semantic time scope in `q`.
- Fetch REST/TypeScript use `renderJs`; Python uses `render_js`. Prefer
  `includeRawContent` → `rawContent`/`contentType`; `includeRawHtml`/`rawHtml` are
  deprecated compatibility fields. HTML is limited to 20 MB, PDFs to 100 MB.
  Choose JS/pro mode deliberately because they change cost.
- Research's API reasoning-depth default is **L**; **M** is a workflow
  recommendation, not the server default. Start polling after 2 seconds and
  back off to at most 10 seconds between checks, never above 1 request/second.
  Extract guidance uses roughly 30-second polling. Bound the client wait,
  retain the job ID on timeout, and never resubmit an active job.
- Current Python SDK source uses `client.research(...)`, `client.get_research(id)`,
  `client.create_tasks(...)`, `client.get_task(id)`, `client.list_tasks(...)`.
  Batch creation requires typed task inputs, not arbitrary REST dictionaries.
  TypeScript uses `research`, `getResearch`, `createTasks`, `getTask`, `listTasks`.
  Namespace-style `.research.create/get` and `.tasks.create/get/list` examples
  are stale. Application helpers in the adapted guides are explicitly pseudocode.
- The verified MCP source names are `linkup-search`, `linkup-fetch`,
  `linkup-research`, `linkup-get-research`. Its Search fixes `searchResults`;
  Research fixes `sourcedAnswer`. Source supports all four Search depths with a
  fast default and Research `reasoningDepth`, but docs/deployments differ.
  Fetch source lacks some documented REST options, including schema/pro mode.
  Inspect the **actual available tool schema**; never infer hosted deployment
  support or invent Extract, Tasks, or Workflow tools.
- Optional `/v1/tasks` batches up to 100 independent requests without dependency
  ordering or a discount. Separate dependent workflow stages. This endpoint's
  existence does not install another skill or tool.

## Errors, Cost, and Validation Limits

Search/Fetch are documented at 10 requests/second per organization; asynchronous
polling is capped at 1 request/second. Treat `429` as potentially insufficient
credits, not automatically transient. Stop for authentication, permission/beta,
or billing problems. Retry genuine transient failures with bounded backoff,
without duplicating successful or still-active job submissions.

Prices are a dated observation, not a quote: Search Flash/fast/standard
$0.005 for results or $0.006 for answer/structured; deep $0.05/$0.055.
Fetch standard costs $0.001 without JS or $0.005 with JS; pro $0.005/$0.01;
schema adds $0.001. Research S/M/L/XL costs $0.25/$0.50/$1.50/$2.50.
Extract typically costs $2–10 and requires a $10 account balance to submit.
Check current pricing and account access before a paid workflow.

The local evaluation is synthetic and offline: it checks a generated application
request plan and its documented endpoint/lifecycle/output policy. It does not
authenticate, execute a workflow, measure retrieval accuracy, establish beta
access, or call Linkup. Run the reference directly and the grader separately
when diagnosing the full skillgrade runner; report runner/rubric blockers
rather than presenting a direct check as a full skillgrade pass.

## Local Validation (2026-10-29)

The direct reference plan scored **1.0**, and **93 stdlib regression tests**
passed for this skill. These checks use synthetic files and make no Linkup calls.

Full `skillgrade 0.5.0 --validate` was attempted and remains **blocked**:
its validation workspace cannot find `bash reference-plan`; the separate LLM
rubric also reports missing `GEMINI_API_KEY`. This is not a full skillgrade pass.
Do not add credentials or weaken the grader to conceal a runner failure.

## Primary Evidence

- [Official five-skill integration page](https://docs.linkup.so/pages/integrations/linkup-skill)
- [Current OpenAPI](https://api.linkup.so/v1/openapi.json)
- [Search reference](https://docs.linkup.so/pages/documentation/endpoints/search/reference)
- [Search best practices](https://docs.linkup.so/pages/documentation/endpoints/search/best-practices)
- [Fetch reference](https://docs.linkup.so/pages/documentation/endpoints/fetch/reference)
- [Fetch agent guide](https://docs.linkup.so/pages/documentation/endpoints/fetch/for-agents)
- [Research agent guide](https://docs.linkup.so/pages/documentation/endpoints/research/for-agents)
- [Extract overview and beta](https://docs.linkup.so/pages/documentation/endpoints/extract/overview)
- [MCP guide](https://docs.linkup.so/pages/integrations/mcp/mcp)
- [Pricing](https://docs.linkup.so/pages/documentation/platform/pricing)
- [Python client source, 1271ffabefa11006e9d3500c936abe586c2fd96f](https://github.com/LinkupPlatform/linkup-python-sdk/blob/1271ffabefa11006e9d3500c936abe586c2fd96f/src/linkup/_client.py)
- [TypeScript client source, 832dbecf81f0728dc9e00cdf9f4355a625e219d5](https://github.com/LinkupPlatform/linkup-js-sdk/blob/832dbecf81f0728dc9e00cdf9f4355a625e219d5/src/linkup-client.ts)
- [MCP source, ba40fdb88067dc5b1784ad4623bb5d96840a81c0](https://github.com/LinkupPlatform/linkup-mcp-server/tree/ba40fdb88067dc5b1784ad4623bb5d96840a81c0/src/tools)

## Local Rename and Provenance Map

| MIT source-pack path | Local guide |
| --- | --- |
| `skills/linkup-extract/SKILL.md` | [references/upstream-skill.md](upstream-skill.md) |
| `skills/linkup-extract/references/LINKUP_SPECIALIZED_ENDPOINTS.md` | [references/linkup-specialized-endpoints.md](linkup-specialized-endpoints.md) |
