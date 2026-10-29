# Linkup Search

Adapted from Linkup's MIT source pack (2026-09-15 snapshot). Read the
[2026-10-29 compatibility corrections](sources-and-compatibility.md) before copying API examples.
Examples require an approved connection/budget; do not run paid calls for validation.

Linkup is a web search API for agents: it turns a natural-language instruction into retrieval actions (web search, scraping, LinkedIn, and more) and returns accurate, cited, real-time results.

A Linkup query is an **instruction to a retrieval system**, not a question to answer. Optimize for *what to find and where*, then do the synthesis yourself.

## How to call it

- If the **`linkup-search` MCP tool** is available, use it: pass a natural-language query and a `depth`. (The Linkup MCP server also exposes `linkup-fetch`, `linkup-research`, and `linkup-get-research`; see the sibling skills.)
- Inspect the actual MCP schema first: the pinned Search tool supports domain filters, but fixes `outputType` to `searchResults`. Use the **REST Search API** when the available tool lacks a needed option, including structured output. REST requires `LINKUP_API_KEY`:

```shell
curl -sS -X POST "https://api.linkup.so/v1/search" \
  -H "Authorization: Bearer $LINKUP_API_KEY" -H "Content-Type: application/json" \
  -d '{"q":"...","depth":"standard","outputType":"searchResults"}'
```

## Reason from the data request (before writing `q`)

Answer three questions in order — each narrows the next and lands you on a `depth`:

1. **What inputs do I already have?** A URL → scrape it directly (don't search to find it). A name/topic only → search. Both → combine (scrape the URL + search for the rest).
2. **Where does the data live?** A single fact usually in snippets (CEO, price, date) → `flash` if latency is the constraint (< 200 ms), otherwise `fast` (~1s, higher quality). A few facts across snippets → `standard`. On full pages (tables, specs, long-form) → you must **scrape**. Unsure → `deep`.
3. **Do I need to chain steps?** All work is parallel → `standard`. Must find a URL *then* scrape it, or scrape multiple/discovered pages → `deep`. When uncertain, `deep`.

`flash` and `fast` pass the query to the index as-is: keep them short and keyword-shaped. They ignore instructions ("scrape…", "then…"); those need `standard` or `deep`.

Then pick the **output type** — `searchResults` (you'll inspect/synthesize sources), `sourcedAnswer` (a human needs a direct cited answer), `structured` (software needs fields; always include a `structuredOutputSchema`). Set **source filters** (`includeDomains`, `excludeDomains`) only when you know the exact domains to target or exclude — from the user or a previous result. Never invent domains. Use fromDate/toDate for intentional index-date filtering and state semantic time scope in q.

Pricing: `flash`/`fast`/`standard` $0.005 (`searchResults`) or $0.006 (`sourcedAnswer`/`structured`); `deep` $0.05 / $0.055.

Key rule: **`standard` cannot discover a URL and then scrape it in the same call** — use `deep` ("first find the official page, then scrape it") or split into two calls.

## Write the query as a retrieval plan

Make the plan visible: target entity/URL, the retrieval action (find/scrape/count/compare), the source scope, the exact fields to return, ordering (deep only), and "return source URLs / say none found." Name distinct facets rather than rewording the same search. Quote and disambiguate ambiguous names (`"Clause AI" legal-tech startup`).

## Worked examples

```text
Input: company name only · need: CEO (one fact) · not sequential
→ depth=fast · q: "{company} CEO"   (depth=flash if this is on a latency-critical path)
```
```text
Input: company name only · need: latest funding amount (lives in snippets) · not sequential
→ depth=standard · q: "Find {company}'s latest funding round amount and date"
```
```text
Input: company name only · need: pricing (lives on a full page) · sequential (find page, then scrape)
→ depth=deep · q: "Find the pricing page for {company}. Scrape it. Extract plan names, prices, and features."
```
```text
Input: a known URL · need: pricing from that page · not sequential
→ depth=standard (or the linkup-fetch skill) · q: "Scrape {url}. Extract plan names, prices, and included features."
```
```text
Input: company name · need: ICP inferred from homepage + blog + case studies · sequential
→ depth=deep · q: "Find and scrape {company}'s homepage, use-case pages, and 2-3 recent blog posts. Extract industries, company sizes, job titles, and pain points."
```

## Read the full knowledge before non-trivial queries

This skill is the summary. For exact depth behavior, query templates, source-constraint rules, LinkedIn wording, local-place rules, and known bad patterns, read the bundled files in this skill's `` directory:

- `linkup-agent-query-mental-model.md` — how to reason from a data request to the right request shape.
- `linkup-prompt-optimizer-knowledge.md` — the detailed rulebook: depth rules, templates, filters, LinkedIn, bad patterns.
- `linkup-api-reference.md` — endpoints, output types, auth, examples.

For reading a known URL (markdown or typed JSON), use the `linkup-fetch` skill. For minutes-long multi-source investigations, use `linkup-research`. For running many searches, fetches, or research jobs as a batch, use `linkup-tasks`. For bulk structured records from one listing page, use `linkup-extract`. To turn a business goal into a multi-step workflow, use `linkup-workflow`.
