# Linkup Agent Query Mental Model

Adapted from Linkup's MIT source pack (2026-09-15 snapshot). Read the
[2026-10-29 compatibility corrections](sources-and-compatibility.md) before copying API examples.
Examples require an approved connection/budget; do not run paid calls for validation.

Use this document before writing a Linkup Search API request. It teaches the agent how to reason from
a user's data request to the right retrieval shape.

This is the thinking guide, not the rulebook. After choosing the request shape, use
`linkup-prompt-optimizer-knowledge.md` for exact depth rules, templates, source constraints,
LinkedIn wording, local-place rules, and bad patterns.

## What This File Is For

Use this file to answer:

- What data does the user already provide?
- What data should Linkup return?
- Which `depth`, `outputType`, and API parameters fit the job?
- How should the query exploit that depth?
- Should the agent make one Linkup call, chain calls outside Linkup, or let `deep` chain work inside
  Linkup?

Do not use this file as the source for exact prompt templates. Use the prompt optimizer knowledge
file for that.

## Core Principle

Do not start by writing `q`.

First choose the request shape:

1. Pick `depth`.
2. Pick `outputType`.
3. Add any necessary API parameters.
4. Decide whether the work is independent or sequential.
5. Then write the query so Linkup's retrieval plan is obvious.

The query should not merely describe the user's goal. It should describe how Linkup should retrieve
the evidence.

## Step 1: Read The Data Request

Convert the user's request into a data contract.

Identify:

- Input: names, URLs, domains, LinkedIn URLs, SIREN/SIRET numbers, date ranges, geography, source
  families, or entity lists.
- Output: facts, fields, source URLs, full page content, a direct answer, or structured data.
- Constraints: official sources, freshness, country, domain, image requirement, confidence rule, or
  "say none found" behavior.
- Consumer: human reader, downstream agent, code, CRM, database, or spreadsheet.

This step prevents the agent from writing broad prompts like "tell me about X" when the real job is
to retrieve specific evidence.

## Step 2: Choose The Request Shape

Choose the smallest Linkup request shape that can produce the required evidence.

Use `flash` when the job is one keyword-shaped lookup, snippets are enough, and latency is the
constraint (under 200 ms). Use `fast` for the same shape when ~1s is acceptable and you want a
higher-quality single pass. Neither follows instructions; the query goes to the index as written.

Use `standard` when all useful retrieval work can be planned upfront.

Use `deep` when later retrieval depends on earlier results.

Use `searchResults` when another agent or application will inspect, rank, chain, or synthesize
sources.

Use `sourcedAnswer` when a human needs a direct answer with citations.

Use `structured` when software needs fields. Always include `structuredOutputSchema`.

Use source filtering only if you know exactly the URLs or domains you are targeting or not targeting.
Do not infer source filters from preferences such as "official sources." Use date filters only for an intentional index-date restriction.
Use other API parameters when the request requires them, such as images or result limits.

Important: when the URL is already known, prefer `/fetch` over a search. Pass `schema` (and optional
`instructions`) to get typed JSON from that page in the same call; leave it out for markdown. Use
Search with `outputType: structured` only when the URL still has to be discovered. `/fetch` reads one
page and does not follow links; for many rows across pagination use `/extract`.

If the request is a batch (many entities, a nightly job), plan the individual calls as usual and then
submit them through `/tasks` instead of looping synchronous calls.

## Step 3: Match Query Shape To Depth

### `standard`: Parallel Work Planned Upfront

Think of `standard` as upfront fan-out.

It is strongest when the query clearly separates independent work:

- several targeted web searches
- adjacent keyword searches for wider news or incident coverage
- a known URL scrape plus independent searches
- a known LinkedIn URL extraction
- structured extraction when the fields and schema are known

The query should list separate facets, entities, source families, or adjacent keywords. This helps
Linkup run separate targeted tool calls instead of one mixed search.

Do not ask `standard` to discover a URL and then scrape that discovered URL in the same request. Use
`deep`, or have the outer agent run one call to discover the URL and another call to read it.

### `deep`: Sequential Result Reuse

Think of `deep` as search, inspect, then reuse.

It is strongest when the query describes a dependency chain:

- find the official page, then scrape it
- find relevant LinkedIn URLs, then extract data from them
- search for likely source pages, then read the best pages
- follow relevant links from a discovered page
- verify a claim across multiple discovered sources

The query should use ordered language such as `first`, `then`, `after that`, and `if the page links
to`. This tells Linkup which results should feed later retrieval.

## Step 4: Decide Whether To Chain Outside Linkup

The agent can choose between one deeper Linkup call and several simpler calls.

Use one `deep` call when Linkup should handle the dependency chain internally.

Use several `standard` or Search API calls when the outer agent should inspect intermediate results,
choose the next URL or entity, apply business logic, or combine Linkup with other tools.

Use known URL scraping or full-content reading only when the page to read is already known, or after
a previous search step has selected the best URL.

## Step 5: Write The Query As A Retrieval Plan

The final query should make these choices visible:

- target entities or URLs
- independent searches that can run in parallel
- sequential steps that must happen in order
- adjacent keywords when coverage matters
- exact output fields
- source preferences and hard evidence rules
- what to do when no relevant source is found

For `sourcedAnswer`, guide which sources matter and how claims should be cited.

For `structured`, make the query fields match the schema fields.

For LinkedIn, name the exact action: profile details, posts, post search, comments, or profile URLs
as evidence. Use `standard` when the LinkedIn URL is known and `deep` when Linkup must find it first.

## Final Check

Before sending the request, verify:

- The depth matches the retrieval pattern.
- `standard` is used for upfront fan-out, not hidden sequencing.
- `deep` is used for result reuse, not just because the request feels important.
- The query names exact fields and source rules.
- Source filters are used only for exact known target or exclusion URLs or domains.
- `structured` has a schema.
- The query does not invent URLs, domains, entities, or facts.

If the agent needs exact wording examples, continue in `linkup-prompt-optimizer-knowledge.md`.
