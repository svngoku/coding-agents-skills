# Linkup Workflow Schema

Adapted from Linkup's MIT source pack (2026-09-15 snapshot). Read the
[2026-10-29 compatibility corrections](../sources-and-compatibility.md) before copying API examples.
Examples require an approved connection/budget; do not run paid calls for validation.

Use this schema for every workflow in the Linkup workflow library.

The goal is to make each workflow useful in three places:

- a human-readable website or sales page
- a Git repository of reusable agent ideas
- an agent that turns a business goal into executable Linkup prompt chains

## File Format

Each workflow is a Markdown file with YAML front matter followed by standard sections.

```yaml
---
id: sales.icp_to_lead_list
title: ICP to Lead List
category: sales
summary: Find companies that match a target customer profile and explain why they fit.
user_goal: I want to find companies that look like my best customers.
inputs:
  - ideal customer profile
  - geography
  - company size
  - target industry
outputs:
  - qualified company list
  - evidence for each company
  - recommended next step
linkup_strength: Finds and verifies public web evidence for companies, people, and markets.
handoff_tools:
  - CRM
  - email finder
  - outbound sequencer
---
```

## Required Fields

- `id`: Stable workflow ID in `{category}.{slug}` format.
- `title`: Human-readable workflow name.
- `category`: Main folder/category.
- `summary`: One sentence describing what the workflow does for the user.
- `user_goal`: Plain-English goal this workflow answers.
- `inputs`: Inputs the agent should ask for before running the workflow.
- `outputs`: Final artifacts the workflow should produce.
- `linkup_strength`: Why Linkup is useful for this workflow.
- `handoff_tools`: Tools that should receive Linkup's output when Linkup is not the final system.

## Required Sections

Every workflow body should use these headings:

1. `## What The User Gets`
2. `## When To Use`
3. `## Inputs To Ask For`
4. `## Linkup Workflow`
5. `## Output Contract`
6. `## Handoffs`
7. `## Failure Modes`

## Linkup Step Format

Every step under `## Linkup Workflow` should use a Search API, Fetch API, or Research API shape.

Use `linkup.search` for narrow retrieval, entity lists, enrichment fields, or steps that feed another
system. Use `linkup.fetch` when the step already has the exact URL.

```yaml
step: 1
name: Discover matching companies
purpose: Build the first candidate list.
linkup.search:
  q: "Find companies that match {ideal_customer_profile}. Run separate web searches for {facet_1}, {facet_2}, and {facet_3}. Return company name, website, evidence, and source URLs."
  depth: standard
  outputType: searchResults
expected_behavior:
  - Multiple independent web search calls if facets are explicit.
uses_previous_step: false
produces:
  - candidate_companies
```

Use `linkup.fetch` when a previous step or the user already supplies the exact URL. Add `schema` when
downstream code needs typed fields from that page; omit it for markdown.

```yaml
step: 2
name: Read the company website
purpose: Get official positioning from the known URL.
linkup.fetch:
  url: "{company_website}"
  renderJs: true
  schema:
    type: object
    properties:
      description:
        type: string
        description: What the company does, in its own words
      target_customers:
        type: array
        items:
          type: string
expected_behavior:
  - Single page read; no searching. Retry with mode "pro" if markdown is empty.
uses_previous_step: company_website
produces:
  - official_company_context
```

Use `linkup.research` for long-running investigations, reports, market maps, sector risk, technical
landscapes, event/prediction investigations, and complex dossiers.

```yaml
step: 1
name: Produce market map
purpose: Build a synthesized report across many source families.
linkup.research:
  q: "Research the market for {market_definition} in {geography}. Identify segments, visible players, growth signals, customer segments, underserved areas, and source URLs. Cite primary and credible secondary sources."
  mode: research
  reasoningDepth: L
  outputType: sourcedAnswer
expected_behavior:
  - Long-running multi-source investigation with a synthesized report.
uses_previous_step: false
produces:
  - market_map_report
```

Use the atomic rules in `../linkup-prompt-optimizer-knowledge.md` for each `linkup.search` object.
Use `../linkup-workflow-optimizer-knowledge.md` for deciding when a step should instead use
`linkup.fetch` or `linkup.research`.

### Batching

A step is not a different shape when it runs over a list. When a workflow runs the same
`linkup.search`, `linkup.fetch`, or `linkup.research` step for many inputs (a company list, a nightly
monitor), note it under `expected_behavior` and submit the calls through `/v1/tasks` (up to 100 per
submission, same parameters and price as the direct calls) instead of looping synchronous requests.

## Output Contract

Each workflow should say what downstream software can expect. Prefer stable field names:

- `company_name`
- `website`
- `person_name`
- `role`
- `source_url`
- `evidence`
- `confidence`
- `recommended_next_step`

For `structured` output, always include a shallow JSON schema in the workflow. Do not mark a Search
API or Research API step as `structured` unless the schema is present.

## Quality Bar

A workflow is good when:

- every Linkup call has a clear retrieval job
- steps feed into each other without requiring hidden assumptions
- the workflow says where Linkup stops and another tool should take over
- the final output is useful to a business user, not just technically correct
- failure modes are explicit enough for an agent to recover or ask the user for more input
