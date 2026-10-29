---
id: research.company_dossier
title: Company Dossier
category: research
summary: Create a source-backed profile of a company for sales, investing, partnership, or research.
user_goal: I want to understand a company quickly.
inputs:
  - company name
  - company website
  - research objective
outputs:
  - company summary
  - product and market
  - leadership
  - recent news
  - source URLs
linkup_strength: Combines official site scraping with external public context.
handoff_tools:
  - CRM
  - research memo
  - sales briefing
---

# Company Dossier

Adapted from Linkup's MIT source pack (2026-09-15 snapshot). Read the
[2026-10-29 compatibility corrections](../../sources-and-compatibility.md) before copying API examples.
Examples require an approved connection/budget; do not run paid calls for validation.

## What The User Gets

A compact, sourced company profile with official description, product, customers, leadership, recent
news, and open questions.

## When To Use

Use for account research, investment screening, partnership review, or customer discovery.

## Inputs To Ask For

- Company name.
- Website if known.
- Research objective.
- Specific fields the user cares about.

## Linkup Workflow

Use the Search API path for normal account research. Use the Research API path when the user asks for
ownership chains, M&A screening, regulatory context, sector risk, or a board/investor-style memo.

```yaml
step: 1
name: Collect official context
purpose: Start from the company source of truth.
linkup.search:
  q: "Scrape {company_website}. Extract company description, product, target customers, leadership or team page if present, pricing or packaging if present, customer proof, and source URL."
  depth: standard
  outputType: searchResults
expected_behavior:
  - Web scraping for known website.
uses_previous_step: false
produces:
  - official_context
```

```yaml
step: 2
name: Add external evidence
purpose: Verify the company from independent sources.
linkup.search:
  q: "Find external public information about \"{company_name}\" {company_disambiguator}. Run separate web searches for: company profile, funding or investors, recent news, customer stories, and leadership. Return field, evidence, date if available, and source URLs."
  depth: standard
  outputType: searchResults
expected_behavior:
  - Multiple web search calls; exact quoted name reduces entity drift.
uses_previous_step: official_context
produces:
  - external_context
```

```yaml
step: 3
name: Write dossier
purpose: Produce the final research artifact.
linkup.search:
  q: "Create a source-backed company dossier for {company_name} using official context {official_context} and external context {external_context}. Include summary, product, market, customers, leadership, recent news, risks, open questions, and source URLs."
  depth: standard
  outputType: sourcedAnswer
expected_behavior:
  - SourcedAnswer synthesis with citations.
uses_previous_step: external_context
produces:
  - company_dossier
```

```yaml
step: research
name: Produce complex company dossier
purpose: Investigate a company deeply across ownership, market, regulatory, risk, and transaction context.
linkup.research:
  q: "Research \"{company_name}\" {company_disambiguator} for {research_objective}. Include official website, ownership or leadership chain, product and market, customers, recent news, regulatory or sector risks, M&A relevance if applicable, open questions, and source URLs. Use primary sources where possible and mark uncertain findings."
  mode: research
  reasoningDepth: L
  outputType: sourcedAnswer
expected_behavior:
  - Long-running multi-source investigation for a dossier or memo.
uses_previous_step: false
produces:
  - complex_company_dossier
```

## Output Contract

- `company_name`
- `website`
- `description`
- `product`
- `customers`
- `leadership`
- `recent_news`
- `risks`
- `source_urls`
- `complex_company_dossier`

## Handoffs

Send the dossier to CRM, research notes, or meeting prep.

## Failure Modes

- Ambiguous names need quotes and a disambiguator.
- Some details are unavailable from public sources.
- Official messaging may conflict with external sources; show both rather than collapsing them.
- Use `/v1/research` when the user asks for ownership chains, M&A context, regulation, or risk.
