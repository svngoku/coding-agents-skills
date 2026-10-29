---
id: research.market_map
title: Market Map
category: research
summary: Find visible companies, categories, and segments in a market.
user_goal: I want to understand who is active in a market.
inputs:
  - market definition
  - geography
  - customer segment
  - inclusion and exclusion rules
outputs:
  - market segments
  - company list
  - category notes
  - source URLs
linkup_strength: Discovers companies and public category evidence from the web.
handoff_tools:
  - spreadsheet
  - investment research
  - sales planning
---

# Market Map

Adapted from Linkup's MIT source pack (2026-09-15 snapshot). Read the
[2026-10-29 compatibility corrections](../../sources-and-compatibility.md) before copying API examples.
Examples require an approved connection/budget; do not run paid calls for validation.

## What The User Gets

A first-pass map of a market: visible companies, segments, source evidence, and gaps for follow-up
research.

## When To Use

Use when the user is exploring a category, territory, or customer segment.

## Inputs To Ask For

- Market definition.
- Geography.
- Target customer segment.
- Company size or stage.
- Exclusions.

## Linkup Workflow

Use the Search API path when the user needs a first-pass company list. Use the Research API path
when the user wants a synthesized market map, sizing view, underserved segments, or investment-style
assessment.

```yaml
step: 1
name: Discover market segments
purpose: Define how the market is organized.
linkup.search:
  q: "Find how the market for {market_definition} is segmented in {geography}. Run separate web searches for: market reports, category explainers, competitor lists, and industry directories. Return segment, description, example companies, and source URLs."
  depth: standard
  outputType: searchResults
expected_behavior:
  - Multiple web search calls across source families.
uses_previous_step: false
produces:
  - market_segments
```

```yaml
step: 2
name: Find companies by segment
purpose: Build a visible company list.
linkup.search:
  q: "Find companies in {market_definition} for these segments: {market_segments}. Run separate web searches for each segment and preserve {geography} in every search. Return company name, website, segment, evidence, and source URLs."
  depth: standard
  outputType: structured
  structuredOutputSchema:
    type: object
    properties:
      companies:
        type: array
        items:
          type: object
          properties:
            company_name:
              type: string
            website:
              type:
                - string
                - "null"
            segment:
              type: string
            evidence:
              type: string
            source_urls:
              type: array
              items:
                type: string
          required:
            - company_name
            - segment
            - evidence
            - source_urls
    required:
      - companies
expected_behavior:
  - Segment-specific web search calls.
uses_previous_step: market_segments
produces:
  - market_companies
```

```yaml
step: 3
name: Summarize market map
purpose: Make the map useful for planning.
linkup.search:
  q: "Summarize this market map for {market_definition}: {market_companies}. Include visible segments, representative companies, whitespace, uncertainty, and source URLs."
  depth: standard
  outputType: sourcedAnswer
expected_behavior:
  - Sourced synthesis of discovered companies.
uses_previous_step: market_companies
produces:
  - market_map_summary
```

```yaml
step: research
name: Produce full market map report
purpose: Run a long-form investigation across market reports, company pages, news, directories, and credible secondary sources.
linkup.research:
  q: "Research the market for {market_definition} in {geography}. Identify market segments, visible companies, growth signals, customer segments, underserved areas, regulatory or geographic constraints, and source URLs. Include uncertainty and explain where evidence is weak."
  mode: research
  reasoningDepth: L
  outputType: sourcedAnswer
expected_behavior:
  - Long-running multi-source investigation with a synthesized market map.
uses_previous_step: false
produces:
  - market_map_report
```

## Output Contract

- `segment`
- `company_name`
- `website`
- `evidence`
- `source_urls`
- `uncertainty`
- `market_map_report`

## Handoffs

Send company lists to sales planning, investment research, or a spreadsheet.

## Failure Modes

- Market definitions may be too broad.
- Directories can be stale.
- The workflow should show uncertainty and avoid implying full market coverage.
- Use `/v1/research` when the user expects a report or assessment, not only a list of companies.
