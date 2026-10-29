---
id: marketing.competitor_messaging_scan
title: Competitor Messaging Scan
category: marketing
summary: Compare competitor positioning, claims, audiences, and proof points.
user_goal: I want to understand how competitors talk about the market.
inputs:
  - competitor names or domains
  - product category
  - target audience
outputs:
  - competitor messaging matrix
  - repeated claims
  - differentiation opportunities
  - source URLs
linkup_strength: Scrapes competitor pages and finds public positioning evidence.
handoff_tools:
  - positioning doc
  - sales enablement
  - website planning
---

# Competitor Messaging Scan

Adapted from Linkup's MIT source pack (2026-09-15 snapshot). Read the
[2026-10-29 compatibility corrections](../../sources-and-compatibility.md) before copying API examples.
Examples require an approved connection/budget; do not run paid calls for validation.

## What The User Gets

A comparison of how competitors describe their product, target customers, categories, outcomes, and
proof points.

## When To Use

Use when building positioning, sales enablement, landing pages, or competitive battlecards.

## Inputs To Ask For

- Competitor names or websites.
- Product category.
- Target market.
- Claims to compare.

## Linkup Workflow

```yaml
step: 1
name: Scrape competitor homepages
purpose: Collect official positioning.
linkup.search:
  q: "Scrape these competitor websites: {competitor_websites}. Extract headline, subheadline, target audience, category terms, promised outcomes, proof points, and source URLs."
  depth: deep
  outputType: searchResults
expected_behavior:
  - Web scraping calls for known URLs when domains are provided.
uses_previous_step: false
produces:
  - official_messaging
```

```yaml
step: 2
name: Find external messaging context
purpose: Add reviews, articles, and launch pages.
linkup.search:
  q: "Find public information about competitors in {product_category}: {competitors}. Run separate web searches for: competitor reviews, launch announcements, comparison pages, and customer stories. Return claim, source type, competitor, and source URLs."
  depth: standard
  outputType: searchResults
expected_behavior:
  - Multiple web search calls for external context.
uses_previous_step: official_messaging
produces:
  - external_messaging_context
```

```yaml
step: 3
name: Build messaging matrix
purpose: Make the comparison actionable.
linkup.search:
  q: "Using official messaging {official_messaging} and external context {external_messaging_context}, compare competitors in {product_category}. Return competitor, audience, main claim, proof point, differentiation opportunity, and source URLs."
  depth: standard
  outputType: structured
  structuredOutputSchema:
    type: object
    properties:
      competitors:
        type: array
        items:
          type: object
          properties:
            competitor:
              type: string
            audience:
              type: string
            main_claim:
              type: string
            proof_point:
              type: string
            differentiation_opportunity:
              type: string
            source_urls:
              type: array
              items:
                type: string
          required:
            - competitor
            - main_claim
            - source_urls
    required:
      - competitors
expected_behavior:
  - Structured synthesis with source URLs.
uses_previous_step: external_messaging_context
produces:
  - messaging_matrix
```

## Output Contract

- `competitor`
- `audience`
- `main_claim`
- `proof_point`
- `source_urls`
- `differentiation_opportunity`

## Handoffs

Send the matrix to positioning, sales enablement, or website planning.

## Failure Modes

- Competitor domains are missing; use `deep` only if the workflow must discover and scrape websites.
- Competitors use vague claims; mark weak evidence rather than over-interpreting.
- Product category terms may differ across competitors.
