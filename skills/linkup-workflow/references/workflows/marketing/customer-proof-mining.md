---
id: marketing.customer_proof_mining
title: Customer Proof Mining
category: marketing
summary: Find public proof points such as case studies, testimonials, reviews, and customer stories.
user_goal: I need public proof that customers care about a category, product, or company.
inputs:
  - company or product category
  - customer segment
  - proof type
outputs:
  - proof points
  - customer examples
  - quotes or snippets
  - source URLs
linkup_strength: Finds public customer evidence across websites, review pages, and articles.
handoff_tools:
  - website copy
  - sales enablement
  - content brief
---

# Customer Proof Mining

Adapted from Linkup's MIT source pack (2026-09-15 snapshot). Read the
[2026-10-29 compatibility corrections](../../sources-and-compatibility.md) before copying API examples.
Examples require an approved connection/budget; do not run paid calls for validation.

## What The User Gets

A list of public proof points: case studies, testimonials, reviews, customer logos, implementation
stories, or quotes.

## When To Use

Use when marketing or sales needs evidence that a problem, market, or product matters.

## Inputs To Ask For

- Company, competitor, or category.
- Customer segment.
- Proof type.
- Geography if relevant.

## Linkup Workflow

```yaml
step: 1
name: Find proof sources
purpose: Collect pages likely to contain customer evidence.
linkup.search:
  q: "Find public customer proof for {company_or_category} and {customer_segment}. Run separate web searches for: case studies, customer stories, testimonials, reviews, and customer logos. Return proof type, customer name if present, snippet, and source URLs."
  depth: standard
  outputType: searchResults
expected_behavior:
  - Multiple web search calls for proof-source categories.
uses_previous_step: false
produces:
  - proof_sources
```

```yaml
step: 2
name: Extract proof points
purpose: Turn source pages into reusable evidence.
linkup.search:
  q: "Using these proof sources: {proof_sources}, extract public proof points for {company_or_category}. Return customer, proof type, quote or summary, evidence strength, and source URL."
  depth: standard
  outputType: structured
  structuredOutputSchema:
    type: object
    properties:
      proof_points:
        type: array
        items:
          type: object
          properties:
            customer:
              type:
                - string
                - "null"
            proof_type:
              type: string
            evidence:
              type: string
            source_url:
              type: string
            strength:
              type: string
          required:
            - proof_type
            - evidence
            - source_url
    required:
      - proof_points
expected_behavior:
  - Structured extraction from discovered proof sources.
uses_previous_step: proof_sources
produces:
  - proof_points
```

## Output Contract

- `customer`
- `proof_type`
- `evidence`
- `source_url`
- `strength`

## Handoffs

Send proof points to website copy, sales collateral, battlecards, or content briefs.

## Failure Modes

- Proof may be promotional or vendor-authored.
- Customer logos without case studies are weaker evidence.
- The workflow should label evidence strength instead of treating every source equally.
