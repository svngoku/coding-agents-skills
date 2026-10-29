---
id: research.competitor_tracker
title: Competitor Tracker
category: research
summary: Monitor competitors for public product, pricing, messaging, hiring, and news changes.
user_goal: I want to keep track of what competitors are doing.
inputs:
  - competitor list
  - competitor domains
  - signal types
  - date range
outputs:
  - competitor updates
  - evidence URLs
  - impact summary
  - recommended follow-up
linkup_strength: Searches current public web sources and competitor sites for changes.
handoff_tools:
  - Slack alert
  - battlecard
  - product planning
  - sales enablement
---

# Competitor Tracker

Adapted from Linkup's MIT source pack (2026-09-15 snapshot). Read the
[2026-10-29 compatibility corrections](../../sources-and-compatibility.md) before copying API examples.
Examples require an approved connection/budget; do not run paid calls for validation.

## What The User Gets

A recurring update on competitor activity, grouped by signal type and impact.

## When To Use

Use for competitive intelligence, sales enablement, product marketing, and leadership updates.

## Inputs To Ask For

- Competitor names and domains.
- Signal types.
- Date range.
- Product category.

## Linkup Workflow

Use Search API steps for recurring signal collection. Use the Research API when the user wants a
written impact report across competitors, positioning, product, pricing, and market context.

```yaml
step: 1
name: Search competitor changes
purpose: Find recent public updates.
linkup.search:
  q: "Find recent public updates for these competitors: {competitors}. Run separate web searches for: product launches, pricing changes, messaging changes, hiring, partnerships, and funding or news. Return competitor, signal type, date, evidence, and source URLs."
  depth: standard
  outputType: searchResults
expected_behavior:
  - Multiple web search calls across signal categories.
uses_previous_step: false
produces:
  - competitor_updates
```

```yaml
step: 2
name: Scrape known competitor pages
purpose: Check official pages for changed positioning.
linkup.search:
  q: "Scrape these competitor pages: {competitor_pages}. Extract headline, product claims, pricing language, feature claims, customer proof, and source URLs."
  depth: deep
  outputType: searchResults
expected_behavior:
  - Web scraping calls for known URLs.
uses_previous_step: competitor_updates
produces:
  - official_competitor_pages
```

```yaml
step: 3
name: Summarize impact
purpose: Explain what matters.
linkup.search:
  q: "Using competitor updates {competitor_updates} and official page evidence {official_competitor_pages}, summarize what changed, why it matters, recommended response, and source URLs."
  depth: standard
  outputType: sourcedAnswer
expected_behavior:
  - Sourced synthesis for business readers.
uses_previous_step: official_competitor_pages
produces:
  - competitor_tracker_summary
```

```yaml
step: research
name: Produce competitor impact report
purpose: Synthesize competitor activity into a strategic update.
linkup.research:
  q: "Research recent activity for these competitors: {competitors}. Compare product launches, pricing or packaging changes, positioning, hiring, partnerships, funding, customer proof, and market implications since {from_date}. Explain what changed, why it matters, recommended response, and source URLs."
  mode: research
  reasoningDepth: M
  outputType: sourcedAnswer
expected_behavior:
  - Long-running multi-source competitor report with impact assessment.
uses_previous_step: false
produces:
  - competitor_impact_report
```

## Output Contract

- `competitor`
- `signal_type`
- `date`
- `evidence`
- `source_url`
- `recommended_response`
- `competitor_impact_report`

## Handoffs

Send updates to Slack, battlecards, sales enablement, or product planning.

## Failure Modes

- Many competitor changes are subtle and require known URLs to monitor.
- Search results may repeat syndicated news.
- The workflow should separate confirmed changes from weak signals.
- Use `/v1/research` when leadership or product marketing needs a written impact report.
