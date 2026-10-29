---
id: research.funding_news_monitor
title: Funding and News Monitor
category: research
summary: Track companies, markets, or competitors for funding, launches, partnerships, and news.
user_goal: I want to monitor public news about companies or a market.
inputs:
  - companies or market
  - signal types
  - date range
  - geography
outputs:
  - news items
  - funding signals
  - source URLs
  - impact notes
linkup_strength: Finds recent public news and source-backed company events.
handoff_tools:
  - Slack alert
  - CRM
  - investor database
  - newsletter
---

# Funding and News Monitor

Adapted from Linkup's MIT source pack (2026-09-15 snapshot). Read the
[2026-10-29 compatibility corrections](../../sources-and-compatibility.md) before copying API examples.
Examples require an approved connection/budget; do not run paid calls for validation.

## What The User Gets

A recurring source-backed feed of funding, launches, partnerships, acquisitions, leadership changes,
or market news.

## When To Use

Use for investor research, sales triggers, competitor monitoring, market intelligence, or newsletters.

## Inputs To Ask For

- Company list or market.
- Signal types.
- Date range.
- Geography.
- Source preferences.

## Linkup Workflow

Use Search API steps for alerts and feeds. Use the Research API when the user wants an event
investigation, probability-style assessment, or synthesized market/news report.

```yaml
step: 1
name: Find recent events
purpose: Collect public signals in the requested window.
linkup.search:
  q: "Find recent funding and news signals for {companies_or_market} in {geography}. Run separate web searches for: funding announcements, product launches, partnerships, acquisitions, and leadership changes. Return entity, event type, date, summary, and source URLs."
  depth: standard
  outputType: searchResults
expected_behavior:
  - Multiple web search calls with the entity or market preserved in each facet.
uses_previous_step: false
produces:
  - raw_events
```

```yaml
step: 2
name: Verify and deduplicate
purpose: Remove repeated or weak signals.
linkup.search:
  q: "Deduplicate and verify these public events: {raw_events}. Prefer primary sources, company blogs, investor announcements, credible news, and official filings. Return entity, event type, canonical source URL, duplicate URLs, and confidence."
  depth: standard
  outputType: structured
  structuredOutputSchema:
    type: object
    properties:
      events:
        type: array
        items:
          type: object
          properties:
            entity:
              type: string
            event_type:
              type: string
            canonical_source_url:
              type: string
            confidence:
              type: string
            summary:
              type: string
          required:
            - entity
            - event_type
            - canonical_source_url
            - confidence
    required:
      - events
expected_behavior:
  - Structured synthesis from raw search results.
uses_previous_step: raw_events
produces:
  - verified_events
```

```yaml
step: 3
name: Write monitor summary
purpose: Turn events into a useful update.
linkup.search:
  q: "Summarize these verified events for {audience}: {verified_events}. Include what happened, why it matters, recommended follow-up, and source URLs."
  depth: standard
  outputType: sourcedAnswer
expected_behavior:
  - User-facing summary with citations.
uses_previous_step: verified_events
produces:
  - monitor_summary
```

```yaml
step: research
name: Investigate event or news context
purpose: Gather current status, latest facts, official sources, base rates, and impact in one long-running report.
linkup.research:
  q: "Investigate {event_or_market_question}. Gather current status, latest relevant facts and news, applicable base rates or historical context, scheduled dates, official results or announcements, primary sources, and credible reporting. Do not report prediction-market prices or betting odds unless explicitly requested."
  mode: investigate
  reasoningDepth: L
  outputType: sourcedAnswer
expected_behavior:
  - Long-running event investigation with primary-source citations and uncertainty.
uses_previous_step: false
produces:
  - event_investigation_report
```

## Output Contract

- `entity`
- `event_type`
- `date`
- `summary`
- `canonical_source_url`
- `confidence`
- `recommended_follow_up`
- `event_investigation_report`

## Handoffs

Send events to Slack, CRM, investor databases, or newsletters.

## Failure Modes

- Syndicated news can create duplicates.
- Some funding signals are rumors; label confidence.
- State the desired time window in the query for recurring monitors.
- Use `/v1/research` when the user asks for an assessment of an uncertain event, not just alerts.
