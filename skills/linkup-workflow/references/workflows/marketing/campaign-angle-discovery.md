---
id: marketing.campaign_angle_discovery
title: Campaign Angle Discovery
category: marketing
summary: Find timely campaign angles from market news, customer pain, and competitor activity.
user_goal: I need ideas for a campaign that connects to what is happening in the market.
inputs:
  - product category
  - target audience
  - geography
  - time window
outputs:
  - campaign angles
  - supporting evidence
  - source URLs
  - recommended audience
linkup_strength: Finds current market signals and turns them into evidence-backed campaign ideas.
handoff_tools:
  - campaign planning
  - content calendar
  - ad platform
---

# Campaign Angle Discovery

Adapted from Linkup's MIT source pack (2026-09-15 snapshot). Read the
[2026-10-29 compatibility corrections](../../sources-and-compatibility.md) before copying API examples.
Examples require an approved connection/budget; do not run paid calls for validation.

## What The User Gets

A shortlist of campaign angles grounded in current market signals, public pain points, and competitor
activity.

## When To Use

Use when a marketing team needs campaign ideas, not just keyword lists.

## Inputs To Ask For

- Product category.
- Target audience.
- Geography.
- Time window.
- Brand point of view.

## Linkup Workflow

```yaml
step: 1
name: Find market signals
purpose: Discover timely hooks.
linkup.search:
  q: "Find current market signals for {product_category} and {target_audience} in {geography}. Run separate web searches for: recent news, customer pain points, regulatory or market changes, and competitor launches. Return signal, audience affected, date, and source URLs."
  depth: standard
  outputType: searchResults
expected_behavior:
  - Multiple web search calls across signal categories.
uses_previous_step: false
produces:
  - market_signals
```

```yaml
step: 2
name: Turn signals into angles
purpose: Connect evidence to campaign ideas.
linkup.search:
  q: "Using these market signals: {market_signals}, generate campaign angles for {target_audience}. For each angle, include the market truth, why now, suggested message, risk, and source URLs."
  depth: standard
  outputType: sourcedAnswer
expected_behavior:
  - Sourced synthesis from signal evidence.
uses_previous_step: market_signals
produces:
  - campaign_angles
```

```yaml
step: 3
name: Package campaign brief
purpose: Make the output usable for marketing execution.
linkup.search:
  q: "Create a campaign brief from these angles: {campaign_angles}. Return angle name, audience, message, evidence, suggested channels, and source URLs."
  depth: standard
  outputType: structured
  structuredOutputSchema:
    type: object
    properties:
      angles:
        type: array
        items:
          type: object
          properties:
            angle_name:
              type: string
            audience:
              type: string
            message:
              type: string
            evidence:
              type: string
            source_urls:
              type: array
              items:
                type: string
          required:
            - angle_name
            - message
            - source_urls
    required:
      - angles
expected_behavior:
  - Structured campaign planning output.
uses_previous_step: campaign_angles
produces:
  - campaign_brief
```

## Output Contract

- `angle_name`
- `audience`
- `message`
- `evidence`
- `source_urls`
- `suggested_channels`

## Handoffs

Send campaign angles to a content calendar, ad planning workflow, or creative brief.

## Failure Modes

- Signals are too generic.
- The user has not specified an audience.
- The workflow should not claim market momentum without public evidence.
