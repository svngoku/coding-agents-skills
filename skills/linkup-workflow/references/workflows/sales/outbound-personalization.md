---
id: sales.outbound_personalization
title: Outbound Personalization
category: sales
summary: Generate source-backed outreach angles for a target account and buyer.
user_goal: I want better outbound messages based on public evidence.
inputs:
  - company name
  - company website
  - buyer name or role
  - product being sold
  - value proposition
outputs:
  - personalization angles
  - evidence snippets
  - source URLs
  - draft talking points
linkup_strength: Finds recent public context and account-specific evidence for outreach.
handoff_tools:
  - outbound sequencer
  - CRM
  - sales copilot
---

# Outbound Personalization

Adapted from Linkup's MIT source pack (2026-09-15 snapshot). Read the
[2026-10-29 compatibility corrections](../../sources-and-compatibility.md) before copying API examples.
Examples require an approved connection/budget; do not run paid calls for validation.

## What The User Gets

Evidence-backed angles for outreach: recent company activity, buyer-role relevance, pain points, and
short talking points with citations.

## When To Use

Use this after account enrichment and buyer discovery, or when the user already has a target account
and buyer role.

## Inputs To Ask For

- Company name and website.
- Buyer name or buyer role.
- Product and value proposition.
- Tone and channel if a message draft is needed.

## Linkup Workflow

```yaml
step: 1
name: Find account signals
purpose: Collect timely and relevant company context.
linkup.search:
  q: "Find recent public signals for {company_name} that could make {value_proposition} relevant. Run separate web searches for: {company_name} recent news, {company_name} product launches, {company_name} hiring, and {company_name} customer or case studies. Return signal, why it matters, date if available, and source URLs."
  depth: standard
  outputType: searchResults
expected_behavior:
  - Multiple web search calls for independent signal categories.
uses_previous_step: false
produces:
  - account_signals
```

```yaml
step: 2
name: Connect signals to buyer role
purpose: Explain why the buyer would care.
linkup.search:
  q: "Using these account signals for {company_name}: {account_signals}, explain which signals matter to {buyer_role} for a product that {value_proposition}. Return personalization angle, buyer relevance, evidence snippet, and source URL."
  depth: standard
  outputType: sourcedAnswer
expected_behavior:
  - Sourced synthesis from prior search results.
uses_previous_step: account_signals
produces:
  - personalization_angles
```

```yaml
step: 3
name: Create outreach notes
purpose: Produce usable material for a human or sequencer.
linkup.search:
  q: "Turn these personalization angles into outreach notes for {buyer_role} at {company_name}. Return 3 concise opening lines, the evidence behind each line, source URL, and risk if the evidence is weak. Do not invent facts."
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
            opening_line:
              type: string
            evidence:
              type: string
            source_url:
              type: string
            risk:
              type: string
          required:
            - opening_line
            - evidence
            - source_url
    required:
      - angles
expected_behavior:
  - Structured synthesis; no new unsupported claims.
uses_previous_step: personalization_angles
produces:
  - outreach_notes
```

## Output Contract

- `opening_line`
- `evidence`
- `source_url`
- `buyer_relevance`
- `risk`

## Handoffs

Send outreach notes to an outbound sequencer, CRM, or sales copilot. The sending system should handle
message generation, approvals, and delivery.

## Failure Modes

- Company has no recent public signals.
- Signals are generic and not useful for personalization.
- The workflow should mark weak evidence rather than forcing a personalized angle.
