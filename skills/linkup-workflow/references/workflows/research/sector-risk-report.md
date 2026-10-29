---
id: research.sector_risk_report
title: Sector Risk Report
category: research
summary: Produce a sector-specific risk and market watch report for a company or client.
user_goal: I need a sector analysis for a specific company, audit file, or investment review.
inputs:
  - company name
  - sector or activity code
  - geography
  - business context
  - risk areas
outputs:
  - sector trends
  - regulatory risks
  - market risks
  - company-specific implications
  - source URLs
linkup_strength: Synthesizes credible public sources, regulatory context, market signals, and company-specific context.
handoff_tools:
  - audit file
  - investment memo
  - risk register
  - client briefing
---

# Sector Risk Report

Adapted from Linkup's MIT source pack (2026-09-15 snapshot). Read the
[2026-10-29 compatibility corrections](../../sources-and-compatibility.md) before copying API examples.
Examples require an approved connection/budget; do not run paid calls for validation.

## What The User Gets

A structured sector watch report tailored to one company or client, including trends, regulation,
risks, and implications for the business.

## When To Use

Use for audit teams, investors, consultants, or operators who need a long-running public-source
analysis, not a simple company profile.

## Inputs To Ask For

- Company name and location.
- Sector, NAF/NACE/SIC code, or activity description.
- Business context from the user.
- Geography and export markets.
- Risk areas, such as regulation, sanctions, demand, competition, supply chain, or labor.

## Linkup Workflow

```yaml
step: research
name: Produce sector risk report
purpose: Investigate market, regulatory, and company-specific risks across credible sources.
linkup.research:
  q: "Produce a sector risk report for {company_name}, active in {sector_or_activity_code}, located in {geography}. Use this business context to target the research: {business_context}. Analyze market trends, demand, competitors, regulation, export or sanctions exposure, supply chain, labor, and company-specific implications. Prefer official, regulatory, industry, and credible news sources. Return risks, opportunities, evidence, confidence, and source URLs."
  mode: research
  reasoningDepth: L
  outputType: structured
  structuredOutputSchema:
    type: object
    properties:
      risks:
        type: array
        items:
          type: object
          properties:
            risk_area:
              type: string
            finding:
              type: string
            company_implication:
              type: string
            confidence:
              type: string
            source_urls:
              type: array
              items:
                type: string
          required:
            - risk_area
            - finding
            - source_urls
      opportunities:
        type: array
        items:
          type: object
          properties:
            opportunity:
              type: string
            evidence:
              type: string
            source_urls:
              type: array
              items:
                type: string
          required:
            - opportunity
            - evidence
            - source_urls
    required:
      - risks
expected_behavior:
  - Long-running structured sector analysis with credible sources and confidence labels.
uses_previous_step: false
produces:
  - sector_risk_report
```

## Output Contract

- `risk_area`
- `finding`
- `company_implication`
- `confidence`
- `source_urls`
- `opportunity`

## Handoffs

Send the report to an audit file, investment memo, risk register, or client briefing.

## Failure Modes

- The business context may be too generic to target the research.
- Regulatory sources vary by country and sector.
- The workflow should label uncertainty and not overstate company-specific impact from sector-level
  evidence.
- Use `/v1/research` because this workflow is a long-form sector investigation.
