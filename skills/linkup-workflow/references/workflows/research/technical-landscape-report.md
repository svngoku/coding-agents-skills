---
id: research.technical_landscape_report
title: Technical Landscape Report
category: research
summary: Compare tools, frameworks, protocols, or engineering practices from docs and community sources.
user_goal: I want to understand the current state of a technical category or implementation choice.
inputs:
  - technical topic
  - evaluation criteria
  - source preferences
  - time window
outputs:
  - tool or framework comparison
  - tradeoffs
  - recommended options
  - source URLs
linkup_strength: Synthesizes official docs, developer discussions, examples, and current best practices.
handoff_tools:
  - engineering design doc
  - vendor shortlist
  - implementation plan
---

# Technical Landscape Report

Adapted from Linkup's MIT source pack (2026-09-15 snapshot). Read the
[2026-10-29 compatibility corrections](../../sources-and-compatibility.md) before copying API examples.
Examples require an approved connection/budget; do not run paid calls for validation.

## What The User Gets

A source-backed report comparing tools, frameworks, protocols, or engineering approaches, including
tradeoffs and current community sentiment.

## When To Use

Use when the user asks broad technical questions such as which frameworks are recommended, how teams
solve an engineering problem, or what the state of the art looks like.

## Inputs To Ask For

- Technical topic.
- Evaluation criteria.
- Required source types, such as official docs, GitHub, Reddit, or developer forums.
- Time window.
- Constraints, such as language, deployment environment, budget, or scale.

## Linkup Workflow

```yaml
step: research
name: Produce technical landscape report
purpose: Investigate a technical space across official and community sources.
linkup.research:
  q: "Research {technical_topic}. Compare the most useful tools, frameworks, protocols, or approaches according to official documentation, developer discussions, Reddit or community sources if relevant, GitHub examples, and recent best practices. Evaluate them on {evaluation_criteria}. Return recommended options, tradeoffs, risks, maturity, and source URLs."
  mode: research
  reasoningDepth: L
  outputType: sourcedAnswer
expected_behavior:
  - Long-running multi-source technical investigation with citations and tradeoffs.
uses_previous_step: false
produces:
  - technical_landscape_report
```

```yaml
step: 2
name: Extract shortlist
purpose: Convert the report into actionable options.
linkup.search:
  q: "Using this technical landscape report: {technical_landscape_report}, create a shortlist of the best options for {technical_topic}. Return option name, when to use it, tradeoffs, maturity, and source URLs."
  depth: standard
  outputType: structured
  structuredOutputSchema:
    type: object
    properties:
      options:
        type: array
        items:
          type: object
          properties:
            name:
              type: string
            when_to_use:
              type: string
            tradeoffs:
              type: string
            maturity:
              type: string
            source_urls:
              type: array
              items:
                type: string
          required:
            - name
            - when_to_use
            - source_urls
    required:
      - options
expected_behavior:
  - Structured synthesis from the research report.
uses_previous_step: technical_landscape_report
produces:
  - technical_shortlist
```

## Output Contract

- `name`
- `when_to_use`
- `tradeoffs`
- `maturity`
- `source_urls`
- `recommendation`

## Handoffs

Send the report to an engineering design document, vendor selection process, or implementation plan.

## Failure Modes

- Community discussions can be anecdotal.
- Tool names can be ambiguous; quote exact names and add a technical disambiguator.
- Fast-moving topics need a clear time window.
- Use `/v1/research` for the landscape report; use `/v1/search` only for follow-up extraction or
  shortlist formatting.
