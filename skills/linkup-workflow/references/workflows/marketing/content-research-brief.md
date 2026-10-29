---
id: marketing.content_research_brief
title: Content Research Brief
category: marketing
summary: Build a source-backed brief for a blog post, guide, or campaign asset.
user_goal: I want to create content grounded in current market evidence.
inputs:
  - topic
  - audience
  - point of view
  - source preferences
outputs:
  - research brief
  - source list
  - claims to use
  - content angles
linkup_strength: Finds fresh sources, citations, and examples for content planning.
handoff_tools:
  - content writer
  - CMS
  - SEO tool
---

# Content Research Brief

Adapted from Linkup's MIT source pack (2026-09-15 snapshot). Read the
[2026-10-29 compatibility corrections](../../sources-and-compatibility.md) before copying API examples.
Examples require an approved connection/budget; do not run paid calls for validation.

## What The User Gets

A research brief with source-backed claims, examples, opposing views, and suggested content angles.

## When To Use

Use before writing a blog post, landing page, white paper, or campaign asset.

## Inputs To Ask For

- Topic.
- Target reader.
- Desired point of view.
- Preferred source types.
- Competitors or sources to exclude.

## Linkup Workflow

```yaml
step: 1
name: Find current sources
purpose: Collect credible source material.
linkup.search:
  q: "Find current sources about {topic} for {audience}. Run separate web searches for: recent industry reports, expert commentary, examples or case studies, and counterarguments. Return title, source type, key claim, date, and source URL."
  depth: standard
  outputType: searchResults
expected_behavior:
  - Multiple web search calls for independent source categories.
uses_previous_step: false
produces:
  - source_pool
```

```yaml
step: 2
name: Extract useful claims
purpose: Turn sources into content inputs.
linkup.search:
  q: "Using these sources about {topic}: {source_pool}, extract the strongest claims, stats, examples, and caveats for {audience}. Return claim, why it matters, source URL, and confidence."
  depth: standard
  outputType: structured
  structuredOutputSchema:
    type: object
    properties:
      claims:
        type: array
        items:
          type: object
          properties:
            claim:
              type: string
            why_it_matters:
              type: string
            source_url:
              type: string
            confidence:
              type: string
          required:
            - claim
            - source_url
    required:
      - claims
expected_behavior:
  - Structured synthesis from the source pool.
uses_previous_step: source_pool
produces:
  - content_claims
```

```yaml
step: 3
name: Create the brief
purpose: Package the research for writing.
linkup.search:
  q: "Create a content research brief about {topic} for {audience} using these claims: {content_claims}. Include narrative angle, outline, claims to cite, caveats, and source URLs."
  depth: standard
  outputType: sourcedAnswer
expected_behavior:
  - User-facing sourced synthesis.
uses_previous_step: content_claims
produces:
  - content_brief
```

## Output Contract

- `topic`
- `angle`
- `outline`
- `claim`
- `source_url`
- `caveat`

## Handoffs

Send the brief to a content writer, SEO workflow, or CMS planning tool.

## Failure Modes

- The topic is too broad.
- Sources are stale or promotional.
- The workflow should separate evidence from suggested positioning.
