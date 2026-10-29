---
id: marketing.seo_source_research
title: SEO Source Research
category: marketing
summary: Collect credible sources and examples for an SEO page or article.
user_goal: I want a search-informed article brief with sources, not just keywords.
inputs:
  - keyword or topic
  - target audience
  - source preferences
  - geography
outputs:
  - source list
  - questions to answer
  - competing pages
  - evidence-backed outline
linkup_strength: Finds current web sources, competing pages, and credible references for content.
handoff_tools:
  - SEO tool
  - content writer
  - CMS
---

# SEO Source Research

Adapted from Linkup's MIT source pack (2026-09-15 snapshot). Read the
[2026-10-29 compatibility corrections](../../sources-and-compatibility.md) before copying API examples.
Examples require an approved connection/budget; do not run paid calls for validation.

## What The User Gets

A source-backed research pack for an SEO article or landing page.

## When To Use

Use when the user wants to produce content that answers real questions with credible citations.

## Inputs To Ask For

- Topic or keyword.
- Target reader.
- Geography.
- Preferred sources.
- Product or brand angle.

## Linkup Workflow

```yaml
step: 1
name: Find competing and reference pages
purpose: Understand the source landscape.
linkup.search:
  q: "Find useful pages for an SEO article about {topic} for {target_reader}. Run separate web searches for: top explanatory pages, credible reference sources, recent reports or statistics, and common questions. Return title, URL, source type, useful angle, and source URLs."
  depth: standard
  outputType: searchResults
expected_behavior:
  - Multiple web search calls for source categories.
uses_previous_step: false
produces:
  - seo_sources
```

```yaml
step: 2
name: Build source-backed outline
purpose: Turn research into a writing plan.
linkup.search:
  q: "Using these SEO sources for {topic}: {seo_sources}, create a source-backed outline for {target_reader}. Include sections, questions to answer, evidence to cite, and source URLs."
  depth: standard
  outputType: sourcedAnswer
expected_behavior:
  - User-facing synthesis with citations.
uses_previous_step: seo_sources
produces:
  - seo_outline
```

## Output Contract

- `section`
- `question`
- `claim`
- `source_url`
- `source_type`

## Handoffs

Send the outline to a content writer, SEO platform, or CMS.

## Failure Modes

- Search results may be dominated by low-quality SEO pages.
- The workflow should prefer credible references over content farms.
- Use source filtering only if you know exactly the URLs or domains you are targeting or not
  targeting.
