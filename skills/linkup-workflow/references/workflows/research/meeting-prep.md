---
id: research.meeting_prep
title: Meeting Prep
category: research
summary: Prepare for a meeting using public information about people, companies, and recent context.
user_goal: I have a meeting and want useful prep notes.
inputs:
  - attendee names
  - professional emails or company domains
  - meeting goal
outputs:
  - attendee profiles
  - company context
  - recent news
  - suggested questions
linkup_strength: Finds public company and person context with source URLs.
handoff_tools:
  - calendar assistant
  - CRM
  - notes app
---

# Meeting Prep

Adapted from Linkup's MIT source pack (2026-09-15 snapshot). Read the
[2026-10-29 compatibility corrections](../../sources-and-compatibility.md) before copying API examples.
Examples require an approved connection/budget; do not run paid calls for validation.

## What The User Gets

A concise prep note with attendee context, company background, recent developments, and suggested
talking points.

## When To Use

Use before sales calls, partnership calls, customer interviews, investor meetings, or hiring calls.

## Inputs To Ask For

- Attendee names or emails.
- Company names or domains.
- Meeting goal.
- Any known context.

## Linkup Workflow

```yaml
step: 1
name: Resolve company context
purpose: Identify the organization behind the meeting.
linkup.search:
  q: "Find public information about {company_name_or_domain}. Scrape the official website if provided. Also run separate web searches for company profile, product description, leadership, and recent news. Return company summary, product, audience, recent context, and source URLs."
  depth: standard
  outputType: searchResults
expected_behavior:
  - Web scraping if a known URL is provided plus multiple web search calls.
uses_previous_step: false
produces:
  - company_context
```

```yaml
step: 2
name: Research attendees
purpose: Gather public person context without overreaching.
linkup.search:
  q: "Find public professional information about these meeting attendees: {attendees}. Search for role, company, public profile URLs, recent public work, and source URLs. Extract LinkedIn profile URLs as text evidence only. Do not retrieve LinkedIn posts."
  depth: standard
  outputType: searchResults
expected_behavior:
  - web search calls for public profiles; no LinkedIn post fetching unless explicitly requested.
uses_previous_step: company_context
produces:
  - attendee_context
```

```yaml
step: 3
name: Create prep note
purpose: Turn research into a useful meeting artifact.
linkup.search:
  q: "Create meeting prep notes for a meeting with {attendees} about {meeting_goal}. Use company context {company_context} and attendee context {attendee_context}. Include why the meeting matters, recent context, likely priorities, suggested questions, and source URLs."
  depth: standard
  outputType: sourcedAnswer
expected_behavior:
  - User-facing synthesis with citations.
uses_previous_step: attendee_context
produces:
  - meeting_prep_note
```

## Output Contract

- `attendee`
- `role`
- `company`
- `recent_context`
- `suggested_question`
- `source_url`

## Handoffs

Send prep notes to calendar, CRM, or a notes app.

## Failure Modes

- Professional emails may not map cleanly to public profiles.
- People with common names need company/domain disambiguation.
- The workflow should avoid private or invasive personal research.
