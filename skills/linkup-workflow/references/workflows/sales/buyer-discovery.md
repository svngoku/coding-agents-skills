---
id: sales.buyer_discovery
title: Buyer Discovery
category: sales
summary: Identify likely buyer roles and public profile evidence for a target account.
user_goal: I know the account, but I need to find the right people or roles to contact.
inputs:
  - company name
  - company website
  - buyer roles
  - product being sold
outputs:
  - likely buyer roles
  - candidate people
  - profile URLs
  - evidence URLs
linkup_strength: Finds public people, role, team, and profile evidence from the web.
handoff_tools:
  - CRM
  - LinkedIn workflow
  - email finder
---

# Buyer Discovery

Adapted from Linkup's MIT source pack (2026-09-15 snapshot). Read the
[2026-10-29 compatibility corrections](../../sources-and-compatibility.md) before copying API examples.
Examples require an approved connection/budget; do not run paid calls for validation.

## What The User Gets

A list of likely buyers at a target account, with role evidence and public profile URLs where
available.

## When To Use

Use this after account enrichment or when the user already knows the target account.

## Inputs To Ask For

- Company name and website.
- Buyer roles or departments.
- Product category being sold.
- Geography or business unit if relevant.

## Linkup Workflow

```yaml
step: 1
name: Find relevant teams and roles
purpose: Identify which functions likely own the problem.
linkup.search:
  q: "Find the teams and roles at {company_name} that would likely buy {product_category}. Run separate web searches for: {company_name} leadership team, {company_name} {buyer_role_1}, {company_name} {buyer_role_2}, and {company_name} org chart or team page. Return role, person name if found, title, profile URL, and source URLs."
  depth: standard
  outputType: searchResults
expected_behavior:
  - Multiple web search calls for role-specific discovery.
uses_previous_step: false
produces:
  - candidate_buyers
```

```yaml
step: 2
name: Verify public profiles
purpose: Confirm that candidate people match the buyer roles.
linkup.search:
  q: "For these candidate buyers at {company_name}: {candidate_buyers}, find public profile evidence only. Return name, title, department, LinkedIn/profile URL if found, role evidence, and source URLs. Do not retrieve LinkedIn posts."
  depth: standard
  outputType: structured
  structuredOutputSchema:
    type: object
    properties:
      buyers:
        type: array
        items:
          type: object
          properties:
            person_name:
              type: string
            title:
              type: string
            profile_url:
              type: string
            evidence:
              type: string
            source_urls:
              type: array
              items:
                type: string
          required:
            - person_name
            - title
            - evidence
            - source_urls
    required:
      - buyers
expected_behavior:
  - web search for profile discovery; avoids LinkedIn post fetching.
uses_previous_step: candidate_buyers
produces:
  - verified_buyers
```

```yaml
step: 3
name: Recommend buyer map
purpose: Explain who to prioritize.
linkup.search:
  q: "Using this verified buyer evidence for {company_name}: {verified_buyers}, recommend the best buyer roles to contact for {product_category}. Return priority, reason, profile URL, and source URLs."
  depth: standard
  outputType: sourcedAnswer
expected_behavior:
  - Sourced synthesis from verified profile evidence.
uses_previous_step: verified_buyers
produces:
  - buyer_map
```

## Output Contract

- `person_name`
- `title`
- `department`
- `profile_url`
- `buyer_reason`
- `source_urls`
- `priority`

## Handoffs

Send verified buyers to CRM contact records, an email-finding provider, or outbound personalization.

## Failure Modes

- Some companies do not publish team pages.
- LinkedIn profile discovery may be incomplete from public web results.
- Role names vary by company; search for department and responsibility as well as exact titles.
