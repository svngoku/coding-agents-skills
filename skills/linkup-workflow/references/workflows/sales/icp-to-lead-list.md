---
id: sales.icp_to_lead_list
title: ICP to Lead List
category: sales
summary: Find companies that match a target customer profile and explain why they fit.
user_goal: I want to find leads for my business.
inputs:
  - ideal customer profile
  - geography
  - company size
  - target industry
  - buyer roles
outputs:
  - qualified company list
  - fit evidence
  - source URLs
  - recommended next step
linkup_strength: Finds and verifies companies from public web evidence.
handoff_tools:
  - CRM
  - spreadsheet
  - account enrichment agent
---

# ICP to Lead List

Adapted from Linkup's MIT source pack (2026-09-15 snapshot). Read the
[2026-10-29 compatibility corrections](../../sources-and-compatibility.md) before copying API examples.
Examples require an approved connection/budget; do not run paid calls for validation.

## What The User Gets

A sourced list of companies that match an ideal customer profile, with public evidence explaining why
each company is a fit.

## When To Use

Use this when the user can describe a target customer but does not already have a company list.

## Inputs To Ask For

- Target customer description.
- Geography.
- Company size or stage.
- Industry or category.
- Buyer roles.
- Exclusions, such as competitors, existing customers, or unsupported countries.

## Linkup Workflow

```yaml
step: 1
name: Discover candidate companies
purpose: Build the first candidate account list.
linkup.search:
  q: "Find companies matching this ideal customer profile: {ideal_customer_profile}. Focus on {geography}, {company_size}, and {industry}. Run separate web searches for: {industry} startups {geography}, {category} companies {company_size}, and {buyer_problem} software companies. Return company name, website, location, short fit reason, and source URLs."
  depth: standard
  outputType: searchResults
expected_behavior:
  - Multiple independent web search calls because facets are explicit.
uses_previous_step: false
produces:
  - candidate_companies
```

```yaml
step: 2
name: Verify company fit
purpose: Remove weak matches and collect evidence.
linkup.search:
  q: "For each company in {candidate_companies}, find public evidence that it matches {ideal_customer_profile}. Run separate web searches for company website, product description, customer segment, and recent news. Return company name, website, fit evidence, disqualifying evidence, and source URLs."
  depth: standard
  outputType: structured
  structuredOutputSchema:
    type: object
    properties:
      companies:
        type: array
        items:
          type: object
          properties:
            company_name:
              type: string
            website:
              type: string
            fit_reason:
              type: string
            source_urls:
              type: array
              items:
                type: string
            confidence:
              type: string
          required:
            - company_name
            - fit_reason
            - source_urls
    required:
      - companies
expected_behavior:
  - web search calls scoped to each candidate or batch of candidates.
uses_previous_step: candidate_companies
produces:
  - qualified_companies
```

```yaml
step: 3
name: Suggest next action
purpose: Turn the list into a usable sales artifact.
linkup.search:
  q: "Using this qualified company list: {qualified_companies}, summarize the best accounts to prioritize. For each company, explain the reason to contact them, likely buyer roles from {buyer_roles}, and source URLs."
  depth: standard
  outputType: sourcedAnswer
expected_behavior:
  - Uses sourced evidence collected in prior steps.
uses_previous_step: qualified_companies
produces:
  - prioritized_lead_list
```

## Output Contract

- `company_name`
- `website`
- `location`
- `fit_reason`
- `source_urls`
- `confidence`
- `recommended_next_step`

## Handoffs

Send qualified companies to a CRM, spreadsheet, or account enrichment workflow.

## Failure Modes

- The ICP is too broad, producing generic company lists.
- The geography is missing, causing irrelevant regions.
- Company size is not public, so the workflow should use public proxies such as funding stage,
  LinkedIn size snippets, careers pages, or startup databases.
