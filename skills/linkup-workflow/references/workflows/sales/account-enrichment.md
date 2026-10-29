---
id: sales.account_enrichment
title: Account Enrichment
category: sales
summary: Enrich a company with public evidence, fit signals, and useful sales context.
user_goal: I have a company list and want to know which accounts are worth pursuing.
inputs:
  - company name
  - company website
  - ideal customer profile
  - buyer roles
outputs:
  - company profile
  - product and customer summary
  - fit score
  - evidence URLs
linkup_strength: Scrapes known company websites and combines them with public web context.
handoff_tools:
  - CRM
  - lead scoring system
  - outbound personalization agent
---

# Account Enrichment

Adapted from Linkup's MIT source pack (2026-09-15 snapshot). Read the
[2026-10-29 compatibility corrections](../../sources-and-compatibility.md) before copying API examples.
Examples require an approved connection/budget; do not run paid calls for validation.

## What The User Gets

A source-backed profile for one account or a batch of accounts, including what the company does, who
it serves, and why it may fit the user's product.

## When To Use

Use this after the user already has company names or websites.

## Inputs To Ask For

- Company name and website.
- The product being sold.
- Fit criteria.
- Buyer roles.
- Any disqualifiers.

## Linkup Workflow

```yaml
step: 1
name: Read the known company website
purpose: Get official company positioning and product context.
linkup.fetch:
  url: "{company_website}"
  renderJs: true
  schema:
    type: object
    properties:
      description:
        type: string
        description: What the company does, in its own words
      product_categories:
        type: array
        items:
          type: string
      target_customers:
        type: array
        items:
          type: string
      pricing_evidence:
        type: string
        description: Plan names or pricing signals if shown on the page
      integrations:
        type: array
        items:
          type: string
      case_study_customers:
        type: array
        items:
          type: string
expected_behavior:
  - Single page read of the known URL; no search. Retry with mode "pro" if the markdown is empty.
  - For a batch of accounts, submit one fetch per account through /v1/tasks (up to 100 per submission).
uses_previous_step: false
produces:
  - official_company_context
```

```yaml
step: 2
name: Add external context
purpose: Verify the account with sources outside the company website.
linkup.search:
  q: "Scrape {company_website}. Also run separate web searches for: {company_name} company profile, {company_name} customers, {company_name} funding or news, and {company_name} careers. Return fit evidence, company size signals, recent activity, and source URLs."
  depth: standard
  outputType: searchResults
expected_behavior:
  - One web scrape plus multiple web search calls in one standard request.
  - For a batch of accounts, submit one search per account through /v1/tasks.
uses_previous_step: official_company_context
produces:
  - external_company_context
```

```yaml
step: 3
name: Score fit
purpose: Summarize whether the account is worth pursuing.
linkup.search:
  q: "Using this evidence about {company_name}: {official_company_context} and {external_company_context}, decide whether it matches this ICP: {ideal_customer_profile}. Return fit score, reasons to pursue, risks, missing information, and source URLs."
  depth: standard
  outputType: sourcedAnswer
expected_behavior:
  - Sourced synthesis from collected evidence.
uses_previous_step: external_company_context
produces:
  - account_fit_summary
```

## Output Contract

- `company_name`
- `website`
- `description`
- `target_customers`
- `fit_score`
- `fit_evidence`
- `risks`
- `source_urls`

## Handoffs

Write the account profile and fit score to CRM fields, then pass promising accounts to buyer
discovery or outbound personalization.

## Failure Modes

- The website is stale or too thin; use external searches to compensate.
- The fetch returns empty or truncated markdown; retry with `renderJs: true` and then `mode: "pro"`.
- The company name is ambiguous; quote the exact company name and include the website.
- The account cannot be scored confidently; mark missing evidence instead of inventing fit.
