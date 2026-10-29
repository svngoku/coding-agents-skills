---
id: sales.email_discovery_handoff
title: Email Discovery Handoff
category: sales
summary: Prepare buyer and company evidence for an email-finding or verification provider.
user_goal: I found buyers and need a reliable way to continue toward emails.
inputs:
  - verified buyers
  - company domain
  - source URLs
outputs:
  - email-finder input rows
  - public email evidence if available
  - verification notes
linkup_strength: Finds public contact pages, domains, profile URLs, and evidence for downstream email tools.
handoff_tools:
  - email finder
  - email verifier
  - CRM
---

# Email Discovery Handoff

Adapted from Linkup's MIT source pack (2026-09-15 snapshot). Read the
[2026-10-29 compatibility corrections](../../sources-and-compatibility.md) before copying API examples.
Examples require an approved connection/budget; do not run paid calls for validation.

## What The User Gets

A clean handoff package for an email-finding tool: person name, company domain, role, public profile
URL, and any public contact-page evidence Linkup can find.

## When To Use

Use this after buyer discovery. Linkup should prepare evidence and inputs; it should not be presented
as a guaranteed private email database.

## Inputs To Ask For

- Buyer names and profile URLs.
- Company website or domain.
- Company name.
- Required verification standard.

## Linkup Workflow

```yaml
step: 1
name: Confirm company domain
purpose: Make sure the email finder receives the right domain.
linkup.search:
  q: "Scrape {company_website}. Also run a separate web search for {company_name} official website. Return official domain, company name, contact page URL if present, and source URLs."
  depth: standard
  outputType: structured
  structuredOutputSchema:
    type: object
    properties:
      company_name:
        type: string
      domain:
        type: string
      contact_page_url:
        type:
          - string
          - "null"
      source_urls:
        type: array
        items:
          type: string
    required:
      - company_name
      - domain
      - source_urls
expected_behavior:
  - Web scraping for known website plus web search for official-domain confirmation.
uses_previous_step: false
produces:
  - verified_domain
```

```yaml
step: 2
name: Search public contact evidence
purpose: Find public pages that might contain emails or contact patterns.
linkup.search:
  q: "Find public contact information for {company_name}. Run separate web searches for: {company_name} contact page, {company_name} press contact, {company_name} leadership email, and {company_domain} email format. Return public email addresses only if present on public pages, contact URLs, and source URLs."
  depth: standard
  outputType: searchResults
expected_behavior:
  - Multiple web search calls for contact-related public pages.
uses_previous_step: verified_domain
produces:
  - public_contact_evidence
```

```yaml
step: 3
name: Build handoff rows
purpose: Prepare clean rows for another provider.
linkup.search:
  q: "Using these verified buyers: {verified_buyers}, company domain: {verified_domain}, and public contact evidence: {public_contact_evidence}, create email-finder input rows. Return person name, title, company, domain, profile URL, evidence URL, and notes. Do not invent email addresses."
  depth: standard
  outputType: structured
  structuredOutputSchema:
    type: object
    properties:
      rows:
        type: array
        items:
          type: object
          properties:
            person_name:
              type: string
            title:
              type: string
            company:
              type: string
            domain:
              type: string
            profile_url:
              type:
                - string
                - "null"
            evidence_url:
              type:
                - string
                - "null"
            notes:
              type: string
          required:
            - person_name
            - company
            - domain
    required:
      - rows
expected_behavior:
  - Structured synthesis from previous outputs.
uses_previous_step: verified_buyers
produces:
  - email_finder_rows
```

## Output Contract

- `person_name`
- `title`
- `company`
- `domain`
- `profile_url`
- `evidence_url`
- `notes`

## Handoffs

Send rows to an email finder or verifier. Write verified emails back to CRM after the external tool
returns results.

## Failure Modes

- No public email evidence exists.
- The company has multiple domains.
- The buyer's profile is public but not enough to verify employment.
- The workflow must not invent emails from name patterns.
