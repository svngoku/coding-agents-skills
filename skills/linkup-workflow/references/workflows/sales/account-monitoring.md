---
id: sales.account_monitoring
title: Account Monitoring
category: sales
summary: Monitor target accounts for public signals that suggest timing or fit.
user_goal: I want to know when target accounts show buying signals.
inputs:
  - target account list
  - signal types
  - date range
  - buyer roles
outputs:
  - account signals
  - signal evidence
  - urgency score
  - recommended follow-up
linkup_strength: Searches recent public web sources for account-level changes and news.
handoff_tools:
  - CRM
  - Slack alert
  - outbound sequencer
  - account enrichment agent
---

# Account Monitoring

Adapted from Linkup's MIT source pack (2026-09-15 snapshot). Read the
[2026-10-29 compatibility corrections](../../sources-and-compatibility.md) before copying API examples.
Examples require an approved connection/budget; do not run paid calls for validation.

## What The User Gets

A recurring list of target-account signals, such as hiring, launches, funding, leadership changes,
new partnerships, security incidents, or market expansion.

## When To Use

Use this for sales teams that already have target accounts and want timely reasons to reach out.

## Inputs To Ask For

- Target account list.
- Signal categories.
- Date range.
- Geography if relevant.
- What the user sells.

## Linkup Workflow

```yaml
step: 1
name: Search recent signals
purpose: Find account-level events.
linkup.search:
  q: "Find recent public signals for these target accounts: {target_accounts}. Run separate web searches for: {target_accounts} hiring, {target_accounts} product launch, {target_accounts} funding or acquisition, and {target_accounts} leadership change. Return account, signal type, date, evidence, and source URLs."
  depth: standard
  outputType: searchResults
expected_behavior:
  - Multiple web search calls with the account names preserved in each facet.
  - Recurring runs over a long account list: submit one search per account (or small group of accounts) through /v1/tasks each cycle, then poll and diff against the previous run.
uses_previous_step: false
produces:
  - raw_account_signals
```

```yaml
step: 2
name: Score signal relevance
purpose: Decide which signals matter for outreach.
linkup.search:
  q: "Using these recent account signals: {raw_account_signals}, score which ones are relevant for selling {product_category} to {buyer_roles}. Return account, signal, why it matters, urgency, recommended follow-up, and source URLs."
  depth: standard
  outputType: structured
  structuredOutputSchema:
    type: object
    properties:
      signals:
        type: array
        items:
          type: object
          properties:
            account:
              type: string
            signal:
              type: string
            urgency:
              type: string
            reason:
              type: string
            source_url:
              type: string
            recommended_follow_up:
              type: string
          required:
            - account
            - signal
            - reason
            - source_url
    required:
      - signals
expected_behavior:
  - Structured synthesis from the search results.
uses_previous_step: raw_account_signals
produces:
  - scored_signals
```

```yaml
step: 3
name: Prepare alert summary
purpose: Make the result easy for a sales team to act on.
linkup.search:
  q: "Summarize these scored account signals for a sales team: {scored_signals}. Group by urgency and include account, signal, recommended action, and source URL."
  depth: standard
  outputType: sourcedAnswer
expected_behavior:
  - User-facing summary with citations.
uses_previous_step: scored_signals
produces:
  - sales_alert_summary
```

## Output Contract

- `account`
- `signal`
- `signal_type`
- `date`
- `urgency`
- `source_url`
- `recommended_follow_up`

## Handoffs

Send high-urgency signals to CRM tasks, Slack alerts, or an outbound personalization workflow.

## Failure Modes

- Public news may lag behind actual buying intent.
- Generic company news may not be relevant to the product.
- The workflow should preserve target account names in each search facet to avoid unrelated news.
