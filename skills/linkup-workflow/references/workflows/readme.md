# Linkup Workflow Library

Adapted from Linkup's MIT source pack (2026-09-15 snapshot). Read the
[2026-10-29 compatibility corrections](../sources-and-compatibility.md) before copying API examples.
Examples require an approved connection/budget; do not run paid calls for validation.

This library contains reusable agent workflows powered by Linkup.

Use it when a customer says:

- "How can I use Linkup?"
- "What agents could we offer to our users?"
- "Can Linkup help with sales, marketing, research, or enrichment?"
- "I have a business goal, but I do not know what searches to run."

Each workflow decomposes a business job into a chain of Linkup searches or long-running research
jobs, outputs, and handoffs.

## How This Library Fits Together

- `../linkup-workflow-optimizer-knowledge.md` teaches an agent how to choose and assemble workflows.
- `../linkup-prompt-optimizer-knowledge.md` teaches an agent how to optimize each individual Linkup
  call.
- `workflow-schema.md` defines the standard workflow format.
- Category folders contain the actual workflow templates.

## Categories

### Sales

- [ICP to Lead List](sales/icp-to-lead-list.md): Find companies that match a target customer profile.
- [Account Enrichment](sales/account-enrichment.md): Enrich a company with public evidence.
- [Buyer Discovery](sales/buyer-discovery.md): Find likely buyer roles and public profile evidence.
- [Email Discovery Handoff](sales/email-discovery-handoff.md): Prepare inputs for an email-finding
  provider.
- [Outbound Personalization](sales/outbound-personalization.md): Generate evidence-backed outreach
  angles.
- [Account Monitoring](sales/account-monitoring.md): Monitor target accounts for public buying
  signals.

### Marketing

- [Content Research Brief](marketing/content-research-brief.md): Build a source-backed brief for a
  content piece.
- [Competitor Messaging Scan](marketing/competitor-messaging-scan.md): Compare how competitors
  describe their product and market.
- [Campaign Angle Discovery](marketing/campaign-angle-discovery.md): Find timely campaign angles from
  market signals.
- [Customer Proof Mining](marketing/customer-proof-mining.md): Find public proof points, case studies,
  reviews, and testimonials.
- [SEO Source Research](marketing/seo-source-research.md): Collect credible sources for SEO content.

### Research

- [Meeting Prep](research/meeting-prep.md): Prepare for a meeting from company and person inputs.
- [Company Dossier](research/company-dossier.md): Create a source-backed company profile.
- [Market Map](research/market-map.md): Map a category and its visible players.
- [Competitor Tracker](research/competitor-tracker.md): Monitor competitors for updates.
- [Funding and News Monitor](research/funding-news-monitor.md): Track funding, launches, and news.
- [Technical Landscape Report](research/technical-landscape-report.md): Compare tools, frameworks,
  protocols, and engineering practices.
- [Sector Risk Report](research/sector-risk-report.md): Produce a sector-specific market and risk
  report for a company or client.

## How To Use The Library

For a vague customer request, first select 3-7 workflows that match their goal. Then show each as an
agent template:

1. What the agent produces.
2. What input the user provides.
3. What Linkup searches happen.
4. What the agent writes back.
5. What tool receives the output next.

For a concrete customer goal, select one workflow and fill the placeholders with the customer's
industry, geography, target accounts, buyer personas, or source constraints.

Use `/v1/search` for narrow discovery, enrichment, and handoff fields. Use `/v1/research` for
long-running reports such as market maps, sector risk, technical landscapes, event investigations,
and complex dossiers.

## Example

Customer request:

> We help users run AI agents. What Linkup-powered agents could we suggest for sales teams?

Good answer:

- Lead List Builder: turns an ICP into a sourced list of companies.
- Account Enrichment Agent: researches each company and explains fit.
- Buyer Discovery Agent: identifies likely buyer roles and public profile evidence.
- Outbound Personalization Agent: finds recent signals for tailored outreach.
- Account Monitoring Agent: watches target accounts for triggers.

Each of those should expand into a Linkup prompt chain from the corresponding workflow file.

## Contribution Rules

When adding a workflow:

- follow `workflow-schema.md`
- keep every Linkup step narrow and verifiable
- include source URLs in outputs
- state expected handoffs
- add failure modes
- avoid claiming Linkup has private data or guaranteed email coverage
