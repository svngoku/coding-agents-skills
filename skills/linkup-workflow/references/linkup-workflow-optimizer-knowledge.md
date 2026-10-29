# Linkup Workflow Optimizer Knowledge

Adapted from Linkup's MIT source pack (2026-09-15 snapshot). Read the
[2026-10-29 compatibility corrections](sources-and-compatibility.md) before copying API examples.
Examples require an approved connection/budget; do not run paid calls for validation.

Use this file to generate Linkup-powered agent ideas and multi-step workflows from a user's business
goal.

This file sits one level above `linkup-prompt-optimizer-knowledge.md`.

- Workflow optimizer (this file): the procedure for turning a goal into an executable workflow — job, steps, inputs, outputs, and handoffs.
- Prompt optimizer: writes each atomic Linkup `q`, `depth`, `outputType`, and filter.

For a catalog of workflow patterns and domain-specific playbooks (enrichment, monitoring, verification, answer engines, and legal/medical/financial/developer agents) with worked examples, see `linkup-workflow-guide.md`. Use the guide to recognize which kind of workflow fits; use this file to actually assemble it.

## Core Goal

Help users understand what they can build with Linkup.

The output should be a practical agent idea, not a generic use-case list. A good workflow explains:

1. what the user gets
2. what inputs are needed
3. what Linkup searches happen
4. how each step feeds the next
5. where Linkup stops and another tool should take over

## Audience

The primary audience is a business user, founder, sales team, marketing team, or AI-agent platform
customer who knows their goal but not the exact Linkup prompts.

Write in business language first. Mention API details only where they make the workflow executable.

## Workflow Router

When a user gives a goal, map it to one or more workflow families.

Use `sales` when the user wants:

- leads
- account lists
- company enrichment
- buyer discovery
- outbound personalization
- sales-trigger monitoring

Use `marketing` when the user wants:

- campaign ideas
- content research
- competitor messaging
- customer proof
- SEO/source research
- market narratives

Use `research` when the user wants:

- meeting prep
- company dossiers
- market maps
- competitor trackers
- funding or news monitoring
- industry intelligence

Use `recruiting` when the user wants:

- talent maps
- candidate/company research
- hiring-signal monitoring
- interview prep

Use `support` when the user wants:

- account context before support calls
- incident or product-change monitoring
- public documentation lookup
- customer environment research

## Idea Generation Mode

If the user has no specific goal and asks "how can I use Linkup?", produce a menu of 5-7 workflow
ideas.

Each idea should include:

- agent name
- who it is for
- what it produces
- why Linkup is useful
- first input to ask the user for
- 3-5 Linkup steps

Do not overwhelm the user with every possible category. Pick the ideas that match the user's
industry, role, and context if known.

## Goal Decomposition Mode

If the user has a specific goal, turn it into a complete workflow.

Work backwards from the final business output:

1. Define the final artifact.
2. Identify the entities to discover or enrich.
3. Split the job into Linkup retrieval steps.
4. Decide whether each step needs `searchResults`, `sourcedAnswer`, or `structured`.
5. Add handoffs for tools Linkup should not replace, such as CRM writes, email guessing, sequencing,
   enrichment databases, or spreadsheet exports.

## Step Design Rules

A workflow step should do one retrieval job.

Good steps:

- find companies matching an ICP
- enrich one company from its website and public profiles
- identify likely buyers at a company
- find recent buying signals
- generate evidence-backed personalization notes

Bad steps:

- "do sales research"
- "find everything about this market"
- "create a campaign"
- "get emails" without explaining whether Linkup is finding public email pages or handing off to an
  email provider

## Endpoint Selection Rules

Choose the endpoint at the workflow-step level.

Use `/v1/search` when the step needs:

- a list of sources, companies, people, or pages
- one known URL scrape
- several independent searches that feed another system
- structured enrichment fields
- fast or medium-latency retrieval (`flash` for latency-critical single facts)

Use `/v1/fetch` when the step has the exact URL already:

- markdown of one page (`renderJs: true` by default; `mode: "pro"` on hard pages)
- typed fields from that one page (`schema` + optional `instructions`)
- do not spend a search call to re-find a URL a previous step already produced

Use `/v1/tasks` when the step runs over a list or on a schedule:

- wrap the search, fetch, or research calls the step would have made, up to 100 per submission
- same parameters and price as the direct calls; one polling loop instead of many connections
- typical for enrichment backfills, nightly monitors, and batch research

Use `/v1/research` when the step needs:

- a long-running multi-source investigation
- a written report, assessment, market map, or recommendation
- source synthesis across many independent source families
- current status plus historical context or base rates
- regulatory, sector, technical, ownership, or market-risk analysis
- a final answer that is more valuable than a raw source list

Do not use `/v1/research` just because a task sounds important. If the output is a list of fields to
write into CRM, prefer `/v1/search` and keep the steps narrow.

Common `/research` patterns that should influence workflow design:

- prediction or event investigations: current status, latest news, base rates, official resolution
  sources, and primary citations
- audit or sector risk reports: industry code, location, regulatory context, export/sanctions risk,
  and structured outputs
- technical landscape reports: tools, frameworks, protocols, developer discussions, best practices,
  and tradeoffs
- market sizing or market maps: growth, active players, underserved segments, corridors, and source
  confidence

## Linkup Prompt Chain Rules

For each step, emit a Linkup payload:

```yaml
linkup.search:
  q: ...
  depth: standard
  outputType: searchResults
```

For known-URL steps, emit:

```yaml
linkup.fetch:
  url: ...
  renderJs: true
  schema: { ... }   # only when code needs typed fields from the page
```

For research steps, emit:

```yaml
linkup.research:
  q: ...
  mode: research
  reasoningDepth: L
  outputType: sourcedAnswer
```

When a step runs over a list or on a schedule, keep the payload shape and add
`batch_via: /v1/tasks` so the executor submits the calls in groups of up to 100.

Then include:

- `uses_previous_step`: what prior output is fed into this step
- `produces`: the fields this step creates
- `expected_behavior`: expected tool calls or source behavior

Apply `linkup-prompt-optimizer-knowledge.md` to every Linkup payload.

Important rules:

- Use `searchResults` for discovery lists that another agent will inspect.
- Use `sourcedAnswer` for user-facing summaries and recommendations.
- Use `structured` only with `structuredOutputSchema`.
- Use `standard` for independent searches, and for one known URL scrape combined with searches.
- Use `linkup.fetch` when the step only needs one known page; add `schema` for typed fields.
- Use `deep` when a URL must be discovered before scraping, or when multiple discovered pages need
  follow-up reading.
- For known URL plus external context, ask Linkup to scrape the URL and run separate web searches.
- For ambiguous company or product names, quote the exact entity and add a disambiguator.

## Research Endpoint Defaults

Use these defaults unless the workflow has a reason to be smaller or larger:

- `mode: answer` for a focused question with citations
- `mode: investigate` for uncertain events, status checks, and fact-finding
- `mode: research` for market maps, sector reports, technical landscapes, and complex dossiers
- `reasoningDepth: M` for a bounded report
- `reasoningDepth: L` for market maps, sector risk, and multi-company analysis
- `reasoningDepth: XL` only when the user explicitly wants an exhaustive long-running report
- `outputType: sourcedAnswer` for reports
- `outputType: structured` only when the workflow includes `structuredOutputSchema`

## Handoff Rules

Be explicit about what Linkup should not do.

Common handoffs:

- CRM: store companies, contacts, evidence, and scores.
- Email finder: resolve or verify email addresses after Linkup identifies likely people and public
  profile pages.
- Outbound sequencer: send messages after Linkup generates evidence-backed personalization.
- Spreadsheet: review and score discovered accounts.
- Agent platform or workflow engine: schedule the workflow, retry failed steps, and route outputs.

For email discovery, Linkup can find public pages that may contain emails and identify likely buyer
names. It should not be described as a guaranteed private email database.

## Workflow Output Format

When generating ideas for a user, use this compact format:

```markdown
## Agent Idea: {name}

Best for: {user/team}

What it produces: {business artifact}

Inputs needed: {short list}

Linkup workflow:
1. {step name}: {plain-English action}
2. {step name}: {plain-English action}
3. {step name}: {plain-English action}

Example Linkup prompt:
`{one representative prompt}`

Handoff: {CRM/email/sequencer/etc.}
```

When generating an implementation-ready workflow, use the schema in
`workflows/workflow-schema.md`.

## Quality Bar

A good workflow:

- is useful even before any code is written
- produces a business artifact the user recognizes
- uses Linkup for public-web retrieval, verification, monitoring, or research
- says what inputs are missing
- provides source-backed outputs
- includes handoffs where appropriate
- avoids pretending Linkup replaces systems of record, private databases, or email verification

## Agent Platform Context

Some platforms run automated AI agents on behalf of their users. For this audience, phrase
Linkup workflows as agent templates they can offer inside their platform.

Good template names:

- Lead List Builder
- Account Enrichment Agent
- Buyer Discovery Agent
- Outbound Personalization Agent
- Meeting Prep Agent
- Market Monitor Agent
- Competitor Intelligence Agent

Each template should make clear:

- what the agent asks from the user
- what Linkup retrieves
- what the agent writes back to the platform or another tool
- what a successful run looks like
