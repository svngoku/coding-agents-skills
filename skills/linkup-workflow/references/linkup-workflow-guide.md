# Linkup Workflow Guide

Adapted from Linkup's MIT source pack (2026-09-15 snapshot). Read the
[2026-10-29 compatibility corrections](sources-and-compatibility.md) before copying API examples.
Examples require an approved connection/budget; do not run paid calls for validation.

Use this document to map user business objectives to Linkup-powered agent workflows. This is the orchestration layer: it helps you decide which Linkup patterns to compose, what additional tooling you need, and how to structure multi-step workflows.

This guide is a **pattern catalog**: browse it to see the kinds of workflows you can build, with worked examples, decision tables, and domain-specific playbooks. To turn one specific goal into a concrete, step-by-step workflow with defined inputs, outputs, and handoffs, use `linkup-workflow-optimizer-knowledge.md` and pull ready-made recipes from `workflows/`.

For individual query construction, use the separately available `linkup-search` skill for its mental model, then this bundle's [query templates](linkup-prompt-optimizer-knowledge.md). For specialized endpoints, use `linkup-research` or `linkup-extract`. These are skill-routing recommendations, not missing local-file dependencies; the [local compatibility guide](sources-and-compatibility.md) summarizes endpoint contracts.

## What This File Is For

This guide answers:

- What business problem is the user trying to solve?
- Which Linkup workflow patterns fit that problem?
- What other tools/APIs integrate with Linkup in this workflow?
- Should the workflow be synchronous (real-time) or asynchronous (batch)?

## Core Workflow Patterns

### Pattern 1: Data Enrichment

**User says:** "Enrich my leads/accounts/contacts" or "Add company intelligence to my CRM" or "Augment my company records with web data"

**What they need:** Take a structured record (company name, domain, person) and augment it with web-sourced fields.

**Linkup role:** Fetch firmographics, funding news, key people, technographics, or recent events.

**Two valid approaches:**

#### Approach A: Parallel Standard Calls

Best when facets are independent, you need granular caching, or want per-facet error handling.

```text
Input: structured record (name, domain, LinkedIn URL)
  ↓
Parallel Linkup calls (standard depth):
  - Company profile from official sources
  - Recent news/funding (last 90 days)
  - LinkedIn company page details
  - Key executive profiles (if LinkedIn URLs known)
  ↓
Merge + validate: cross-check conflicting fields, prefer official sources
  ↓
Output: enriched record with confidence scores and source URLs
```

**When to use:**
- You cache different facets at different TTLs (firmographics 30d, news 1d)
- Some facets fail often (LinkedIn rate limits) and you want partial enrichment
- You need per-facet metrics (which data sources are slow/unreliable)
- Facets target different exact URLs or domains already supplied by the user

#### Approach B: Single Deep Call

Best when facets share context, discovery is needed first, or you want coherent cross-facet validation.

```text
Input: structured record (name, domain, LinkedIn URL)
  ↓
Linkup (deep depth):
  - First find official website, scrape firmographics
  - Search recent news and funding with company context
  - Find and extract executive information
  - Cross-validate: is news about this specific company?
  ↓
Output: enriched record with globally-validated fields and source URLs
```

**When to use:**
- You start with only a company name (discovery needed before any enrichment)
- Cross-facet validation matters (ensure news mentions match the right company)
- You want the LLM to reason across facets ("this funding round aligns with CEO change")
- Lower latency matters more than per-facet caching

#### Approach C: Research Endpoint (Async)

Best when enrichment quality is critical and 2-10 minute latency is acceptable (batch processes, overnight jobs, or research workflows).

```text
Input: structured record (name, domain, LinkedIn URL)
  ↓
POST /v1/research (mode=investigate, reasoningDepth=M or L):
  - Comprehensive multi-source investigation
  - Built-in cross-checking and verification
  - Structured output matching your schema
  ↓
Output: enriched record with verified fields, confidence reasoning, and sources
```

**When to use:**
- Data quality > speed (can wait minutes for thorough investigation)
- Conflicting sources are common (Research resolves contradictions)
- You need verified, cited answers for high-stakes decisions
- Running batch enrichment overnight or in background jobs

Use the separately available `linkup-research` skill for Research endpoint details.

**Key decisions:**

| Question | Guidance |
|----------|----------|
| Real-time or batch? | Real-time for single-record lookup; for bulk lists submit the same calls through `/tasks` (≤100 per submission, no surcharge) and poll |
| Which approach? | Parallel standards for known domains with independent facets; deep for discovery-heavy or cross-facet reasoning |
| Output type? | `structured` with schema matching your CRM fields |
| Cache strategy? | Parallel: per-facet TTLs; Deep: single cache key with shorter TTL |

**Example prompt (deep approach):**

```text
First find the official website for {company_name}. Scrape it to extract: year founded,
employee count, headquarters location, and industry. Then search for recent news and funding
rounds about this company from the last 6 months. Also find LinkedIn profiles for executives
mentioned on the website. Cross-check that news mentions refer to this specific company.
Return structured enrichment data with source URLs.
```

---

### Pattern 2: Research & Intelligence

**User says:** "Research this company/market/topic" or "Build a briefing dossier" or "What should I know before this meeting?"

**What they need:** Comprehensive synthesis from multiple sources, organized for human consumption.

**Linkup role:** Gather evidence from web, news, LinkedIn, financial sources; provide citations.

**Workflow shape:**

```text
Input: entity name, topic, or meeting context
  ↓
Linkup (deep depth):
  - Find official sources and scrape key pages
  - Search recent news and developments
  - Identify key people and their backgrounds
  ↓
Synthesis layer: organize by category (company overview, recent news, people, competitors)
  ↓
Optional: store in knowledge base, generate PDF, or return as structured JSON
```

**Key decisions:**

| Question | Guidance |
|----------|----------|
| Pre-read vs. live search? | Pre-read for meeting prep (batch); live search for ad-hoc questions |
| Depth? | `deep` for comprehensive research; `standard` for quick summaries |
| Output type? | `sourcedAnswer` for human reading; `searchResults` if another agent will synthesize |
| Follow-up? | Store findings with source URLs so later queries can reference cached research |

**Example prompt for pre-meeting research:**

```text
First find the official website for {company_name}. Scrape the about page, product page, and any
news/press releases. Then search for recent news about {company_name} from the last 3 months.
Also find LinkedIn profiles for executives mentioned on the website.
Return a structured briefing with source URLs.
```

---

### Pattern 3: Monitoring & Alerts

**User says:** "Alert me when X happens" or "Monitor my competitors" or "Track mentions of my product"

**What they need:** Continuous or periodic surveillance with signal extraction.

**Linkup role:** Execute scheduled searches, extract new findings, detect changes.

**Workflow shape:**

```text
Schedule trigger (hourly, daily, weekly)
  ↓
Linkup (standard or deep, depending on complexity):
  - Search for new content since last check
  - Scrape changed pages if tracking specific URLs
  ↓
Diff layer: compare to previous snapshot, identify new items
  ↓
Alert if: new items match user criteria (funding, product launch, leadership change, etc.)
```

**Key decisions:**

| Question | Guidance |
|----------|----------|
| Frequency? | News: hourly; company changes: daily; market shifts: weekly |
| Depth? | `standard` for keyword monitoring; `/fetch` (with `schema`) for tracked pages whose URL you already know; `deep` only when the page must be discovered first |
| Scheduling? | Put each run's calls in one `/tasks` submission; poll and diff against the previous run |
| Signal detection? | Use `searchResults` and apply custom logic to detect meaningful changes |

**Example prompt for competitor monitoring:**

```text
Search for any news about {competitor_name} from the last 24 hours. Look for: product launches,
pricing changes, funding announcements, or leadership changes. Return findings with source URLs
and publication dates.
```

---

### Pattern 4: Verification & Fact-Checking

**User says:** "Is this claim true?" or "Verify this data before I act on it" or "Ground this AI output"

**What they need:** Evidence validation with source provenance.

**Linkup role:** Search for corroborating or contradicting evidence.

**Workflow shape:**

```text
Input: claim or AI-generated content to verify
  ↓
Decompose: break into individual factual assertions
  ↓
Parallel Linkup calls (standard depth):
  - Search for each assertion independently
  - Prefer authoritative sources (official sites, news, academic)
  ↓
Scoring layer: confidence based on source authority, consistency across sources, recency
  ↓
Output: verified claims with citations, flagged discrepancies, confidence scores
```

**Key decisions:**

| Question | Guidance |
|----------|----------|
| Source filtering? | Use only if you know exactly the URLs or domains you are targeting or not targeting |
| Depth? | `standard` for straightforward facts; `deep` for complex multi-page verification |
| Output type? | `searchResults` to inspect sources; `sourcedAnswer` for direct verification |
| Confidence threshold? | Define rules: 2+ corroborating sources = high confidence; 1 source = medium; none = unverified |

**Example prompt for fact verification:**

```text
Verify the claim: "{specific claim}". Search for official sources, recent news coverage, and
any conflicting information. Return whether the claim is supported, contradicted, or unverified,
with source URLs and confidence reasoning.
```

---

### Pattern 5: Content Generation with Grounding

**User says:** "Write me a sales email with current context" or "Draft a proposal citing recent news" or "Generate content based on real data"

**What they need:** AI-generated content that references current, real-world information.

**Linkup role:** Fetch current context so the LLM generates grounded, relevant content.

**Workflow shape:**

```text
Input: content request + target recipient/context
  ↓
Linkup (standard depth):
  - Research recipient company/person for personalization
  - Gather relevant recent news or market data
  ↓
Prompt assembly: inject retrieved context into generation prompt with source citations
  ↓
LLM generates content with inline citations
  ↓
Output: draft content + bibliography of sources used
```

**Key decisions:**

| Question | Guidance |
|----------|----------|
| Context scope? | Narrow for personalization; broad for market-aware content |
| Depth? | `standard` for known targets; `deep` to discover and research unknown targets |
| Output type? | `sourcedAnswer` for concise context; `searchResults` if you need raw sources for complex synthesis |
| Citation format? | Include source URLs in final output for user verification |

**Example prompt for sales research:**

```text
Find {target_company} current business priorities, recent news from the last 3 months, and
key executive backgrounds. Return structured findings I can use to personalize a sales outreach email.
```

---

### Pattern 6: Legal/Compliance/Procurement Intelligence

**User says:** "Research this vendor for procurement" or "Check regulatory compliance" or "Due diligence on a supplier"

**What they need:** Risk assessment, compliance verification, or vendor intelligence.

**Linkup role:** Gather regulatory filings, news about incidents, company background, executive history.

**Workflow shape:**

```text
Input: vendor/entity name + specific concern areas (security, financial, regulatory)
  ↓
Parallel Linkup calls (standard depth):
  - Company background and leadership
  - Recent news (incidents, breaches, lawsuits)
  - Regulatory filings if applicable
  - Industry-specific sources (security, finance, healthcare)
  ↓
Risk scoring: apply domain-specific heuristics
  ↓
Output: risk assessment with evidence and source URLs
```

**Key decisions:**

| Question | Guidance |
|----------|----------|
| Source filtering? | Use only if you know exactly the URLs or domains you are targeting or not targeting |
| Depth? | `standard` for checklist items; `deep` for comprehensive due diligence |
| Output type? | `structured` with risk categories and evidence fields |

**Example prompt for vendor due diligence:**

```text
Research {vendor_name} for procurement risk assessment. Find: company background, leadership team,
any security incidents or data breaches in the last 2 years, financial stability indicators,
and customer reviews. Prefer official sources and industry publications. Return structured
risk assessment with source URLs.
```

---

### Pattern 7: Answer Engines & General-Purpose Chatbots

**User says:** "Build a ChatGPT-like chatbot but with real-time web access" or "Answer engine with citations" or "Conversational AI that can look things up"

**What they need:** General-purpose Q&A system that grounds responses in current web data, not just training knowledge.

**Linkup role:** Retrieve current context on demand, provide citations, enable follow-up with source browsing.

**Workflow shape:**

```text
User query → Intent classification
  ↓
If factual/current question:
  Linkup (standard depth, searchResults):
    - Run 2-3 parallel searches from different angles
    - Return top N sources with snippets
  ↓
Synthesis: LLM reads sources, generates answer with inline citations
  ↓
Output: conversational response + source cards for verification

If opinion/complex reasoning:
  Skip Linkup, use base model knowledge

If "tell me more about [source]":
  Linkup fetch/scrape of specific URL
  ↓
  Return detailed summary with deeper citations
```

**Key decisions:**

| Question | Guidance |
|----------|----------|
| Trigger condition? | Classify: "current events", "specific facts", "recent developments" → use Linkup; "opinion", "creative", "general knowledge" → skip |
| Depth? | `standard` for most queries; `deep` for "research this topic thoroughly" or multi-source synthesis |
| Output type? | `searchResults` to let your LLM synthesize with your tone/persona; `sourcedAnswer` for direct answers |
| Source presentation? | Show 3-5 source cards; let users click to expand; offer "deep dive" follow-up |
| Follow-up handling? | "What does [source] say about X?" → fetch that URL; "Is this true?" → verification pattern |

**Anti-patterns to avoid:**

- Don't call Linkup for every turn (expensive, slow). Classify first.
- Don't send user query verbatim to Linkup. Extract key search terms.
- Don't hide sources. Users trust answers they can verify.

**Example query construction:**

```text
User asks: "What happened with Tesla yesterday?"
Extracted searches: ["Tesla news yesterday", "Tesla stock price yesterday", "Tesla announcements June 30 2026"]
```

---

### Pattern 8: Verticalized Domain Agents

**User says:** "Build a legal research assistant" or "Medical guideline chatbot" or "Financial analyst agent" or "Dev tools documentation assistant"

**What they need:** Specialized agent with domain-specific retrieval, source authority rules, and structured reasoning.

**Linkup role:** Domain-specific retrieval, authoritative source ranking, structured data extraction for domain workflows.

Use source filtering only if you know exactly the URLs or domains you are targeting or not targeting.
Otherwise, describe the desired source types in the query and search broadly.

#### 8A: Legal Research Agent

**Workflow:**

```text
User query → Parse legal entities (statutes, cases, jurisdictions)
  ↓
Linkup (standard or deep):
  - Search official legislative sources for statutes
  - Search recognized case-law sources for decisions
  - Search legal publications for commentary
  ↓
Synthesis with legal reasoning:
  - Cite jurisdiction and authority level
  - Flag precedential weight
  - Surface conflicting interpretations
  ↓
Output: legal memo format with full citations, warning if primary sources not found
```

**Special considerations:**

- Always prefer primary sources (statutes, regulations, case law) over secondary
- Flag when LLM is summarizing vs. quoting
- Never hallucinate case citations — verify with Linkup

**Example prompt:**

```text
Find the current text of {statute_name} in {jurisdiction}. Search official legislative databases
first. Return the relevant sections verbatim with citations, then summarize the implications
for {legal_scenario}.
```

---

#### 8B: Medical/Clinical Decision Support

**Workflow:**

```text
User query → Extract condition, drug, procedure, population
  ↓
Linkup (standard):
  - Search medical databases and professional societies for guidelines
  - Search regulatory and clinical sources for drug information
  - Search medical literature for recent trials
  ↓
Synthesis with clinical framing:
  - Cite evidence level (guideline vs. trial vs. review)
  - Flag contraindications and populations
  - Include "not medical advice" disclaimer
  ↓
Output: structured clinical summary with evidence grading
```

**Special considerations:**

- Never rely on single sources for safety-critical information
- Cross-check drug interactions across multiple databases
- Always include disclaimer: information for discussion with qualified professionals

**Example prompt:**

```text
Search PubMed and clinical guidelines for "{condition} treatment {population}". Prioritize
2023-2026 guidelines and systematic reviews. Return: recommended first-line treatments,
contraindications, and evidence quality. Include source URLs to guidelines and key studies.
```

---

#### 8C: Financial Analysis Agent

**Workflow:**

```text
User query → Parse ticker, metric, time period, comparison context
  ↓
Linkup (standard or deep depending on data location):
  - Search regulatory filings and investor relations for company fundamentals
  - Search financial reporting for market data
  - Search analyst publications for research
  ↓
Synthesis with financial framing:
  - Distinguish reported vs. estimated figures
  - Cite reporting lag (Q1 earnings published in April)
  - Flag consensus vs. outlier views
  ↓
Output: financial brief with data attribution and confidence levels
```

**Special considerations:**

- SEC filings are authoritative; news is interpretive — weight accordingly
- Cross-check material changes across multiple financial news sources
- Flag forward-looking statements vs. reported results

**Example prompt:**

```text
Find {ticker} latest quarterly earnings (revenue, EPS, guidance) from SEC filings and official
investor relations. Then search for analyst reactions and consensus changes. Distinguish
reported figures from analyst estimates. Return structured financial summary with source URLs.
```

---

#### 8D: Developer Documentation Agent

**Workflow:**

```text
User query → Parse library/framework, function/version, error message
  ↓
Linkup (standard):
  - Search official documentation
  - GitHub issues/PRs for edge cases and bugs
  - Stack Overflow for common patterns
  ↓
Synthesis with code context:
  - Prioritize official docs over community answers
  - Flag version-specific behavior
  - Include code examples with source attribution
  ↓
Output: technical answer with code snippets, version notes, and "see also" references
```

**Special considerations:**

- Prioritize official documentation over Stack Overflow
- Include version numbers in queries ("React 19", "Python 3.12")
- Link to GitHub issues for known bugs or workarounds

**Example prompt:**

```text
Find the official documentation for "{library} {function}" in version {version}. Search
docs sites and GitHub issues. Return: function signature, parameters, common pitfalls,
and any version-specific changes. Prioritize official documentation over community posts.
```

---

#### Vertical Agent Comparison

| Vertical | Source Guidance | Recency Needs | Special Output Needs |
|----------|------------------------|---------------|---------------------|
| Legal | Primary sources over commentary | Medium (statutes change slowly) | Full citations, jurisdiction flagging |
| Medical | Peer-reviewed + guidelines over blogs | High (treatments evolve) | Evidence grading, safety disclaimers |
| Financial | SEC/regulatory over analyst opinion | High (quarterly reporting) | Reported vs. estimated distinction |
| Developer | Official docs over Stack Overflow | Very high (versions matter) | Version specificity, code attribution |

---

## Workflow Composition Rules

### When to Chain Multiple Linkup Calls

**Parallel calls (independent):**

- Different facets of the same entity (funding + news + people)
- Different entities (enriching a list of companies)
- Different source families (web + LinkedIn + news)

**Sequential calls (dependent):**

- Discover URL → scrape it → extract fields
- Initial research → identify key people → research those people
- Broad search → identify relevant sources → deep-read those sources

### When to Keep Workflows Outside Linkup

Use your orchestration layer (agent, workflow engine) when you need to:

- Apply business logic between steps (scoring, filtering, deduplication)
- Integrate with other APIs (CRM, email, databases)
- Handle user interaction (ask clarifying questions, present options)
- Manage state across long-running workflows
- Cache and reuse results intelligently

### Depth vs. Orchestration Trade-off

| Scenario | Recommendation |
|----------|----------------|
| Simple lookup | Single `standard` call |
| Known multi-facet research | Parallel `standard` calls |
| Discover-then-extract | One `deep` call OR two `standard` calls with orchestration |
| Complex multi-step reasoning | Multiple Linkup calls with your logic layer between |
| Human-in-the-loop decisions | Orchestrate outside; use Linkup for retrieval only |
| One known URL, content or typed fields | `/fetch` (`renderJs: true`; add `schema` for JSON) |
| Latency-critical single fact (chat, voice) | `/search` with `flash` |
| Bulk or scheduled run of any of the above | Submit through `/tasks` (≤100 per submission, same price) |
| Deep investigation (5-20 min acceptable) | Consider `/research` endpoint — see specialized endpoints guide |
| Bulk extraction from known listing pages, following links | Consider `/extract` endpoint (closed beta) — see specialized endpoints guide |

**Escalation path:** When deep search chains become complex or you need synthesis across 5+ sources with verification, escalate to the Research endpoint. When you need typed fields from one known page, use Fetch with `schema`. When extracting many records that span pagination or detail pages (team directories, product catalogs, job listings), use the Extract endpoint. When any step runs over a list of hundreds of items or on a schedule, batch it through Tasks rather than looping synchronous calls.

---

## Natural Language Intent Mapping

When users describe goals in natural language, map to patterns:

| User Intent | Pattern | Key Linkup Calls |
|-------------|---------|------------------|
| "Enrich my leads" | Pattern 1 | Company/person lookups (parallel standard or single deep) |
| "Research before my meeting" | Pattern 2 | Deep research on attendees + company |
| "Track my competitors" | Pattern 3 | Scheduled search with diff detection |
| "Verify this AI output" | Pattern 4 | Parallel fact-checking searches |
| "Write personalized outreach" | Pattern 5 | Research target + generate content |
| "Vet this vendor" | Pattern 6 | Risk-focused research |
| "ChatGPT but with live web" | Pattern 7 | Triggered searchResults for current queries |
| "Build a legal research assistant" | Pattern 8A | Statute and case-law research |
| "Medical/clinical decision support" | Pattern 8B | PubMed + guideline research |
| "Financial analyst agent" | Pattern 8C | SEC filings + market data |
| "Dev tools documentation bot" | Pattern 8D | Official docs with version filtering |
| "Plug my legal assistant to web sources" | Pattern 8A | Legal source research + verification |
| "Use Linkup for my AI procurement product" | Pattern 6 | Vendor intelligence workflows |

---

## Implementation Checklist

Before building a Linkup workflow:

- [ ] What business outcome does the user want?
- [ ] Which pattern(s) match their intent?
- [ ] What inputs do they provide (names, URLs, lists)?
- [ ] What outputs do they need (structured, prose, citations)?
- [ ] Real-time or batch?
- [ ] What other tools integrate (CRM, email, DB, vector store)?
- [ ] Caching strategy for repeated lookups?
- [ ] Error handling when Linkup returns no results?
- [ ] Can this accept async latency (2-20 min)? → Consider `/research` endpoint
- [ ] Extracting from a known listing page? → Consider `/extract` endpoint

After selecting a pattern, use the separately available `linkup-search` skill to design individual Linkup calls, then the bundled [query templates](linkup-prompt-optimizer-knowledge.md).

---

## Chatbot Integration Guide

For the "describe what you want" chatbot use case:

### Intent Classification

First, classify the user's natural language request:

1. **Enrichment intent** → keywords: "enrich", "augment", "add data to", "fill in", "complete", "enrichment"
2. **Research intent** → keywords: "research", "learn about", "find out", "investigate", "what do you know", "briefing", "dossier"
3. **Monitoring intent** → keywords: "track", "monitor", "alert", "watch", "notify me", "keep an eye on"
4. **Verification intent** → keywords: "verify", "check", "confirm", "is it true", "ground", "fact-check"
5. **Content intent** → keywords: "write", "draft", "generate", "create", "help me with", "personalize"
6. **Answer engine intent** → keywords: "chatbot", "answer engine", "like ChatGPT", "conversational AI", "Q&A system"
7. **Vertical agent intent** → keywords: "legal assistant", "medical", "financial analyst", "developer tools", "compliance"

### Clarifying Questions

If intent is ambiguous, ask:

- "Do you want real-time lookups or scheduled monitoring?"
- "What data do you already have (company names, domains, LinkedIn URLs)?"
- "Where should the results go (CRM, email, database, document)?"

### Workflow Generation

Once intent is clear:

1. Select the matching pattern
2. Generate the workflow description (natural language)
3. Show example Linkup queries they'll use
4. Provide integration code (API calls, webhook setup, etc.)
5. Offer to customize based on their specific tools

---

## Summary

This guide sits above individual query optimization. Use it to:

1. **Understand the user's business goal** (not just the data request)
2. **Select the right workflow pattern** from the 8 patterns (6 business workflows + 2 infrastructure/vertical patterns)
3. **Decide on endpoint and orchestration** (sync `/search`, async `/research` for deep investigation, `/extract` for bulk extraction)
4. **Apply domain-specific guidance** (verticalized agents need source-quality rules and output formatting)
5. **Build natural language interfaces** that translate vague requests to concrete Linkup implementations

For each Linkup call within a workflow, descend into:
- The separately available `linkup-search` skill (choose depth, output type, chaining)
- The bundled [query templates](linkup-prompt-optimizer-knowledge.md) (exact templates and rules)
- The separately available `linkup-research` / `linkup-extract` skills (specialized endpoint details)
