# Linkup Specialized Endpoints Guide

Adapted from Linkup's MIT source pack (2026-09-15 snapshot). Read the
[2026-10-29 compatibility corrections](sources-and-compatibility.md) before copying API examples.
Examples require an approved connection/budget; do not run paid calls for validation.

Use this document to decide when to use the `/research`, `/tasks`, and `/extract` endpoints instead of the synchronous `/search` API, and when Fetch with `schema` is enough.

For individual Search queries, use the separately available `linkup-search` skill. For workflow orchestration, use `linkup-workflow`; neither is a filesystem dependency of this guide.

## Quick Decision Matrix

| Your Need | Endpoint | Why |
|-----------|----------|-----|
| One fact, lowest latency (< 200 ms) | `/search` with `flash` | Keyword-only retrieval; synthesis may add an LLM |
| Quick lookup (< 10s), simple facts | `/search` with `fast` or `standard` | Low latency, synchronous |
| Discover URL → scrape content | `/search` with `deep` | Handles chaining in one request |
| Typed JSON from one known page | `/fetch` with `schema` | Same call as fetch, +$0.001 |
| Hundreds of search/fetch/research calls, or a nightly batch | `/tasks` | Async, up to 100 per submission, no surcharge |
| Complex report requiring 5+ minutes of investigation | `/research` | Multi-source, iterative, thorough |
| Bulk extract structured rows from a known listing page, following links/pagination | `/extract` (closed beta) | Purpose-built for extraction |
| Verified answer to a precise question requiring cross-checking | `/research` mode=`answer` | Built-in verification |
| Focused investigation of a defined entity | `/research` mode=`investigate` | Systematic subject analysis |
| Broad market analysis across many entities | `/research` mode=`research` | Parallel multi-subject coverage |

---

## Research Endpoint (`/v1/research`)

The Research endpoint is an **autonomous research agent** that runs asynchronously for minutes, not seconds. It investigates the web iteratively, gathering evidence from multiple sources in parallel, cross-checking claims, and producing a synthesized, cited answer.

### When to Choose Research over Deep Search

| Factor | Deep Search (`/search` with `depth=deep`) | Research (`/v1/research`) |
|--------|-------------------------------------------|---------------------------|
| **Latency** | Seconds (typically 5-30s) | Minutes (2-20 min depending on depth) |
| **Investigation style** | Bounded sequential retrieval, up to 10 iterations | Longer-running, multi-turn investigation |
| **Source coverage** | Limited by single-request context | Extensive parallel source gathering |
| **Cross-checking** | Limited | Built-in verification across sources |
| **Synthesis** | Requested output type, optionally cited or structured | Structured synthesis with reasoning |
| **Cost model** | Per-request | Per-call with depth-based pricing ($0.25-$2.50) |
| **Best for** | "Find X, then scrape it" chains | "Investigate this thoroughly" questions |

### Research Modes

Choose the mode that matches your question type:

| Mode | Use When | Example Query | Behavior |
|------|----------|---------------|----------|
| `answer` | Precise question with a single correct answer, requires verification | "Which 12 S&P 500 companies gained >50% with market cap >$5B in Q3 2025?" | Agent self-verifies, cross-references evidence, reasons against alternative candidates |
| `investigate` | In-depth analysis of a defined subject | "Risk profile and regulatory history of Stripe" | Follows discovered threads, explores new trails, verifies claims along the way |
| `research` | Broad, multi-angle exploration | "State of the European generative AI market in 2026" | Searches multiple threads in parallel, covers topics/entities broadly |

**Rule of thumb:** If the question ends with "?" and expects a single answer → `answer`. If it asks about "the state of" or "analysis of" → `research`. If it focuses on one entity comprehensively → `investigate`.

**Important:** Always set `mode` explicitly for predictable latency, cost, and output shape. Omitting it lets the agent auto-classify, which is convenient but less predictable.

### Reasoning Depths

| Depth | Time | Cost | Use When |
|-------|------|------|----------|
| `S` | 2-5 min | $0.25 | Light coverage, short multi-step investigations |
| `M` | 3-7 min | $0.50 | Balanced cost/quality, routine use |
| `L` | 5-10 min | $1.50 | Thorough investigation (default) |
| `XL` | 10-20 min | $2.50 | Exhaustive coverage, deliverables where completeness trumps speed |

### Prompt Detail Levels

Research runs as an agentic loop: the agent interprets the question, plans retrieval, executes parallel searches, verifies claims, and synthesizes. Both terse and detailed inputs work — more precise input produces more thorough, predictable, aligned output.

**Dimensions to specify for best results:**

| Dimension | What to Include |
|-----------|-----------------|
| Angles to cover | Specific aspects of the topic to investigate |
| Leads to pursue | Known starting points, companies, people, documents |
| Facts to verify | Claims that need cross-referencing |
| Entities to compare | Side-by-side comparison targets |
| Constraints | Exclusions, scope boundaries, required sources |
| Output structure | Format, sections, length expectations |

**Example progression:**

```text
# Terse — agent chooses angles and detail level
What's going on in the European AI inference market?

# Medium — angle and deliverable constrained
Survey the European AI inference market in 2026: identify the main hyperscalers
and independent inference providers, their pricing posture, and regulatory pressure
under the EU AI Act.

# Fully specified — all dimensions defined
Produce a competitive landscape of European AI inference providers operating in 2026.

Scope:
- Cover at minimum: Mistral, Aleph Alpha, Silo AI, OVHcloud, Scaleway
- Exclude US-headquartered hyperscalers unless they operate EU sovereign offerings

For each provider, surface:
- Headquarters and primary inference regions
- Models served and deployment modes
- Disclosed pricing normalized per million tokens
- Sovereignty and EU AI Act posture
- Latest disclosed funding

Verify all numeric claims against primary sources. Flag figures only found in secondary aggregators.
```

### Common Research Patterns

#### Pattern A: Verified Fact-Finding

**Scenario:** User asks a precise question requiring cross-checking multiple authoritative sources.

```text
POST /v1/research
{
  "q": "What was Amazon's total advertising revenue in Q4 2024, and how did it compare to Q4 2023?",
  "mode": "answer",
  "reasoningDepth": "M",
  "outputType": "sourcedAnswer"
}
```

**Why Research wins:** The agent will find Amazon's official earnings, cross-check against analyst reports, and resolve any discrepancies with reasoning.

#### Pattern B: Risk & Compliance Report

**Scenario:** Generate a comprehensive risk profile that requires examining multiple angles (financial, legal, operational).

```text
POST /v1/research
{
  "q": "Comprehensive risk assessment for vendor {company_name}: financial stability, security incidents, regulatory actions, leadership changes, and customer complaints from 2022-2026",
  "mode": "investigate",
  "reasoningDepth": "L",
  "outputType": "structured",
  "structuredOutputSchema": "{\"type\":\"object\",\"properties\":{\"financialRisks\":{\"type\":\"array\",\"items\":{\"type\":\"string\"}},\"securityIncidents\":{\"type\":\"array\",\"items\":{\"type\":\"object\"}},\"regulatoryActions\":{\"type\":\"array\",\"items\":{\"type\":\"object\"}},\"overallRiskRating\":{\"type\":\"string\",\"enum\":[\"low\",\"medium\",\"high\",\"critical\"]}}}"
}
```

**Why Research wins:** Multi-angle investigation with cross-referencing between financial news, security databases, and regulatory filings.

#### Pattern C: Market Landscape Analysis

**Scenario:** Understand a market space with multiple players, trends, and dynamics.

```text
POST /v1/research
{
  "q": "European generative AI market 2026: key players, funding rounds, enterprise adoption rates, regulatory impact, and competitive dynamics",
  "mode": "research",
  "reasoningDepth": "XL",
  "outputType": "sourcedAnswer"
}
```

**Why Research wins:** Breadth across many entities requires parallel investigation that would exceed deep search's single-request context.

#### Pattern D: Due Diligence Dossier

**Scenario:** Pre-meeting or pre-investment deep dive on one company.

```text
POST /v1/research
{
  "q": "Complete company profile for {target_company}: business model, revenue estimates, competitive positioning, leadership background, recent strategic moves, and customer sentiment",
  "mode": "investigate",
  "reasoningDepth": "L",
  "outputType": "structured",
  "structuredOutputSchema": { ... }
}
```

**Why Research wins:** Comprehensive, structured output on a single subject with thorough source coverage.

### Research Anti-Patterns

Don't use Research when:

- **Latency matters** → Use `/search` with `fast` or `standard`
- **Simple lookup** → "What is Stripe's CEO name?" is overkill for Research
- **Real-time chat** → Research's 2+ minute latency breaks conversational flow
- **Known URL extraction** → Use `/extract` or `/fetch` for structured data from a specific page
- **Already have sources** → If you have the URLs, use `/fetch` or `/search` with direct scraping

### Implementation Notes

**Async Lifecycle:**

```text
POST /v1/research → immediate response with {id, status: "pending"}
  ↓
Poll GET /v1/research/:id with backoff
  ↓
When status is "completed", output contains answer + sources
  ↓
When status is "failed", check error field for details (no charge for failures)
```

**Polling strategy (exponential backoff):**

```python
import time


def wait_for_research(client, task_id, max_wait_seconds=600):
    """Poll an existing job only; a timeout never submits another job."""
    deadline = time.monotonic() + max_wait_seconds
    interval = 2
    while time.monotonic() < deadline:
        time.sleep(min(interval, max(0, deadline - time.monotonic())))
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        result = client.get_research(task_id, timeout=min(30, remaining))
        if result.status in ("completed", "failed"):
            return result
        interval = min(interval * 2, 10)
    raise TimeoutError(f"Keep research id {task_id}; do not resubmit this job.")
```

**Expected completion times:**
- S depth: 2-5 minutes
- M depth: 3-7 minutes
- L depth: 5-10 minutes
- XL depth: 10-20 minutes

**Rate limiting:** Above 1 request/second will be rate-limited. Use backoff, not fixed-interval rapid polling.

**Many research jobs at once:** Submit them as `"research"` tasks through `/tasks` (below) and poll one list instead of many individual ids.

---

## Tasks Endpoint (`/v1/tasks`)

Tasks is an **asynchronous batch wrapper** around Search, Fetch, and Research. One `POST /v1/tasks` accepts up to 100 `{ type, input }` objects, in any mix of `"search"`, `"fetch"`, and `"research"`. Each task takes the same parameters and is billed at the same rate as the direct call. No surcharge, no discount.

### When to Use Tasks

| Scenario | Why Tasks |
|----------|-----------|
| Bulk workloads: CRM enrichment, backfills, batch research over hundreds of queries | One submission per 100 items instead of hundreds of open connections |
| Long-running jobs | Submit now, poll later; no blocking HTTP call |
| Scheduled pipelines | Kick off nightly, collect results in the morning |
| Mixed batches | Search + fetch + research in one submission, each billed at its own rate |
| Concurrency overflow | Workload exceeds the synchronous concurrency budget |
| 2–3 calls where one polling loop is simpler than several sync calls | No surcharge, so batching small sets is free |

### When NOT to Use Tasks

Tasks does **not** make individual calls faster or cheaper. For interactive single-shot calls (chat UI, one agent step), call the synchronous endpoint directly.

### Request Shape

```text
POST /v1/tasks
[
  { "type": "search",   "input": { "q": "...", "depth": "standard", "outputType": "structured", "structuredOutputSchema": { ... } } },
  { "type": "fetch",    "input": { "url": "https://...", "renderJs": true, "schema": { ... } } },
  { "type": "research", "input": { "q": "...", "mode": "investigate", "reasoningDepth": "M", "outputType": "sourcedAnswer" } }
]
```

Each task type accepts exactly the parameters of its synchronous endpoint:

| `type` | `input` parameters |
|--------|--------------------|
| `"search"` | `q`, `depth`, `outputType`, `structuredOutputSchema`, `includeDomains`, `excludeDomains`, `maxResults`, ... |
| `"fetch"` | `url`, `mode`, `renderJs`, `includeRawContent`, `extractImages`, `schema`, `instructions` |
| `"research"` | `q`, `mode`, `reasoningDepth`, `outputType`, `structuredOutputSchema` |

### Async Lifecycle

```text
POST /v1/tasks → array of envelopes, each {id, type, status: "pending", input, output: null, error: null}
  ↓
Poll GET /v1/tasks/{id} for one task, or GET /v1/tasks to list all
  ↓
status: "pending" → "processing" → "completed" | "failed"
  ↓
"completed": output has the same shape as the synchronous endpoint's response
"failed": error is a string; no charge
```

**Polling guidance:** Search and fetch tasks finish in seconds; research tasks in minutes. Poll the list endpoint with backoff (start ~5s, cap ~30s) and stop once every id in the submission is terminal. Split workloads larger than 100 items across submissions.

`post_task_batch`, `chunks`, and persistence functions below are application-defined
REST pseudocode helpers, not Linkup SDK methods. Python SDK `create_tasks` instead
requires typed task inputs. External writes require separate authorization.

### Tasks Pattern: Overnight Enrichment

```python
# 1. Chunk the batch into submissions of <= 100
for chunk in chunks(companies, 100):
    envelopes = post_task_batch([
        {
            "type": "search",
            "input": {
                "q": f"Find {c['name']}'s headquarters, employee count, and latest funding round",
                "depth": "standard",
                "outputType": "structured",
                "structuredOutputSchema": ENRICHMENT_SCHEMA,
            },
        }
        for c in chunk
    ])
    store_task_ids(chunk, envelopes)

# 2. Later: poll GET /v1/tasks, write completed outputs to the CRM, retry failures
```

---

## Extract Endpoint (`/v1/extract`)

The Extract endpoint is a **structured data extraction agent** that transforms web pages into tables of records. Given a seed URL and a natural-language description of what rows you want, it extracts matching records and returns them as an NDJSON file.

Extract is in **closed beta**. Access is limited; request it at contact@linkup.so. Parameters, behavior, and response shape may change. For typed JSON from a single known page with no link-following, use Fetch with `schema` instead (generally available, $0.001 extra).

### When to Use Extract

| Scenario | Example |
|----------|---------|
| **Directory/List pages** | Extract all team members from /team page |
| **Product catalogs** | Extract all products with price, SKU, description from a catalog |
| **Job listings** | Extract all open roles with title, location, department from /careers |
| **Conference speakers** | Extract all speakers with name, company, topic from event page |
| **Paginated lists** | Extract all items across multiple pages of results |
| **Verification workflows** | Extract + verify URLs are reachable |

### When NOT to Use Extract

| Instead Use | For |
|-------------|-----|
| `/search` | Finding information across the web (not extracting from one known page) |
| `/fetch` | Reading a single article or document |
| `/fetch` with `schema` | Typed JSON from one known page, no link-following or pagination |
| `/research` | Synthesizing information from multiple sources |

### Extract vs. Search Scraping

| | Extract | Search with scraping |
|--|-----------|----------------------|
| **Starting point** | You provide the exact URL | Search discovers URLs |
| **Output format** | NDJSON file with structured rows | Inline response with content |
| **Scale** | Designed for 10s-1000s of records | Designed for 1-10 pages |
| **Pagination** | Handles automatically | Must specify or handle manually |
| **Schema enforcement** | Optional strict schema | Inferred from query |
| **Best for** | Bulk extraction from known listing pages | Research and discovery workflows |

### Common Extract Patterns

#### Pattern A: Team Directory Extraction

**Scenario:** Enrich company data with full team information.

```text
POST /v1/extract
{
  "q": "All leadership team members with name, job title, bio summary, and LinkedIn profile URL",
  "url": "https://example.com/about/leadership",
  "schema": {
    "type": "object",
    "properties": {
      "name": {"type": "string"},
      "title": {"type": "string"},
      "bio": {"type": "string"},
      "linkedinUrl": {"type": "string", "format": "uri"}
    },
    "required": ["name", "title"]
  },
  "verifyUrls": true
}
```

**Result:** Download NDJSON file with one row per executive, LinkedIn URLs verified reachable.

#### Pattern B: Product Catalog Extraction

**Scenario:** Competitive pricing analysis from competitor's product page.

```text
POST /v1/extract
{
  "q": "All pricing plans with plan name, monthly price, annual price, key features included, and any usage limits",
  "url": "https://competitor.com/pricing",
  "schema": {
    "type": "object",
    "properties": {
      "planName": {"type": "string"},
      "monthlyPrice": {"type": "string"},
      "annualPrice": {"type": "string"},
      "features": {"type": "array", "items": {"type": "string"}},
      "limits": {"type": "object"}
    }
  }
}
```

**Result:** Structured pricing comparison data for analysis.

#### Pattern C: Job Listing Monitoring

**Scenario:** Track competitor hiring patterns.

```text
POST /v1/extract
{
  "q": "All open positions with job title, department, location, posting date, and required skills",
  "url": "https://competitor.com/careers",
  "schema": {
    "type": "object",
    "properties": {
      "title": {"type": "string"},
      "department": {"type": "string"},
      "location": {"type": "string"},
      "postedDate": {"type": "string"},
      "skills": {"type": "array", "items": {"type": "string"}}
    }
  }
}
```

**Result:** Weekly extract to track hiring velocity by department.

### Extract Parameters

| Parameter | When to Use |
|-----------|-------------|
| `schema` | Always provide for production use. Ensures consistent output structure and helps the agent understand required fields. |
| `verifyUrls` | Enable when extracted URLs will be used downstream (prevents broken links). Adds latency. |

### Implementation Notes

**Async Lifecycle:**

```text
POST /v1/extract → immediate response with {id, status: "pending"}
  ↓
Poll GET /v1/extract/:id every 30 seconds
  ↓
When status is "completed", output contains resultUrl (24h expiry)
  ↓
GET resultUrl → download NDJSON file
  ↓
Parse: one JSON object per line
```

**Pricing:** Variable based on page size, rows extracted, pagination depth, and verifyUrls. Typical: $2-10 per task. Minimum $10 account balance required.

**Polling guidance:** Poll every 30s (slower than Research due to typically longer crawl times).

---

## Choosing Your Endpoint: Decision Tree

```text
Is this a batch (many items, nightly job, or more calls than your concurrency budget)?
├── YES → Wrap the calls below in /tasks (<= 100 per submission, same params, same price)
└── NO  → continue

Do you already have the exact URL?
├── YES
│   ├── Need its content as markdown → /fetch (renderJs: true; retry mode: "pro" if empty)
│   ├── Need typed JSON from that one page → /fetch with schema (+ instructions)
│   └── Need many rows, following links or pagination (team directory, catalog, careers)
│       └── /extract (closed beta; provide schema)
│
└── NO (discovery needed, or research question)
    └── Is this a simple, quick lookup?
        ├── YES (CEO name, stock price, simple fact)
        │   ├── One keyword-shaped fact, latency-critical → /search depth=flash (< 200 ms)
        │   └── Otherwise → /search depth=fast or depth=standard
        │       └── Synchronous, < 10s response
        │
        └── NO (complex, multi-source investigation)
            └── Does the answer require < 30 seconds?
                ├── YES (chatbot, real-time UI)
                │   └── Use /search with depth=deep
                │       └── Bounded sequential retrieval, up to 10 iterations
                │
                └── NO (report, analysis, verification can wait minutes)
                    └── Use /research
                        ├── Mode = answer → precise verified facts
                        ├── Mode = investigate → focused entity analysis
                        └── Mode = research → broad multi-subject coverage
```

---

## Workflow Integration

### Research in Enrichment Workflows

Replace the enrichment workflow's Linkup calls with Research when:

1. **Data quality > speed** — Research's cross-checking produces more reliable firmographics
2. **Conflicting sources common** — Research resolves contradictions with reasoning
3. **Batch/async acceptable** — Can run overnight, not needed in real-time

**Example:** Overnight enrichment batch using Research:

```python
# Queue companies for research enrichment via /tasks (<= 100 per submission)
for chunk in chunks(batch, 100):
    envelopes = post_task_batch([
        {
            "type": "research",
            "input": {
                "q": f"Complete company profile for {c['name']}: firmographics, funding, leadership, recent news",
                "mode": "investigate",
                "reasoningDepth": "M",
                "outputType": "structured",
                "structuredOutputSchema": ENRICHMENT_SCHEMA,
            },
        }
        for c in chunk
    ])
    store_task_ids(chunk, envelopes)

# Poll GET /v1/tasks, store results when complete
```

### Extract in Monitoring Workflows

Combine Extract with Research for comprehensive competitor tracking:

```text
Weekly job:
  1. /extract on competitor /careers → hiring patterns (structured)
  2. /research mode=investigate on competitor → strategic moves (synthesized)
  3. /search depth=standard for recent competitor news (quick)
  4. Merge into competitor intelligence report
```

### Hybrid: Search Discovery + Fetch/Extract

When you don't know the URL but know the data structure you need:

```text
Step 1: /search depth=standard
  "Find the careers page URL for {company_name}"

Step 2a: All rows are on that one page → /fetch with schema
  POST /v1/fetch { url: <result_from_step_1>, renderJs: true, schema: { ... } }

Step 2b: Rows span pagination or detail pages → /extract (closed beta)
  POST /v1/extract with url=<result_from_step_1>
```

For a list of companies, put every Step 2a fetch into one `/tasks` submission.

---

## Summary Table: All Linkup Endpoints

| Endpoint | Type | Latency | Best For | Output | Cost |
|----------|------|---------|----------|--------|------|
| `/search?depth=flash` | Sync | < 200 ms | One keyword-shaped fact, lowest latency | Results + snippets | $0.005 / $0.006 |
| `/search?depth=fast` | Sync | ~1s | Simple lookups, snippets | Quick answer | $0.005 / $0.006 |
| `/search?depth=standard` | Sync | 1-3s | Parallel facts, one known URL | Sourced answer | $0.005 / $0.006 |
| `/search?depth=deep` | Sync | 5-30s | Discover→scrape chains | Sourced answer | $0.05 / $0.055 |
| `/fetch` | Sync | 2-5s | Read specific URL content; typed JSON with `schema` | Markdown (+ `data`) | $0.001–$0.011 |
| `/research` | Async | 2-20 min | Complex investigation, reports | Synthesized report | $0.25–$2.50 |
| `/tasks` | Async | Same as wrapped calls | Bulk / scheduled search, fetch, research | Array of envelopes | Same as wrapped calls |
| `/extract` (closed beta) | Async | 1-5 min | Bulk structured extraction | NDJSON file | ~$2–10 |

Search costs are `searchResults` / `sourcedAnswer`-or-`structured`.

**Integration principle:** Use the fastest endpoint that can reliably accomplish the task. Escalate to slower, more thorough endpoints when data quality or synthesis depth matters more than speed.
