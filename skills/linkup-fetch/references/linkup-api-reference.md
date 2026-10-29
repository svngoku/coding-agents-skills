# Linkup API Context for Coding Agents

Adapted from Linkup's MIT source pack (2026-09-15 snapshot). Read the
[2026-10-29 compatibility corrections](sources-and-compatibility.md) before copying API examples.
Examples require an approved connection/budget; do not run paid calls for validation.

## What is Linkup?

Linkup is a web search API built specifically for AI applications and agents.

It provides:

- Real-time web search

- Citation-backed answers

- Deep research capabilities

- Web page fetching and extraction

- Structured JSON outputs (from the web via Search, from a known page via Fetch)

- Domain filtering and source control

- Asynchronous batching (Tasks)

Think of Linkup as **internet access for LLMs and AI agents**.

---

# Core APIs

## 1. Search API

Search the live web and return:

- Search results (URLs + snippets)

- Citation-backed natural language answers

- Structured JSON matching a schema

### Endpoint

```http

POST https://api.linkup.so/v1/search

```

### Typical use cases

- Agent web search

- Retrieval Augmented Generation (RAG)

- Company research

- Lead enrichment

- Market intelligence

- Fact verification

### Example

```json

{

  "q": "Who is the CEO of Stripe?",

  "depth": "standard",

  "outputType": "sourcedAnswer"

}

```

### Output Types

#### searchResults

Returns URLs and snippets.

```json

{

  "outputType": "searchResults"

}

```

Best when:

- Agent wants to inspect sources itself

- Multi-step workflows

---

#### sourcedAnswer

Returns a generated answer with citations.

```json

{

  "outputType": "sourcedAnswer"

}

```

Best when:

- User-facing chat

- Question answering

- Agent reasoning

---

#### structured

Returns JSON matching a user schema.

```text

{

  "outputType": "structured",

  "structuredOutputSchema": {

    ...

  }

}

```

Best when:

- CRM enrichment

- Lead generation

- Data extraction

- Workflow automation

---

# Search Depth

Four depths. `flash` (Alpha) and `fast` pass the query to the index as-is (no retrieval-stage LLM; synthesized output types can add one): keep
prompts short and keyword-shaped. `standard` and `deep` are agentic and follow
instruction-style prompts.

| Depth | Best for | Latency |
|-------|----------|---------|
| `flash` | One piece of information, lowest latency | < 200 ms |
| `fast` | Higher-quality one-shot lookup when ~1s is acceptable | ~1s |
| `standard` | Instruction-style retrieval, several topics, or one provided URL | 1–3s |
| `deep` | Sequential search-then-scrape chains | 5–30s |

## Flash

Lowest-latency search: ranked sources and snippets in under 200 ms. Keyword-only.
No LLM, no query reinterpretation, no scraping, no chaining.

Use for:

- Real-time chat, voice, autocomplete, and other latency-critical paths

- Simple keyword-shaped queries with one target fact

```json

{

  "depth": "flash"

}

```

---

## Fast

Higher-quality one-shot keyword search in about a second. No LLM, no scraping, no chaining.

Use for:

- Simple lookups where ~1s is acceptable and quality matters more than raw speed

- High-volume pipelines of one-fact queries

```json

{

  "depth": "fast"

}

```

---

## Standard

Single-iteration agentic search, ~1-3s. Can run parallel sub-searches and
scrape one URL provided in the query.

Use for:

- Most agent searches

- Enrichment

- Simple questions

```json

{

  "depth": "standard"

}

```

---

## Deep

Up to 10 iterations, ~5-30s. Can scrape multiple URLs and chain
search-then-scrape sequentially. Use when the next step depends on the
previous step's output.

Use for:

- Competitive research

- Complex investigations

- Multi-source validation

```json

{

  "depth": "deep"

}

```

---

# Structured Output

One of Linkup's most powerful features.

Provide a JSON schema and Linkup returns validated structured data.

Example:

```json
{
  "q": "Find information about OpenAI",
  "depth": "standard",
  "outputType": "structured",
  "structuredOutputSchema": "{\"type\":\"object\",\"properties\":{\"company_name\":{\"type\":\"string\"},\"founding_year\":{\"type\":\"number\"},\"headquarters\":{\"type\":\"string\"}}}"
}
```

Possible response:

```json
{
  "company_name": "OpenAI",
  "founding_year": 2015,
  "headquarters": "San Francisco, California"
}
```

---

# Domain Controls

Limit or exclude sources.

Use source filtering only if you know exactly the URLs or domains you are targeting or not targeting.
Do not infer filters from a general preference for a source type. Use `fromDate`/`toDate` when intentional index-date restrictions are required.

## Include Domains

```json
{
  "includeDomains": [
    "wikipedia.org",
    "openai.com"
  ]
}
```

Only search these domains.

---

## Exclude Domains

```json
{
  "excludeDomains": [
    "reddit.com"
  ]
}
```

Ignore specific sources.

---

# Fetch API

Retrieve clean content from any URL.

Useful when an agent already knows which page it wants.

### Endpoint

```http

POST https://api.linkup.so/v1/fetch

```

### Capabilities

- HTML extraction (up to 20 MB) and PDF (up to 100 MB)

- Markdown conversion

- JavaScript rendering (`renderJs`)

- Two access modes (`mode`): `"standard"` (default) or `"pro"` for hard-to-retrieve pages

- Structured JSON from the page (`schema` + optional `instructions`)

- Raw content (`includeRawContent`) and image extraction (`extractImages`)

Fetch reads the given URL only. It does not follow links or crawl. It cannot read LinkedIn.

### Example

```json
{
  "url": "https://openai.com",
  "renderJs": true
}
```

### Parameters

| Parameter | Values | Notes |
|-----------|--------|-------|
| `url` | string | Required. |
| `renderJs` | `false` (default) / `true` | Execute JavaScript before extraction. Default to `true` in agent pipelines; turn off only for known-static pages. |
| `mode` | `"standard"` (default) / `"pro"` | How the page is accessed. `"pro"` has significantly higher success on hard-to-retrieve pages. Independent of `renderJs`. Start on `"standard"`; retry with `"pro"` when markdown comes back empty or truncated. |
| `schema` | JSON Schema object (`type: "object"`) | Turns on structured output. Response keeps `markdown` and adds `data`. Field `description`s tell the model what to look for. Fields with no grounded value are omitted, even if `required`. |
| `instructions` | string, max 4,000 chars | Optional. Requires `schema`. Global rules the schema cannot express ("public list prices only", "one item per city"). |
| `includeRawContent` | boolean | Return raw content alongside markdown as `rawContent` with `contentType`. |
| `extractImages` | boolean | Return image URLs found on the page. |

### Structured output from a known URL

```json
{
  "url": "https://www.linkup.so/careers",
  "schema": {
    "type": "object",
    "properties": {
      "jobs": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "position": {
              "type": "string",
              "description": "The job title"
            },
            "city": {
              "type": "string",
              "description": "City where the job is located"
            }
          }
        }
      }
    }
  },
  "instructions": "If a position is listed in multiple cities, emit one jobs item per city."
}
```

Search keeps `structuredOutputSchema`; Fetch uses `schema` and `instructions`. Keep schemas shallow
(primitive fields and one level of arrays).

| Job | Use |
|-----|-----|
| One known URL, markdown | Fetch |
| One known URL, typed JSON from that page | Fetch with `schema` |
| Many rows from a listing page, possibly following links | Extract (closed beta) |
| No URL yet, structured JSON from the web | Search with `outputType: "structured"` |

### Errors specific to Fetch

- `400`: page over the size limit (HTML > 20 MB, PDF > 100 MB), unsupported content type, target
  URL not found or unreachable, `instructions` without `schema`, `schema` not a JSON Schema object,
  or structured extraction failed after the page was fetched.

Typical workflow:

1. Search for a source

2. Fetch the page (with `schema` if code will consume the result)

3. Feed content to an LLM

---

# Research API

Deep research as an API.

Designed for:

- Long-horizon investigations

- Multi-hop research

- Comprehensive reports

### Endpoint

```http

POST https://api.linkup.so/v1/research

```

Typical use cases:

- Due diligence

- Market research

- Competitive analysis

- Investment memos

- Enterprise research agents

Compared to Search:

| Search | Research |

|----------|----------|

| Fast | Slower |

| Single query | Multi-step investigation |

| Simple retrieval | Deep synthesis |

| Low cost | Higher cost |

---

# Tasks API

Asynchronous batch wrapper around Search, Fetch, and Research.

### Endpoint

```http

POST https://api.linkup.so/v1/tasks

```

- One submission accepts up to **100 tasks**, in any mix of `"search"`, `"fetch"`, and `"research"`.

- Each task carries the same `input` parameters as the corresponding synchronous endpoint and is
  billed at exactly the same rate. No surcharge, no discount.

- Returns task envelopes immediately with `status: "pending"`. Poll `GET /v1/tasks/{id}` (or
  `GET /v1/tasks` to list). `status` moves through `"pending"` → `"processing"` → `"completed"` or
  `"failed"`. `output` has the same shape as the synchronous response; `error` is a string on failure.

### Example

```json
[
  {
    "type": "search",
    "input": {
      "q": "Microsoft FY2024 revenue",
      "depth": "standard",
      "outputType": "sourcedAnswer"
    }
  },
  {
    "type": "fetch",
    "input": {
      "url": "https://docs.linkup.so",
      "renderJs": true
    }
  },
  {
    "type": "research",
    "input": {
      "q": "Compare 2024 cloud revenue growth of Microsoft, Amazon, and Google.",
      "mode": "investigate",
      "reasoningDepth": "M",
      "outputType": "sourcedAnswer"
    }
  }
]
```

Use Tasks for:

- Bulk workloads (CRM enrichment, backfills, batch research over hundreds of queries)

- Long-running jobs you would rather poll than hold an HTTP connection open for

- Scheduled pipelines (submit nightly, collect in the morning)

- Mixed batches of search + fetch + research

- Concurrency overflow beyond the synchronous budget

Tasks does not make individual calls faster or cheaper. For interactive single-shot calls (chat
UIs, one agent step), call the synchronous endpoints directly. Because there is no surcharge, Tasks
is still worth it for 2–3 calls when one polling loop is simpler than several synchronous ones.

---

# Pricing

Billed per successful call. No charge on errors or empty results. Prices in USD.

### Search (depends on `depth` and `outputType`)

| `depth` | `outputType` | Cost |
|---------|--------------|------|
| `flash` / `fast` / `standard` | `searchResults` | $0.005 |
| `flash` / `fast` / `standard` | `sourcedAnswer` / `structured` | $0.006 |
| `deep` | `searchResults` | $0.05 |
| `deep` | `sourcedAnswer` / `structured` | $0.055 |

### Fetch (depends on `mode`, `renderJs`, and `schema`)

| `mode` | `renderJs` | Markdown only | With `schema` |
|--------|------------|---------------|---------------|
| `standard` | `false` | $0.001 | $0.002 |
| `standard` | `true` | $0.005 | $0.006 |
| `pro` | `false` | $0.005 | $0.006 |
| `pro` | `true` | $0.01 | $0.011 |

### Research (depends on `reasoningDepth`)

| `reasoningDepth` | Cost |
|------------------|------|
| `S` | $0.25 |
| `M` | $0.50 |
| `L` | $1.50 |
| `XL` | $2.50 |

### Tasks and Extract

Tasks bill each task exactly like the direct call. Extract (closed beta) is variable, typically
$2–10 per task, with the exact amount returned as `creditsUsed`.

---

# Citations

Linkup returns source citations whenever possible.

Agents should:

- Preserve citations when presenting answers

- Surface URLs to end users

- Use citations for traceability

This is a key advantage over raw LLM generation.

---

# Authentication

Use a Linkup API key.

Header:

```http

Authorization: Bearer LINKUP_API_KEY

```

---

# Recommended Agent Patterns

## Pattern 1: RAG Search

```text

User Question

      ↓

Linkup Search

      ↓

Cited Results

      ↓

LLM Answer

```

---

## Pattern 2: Structured Enrichment

```text

Company Name

      ↓

Linkup Structured Search

      ↓

Validated JSON

      ↓

CRM / Database

```

---

## Pattern 3: Search + Fetch

```text

Search

   ↓

Relevant URL

   ↓

Fetch

   ↓

Page Content

   ↓

LLM

```

---

## Pattern 4: Deep Research

```text

Research API

      ↓

Multi-step Investigation

      ↓

Comprehensive Report

```

---

# When to Use Which Endpoint

| Goal | Endpoint |

|--------|----------|

| Find information on the web | Search |

| Get cited answers | Search |

| Generate structured JSON | Search |

| Extract content from a URL | Fetch |

| Read a specific page | Fetch |

| Typed JSON from a known page | Fetch with `schema` |

| Analyze PDFs | Fetch |

| Build research agents | Research |

| Due diligence | Research |

| Market intelligence | Research |

| Run hundreds of search/fetch/research calls in bulk | Tasks |

| Many structured rows from one listing page | Extract (closed beta) |

---

# Best Practices

1. Use `structured` output whenever data will feed another system.

2. Use `sourcedAnswer` for user-facing experiences.

3. Use `searchResults` when the agent needs full control over reasoning.

4. Use `fetch` when a URL is already known; add `schema` when code needs fields from that page.

5. Use `research` for complex investigations rather than chaining many search calls.

6. Use `tasks` for bulk or scheduled work instead of loops of synchronous calls.

7. Use `flash`/`fast` only for short keyword-shaped queries; they ignore instructions.

8. Preserve citations whenever possible.

9. Use source filtering only for exact known target or exclusion URLs or domains.

10. Treat Linkup as the source of truth for real-time information, not the LLM.

---

# Mental Model

Linkup provides three layers, plus a batch wrapper:

```text

Search

  ↓

Fetch

  ↓

Research

  ═══

Tasks (batch any of the above)

```

Search finds information.

Fetch retrieves information.

Research investigates information.

Tasks runs any of them in bulk, asynchronously.

Together they give AI agents reliable access to the live web.
