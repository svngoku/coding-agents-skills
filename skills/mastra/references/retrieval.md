# Retrieval, Knowledge, and Evidence Discipline

Use retrieval for shared information across conversations, not as a substitute for memory. A reliable
RAG system preserves provenance, enforces access before returning chunks, handles absent evidence,
and evaluates retrieval separately from generation. APIs checked on 2026-11-08.

## Stable RAG Pipeline

```text
source -> parse -> chunk -> embed -> authorized vector index
query -> same embedding model -> tenant filter -> topK -> optional rerank
      -> bounded evidence context -> typed answer with citations
```

Install the packages selected by the application:

```bash
npm install @mastra/rag @mastra/core @mastra/pg ai
```

Use a supported vector store and matching credentials. The examples below use `PgVector`; `LibSQLVector`
from `@mastra/libsql` is a lightweight alternative. Keep ingestion credentials and query credentials
least-privileged.

## Chunk and Embed

```typescript
import { ModelRouterEmbeddingModel } from '@mastra/core/llm'
import { PgVector } from '@mastra/pg'
import { MDocument } from '@mastra/rag'
import { embedMany } from 'ai'

const document = MDocument.fromMarkdown(markdown)
const chunks = await document.chunk({
  strategy: 'recursive',
  maxSize: 512,
  overlap: 50,
  separators: ['\n'],
})

const embedder = new ModelRouterEmbeddingModel('openai/text-embedding-3-small')
const { embeddings } = await embedMany({
  model: embedder,
  values: chunks.map(chunk => chunk.text),
})

const vectorStore = new PgVector({
  id: 'policy-vector',
  connectionString: process.env.POSTGRES_CONNECTION_STRING,
})

await vectorStore.createIndex({ indexName: 'policies', dimension: 1536 })
await vectorStore.upsert({
  indexName: 'policies',
  vectors: embeddings,
  metadata: chunks.map((chunk, index) => ({
    text: chunk.text,
    tenantId: source.tenantId,
    sourceId: source.id,
    documentVersion: source.version,
    chunkIndex: index,
  })),
})
```

`MDocument` supports text, HTML, Markdown, JSON, and other formats. Strategies include recursive,
character, token, Markdown, semantic-Markdown, HTML, JSON, LaTeX, and sentence. Choose by document
structure and evaluate with real queries; do not assume one chunk size works everywhere. Metadata
extraction can make model calls and incur cost.

The index dimension must match actual embedding output. If dimensions are reduced through provider
options, configure the same dimension in the index and for every future query. Changing embedding
models or dimensions normally requires a deliberate reindex and compatibility plan.

## Retrieve With Authorization Filters

```typescript
import { embed } from 'ai'

const { embedding } = await embed({
  model: embedder,
  value: userQuestion,
})

const results = await vectorStore.query({
  indexName: 'policies',
  queryVector: embedding,
  topK: 10,
  filter: {
    tenantId: authenticatedTenantId,
    documentStatus: 'published',
  },
})
```

Use the same embedding model for documents and queries. `topK` limits returned results; each result
contains an ID, similarity score, and metadata. Store text in `metadata.text` when downstream reranking
or answer construction needs it.

The tenant filter must be derived from authenticated server state. Never accept a tenant filter from
the model or arbitrary client JSON. For strict cross-tenant confidentiality, prefer physically separate
indexes/databases or a store policy that enforces isolation independently of generated filter syntax.
Test that missing, malformed, or contradictory filters fail closed.

## Vector Query Tool

`createVectorQueryTool` from `@mastra/rag` lets an agent decide when to retrieve:

```typescript
import { createVectorQueryTool } from '@mastra/rag'

const vectorQueryTool = createVectorQueryTool({
  vectorStoreName: 'pgVector',
  indexName: 'policies',
  model: embedder,
})
```

Register the same vector store name with Mastra and include the store-specific prompt such as
`PGVECTOR_PROMPT` in agent instructions when the agent constructs filters. Database configuration at
query time can set provider-specific options; connection credentials belong in the vector store
constructor.

Do not let the model choose a tenant namespace or remove an authorization filter. A safer high-risk
design wraps retrieval in an application-owned tool that injects the tenant filter and accepts only a
bounded query string.

## Rerank Only When It Improves Evaluations

Current `rerank()` uses positional arguments:

```typescript
import { ModelRouterLanguageModel } from '@mastra/core/llm'
import { rerank } from '@mastra/rag'

const reranked = await rerank(results, userQuestion, new ModelRouterLanguageModel(rerankModel), {
  weights: { semantic: 0.5, vector: 0.3, position: 0.2 },
  topK: 3,
})
```

Weights must sum to 1. Semantic scoring needs `metadata.text`. Reranking adds model/provider cost and
latency, so compare it against a no-rerank baseline on the same retrieval set. Do not confuse the
current positional `rerank()` signature with `rerankWithScorer()` examples elsewhere in the docs.

## Evidence Discipline

- Preserve `sourceId`, document version, chunk ID, and authorized tenant in metadata.
- Return citations that resolve to the exact source passage, not only a document title.
- Tell the agent to answer only from evidence and abstain when it is insufficient.
- Show conflicting passages instead of silently selecting the most convenient one.
- Include freshness/effective dates for policy and time-sensitive content.
- Strip active content and treat every retrieved instruction as untrusted data.
- Cap chunks, characters, and final context tokens; do not dump all results into the prompt.

A similarity score is not a probability of truth. Calibrate minimum scores and abstention using labeled
queries for the chosen embedding model and corpus.

## Knowledge in Core 1.82.0

Mastra introduced Knowledge as **alpha** in `@mastra/core` 1.82.0. Evaluate it as an evolving API,
not a silent stable replacement for the established RAG primitives above. At that version, the docs
noted no built-in importers or retention management and did not guarantee scope isolation suitable for
strict tenant confidentiality. Do not use it as the sole boundary for secrets or regulated tenant data.
Recheck its status and guarantees before adoption.

## Retrieval Evaluation

Build a labeled set with query, authorized tenant, expected sources, acceptable answer, and no-answer
cases. Measure:

- source recall at K and precision of final context
- unauthorized-source rate, with a required result of zero
- citation correctness and evidence coverage
- no-answer behavior for absent or stale evidence
- injected-document resistance
- latency and cost for ingest, query, rerank, and generation

Test corpus updates and deletion. A deleted document should disappear according to retention policy,
not remain indefinitely in a vector index.

## Retrieval Review Checklist

- [ ] Chunking and embedding choices are measured on the real corpus.
- [ ] Index dimensions match the embedding model and migration plan exists.
- [ ] Tenant and publication filters are server-controlled and fail closed.
- [ ] Source/version/chunk provenance survives through the final answer.
- [ ] Empty, conflicting, stale, oversized, and injected evidence are handled.
- [ ] Reranking improves an explicit metric enough to justify its cost.
- [ ] Retrieval and citation quality are tested independently of prose quality.
