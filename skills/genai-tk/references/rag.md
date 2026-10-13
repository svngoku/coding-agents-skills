# RAG — references/rag.md

genai-tk's RAG layer is built around two abstractions: **`RetrieverFactory`** (reads YAML, returns a configured retriever) and **`ManagedRetriever`** (async-first wrapper that adds `aquery`, `aadd_documents`, `adelete_store`). Six retriever types compose freely under the `retrievers:` YAML key.

## The two-class architecture

```
┌──────────────────────────────────────────────────────────┐
│                     Your code / agent                    │
└────────────────────────┬─────────────────────────────────┘
                         │  RetrieverFactory.create("my_tag")
┌────────────────────────▼─────────────────────────────────┐
│              ManagedRetriever                            │
│  aquery()  aadd_documents()  adelete_store()  get_stats()│
└──────┬─────────────────┬────────────────────┬────────────┘
       │                 │                    │
  VectorStore      BM25DocumentStore    EnsembleRetriever
  (Chroma /         (bm25s, disk        (weighted RRF
   InMemory /        cache)              fusion)
   PgVector)
       │
  RecordManager (optional dedup)
```

Key design points:

- **Async-first** — `aquery` and `aadd_documents` are the canonical API. Sync wrappers (`query`, `add_documents`) call `asyncio.run()` and only work outside an event loop.
- **YAML-driven** — every retriever is a named tag; no Python changes to swap retrievers.
- **Composable** — `ensemble` and `reranked` types reference other named retrievers, allowing arbitrarily deep composition.
- **Read vs. write paths separated** — `ManagedRetriever.retriever` handles read (search), `ManagedRetriever.store` handles write (ingestion). Read-only retrievers (ZeroEntropy, reranked) have `store = None` and `has_store == False`.
- **`EmbeddingsStore` is a pure factory** — it produces `VectorStore` instances; all query/ingest logic lives in `ManagedRetriever`.

## The six retriever types

### `vector` — dense similarity

Backed by an `EmbeddingsStore` (Chroma, InMemory, or PgVector). Direct `asimilarity_search` with optional metadata filter.

```yaml
retrievers:
  my_vec:
    type: vector
    embeddings_store: chroma_indexed   # key in embeddings_store: section
    top_k: 4
    search_type: similarity            # similarity | mmr
    record_manager_url: ~              # null → auto SQLite for dedup
```

If `record_manager_url` is null and the backing store is persistent, a SQLite record manager is auto-created at `data/record_manager/<config_tag>.db` for deduplicating ingested documents.

### `bm25` — sparse keyword

Uses the `bm25s` library. Index is built from documents and persisted to disk; rebuilt from scratch on each `aadd_documents()` call.

```yaml
retrievers:
  bm25_local:
    type: bm25
    k: 4
    preprocessing: default     # default | spacy
    spacy_model: en_core_web_sm
    cache_dir: ~               # null → data/bm25_cache/<tag>/
```

Index files written to disk:

| File | Content |
|------|---------|
| `data/bm25_cache/<tag>/bm25_index/` | bm25s vectorizer (pickle) |
| `data/bm25_cache/<tag>/documents.json` | original documents with metadata |

`get_or_load_retriever()` lazy-loads from disk on first query after a restart.

### `ensemble` — weighted fusion

Combines N other retrievers using LangChain's `EnsembleRetriever` (Reciprocal Rank Fusion).

```yaml
retrievers:
  hybrid:
    type: ensemble
    retrievers:
      - { ref: my_vec, weight: 0.7 }
      - { ref: bm25_local, weight: 0.3 }
```

Weights are normalised to sum to 1.0 before being passed to `EnsembleRetriever`. Each `ref` is resolved by recursively calling `RetrieverFactory.create()`.

### `reranked` — contextual compression

Wraps another retriever with a reranking step that re-scores and filters results.

```yaml
retrievers:
  best_results:
    type: reranked
    retriever: hybrid           # any key in retrievers:
    reranker: embeddings        # embeddings | cohere | cross_encoder
    top_k: 3
    fetch_k: 10                 # how many docs the base retriever fetches
    reranker_model: ~           # optional model name for cohere/cross_encoder
```

| `reranker` | Backend | Extra dependency |
|-----------|---------|-----------------|
| `embeddings` | `EmbeddingsFilter` (semantic similarity ≥ 0.7) | none |
| `cohere` | `CohereRerank` | `uv add langchain-cohere` |
| `cross_encoder` | `HuggingFaceCrossEncoder` | `uv add sentence-transformers` |

### `pg_hybrid` — PostgreSQL vector + full-text

Combines pgvector similarity with PostgreSQL full-text search (tsvector) in a single query — fastest hybrid option when you have Postgres.

```yaml
retrievers:
  pg_hybrid:
    type: pg_hybrid
    embeddings: default        # key in embeddings: section
    postgres: default          # key in postgres: section
    table_name_prefix: embeddings
    hybrid_search: true
    top_k: 4
    hybrid_search_config:
      tsv_lang: pg_catalog.english
      fusion_function_parameters:
        primary_results_weight: 0.7   # vector weight
        secondary_results_weight: 0.3  # full-text weight
```

Required dependencies:

```bash
uv add langchain-postgres psycopg2-binary asyncpg
# Or for embedded Postgres (no server):
uv add pgembed pgembed-pgvector
```

`postgres:` block schema:

```yaml
postgres:
  default:
    mode: external
    url: postgresql+asyncpg://${oc.env:POSTGRES_USER,postgres}:${oc.env:POSTGRES_PASSWORD,password}@localhost:5432/genai

  embedded:
    mode: pgembed
    data_dir: ${paths.data_root}/pgembed
    extensions: [vector]
```

### `zero_entropy` — ZeroEntropy SDK

Read-only external retriever backed by ZeroEntropy's hosted document search.

```yaml
retrievers:
  ze_docs:
    type: zero_entropy
    collection_name: my_collection
    k: 5
    retrieval_type: documents
```

`store = None` for this type — ingestion happens in ZeroEntropy directly, not via `aadd_documents`.

---

## Short alias vs. fully-qualified type names

The `type` field accepts either form:

```yaml
type: vector                                              # short alias
type: genai_tk.core.retrievers.VectorRetriever            # qualified
```

Use qualified names when registering custom retriever builders from your own codebase (see "Custom retrievers" below).

## `embeddings_store:` blocks (used by `vector`)

Referenced from `retrievers.<tag>.embeddings_store`:

```yaml
embeddings_store:
  in_memory_chroma:
    backend: Chroma                # short alias
    embeddings: default
    table_name_prefix: embeddings
    config:
      storage: '::memory::'        # ephemeral — useful for tests

  chroma_indexed:
    backend: Chroma
    embeddings: default
    table_name_prefix: embeddings
    config:
      storage: ${paths.data_root}/vector_store

  local_fast:
    backend: Chroma
    embeddings: bge-small-en@local
    table_name_prefix: embeddings
    config:
      storage: ${paths.data_root}/vector_store_local_fast
```

Backends:

| Short alias | Qualified name | Notes |
|-----------|---|---------|
| `Chroma` | `genai_tk.core.vector_backends.ChromaBackend` | In-memory or persistent |
| `InMemory` | `genai_tk.core.vector_backends.InMemoryBackend` | Ephemeral in-process |
| `PgVector` | `genai_tk.core.vector_backends.PgVectorBackend` | Postgres + pgvector |

---

## A complete YAML example

```yaml
# config/baseline.yaml (or merged from any file in :merge)

retrievers:
  default:
    type: vector
    embeddings_store: in_memory_chroma
    top_k: 4

  persistent:
    type: vector
    embeddings_store: chroma_indexed
    top_k: 8

  bm25_local:
    type: bm25
    k: 8
    preprocessing: spacy

  hybrid:
    type: ensemble
    retrievers:
      - { ref: persistent, weight: 0.7 }
      - { ref: bm25_local, weight: 0.3 }

  hybrid_reranked:
    type: reranked
    retriever: hybrid
    reranker: embeddings
    top_k: 4
    fetch_k: 12

embeddings_store:
  in_memory_chroma:
    backend: Chroma
    embeddings: default
    config:
      storage: '::memory::'
  chroma_indexed:
    backend: Chroma
    embeddings: default
    config:
      storage: ${paths.data_root}/vector_store
```

---

## Python API

### Basic usage

```python
from genai_tk.core.retriever_factory import RetrieverFactory

# Async (canonical)
managed = RetrieverFactory.create("hybrid_reranked")
docs = await managed.aquery("What is hybrid search?", k=5)
await managed.aadd_documents(documents)        # if has_store

# Sync (CLI / notebook only — not safe inside an event loop)
docs = managed.query("question", k=5)
managed.add_documents(documents)
```

### Filters and overrides

```python
# Metadata filter (only supported by vector stores)
docs = await managed.aquery(
    "pricing",
    k=10,
    filter={"source": "contracts", "year": 2024}
)

# Get raw retriever for use in chains
raw = managed.retriever                          # langchain BaseRetriever
```

### Introspection

```python
print(managed.has_store)                         # True/False — can ingest?
print(managed.default_k)                         # 4
print(managed.config_tag)                        # "hybrid_reranked"
print(managed.get_stats())                       # diagnostic dict

print(RetrieverFactory.list_available_configs()) # ['default', 'bm25_local', ...]
```

### Deletion

```python
success = await managed.adelete_store()          # async
managed.delete_store()                            # sync
```

Chroma collections and BM25 cache directories are cleaned up. PgVector deletion is not yet implemented.

### Building a RAG chain

```python
from genai_tk.core.retriever_factory import RetrieverFactory
from genai_tk.core.llm_factory import get_llm
from genai_tk.core.prompts import def_prompt
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

managed = RetrieverFactory.create("hybrid_reranked")
llm = get_llm()

prompt = def_prompt(
    system="Answer using only the provided context.",
    user="Context:\n{context}\n\nQuestion: {question}",
)

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

chain = (
    {"context": managed.retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

answer = chain.invoke("What is hybrid search?")
```

### Document stores directly

`BM25DocumentStore` and `VectorDocumentStore` are usable independently — useful for tests or custom pipelines.

```python
from pathlib import Path
from genai_tk.core.retriever_factory import BM25DocumentStore

store = BM25DocumentStore(cache_dir=Path("data/bm25_cache/my_index"))
await store.aadd_documents(docs)
results = await store.aget_relevant_documents("query", k=3)
```

---

## CLI

```bash
cli rag list-retrievers                                   # show configured tags
cli rag info hybrid_reranked                              # stats for a retriever
cli rag add-files ./docs/ -r persistent                   # ingest a directory
cli rag add-files ./docs/ -r persistent \
    --include "**/*.md" --exclude "**/drafts/**"          # filtered ingest
cli rag add-files ./docs/ -r persistent --force           # reprocess all
cli rag query "What is RAG?" -r hybrid_reranked --k 10
cli rag query "pricing" -r persistent --filter '{"source":"contracts"}'
cli rag query "setup" -r persistent --full                # show full content
cli rag embed persistent --text "snippet to index" --metadata '{"src":"manual"}'
cli rag delete persistent --force                          # clear store
```

`add-files` defaults: `--recursive`, `--batch-size 10`, `--chunk-size 512`. Files are content-hashed to skip unchanged ones unless `--force`.

---

## Batch ingestion — Prefect flow

For large-scale ingestion, use the bundled Prefect flow:

```python
from genai_tk.extra.prefect.runtime import run_flow_ephemeral
from genai_tk.extra.rag.rag_prefect_flow import rag_file_ingestion_flow

result = run_flow_ephemeral(
    rag_file_ingestion_flow,
    root_dir="./documents",
    retriever_name="persistent",
    max_chunk_tokens=512,
    include_patterns=["**/*.md", "**/*.txt"],
    exclude_patterns=["**/node_modules/**"],
    recursive=True,
    force=False,
    batch_size=10,
)
# → {"total_files": 142, "processed_files": 38, "skipped_files": 104, "total_chunks": 512}
```

Content hashing skips unchanged files (unless `force=True`). Hashing only happens for persistent Chroma stores.

---

## Using a retriever as an agent tool

`RAGToolFactory` wraps a `ManagedRetriever` as an async LangChain tool that any agent can use:

```python
from genai_tk.tools.langchain.rag_tool_factory import RAGToolFactory

rag_tool = RAGToolFactory.create(
    retriever_name="hybrid_reranked",
    tool_name="search_docs",
    tool_description="Search the company documentation for relevant passages.",
)

# Then attach to an agent
from genai_tk.agents.langchain import LangchainAgent
agent = LangchainAgent("Research", tools=[rag_tool])
```

Or in YAML:

```yaml
tools:
  - spec: rag_search
    config:
      retriever_name: hybrid_reranked
      tool_name: search_docs
      tool_description: Search the company documentation.
```

---

## Custom retrievers

Define your own retriever builder anywhere in your codebase and reference it by qualified class name in YAML — no need to fork genai-tk.

A builder needs two members:

- **`config_model`** — a Pydantic v2 class for parsing the YAML sub-dict
- **`build(cfg, config_tag, resolver)`** — classmethod returning a `ManagedRetriever`

```python
# myapp/retrievers/custom.py
from pydantic import BaseModel
from collections.abc import Callable
from typing import Any
from genai_tk.core.retriever_factory import ManagedRetriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document


class MyCustomConfig(BaseModel):
    service_url: str
    api_key: str
    top_k: int = 4


class MyCustomRetriever:
    config_model = MyCustomConfig

    @classmethod
    def build(cls, cfg, config_tag, resolver: Callable[[str], Any]) -> ManagedRetriever:
        class CustomServiceRetriever(BaseRetriever):
            async def _aget_relevant_documents(self, query: str, **kwargs) -> list[Document]:
                # call your service here
                return [Document(page_content="...", metadata={})]

            def _get_relevant_documents(self, query, **kwargs):
                return [Document(page_content="...", metadata={})]

        return ManagedRetriever(
            retriever=CustomServiceRetriever(),
            store=None,                          # read-only
            default_k=cfg.top_k,
            config_tag=config_tag,
        )
```

Reference in YAML:

```yaml
retrievers:
  my_custom_service:
    type: myapp.retrievers.custom.MyCustomRetriever
    service_url: https://search.example.com/api
    api_key: ${oc.env:MY_SERVICE_API_KEY}
    top_k: 5
```

The `resolver` callable passed to `build()` is `RetrieverFactory.create` — use it if your retriever needs to compose with others (e.g., wrapping another tag).

---

## Patterns

### Hybrid search with reranking

The default "best of both worlds" config:

```yaml
retrievers:
  vec: { type: vector, embeddings_store: chroma_indexed, top_k: 12 }
  bm:  { type: bm25, k: 12, preprocessing: spacy }
  hybrid:
    type: ensemble
    retrievers: [{ ref: vec, weight: 0.6 }, { ref: bm, weight: 0.4 }]
  best:
    type: reranked
    retriever: hybrid
    reranker: cross_encoder
    reranker_model: cross-encoder/ms-marco-MiniLM-L-6-v2
    top_k: 4
    fetch_k: 12
```

### In-memory test store

Useful for unit tests — no persistence, no API calls if you use `embeddings: default` paired with a local model:

```yaml
embeddings_store:
  test_store:
    backend: InMemory
    embeddings: bge-small-en@local
retrievers:
  test_vec:
    type: vector
    embeddings_store: test_store
    top_k: 3
```

### Switching environments

Per-environment retriever swaps go in `overrides.yaml` (which is the last file in the `:merge` list and git-ignored by default). Use `BLUEPRINT_CONFIG=production` to switch the active config.
