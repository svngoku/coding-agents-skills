# RAG — references/rag.md

genai-tk's retrieval layer separates **where vectors live** (`embeddings_store:`), **how to search**
(`retrievers:`), and **how documents are split** (`chunkers:`). All three are YAML-declared and resolved by
fully qualified class paths.

## Imports

```python
from genai_tk.core.factories import ChunkerFactory, ManagedRetriever, RetrieverFactory
from genai_tk.core.embeddings_store import EmbeddingsStore
```

The old `genai_tk.core.retriever_factory`, `genai_tk.tools.langchain.rag_tool_factory`, and
`genai_tk.extra.rag.*` modules no longer exist.

## Embeddings Stores

```yaml
embeddings_store:
  in_memory:
    backend: genai_tk.core.vector_backends.InMemoryBackend
  scratch:
    backend: genai_tk.core.vector_backends.ChromaBackend
    embeddings: default                       # embeddings tag or model_id@provider
    config: {storage: "::memory::"}
  chroma:
    backend: genai_tk.core.vector_backends.ChromaBackend
    embeddings: default
    table_name_prefix: embeddings
    config: {storage: "${paths.data_root}/vector_store"}
  pg:
    backend: genai_tk.core.vector_backends.PgVectorBackend
    embeddings: default
    config:
      postgres: default                       # → postgres.default section
      postgres_schema: public
      hybrid_search: true

postgres:
  default:
    mode: external                            # external | pgembed
    url: ${oc.env:POSTGRES_URL}
```

| Backend | `config:` keys | Extra |
|---------|----------------|-------|
| `genai_tk.core.vector_backends.InMemoryBackend` | none | core |
| `genai_tk.core.vector_backends.ChromaBackend` | `storage` (`::memory::` or a path) | `chromadb` |
| `genai_tk.core.vector_backends.PgVectorBackend` | `postgres`, `postgres_schema`, `metadata_columns`, `hybrid_search`, `hybrid_search_config` | `postgres` |

Store-level fields: `backend`, `embeddings`, `table_name_prefix`, `collection_metadata`, `config`.

**Short names (`backend: Chroma`, `InMemory`, `PgVector`) no longer resolve**; always use the qualified class path.
ZVec appears in upstream marketing but has no YAML backend class yet.

```python
store = EmbeddingsStore.create_from_config("chroma")   # direct instantiation raises
vector_store = store.get_vector_store()                # LangChain VectorStore
```

## Retrievers

```yaml
retrievers:
  docs:
    type: genai_tk.core.retrievers.VectorRetriever
    embeddings_store: chroma
    top_k: 4
    record_manager_url: "sqlite:///${paths.data_root}/record_manager/docs.db"   # dedup on re-ingest

  docs_bm25:
    type: genai_tk.core.retrievers.BM25Retriever
    k: 4
    preprocessing: default                  # default | spacy (needs the nlp extra)

  hybrid:
    type: genai_tk.core.retrievers.EnsembleRetriever
    retrievers:
      - {ref: docs, weight: 0.7}
      - {ref: docs_bm25, weight: 0.3}       # weights are normalized

  best:
    type: genai_tk.core.retrievers.RerankedRetriever
    retriever: hybrid
    reranker: embeddings                    # embeddings | cohere | cross_encoder
    top_k: 3
```

| `type` | Fields |
|--------|--------|
| `genai_tk.core.retrievers.VectorRetriever` | `embeddings_store` (required), `top_k=4`, `record_manager_url` |
| `genai_tk.core.retrievers.BM25Retriever` | `k=4`, `preprocessing=default\|spacy`, `spacy_model`, `cache_dir` (default `${paths.data_root}/bm25_cache/<tag>`), `bm25_params` |
| `genai_tk.core.retrievers.EnsembleRetriever` | `retrievers: [{ref, weight=0.5}]` (reciprocal-rank fusion) |
| `genai_tk.core.retrievers.RerankedRetriever` | `retriever` (required), `reranker`, `top_k=3`, `reranker_model`, `embeddings` |
| `genai_tk.core.retrievers.PgHybridRetriever` | `embeddings`, `postgres`, `schema_name`, `table_name_prefix`, `top_k`, `metadata_columns`, `hybrid_search=True`, `hybrid_search_config`, `record_manager_url` |
| `genai_tk.core.retrievers.ZeroEntropyRetriever` | `collection_name` (required), `k=5`, `retrieval_type` (read-only, ZeroEntropy SDK) |

Known quirks at `7fbd7a3`:

- `type: vector` / `bm25` / `ensemble` short aliases do not resolve (the alias fallback is dead code).
- `fetch_k` on `RerankedRetriever` and `search_type` on `VectorRetriever` are accepted but have no effect.
- The automatic SQLite record manager for persistent stores no longer triggers; set `record_manager_url`
  explicitly if you re-ingest the same files.
- The `embeddings` reranker is an `EmbeddingsFilter` (similarity threshold 0.7); `cross_encoder` needs
  `langchain-huggingface` (`nlp` extra); `cohere` needs a Cohere key.

## `ManagedRetriever`

`RetrieverFactory.create(tag)` returns a `ManagedRetriever`, an async-first wrapper:

```python
from langchain_core.documents import Document

from genai_tk.core.factories import RetrieverFactory

retriever = RetrieverFactory.create("docs")
await retriever.aadd_documents([Document(page_content="genai-tk is YAML-driven.", metadata={"src": "a.md"})])
docs = await retriever.aquery("What drives genai-tk?", k=5, filter={"src": "a.md"})
print(retriever.get_stats(), retriever.has_store)

RetrieverFactory.list_available_configs()     # all retriever tags
```

| Method | Notes |
|--------|-------|
| `aquery(query, k=None, filter=None)` / `query` | `k`/`filter` only apply to vector- or BM25-backed retrievers |
| `aadd_documents(docs)` / `add_documents` | Raises `RuntimeError` when there is no writable store (`has_store` is false) |
| `adelete_store()` / `delete_store` | Chroma and BM25 only; otherwise `NotImplementedError` |
| `get_stats()`, `.retriever`, `.store` | Introspection; `.retriever` is the LangChain `BaseRetriever` |

Ensemble and ZeroEntropy retrievers go through `retriever.ainvoke(query)`, so `k` and `filter` are ignored. A
reranked retriever exposes its base store, so `aquery` on it **bypasses the reranker**; call
`await retriever.retriever.ainvoke(query)` when you need reranked results.

Ingest into the **base** retrievers (`docs`, `docs_bm25`); ensembles and rerankers read from them.

## Chunkers

```yaml
chunkers:
  recursive:
    class: langchain_text_splitters.RecursiveCharacterTextSplitter
    params: {chunk_size: 512, chunk_overlap: 50}
  markdown:
    class: genai_tk.workflow.rag.chonkie_splitter.ChonkieTextSplitter
    params: {chunker_type: markdown, max_tokens: 300, min_tokens: 50, merge_small_chunks: true, encoding_name: o200k_base}
  auto_map:                     # extension → chunker; used by --chunker auto
    ".md": markdown
    ".txt": recursive
    ".default": recursive
```

```python
from genai_tk.core.factories import ChunkerFactory

splitter = ChunkerFactory.create("markdown")
splitter = ChunkerFactory.create_for_file("notes/intro.md")    # chunker_name="auto" → auto_map
chunks = splitter.split_documents(docs)
```

## CLI — `cli rag`

```bash
uv run cli rag list-retrievers
uv run cli rag list-configs
uv run cli rag add-files ./docs -r docs                          # base dir is positional
uv run cli rag add-files ./docs -r docs -p "**/*.md" -p "!drafts/**" --chunker markdown --dry-run
uv run cli rag add-files ./docs -r docs --force --batch-size 20 --chunk-size 512
uv run cli rag query "What drives genai-tk?" -r docs --k 5
uv run cli rag query "pricing" -r docs --filter '{"src": "a.md"}' --full
uv run cli rag embed docs -t "one-off text" -m '{"src": "cli"}'
uv run cli rag info docs
uv run cli rag delete docs -f
```

- `add-files BASE_DIR [-p/--pathspec ...] [-r/--retriever NAME] [-f/--force] [--dry-run] [-b/--batch-size 10]
  [--chunk-size 512] [--chunker auto]`. Pathspecs are gitignore-style and repeatable; prefix `!` to exclude.
  `--include`, `--exclude`, and `--recursive` are gone.
- `add-files` runs the Prefect flow `rag_file_ingestion_flow` and auto-starts a Prefect server when
  `prefect.auto_start` is true. Unchanged files are skipped via a manifest unless `--force` is set.
- `query [TEXT] [-r NAME] [--k 4] [--filter JSON] [--full] [-l/--max-length 100]`.

Prefer `add-files` over the bundled `rag_ingest` workflow: at upstream commit `7fbd7a3` that workflow passes
`chunk_size`/`chunker` to a flow that expects `max_chunk_tokens`/`chunker_name` and fails. The `md_ingest` example
in [workflows.md](workflows.md) shows a working definition.

```python
from genai_tk.workflow.prefect.flows.rag_flow import rag_file_ingestion_flow

rag_file_ingestion_flow(base_dir="./docs", retriever_name="docs", max_chunk_tokens=512, pathspecs=["**/*.md"])
```

## RAG as an Agent Tool

```yaml
agents:
  docs_qa:
    harness: langchain
    type: react
    llm: gpt-4o-mini@openai
    system_prompt: "Answer only from the documentation. Cite file names."
    tools:
      - factory: genai_tk.agents.tools.langchain.rag_tool_factory.create_rag_tool_from_config
        config:
          retriever: hybrid
          tool_name: search_docs
          tool_description: "Search the product documentation."
          top_k: 5
```

```python
from genai_tk.agents.tools.langchain.rag_tool_factory import RAGToolConfig, RAGToolFactory
from genai_tk.core.factories import get_llm

factory = RAGToolFactory(get_llm())                      # takes a BaseChatModel, not an id
tool = factory.create_tool(RAGToolConfig(retriever="hybrid", tool_name="search_docs", top_k=5))
result = await tool.ainvoke({"query": "vector search", "filter": '{"src": "a.md"}'})
```

`RAGToolConfig` fields: `retriever`, `tool_name="rag_search"`, `tool_description`, `default_filter`, `top_k=4`.
The older `spec: rag_search` / `retriever_name:` form (still shown in upstream `docs/rag.md`) fails validation.

## Choosing a Setup

| Situation | Setup |
|-----------|-------|
| Prototype or tests | `InMemoryBackend` or Chroma `::memory::` + `VectorRetriever`; fake embeddings |
| Local persistent corpus | Chroma on disk + `record_manager_url` |
| Keyword-heavy content (codes, part numbers) | `EnsembleRetriever` of vector + BM25 |
| Precision matters more than latency | `RerankedRetriever` over the ensemble |
| Production on Postgres | `PgVectorBackend` or `PgHybridRetriever` (pgvector + tsvector in one query) |
| Documents are PDFs/Office files | Convert first with the `markdownize` workflow, then ingest the Markdown |
