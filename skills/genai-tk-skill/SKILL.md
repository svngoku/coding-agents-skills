---
name: genai-tk
description: Build GenAI and agentic applications with the genai-tk toolkit (https://github.com/tclatos/genai-tk) — a YAML-driven wrapper over LangChain, LangGraph, and 100+ LLM providers. Use this skill whenever the user mentions genai-tk, genai_tk, the GenAI Toolkit, `cli init`, `LangchainAgent`, `get_llm`/`get_embeddings`, `RetrieverFactory`/`ManagedRetriever`, the four bundled agent frameworks (ReAct, Deep, Deer-flow, SmolAgents), the OpenSandbox Docker integration, the `model_id@provider` identifier format, the `global_config()`/`OmegaConfig` system with `app_conf.yaml` and `:merge`, BAML structured extraction, SkillsMiddleware, writing or editing the toolkit's YAML profiles (langchain.yaml, deerflow.yaml, llm.yaml, retrievers.yaml), composing retrievers (vector/bm25/ensemble/reranked/pg_hybrid/zero_entropy), or extending the CLI with `CliTopCommand`. Trigger even when the user only says "the toolkit" in context.
---

# genai-tk — GenAI & Agentic Toolkit

`genai-tk` (https://github.com/tclatos/genai-tk) is a Python toolkit that wraps LangChain, LangGraph, and 100+ LLM providers into a YAML-driven, configuration-first architecture. It's not a framework you build *into* — it's an inversion-of-control layer where **profiles in YAML** drive **factories** that produce LangChain runtime objects (LLMs, embeddings, retrievers, agents).

The mental model:

```
config/*.yaml ──► global_config() ──► Factory ──► LangChain object ──► your code
```

## Quick Reference

| Topic | Reference |
|-------|-----------|
| Agent frameworks: ReAct, Deep, Deer-flow, SmolAgents — profiles, tools, middleware, MCP, sandbox, skills | [agents.md](references/agents.md) |
| RAG: `RetrieverFactory`, `ManagedRetriever`, the six retriever types, ingestion | [rag.md](references/rag.md) |
| Configuration: `global_config()`, `app_conf.yaml`, `:merge`, environments, `model_id@provider` | [configuration.md](references/configuration.md) |
| CLI: `cli init`, command groups, extending the CLI with `CliTopCommand` | [cli-and-init.md](references/cli-and-init.md) |
| BAML structured extraction with `BamlStructuredProcessor` | [baml-structured.md](references/baml-structured.md) |

## Installation

The toolkit is installed via `uv` and is **not on PyPI** — install from the GitHub repo:

```bash
# Add to existing project
uv add git+https://github.com/tclatos/genai-tk@main

# With extras (postgres, browser, evals)
uv add "genai-tk[extra] @ git+https://github.com/tclatos/genai-tk@main"

# Development (clone & edit)
git clone https://github.com/tclatos/genai-tk.git && cd genai-tk
uv sync                    # core + dev
uv sync --all-groups       # + postgres, browser, evals
```

After install, scaffold a project:

```bash
uv run cli init --name "My AI Project"   # full scaffold (recommended)
uv run cli init --minimal                 # config + Makefile only
uv run cli init --deer-flow               # also clone Deer-flow backend
uv sync
```

`cli init` creates `config/`, `Makefile`, a Python package with example CLI commands and an LCEL chain, an `AGENTS.md`, and `.github/copilot-instructions.md`. See [cli-and-init.md](references/cli-and-init.md) for the full file inventory.

API keys go in `.env` at the project root (auto-loaded):

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=...
GROQ_API_KEY=...
TAVILY_API_KEY=...    # for web search
```

## The four pillars

### 1. LLM & Embeddings factories

Models are referenced as `model_id@provider`. The `model_id` is a short alias declared in `config/providers/llm.yaml`; the provider is one of `openai`, `openrouter`, `groq`, `ollama`, `fake`, etc. The toolkit auto-resolves model capabilities from a built-in `models.dev` database covering 1000+ models.

```python
from genai_tk.core.llm_factory import get_llm
from genai_tk.core.embeddings_factory import get_embeddings

llm = get_llm()                          # default from baseline.yaml
llm = get_llm("gpt41mini@openai")        # explicit alias
llm = get_llm("fast_model")              # named tag from baseline.yaml
llm = get_llm("gpt-4o-mini")             # raw model name — fuzzy resolved
llm = get_llm("parrot_local@fake")       # fake model — no API key, returns input

# kwargs forward to the underlying ChatModel
llm = get_llm("gpt41mini@openai", temperature=0.7, streaming=True, json_mode=True)

# Reasoning/thinking models
llm = get_llm("o3@openai", reasoning=True)

embeddings = get_embeddings()                            # default
embeddings = get_embeddings("ada_002@openai")            # explicit
vectors = embeddings.embed_documents(["doc 1", "doc 2"])
```

The returned object is a regular LangChain `BaseChatModel` / `Embeddings`, so it composes with everything else in the LC ecosystem (LCEL, prompts, output parsers, LangGraph nodes, etc.).

`get_llm()` accepts `cache="sqlite"`, `"memory"`, or `"no_cache"` to override the global cache. Defaults are set in `baseline.yaml` (`llm.cache: sqlite`, `llm.cache_path: data/llm_cache/langchain.db`).

To list every model the toolkit knows about:

```bash
cli info models                              # all declared + db-discovered
cli info llm-profile gpt41mini@openai        # exact match
cli info llm-profile gpt-4o                  # fuzzy match
cli info config                              # show resolved config + API key status
```

### 2. Agents — four frameworks, one profile system

| Framework | Best for | Profile location |
|-----------|----------|------------------|
| **ReAct** (LangChain) | General tasks, tool use | `config/agents/langchain.yaml` (`type: react`) |
| **Deep** (LangChain + `deepagents`) | Multi-step planning, subagents, sandbox | `config/agents/langchain.yaml` (`type: deep`) |
| **Deer-flow** (ByteDance, embedded) | Deep web research, native search | `config/agents/deerflow.yaml` |
| **SmolAgents** (HuggingFace) | Code-first automation | invoked via `cli agents smolagents` |

All four share the same LLM factory, MCP server registry, sandbox config, and skills directories. The unified entry point for LangChain agents is `LangchainAgent`:

```python
from genai_tk.agents.langchain import LangchainAgent

# From a YAML profile
agent = LangchainAgent("Research")
result = agent.run("Summarise recent AI news")

# Async / streaming
result = await agent.arun("Explain RAG")
async for chunk in agent.astream("Tell me a story"):
    print(chunk, end="", flush=True)

# Ad-hoc agent (no profile needed when llm is given)
agent = LangchainAgent(llm="gpt41mini@openai", tools=[my_tool])
```

A profile in `langchain.yaml`:

```yaml
langchain_agents:
  default_profile: "Research"
  profiles:
    - name: "Research"
      type: deep
      llm: gpt_41@openai
      enable_planning: true
      tools:
        - spec: web_search
          config: { provider: serper }
      mcp_servers: [tavily-mcp]
      skills:
        directories:
          - ${paths.project}/skills
      backend:
        type: aio_sandbox             # optional Docker sandbox
      checkpointer:
        type: memory                  # memory | postgres | sqlite
```

CLI equivalents:

```bash
cli agents langchain --list                               # show profiles
cli agents langchain -p Research --chat                   # interactive
cli agents langchain -p Research "Query"                  # one-shot
cli agents langchain --sandbox docker -p Research "..."   # sandboxed exec
cli agents deerflow --chat                                # Deer-flow
cli agents smolagents --executor docker "..."             # code-first
```

For the full agent reference — type semantics, middlewares, tool specs, MCP wiring, sandbox setup, skill loading — read [agents.md](references/agents.md).

### 3. RAG — `RetrieverFactory` + `ManagedRetriever`

The RAG layer is built around two abstractions:

- **`RetrieverFactory.create(tag)`** — reads `retrievers.<tag>` from YAML, returns a `ManagedRetriever`
- **`ManagedRetriever`** — async-first wrapper with `aquery`, `aadd_documents`, `adelete_store`, `get_stats`

Six retriever types compose freely under the `retrievers:` YAML key:

| Type | Purpose |
|------|---------|
| `vector` | Dense similarity (Chroma, InMemory, PgVector) |
| `bm25` | Sparse keyword (`bm25s` library, persisted to disk) |
| `ensemble` | Reciprocal-rank fusion of N other retrievers |
| `reranked` | Wraps another retriever with embeddings / Cohere / cross-encoder reranking |
| `pg_hybrid` | PostgreSQL pgvector + tsvector full-text in a single query |
| `zero_entropy` | Read-only ZeroEntropy SDK retriever |

```python
from genai_tk.core.retriever_factory import RetrieverFactory

retriever = RetrieverFactory.create("hybrid_with_rerank")  # tag from YAML
docs = await retriever.aquery("question", k=5)
await retriever.aadd_documents(documents)                  # if has_store
```

A composed YAML config:

```yaml
retrievers:
  my_vec:
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
      - { ref: my_vec, weight: 0.7 }
      - { ref: bm25_local, weight: 0.3 }
  hybrid_with_rerank:
    type: reranked
    retriever: hybrid
    reranker: embeddings
    top_k: 4
    fetch_k: 12
```

CLI:

```bash
cli rag list-retrievers                                       # show tags
cli rag add-files my_vec ./docs/                              # ingest
cli rag query my_vec "What does the doc say about X?"
```

For the full RAG reference — every retriever type's full schema, the document store separation, ingestion semantics, PgHybrid setup, and using retrievers as agent tools — read [rag.md](references/rag.md).

### 4. Configuration — `global_config()` + `app_conf.yaml`

The config system uses [OmegaConf](https://omegaconf.readthedocs.io/) with auto-discovery: on import, `global_config()` walks up from the current working directory until it finds `config/app_conf.yaml`. **This means notebooks, subdirectories, and deployed containers all work without setup.**

```python
from genai_tk.utils.config_mngr import global_config

cfg = global_config()
default_llm = cfg.get("llm.default")            # str
top_k = cfg.get("retrievers.my_vec.top_k", 4)   # with default
agent_profiles = cfg.get_dict("langchain_agents.profiles")
```

`app_conf.yaml` is the entry point. It declares a `:merge` list of YAML files that are deep-merged in order (later wins):

```yaml
default_config: ${oc.env:BLUEPRINT_CONFIG,baseline}

:merge:
  - ${paths.config}/baseline.yaml
  - ${paths.config}/providers/llm.yaml
  - ${paths.config}/providers/embeddings.yaml
  - ${paths.config}/overrides.yaml
  - ${paths.config}/mcp_servers.yaml

:env:
  LOGURU_LEVEL: INFO

paths:
  project: ${oc.env:PWD}
  config: ${paths.project}/config
  data_root: ${paths.project}/data
```

Switch named environments without touching code: `BLUEPRINT_CONFIG=production` (env var) or `cfg.select_config("production")` (Python). Env vars in `${oc.env:VAR,default}` come from `.env`.

For the full configuration reference — directive semantics, override patterns, computed paths, environment switching — read [configuration.md](references/configuration.md).

## When to use what — a decision guide

| You want to… | Reach for | Doc |
|--------------|-----------|-----|
| Call an LLM with a one-line config swap | `get_llm("alias@provider")` | `core.md` (in repo) |
| Build a tool-using agent | `LangchainAgent("ProfileName")` (`type: react`) | [agents.md](references/agents.md) |
| Plan + delegate to subagents | `type: deep` profile + Docker sandbox | [agents.md](references/agents.md) |
| Do deep web research with reports | `cli agents deerflow` | [agents.md](references/agents.md) |
| Code execution / data analysis agent | `cli agents smolagents --executor docker` | [agents.md](references/agents.md) |
| Build a RAG system | `RetrieverFactory.create()` with a `vector` or `ensemble` retriever | [rag.md](references/rag.md) |
| Get strict typed output from an LLM | BAML + `BamlStructuredProcessor` | [baml-structured.md](references/baml-structured.md) |
| Add a CLI command to your project | Subclass `CliTopCommand`, register in `app_conf.yaml` | [cli-and-init.md](references/cli-and-init.md) |
| Anonymise PII before sending to LLM | `AnonymizationMiddleware` (Presidio + Faker) | repo: `docs/middleware-pii-and-routing.md` |
| Run untrusted code | `--sandbox docker` (uses OpenSandbox) | repo: `docs/sandbox_support.md` |

## Coding conventions (from the project's AGENTS.md)

When generating code that lives inside a genai-tk project, follow the project's house style:

- **Python 3.12+** — use `str | None`, `list[str]` (not `Optional[str]`, `List[str]`)
- **Imports always absolute** — `from genai_tk.core import LLMFactory`, never relative
- **Pydantic v2** for all structured data — DTOs, configs, results. Use `model_config = {...}` and `model_post_init()`, never inner `class Config` or `__init__`
- **`loguru` for logging** — not the stdlib `logging`
- **Ruff** for formatting + linting, line length 120
- **Async-first** for I/O — sync wrappers exist for CLI/notebook convenience but the canonical API is async
- **Docstrings**: Google style, no type info (it's in the signature), no `Raises:` sections
- **Skills over system prompts**: domain knowledge goes in `skills/<name>/SKILL.md` files loaded on demand by `SkillsMiddleware`, not embedded in agent prompts
- **YAML-first**: when adding a new agent, retriever, or LLM, add a config entry — don't hard-code in Python

## Key gotchas

**`model_id@provider` is the canonical identifier.** Anywhere a model is named — CLI flags, YAML, Python — use this format. Bare model names (`gpt-4o-mini`) work via fuzzy lookup but are slower and may resolve unpredictably across versions. Prefer declared aliases in `llm.yaml`.

**Config auto-discovery walks up from the CWD.** If `global_config()` raises `ConfigFileNotFoundError`, the cwd is wrong or `config/app_conf.yaml` doesn't exist yet — run `cli init` to scaffold it.

**The `:merge` order matters.** Later files override earlier ones. `overrides.yaml` is meant to be the last merged file (and is git-ignored by default), so per-environment tweaks go there.

**Deep agents need an extra package.** `type: deep` requires `pip install deepagents`. Ad-hoc `LangchainAgent(llm=...)` defaults to `react`.

**Deer-flow runs in-process but needs the backend cloned.** Run `cli init --deer-flow` once and set `DEER_FLOW_PATH` in `.env`. The toolkit embeds it; there's no separate server to start.

**The OpenSandbox warm-up is worth it.** `cli sandbox start && cli sandbox pull` once per boot cuts container startup from ~28s to ~5s on subsequent agent runs.

**Skills are loaded on demand, not injected into every prompt.** This is the project's preferred pattern for domain knowledge — write a `SKILL.md`, point an agent profile at the directory, and `SkillsMiddleware` lets the agent read them only when relevant. This is genuinely different from "stuffing the system prompt" and worth using.

**`get_llm` accepts deprecated `llm_id`/`llm_tag` kwargs that warn — use `llm=` instead.** The same applies to `get_embeddings(embeddings=...)`.

## Anti-Patterns to Avoid

- **Hard-coding agents/LLMs in Python instead of YAML** — the toolkit's whole point is profile-driven config; adding an agent means adding a profile, not a Python class
- **Bare model names instead of `model_id@provider`** — fuzzy lookup is slower and can resolve unpredictably; declare aliases in `llm.yaml` and use the canonical identifier
- **Stuffing the system prompt with domain knowledge** — write `SKILL.md` files and point profiles at them; `SkillsMiddleware` loads them on demand
- **`type: deep` without `deepagents` installed** — deep agents need the extra package; ad-hoc agents default to `react`
- **Running untrusted code without the Docker sandbox** — `--sandbox docker` (OpenSandbox) is the isolation boundary; `local` executes in your process
- **Putting per-environment overrides in tracked config files** — `overrides.yaml` is merged last and git-ignored; that's where environment tweaks belong
- **Skipping the OpenSandbox warm-up** — `cli sandbox start && cli sandbox pull` once per boot cuts container startup from ~28s to ~5s
- **Using deprecated `llm_id`/`llm_tag` kwargs** — they warn; use `llm=` / `embeddings=` instead

## When to Use / Not Use

**Use this skill when:**

- Building genai-tk projects: YAML-driven agents, retrievers, LLM factories
- Writing `config/*.yaml` profiles (langchain.yaml, deerflow.yaml, llm.yaml, retrievers.yaml)
- Wiring RAG (`RetrieverFactory`), MCP servers, sandboxes, or BAML structured extraction
- Scaffolding with `cli init` or extending the CLI with `CliTopCommand`

**Do NOT use this skill when:**

- The project uses plain LangChain/LangGraph without the genai-tk layer — use the langchain skill
- You need a quick single agent and don't want the config-first machinery — smolagents or langchain skills fit better
- The user is asking about a different agent framework entirely
