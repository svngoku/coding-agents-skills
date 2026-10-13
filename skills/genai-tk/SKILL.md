---
name: genai-tk
description: >
  Build GenAI and agentic apps with genai-tk (tclatos/genai-tk), a YAML-driven IoC toolkit over LangChain,
  DeepAgents, DeerFlow, and Prefect. Use this skill whenever the user mentions genai-tk, genai_tk, `cli init`,
  `cli agents run`, `LangchainAgent`, `create_harness`, `get_llm`, `global_config()`, `model_id@provider`,
  `agents:`/`agent_defaults:` profiles, `RetrieverFactory`, `cli workflow`, or BAML extraction. Also trigger
  for app_conf.yaml, GENAITK_PROFILE, CodeAct, OpenSandbox, decision models, or "the toolkit".
---

# genai-tk — GenAI & Agentic Toolkit

[genai-tk](https://github.com/tclatos/genai-tk) is a Python 3.12+ toolkit where **YAML config drives everything**:
LLM providers, agent profiles, retrievers, workflows, and CLI commands are declared in `config/*.yaml` and
resolved at runtime through an Inversion-of-Control registry (`ImportResolver`). You extend it by adding YAML
entries that point at dotted Python paths, not by editing the library.

```text
config/app_conf.yaml ─► global_config() ─► ImportResolver ─► factories / harness ─► LangChain objects
```

> Verified against upstream commit `7fbd7a3` (2026-10-12). The toolkit moves fast and **several upstream docs lag
> the code** (see "Upstream doc drift" below). When docs and code disagree, trust the code.

## Quick Reference

| Topic | Reference |
|-------|-----------|
| Install, extras, `cli init`, full CLI command tree, `CliTopCommand`, `cli skills`, house conventions | [cli-and-init.md](references/cli-and-init.md) |
| `app_conf.yaml`, `:merge`, `:profiles`, `GENAITK_PROFILE`, typed config access, IoC | [configuration.md](references/configuration.md) |
| `get_llm` / `get_embeddings`, `model_id@provider`, providers.yaml, reasoning effort, routing, decision models | [models.md](references/models.md) |
| Agent profiles, harness layer, tools, middlewares, skills, sandbox, MCP, DeerFlow, CodeAct, trajectories | [agents.md](references/agents.md) |
| `RetrieverFactory`, `ManagedRetriever`, embeddings stores, chunkers, `cli rag`, RAG as a tool | [rag.md](references/rag.md) |
| YAML workflows, Prefect, ingest routing, markdownize converters | [workflows.md](references/workflows.md) |
| BAML structured extraction: `BamlStructuredProcessor`, `baml_invoke`, `cli baml` | [baml-structured.md](references/baml-structured.md) |

## Core Workflow

### 1. Install and scaffold

genai-tk is **not on PyPI**. Install from git with `uv`, then scaffold:

```bash
uv init my-app && cd my-app
uv add "genai-tk @ git+https://github.com/tclatos/genai-tk@main"
uv run cli init --name "My Agent App"                  # scaffolds config/, skills/, justfile; runs uv sync
uv run cli init --extra harnessing --extra browser     # also install DeepAgents/DeerFlow + Playwright
just run                                               # uv run cli agent chat
```

Extras: `harnessing` (deepagents, DeerFlow, OpenSandbox), `browser`, `nlp`, `postgres`, `streamlit`, `baml`,
`chromadb`, `docling`, `monitoring`, and `all` (everything except `monitoring`). Put API keys in `.env`
(auto-loaded, plus `~/.env` for the CLI). Run `uv run cli info doctor` as a pre-flight check.

### 2. Call models

```python
from genai_tk.core.factories import get_llm, get_embeddings

llm = get_llm()                                  # llm.models.default from the active profile
llm = get_llm("fast_model")                      # a tag under llm.models
llm = get_llm("gpt-4o-mini@openai")              # model_id@provider (resolved via models.dev)
llm = get_llm("parrot_local@fake")               # offline fake model, no API key
llm = get_llm("o3 (high)@openai")                # inline reasoning effort
llm = get_llm("claude-haiku@openrouter:speed")   # gateway routing strategy
llm = get_llm("gpt-4o-mini@openai", streaming=True, json_mode=True, cache="sqlite")

embeddings = get_embeddings()                    # embeddings.models.default
```

Both return plain LangChain `BaseChatModel` / `Embeddings` objects. CLI: `cli core llm -i "hello" -m parrot_local@fake`,
`cli info models`, `cli info llm-profile <id>`. Details in [models.md](references/models.md).

### 3. Define an agent in YAML, run it anywhere

Profiles live under a top-level `agents:` **dict keyed by slug**, in `config/agents.yaml` or `config/agents/*.yaml`.
`agent_defaults:` supplies fallbacks for LangChain profiles.

```yaml
# config/agents.yaml
agent_defaults:
  type: react
  default_profile: research
agents:
  research:
    harness: langchain            # langchain (default) | deerflow
    name: "Research"
    type: deep                    # react | deep | custom
    llm: gpt-4o-mini@openai
    system_prompt: "You are an expert researcher. Cite sources."
    tools:
      - genai_tk.agents.tools.langchain.search_tools_factory.create_search_function:
          verbose: false
    mcp_servers: [filesystem]     # names from mcpServers: in mcp_servers.yaml
    skill_directories: ["${paths.project}/skills"]
    enable_planning: true
    enable_file_system: true
    checkpointer: {type: memory}  # none | memory | class
```

```python
from genai_tk.agents.langchain import LangchainAgent

agent = LangchainAgent("research", llm="claude-haiku@openrouter")   # slug or display name, + overrides
print(agent.run("Summarise this week's AI news"))
```

```bash
uv run cli agents list
uv run cli agents run research "Summarise this week's AI news"
uv run cli agents run research --chat
```

For a harness-agnostic API (LangChain and DeerFlow behind one event stream), use
`genai_tk.agents.harness.create_harness("research")`. See [agents.md](references/agents.md).

### 4. Add RAG, workflows, or structured extraction

```yaml
embeddings_store:
  chroma:
    backend: genai_tk.core.vector_backends.ChromaBackend
    embeddings: default
    config: {storage: "${paths.data_root}/vector_store"}
retrievers:
  docs: {type: genai_tk.core.retrievers.VectorRetriever, embeddings_store: chroma, top_k: 4}
```

```bash
uv run cli rag add-files ./docs -r docs -p "**/*.md"
uv run cli rag query "What does the doc say about X?" -r docs
uv run cli workflow run markdownize/fast --set sources=./inbox --set md_output_dir=./data/md
uv run cli baml run ExtractResume -i "John Smith; 10 years Python"         # a function from your baml_src/
```

## Configuration in One Minute

`config/app_conf.yaml` is the root. It is loaded from `config/app_conf.yaml` **relative to the current working
directory** (it does not walk up parent directories), so run commands from the project root.

```yaml
profile: ${oc.env:GENAITK_PROFILE,local}
paths:
  project: ${oc.env:PWD}
  config: ${paths.project}/config
  data_root: ${paths.project}/data
:merge: ["**/*.yaml", "!demos/**", "!examples/agents/**"]   # gitignore-style globs under config/
:profiles:
  local:
    llm: {models: {default: gpt-4o-mini@openai, fake: parrot_local@fake}, cache: sqlite}
    embeddings: {models: {default: embeddings_768@fake}}
:env: {LOGURU_LEVEL: INFO}
```

```python
from genai_tk.config_mgmt.config_mngr import global_config, switch_profile

cfg = global_config()
cfg.get_str("llm.models.default")
cfg.section("my_feature", MyFeatureConfig)      # typed Pydantic access; empty model if absent
switch_profile("pytest")                         # sets GENAITK_PROFILE and reloads
```

Full semantics in [configuration.md](references/configuration.md).

## Coding Conventions (from upstream AGENTS.md)

- Python 3.12+: `str | None`, `list[str]`; **avoid `Any`** (use discriminated unions or `TypeAlias`)
- Absolute imports only; ruff with line length 120
- Pydantic v2 for all structured data: `model_config = {...}`, `model_post_init()`, `model_dump_json()`; no
  dataclasses, unnamed tuples, or untyped dicts
- `loguru` for logging; Google-style docstrings without types or `Raises:`
- Read config through typed models: `global_config().section(...)` / `.section_dict(...)`
- Tests: use the `fake_llm_id` / `fake_llm` fixtures and fake providers instead of mocks; never hard-code model IDs
- Domain knowledge goes in `SKILL.md` files loaded on demand, not in system prompts

## Upstream Doc Drift

These upstream docs or bundled skills contradict the code at `7fbd7a3`. Use the right-hand column.

| Docs say | Code does |
|----------|-----------|
| Config search walks up parent directories | Only `config/app_conf.yaml` under the CWD |
| `cli init -t agent-app\|rag-app\|workflow-app\|minimal`, `--with-deer-flow` | Flags are `--name`, `--extra`, `--with-graph`, `--graph-path`, `--force` |
| `cli mcp serve\|list\|generate` | The group is `cli mcpserver start\|serve\|list\|generate` |
| `cli core decision` | `cli core classifier` |
| `get_llm(..., reasoning={...})` | Raises; use `"model (high)@provider"` or `reasoning_effort="high"` |
| Workflow DSL with `steps:` / `uses:` / `needs:` | `run:` or `pipeline:` with `id`, `run`, `after`, `with` |
| `cli rag add-files <retriever> <dir>`, `--include` | `cli rag add-files <dir> -r <retriever> -p <pathspec>` |
| Short retriever/backend aliases (`type: vector`, `backend: Chroma`) | Fully qualified class paths are required |
| `langchain_agents:` profile lists | `agents:` dict + `agent_defaults:` |
| `baml_invoke(inputs=...)`, `cli baml extract --recursive --include` | `baml_invoke(name, params, ...)`, `-p <pathspec>` (default `**/*.md`) |
| Bundled `baml_extract`, `rag_ingest`, `office2pdf` workflows (and `cli baml extract`) just work | Their YAML param names don't match the flow signatures; see [workflows.md](references/workflows.md) |

## Anti-Patterns to Avoid

- **Using pre-2026 import paths** — `genai_tk.core.llm_factory`, `genai_tk.utils.config_mngr`, and
  `genai_tk.core.retriever_factory` are gone with no shims; import from `genai_tk.core.factories` and
  `genai_tk.config_mgmt.config_mngr`
- **Writing `langchain_agents:` / `profiles:` lists or `spec: web_search` tools** — the old profile schema is
  rejected; use the `agents:` dict and qualified factory paths for tools
- **Putting `skills:` on a profile** — it is silently ignored; use `skill_directories:` (deep agents only)
- **Bare model names in YAML** — `gpt-4o-mini` triggers fuzzy lookup; always write `model_id@provider`
- **Short aliases for retrievers and vector backends** — they fail to resolve; use
  `genai_tk.core.retrievers.VectorRetriever`, `genai_tk.core.vector_backends.ChromaBackend`, etc.
- **Hard-coding agents or LLMs in Python** — add a YAML entry; `LangchainAgent("slug")` only loads and overrides
- **`type: deep` or DeerFlow without the `harnessing` extra** — `deepagents` and `deerflow-harness` live there
- **Expecting `--sandbox` on `cli agents run` to isolate a LangChain profile** — it only applies to DeerFlow; set
  `backend: {type: docker}` (or `LangchainAgent(..., sandbox="docker")`) for LangChain profiles
- **Running from a subdirectory** — config discovery is CWD-relative; `ConfigFileNotFoundError` usually means
  you are not at the project root
- **Copying examples from upstream `docs/` without checking the code** — see "Upstream Doc Drift"

## When to Use / Not Use

**Use this skill when:**

- Working inside a genai-tk project (a `config/app_conf.yaml` and `genai_tk` imports are present)
- Writing or fixing agent profiles, provider/LLM config, retrievers, workflows, or `cli.commands` entries
- Scaffolding with `cli init`, extending the CLI, or wiring MCP servers, sandboxes, BAML, or trajectories
- Migrating code written for older genai-tk versions (old imports, `langchain_agents:`, SmolAgents)

**Do NOT use this skill when:**

- The project uses plain LangChain/LangGraph without genai-tk — use the langchain skill
- You need a quick standalone agent without config-first machinery — the smolagents or langchain skills fit better
- The question is about DeerFlow or DeepAgents used directly, outside genai-tk's harness
