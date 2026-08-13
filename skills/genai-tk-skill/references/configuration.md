# Configuration — references/configuration.md

genai-tk's configuration system uses [OmegaConf](https://omegaconf.readthedocs.io/) with hierarchical YAML files, environment variable substitution, and **auto-discovery from parent directories**. This means notebooks, subdirectories, deployed containers, and CLI invocations all resolve config the same way without any setup.

## How it works

1. `cli init` (or `git clone`) places a `config/` directory next to your project root.
2. On first call, `global_config()` walks **up from the current working directory** until it finds `config/app_conf.yaml`.
3. `app_conf.yaml` declares a `:merge` list — each listed YAML file is loaded and **deep-merged in order** (later files win).
4. `${oc.env:VAR,default}` expressions are resolved from the environment (and from `.env` if present in the project root or any parent).
5. The merged config is wrapped in a singleton `OmegaConfig` object accessible via `global_config()`.

The result: code can be written without thinking about *where it's running from*. A function in `notebooks/explore.ipynb` and a CLI command in `cli` both see the same resolved config.

## File layout (after `cli init`)

```
config/
├── app_conf.yaml              # Entry point: default_config + :merge list
├── baseline.yaml              # Defaults: default LLM, embeddings, cache, vector store
├── overrides.yaml             # Local/per-environment overrides (git-ignored)
├── webapp.yaml                # Streamlit app navigation
├── mcp_servers.yaml           # MCP server definitions
├── providers/
│   ├── llm.yaml               # LLM model declarations
│   ├── embeddings.yaml        # Embeddings model declarations
│   └── providers.yaml         # API key env vars + provider class mappings
└── agents/
    ├── langchain.yaml         # LangChain agent profiles
    └── deerflow.yaml          # Deer-flow profiles
```

Agent and demo configs are **not merged globally** — they're loaded on demand by their respective loaders (`load_unified_config`, `load_deer_flow_profiles`, etc.).

## `app_conf.yaml` — the entry point

```yaml
default_config: ${oc.env:BLUEPRINT_CONFIG,baseline}

:merge:
  - ${paths.config}/baseline.yaml
  - ${paths.config}/providers/llm.yaml
  - ${paths.config}/providers/embeddings.yaml
  - ${paths.config}/overrides.yaml
  - ${paths.config}/mcp_servers.yaml
  - ${paths.config}/webapp.yaml

:env:
  LOGURU_LEVEL: INFO
  LANGCHAIN_TRACING_V2: "false"
  DEER_FLOW_PATH: ${oc.env:DEER_FLOW_PATH,${paths.project}/ext/deer-flow}

paths:
  home: ${oc.env:HOME}
  project: ${oc.env:PWD}        # auto-detected at runtime
  config: ${paths.project}/config
  data_root: ${paths.project}/data

# CLI command registration
cli:
  commands:
    - genai_tk.core.commands_core.CoreCommands
    - genai_tk.core.commands_info.InfoCommands
    - genai_tk.extra.commands_extra.ExtraCommands
    - genai_tk.agents.commands_agents.AgentCommands
    # add your project's CliTopCommand subclasses here
```

### Pseudo-keys

Three special keys at the top level have meaning to OmegaConfig:

- **`:merge`** — list of YAML files to deep-merge into the root. Order matters; later files win on conflict. Files can themselves contain a `:merge` list (recursive).
- **`:env`** — environment variables to set on import (does **not** override existing env vars).
- **`default_config`** — name of the active environment (see "Environments" below).

These are stripped from the config object before validation, so `cfg.get(":merge")` won't work — they're directives, not data.

### Interpolation — `${oc.env:VAR,default}`

OmegaConf's standard env-var resolver, evaluated lazily:

```yaml
api_key: ${oc.env:OPENAI_API_KEY}                          # raises if unset
api_key: ${oc.env:OPENAI_API_KEY,sk-default-do-not-use}    # with fallback
url: postgresql://${oc.env:DB_USER,postgres}:${oc.env:DB_PASS}@localhost/db
```

Path-style references resolve against other config keys:

```yaml
paths:
  project: ${oc.env:PWD}
  data_root: ${paths.project}/data       # → /home/user/proj/data
```

## `baseline.yaml` — defaults

```yaml
llm:
  default: gpt_oss120@openrouter        # used when get_llm() is called bare
  tags:
    cheap_model: claude-haiku@openrouter
    fast_model: claude-haiku@openrouter
    fake: parrot_local@fake             # no API key — useful for tests
  cache: sqlite                          # default | in_memory | sqlite | no_cache
  cache_path: data/llm_cache/langchain.db

embeddings:
  default: ada_002@openai
  tags:
    local: artic_22@ollama
    fake: embeddings_768@fake
```

The `cache_path` is relative to the project root. With `cache: sqlite`, identical `(model, prompt, params)` tuples hit the cache instead of the API — invaluable for development.

## `providers/llm.yaml` — declaring models

The `model_id@provider` format is the canonical way to reference models throughout the toolkit.

```yaml
llm:
  exceptions:                              # custom aliases not in models.dev DB

    - model_id: gpt41mini                  # logical name (Python-identifier-safe)
      providers:
        - openai: gpt-4.1-mini-2025-04-14  # direct — uses native API

    - model_id: haiku
      providers:
        - openrouter: anthropic/claude-haiku-4-5  # via gateway

    - model_id: mistral_7b
      providers:
        - custom:                          # full ChatOpenAI constructor params
            model: mistralai/Mistral-7B-Instruct-v0.3
            base_url: https://my-gateway.example.com/v1
            api_key: ${oc.env:MY_GATEWAY_KEY}

    - model_id: parrot_local               # built-in fake — echoes input
      providers:
        - fake: parrot
```

**`providers:` is a priority list.** The runtime selects the first provider whose API key is available. This is how you build graceful fallback — list a hosted provider first and a local one second; the local one only runs when the hosted key is missing.

You don't need a declaration for models in the [models.dev](https://models.dev) database — those are auto-resolved by fuzzy match. Declare an alias only when:

- The model isn't in models.dev (private gateways, custom endpoints)
- You want a short alias instead of the long native name
- You need provider routing or fallback

### Tags vs. exceptions

```yaml
llm:
  default: gpt41mini@openai

  tags:                   # user-friendly tag → existing model_id@provider
    fast_model: gpt41mini@openai
    smart_model: gpt_4o@openai

  exceptions:             # new model_id declarations
    - model_id: gpt41mini
      providers: [{ openai: gpt-4.1-mini-2025-04-14 }]
```

Tags resolve to declared models; declarations introduce new ones. `get_llm("fast_model")` and `cli core llm -m fast_model` both work.

## `providers/embeddings.yaml`

Same shape, for embedding models:

```yaml
embeddings:
  exceptions:
    - model_id: ada_002
      providers: [{ openai: text-embedding-ada-002 }]
    - model_id: artic_22
      providers: [{ ollama: snowflake-arctic-embed2:22m }]
    - model_id: embeddings_768
      providers: [{ fake: dim768 }]
```

## Environments

You can have multiple named environments alongside `baseline` in one `app_conf.yaml`:

```yaml
default_config: ${oc.env:BLUEPRINT_CONFIG,baseline}

baseline:
  llm:
    default: gpt41mini@openai
    cache: sqlite

production:
  llm:
    default: gpt_4o@openai
    cache: redis

testing:
  llm:
    default: parrot_local@fake             # no API calls in tests
    cache: no_cache
```

Switch active environment three ways:

| Method | Use when |
|--------|----------|
| `BLUEPRINT_CONFIG=production` env var | Per-shell / per-deploy |
| `global_config().select_config("production")` | Per-request / dynamic |
| Set `default_config: production` in `app_conf.yaml` | Persistent default |

## Environment variables (`.env`)

Place a `.env` file in the project root (or any parent — discovery walks upward). It's loaded automatically before any config values resolve.

```bash
# .env — never commit
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk_...
ANTHROPIC_API_KEY=sk-ant-...
TAVILY_API_KEY=tvly-...
DEER_FLOW_PATH=~/deer-flow
BLUEPRINT_CONFIG=production    # optional
```

`cli info config` shows which keys are set and which are missing.

## Python API — `OmegaConfig`

```python
from genai_tk.utils.config_mngr import global_config

cfg = global_config()                              # singleton, auto-discovered
cfg = global_config(reload=True)                   # force reload from disk
```

### Reading values

```python
cfg.get("llm.default")                             # raises if missing
cfg.get("retrievers.my_vec.top_k", 4)              # with default
cfg.get_str("llm.default")                         # type-checked
cfg.get_bool("features.enable_x", False)
cfg.get_list("cli.commands", default=[])
cfg.get_dict("langchain_agents.profiles")          # returns plain dict
cfg.get_dir_path("paths.data_root", create_if_not_exists=True)  # → Path
cfg.get_file_path("llm.cache_path", check_if_exists=False)
cfg.get_dsn("postgres.default", driver="asyncpg")  # build DSN with driver
```

### Mutating at runtime

```python
cfg.set("llm.default", "haiku@openrouter")         # runtime override
cfg.select_config("production")                     # switch environment
cfg.merge_with("/path/to/extra.yaml")               # add another YAML on top
```

### Available helpers

- **`paths_config()`** — typed `PathsConfig` model with `home`, `project`, `config`, `data_root`
- **`select_active_config(name)`** — module-level convenience for `global_config().select_config(name)`
- **`get_raw_config()`** — raw `DictConfig` for OmegaConf-native access (rare)

## Override patterns

### Per-developer overrides

`overrides.yaml` is the **last** file in the default `:merge` list and is git-ignored. Per-developer or per-machine tweaks go there:

```yaml
# config/overrides.yaml — not committed
llm:
  default: claude-sonnet-4@anthropic       # I prefer Claude locally
  cache: in_memory                          # don't hit disk during dev
```

### Per-environment overrides

For deployed environments, use named environment blocks in `app_conf.yaml` and switch via `BLUEPRINT_CONFIG=production`. Don't ship `overrides.yaml` to production.

### Programmatic overrides for tests

```python
from genai_tk.utils.config_mngr import global_config

def setup_module(module):
    cfg = global_config()
    cfg.select_config("testing")             # named env with parrot_local@fake
    cfg.set("llm.cache", "no_cache")
```

## Common pitfalls

**`ConfigFileNotFoundError` on import.** The cwd doesn't have a `config/app_conf.yaml` upward in its path. Either run from the project root, or run `cli init` to scaffold it.

**`${oc.env:VAR}` expands to literal `???` or empty.** The env var isn't set and there's no fallback. Add `${oc.env:VAR,default}` or set the variable.

**Changes to YAML don't take effect.** The config is a singleton — call `global_config(reload=True)` after modifying YAML during a Python session, or restart the process.

**Setting a value via `cfg.set()` doesn't persist.** Runtime overrides live in memory only. To persist, edit the YAML.

**Path keys silently expand `${paths.project}` to wherever you started Python.** This is by design — auto-discovery uses cwd. If you need a fixed root, set `PWD` explicitly before import or override `paths.project` in `app_conf.yaml`.

**Merge order matters.** If two files in `:merge` define the same key, the later one wins. Lists are replaced, not concatenated, by default — use `_deep_merge_with_list_keys` patterns if you need to extend a list across files.

## See also

- `references/cli-and-init.md` — `cli init` scaffolding details, command registration
- `references/agents.md` — agent profile fields
- `references/rag.md` — `retrievers:` and `embeddings_store:` blocks
- repo `docs/llm-selection.md` — full model identifier reference and `cli info` commands
