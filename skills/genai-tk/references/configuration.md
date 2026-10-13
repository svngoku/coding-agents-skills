# Configuration — references/configuration.md

genai-tk uses [OmegaConf](https://omegaconf.readthedocs.io/) with a few custom directives. Everything the toolkit
builds (models, agents, retrievers, workflows, CLI groups, webapp pages) is read from this merged config.

## Imports

```python
from genai_tk.config_mgmt.config_mngr import (
    global_config,         # singleton OmegaConfig
    global_config_reload,
    switch_profile,        # set GENAITK_PROFILE and reload
    use_active_context,    # apply a named overlay without reloading
    get_raw_config,        # underlying DictConfig
    paths_config,
    ImportResolver,        # dotted-path resolver behind the IoC registry
)
```

The old `genai_tk.utils.config_mngr` module no longer exists and there is no compatibility shim.

## Discovery and Loading

- The root file is `config/app_conf.yaml`, looked up **relative to the current working directory**. The loader does
  not walk up parent directories (upstream docs claim it does). If neither path exists,
  `ConfigFileNotFoundError` is raised; run from the project root or `cli init` first.
- `load_dotenv()` runs on import, so `.env` in the CWD is loaded. The `cli` entry point also loads `~/.env`.
- `${oc.env:VAR,default}` reads environment variables; standard OmegaConf interpolation (`${paths.project}`) works
  across merged files.

## `app_conf.yaml` Anatomy

```yaml
profile: ${oc.env:GENAITK_PROFILE,local}      # active profile name

paths:
  home: ${oc.env:HOME}
  project: ${oc.env:PWD}
  config: ${paths.project}/config
  data_root: ${paths.project}/data

:merge:                                        # gitignore-style patterns, relative to config/
  - "**/*.yaml"
  - "!demos/**"
  - "!mcp/**"
  - "!examples/agents/**"

:profiles:                                     # inline overlays; the active one is applied last
  local:
    llm:
      models:
        default: gpt-4o-mini@openai
        fast_model: claude-haiku@openrouter
        fake: parrot_local@fake
      cache: sqlite
      cache_path: data/llm_cache/langchain.db
    embeddings:
      models: {default: embeddings_768@fake}
  production:
    :merge: ["prod/**/*.yaml"]                 # an overlay may pull in extra files
    llm: {models: {default: claude-haiku@openrouter}}
  pytest:
    llm: {models: {default: parrot_local@fake}}

:env:                                          # exported to os.environ at load time
  LOGURU_LEVEL: INFO
  LANGSMITH_TRACING: "false"

cli:
  commands:
    - genai_tk.cli.commands_core.CoreCommands
    - genai_tk.cli.commands_info.InfoCommands
    # ...
```

### Directive semantics

| Directive | Behavior |
|-----------|----------|
| `profile:` | Active profile. Resolution: `GENAITK_PROFILE` env var → `profile:` key → legacy `default_config:` → `local` |
| `:merge:` | Patterns are expanded with `rglob` under `config/`, sorted, then deep-merged in order (later wins). `!pattern` excludes |
| `:profiles:` | Map of overlays. The active profile's block is merged **after** all `:merge:` files and may contain its own `:merge:` |
| `:env:` | Sets environment variables. It **overwrites** variables that are already set (upstream docs say otherwise) |

What changed from older genai-tk versions:

| Old | Now |
|-----|-----|
| `baseline.yaml`, `overrides.yaml` | Inline `:profiles:` blocks |
| `default_config: ${oc.env:BLUEPRINT_CONFIG,baseline}` | `profile: ${oc.env:GENAITK_PROFILE,local}` |
| Explicit file list in `:merge:` | Glob patterns with `!` excludes |
| `cfg.select_config("prod")` | `switch_profile("prod")` (reload) or `use_active_context("prod")` (overlay) |
| `llm.default`, `llm.tags.*` | `llm.models.default`, `llm.models.<tag>` |
| `genai_tk.core.commands_core.CoreCommands` | `genai_tk.cli.commands_core.CoreCommands` |

## Reading Config

```python
from pydantic import BaseModel

from genai_tk.config_mgmt.config_mngr import global_config


class ReportConfig(BaseModel):
    output_dir: str = "data/reports"
    max_items: int = 50


cfg = global_config()

cfg.get("llm.models.default")                         # Any
cfg.get_str("llm.models.default")
cfg.get_bool("prefect.auto_start")
cfg.get_list("cli.commands")
cfg.get_dict("retrievers")
cfg.get_dir_path("paths.data_root", create_if_not_exists=True)   # Path
cfg.get_dsn("postgres.default.url", driver="asyncpg")            # SQLAlchemy URL

report = cfg.section("report", ReportConfig)                     # typed; empty/default model if missing
stores = cfg.section_dict("embeddings_store", StoreConfig)       # dict[str, StoreConfig]; injects name
```

Upstream's convention is to read **every top-level key through a typed Pydantic model** with `section()` /
`section_dict()`. Both return an empty result instead of raising when the key is absent. `section_dict` accepts an
`Annotated[..., Field(discriminator=...)]` union.

### Mutating at runtime

```python
cfg.set("llm.models.default", "parrot_local@fake")    # in-memory only
cfg.merge_yaml("""
retrievers:
  scratch: {type: genai_tk.core.retrievers.VectorRetriever, embeddings_store: memory}
""")                                                  # raw YAML string or a file path
cfg.use_context("production")                         # apply a top-level overlay key
```

### Switching profiles

```bash
GENAITK_PROFILE=production uv run cli agents run research "..."
```

```python
from genai_tk.config_mgmt.config_mngr import switch_profile

switch_profile("pytest")    # sets GENAITK_PROFILE and reloads the singleton
```

In tests, upstream switches to the `pytest` profile in a session-scoped autouse fixture; read test settings with
`genai_tk.config_mgmt.test_config.get_pytest_config()` **inside** fixtures, never at module level.

### Introspection

```bash
uv run cli info config                 # resolved config + API-key status per provider
uv run cli info config-keys            # which file each top-level key came from
uv run cli info config-keys llm
uv run cli info doctor --fix           # environment pre-flight checks
```

`cfg.config_keys_info()` returns the same provenance map in Python.

## Inversion of Control — `ImportResolver`

Config values that name code are **fully qualified dotted paths** resolved with
`ImportResolver.import_from_qualified("pkg.module.Object")`. The resolver tries progressively shorter module
prefixes, so `pkg.module.Class.method`-style paths work. A bare name without dots triggers a slower AST scan of the
project and is fragile; always write the full path.

The same mechanism resolves:

| Config location | What it names |
|-----------------|---------------|
| `providers.<name>.chat_class` / `embeddings_class` / `decision_class` | Provider implementation classes |
| `cli.commands[]` | `CliTopCommand` subclasses or `fn(cli_app)` callables |
| `agents.<key>.tools[]`, `middlewares[].class` | Tool factories and middleware classes |
| `retrievers.<tag>.type`, `embeddings_store.<tag>.backend` | Retriever builders and vector backends |
| `workflows.<name>.run` | Prefect flows or other workflows |
| `markdownize_converters.<name>.class` | Document converters |
| `ui.navigation` | Streamlit pages, including `genai_tk://demos/agent.py` library resources |

To extend the toolkit, write a class in your package and reference it from YAML; do not fork the library.

## Typical File Layout After `cli init`

```text
config/
├── app_conf.yaml          # root: profile, paths, :merge, :profiles, :env, cli.commands
├── agents.yaml            # agents: + agent_defaults:
├── providers/
│   ├── providers.yaml     # provider registry (chat_class, api_key_env_var, api_base, ...)
│   ├── llm.yaml           # llm.exceptions: models not covered by models.dev
│   ├── embeddings.yaml    # embeddings.registry
│   └── decision_models.yaml
├── mcp_servers.yaml       # mcpServers: / mcpProjectServers:
├── webapp.yaml
└── examples/              # bundled examples (agents/, rag.yaml, workflows.yaml, ...)
```

Because `:merge:` globs `**/*.yaml`, any new YAML file you drop under `config/` is picked up automatically unless an
exclude pattern matches it. Keep keys unique across files: later files silently override earlier ones.

## Network Proxies

Corporate proxies are honored through the standard `HTTP(S)_PROXY` variables. Hosts that must bypass the proxy
(local vLLM, Ollama, sandbox) go under `net.proxy_bypass_hosts`; they are added to `NO_PROXY` at load time.
