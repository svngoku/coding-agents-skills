# CLI, Install & Project Init — references/cli-and-init.md

The `cli` entry point (`genai_tk.main.cli:main`) is a Typer app whose command groups are **declared in
`config/app_conf.yaml` under `cli.commands`** and resolved by dotted path. A project extends the CLI by adding one
class and one YAML line.

## Installation

genai-tk is **not published on PyPI**. Install it from git with `uv`; Python `>=3.12` is required.

```bash
uv add "genai-tk @ git+https://github.com/tclatos/genai-tk@main"
uv add "genai-tk[harnessing,browser] @ git+https://github.com/tclatos/genai-tk@main"   # with extras
uv add --editable /path/to/genai-tk                                                  # local clone
```

### Extras

| Extra | Brings in | Needed for |
|-------|-----------|-----------|
| `harnessing` | `deepagents>=0.7.13`, `agent-sandbox`, `opensandbox`, `opensandbox-server`, `deerflow-harness[tui]` (pinned commit) | `type: deep`, DeerFlow, Docker sandbox |
| `browser` | `playwright` | Direct (host) browser tools |
| `nlp` | spaCy + `en_core_web_md`, Presidio, `langchain-huggingface` | PII anonymization, spaCy BM25, cross-encoder rerank |
| `postgres` | `langchain-postgres`, psycopg | PgVector, `pg_hybrid` retriever |
| `streamlit` | `streamlit`, `streamlit-monaco` | `just webapp` |
| `baml` | `baml-py==0.223.0` | BAML structured extraction |
| `chromadb` | `chromadb`, `langchain-chroma` | Chroma vector store |
| `docling` | `docling`, `easyocr` | Docling converter in markdownize |
| `monitoring` | `langfuse`, OpenTelemetry SDK/OTLP exporter, openinference | Langfuse / OTLP tracing |
| `all` | Every extra above **except `monitoring`** | — |

Core already includes Prefect, `prefect-yaml`, LangSmith, NeMo Relay, `mcp>=2.0`, MarkItDown, and FastEmbed.
`dev` and `evals` are dependency groups for contributors, not extras.

Feature-gated code calls `genai_tk.config_mgmt.features.require_feature("<name>")`; install a missing feature later
with `uv sync --extra <name>`.

**Do not** use `[tool.uv] override-dependencies` on genai-tk in a downstream project: it strips forwarded extras.
Pin sibling checkouts with `[tool.uv.sources]` instead.

## `cli init` — Scaffold a Project

```bash
uv init my-app && cd my-app
uv add "genai-tk @ git+https://github.com/tclatos/genai-tk@main"
uv run cli init --name "My Agent App"
uv run cli init --extra harnessing --extra browser      # repeatable; runs uv sync --extra X
uv run cli init --with-graph --graph-path ../genai-graph # knowledge-graph + benchmark scaffolding
uv run cli init --force                                  # overwrite existing files
```

The only flags are `--name/-n`, `--extra/-e`, `--with-graph/-g`, `--graph-path`, and `--force/-f`. Upstream docs
mention `-t/--template` presets, `--minimal`, and `--with-deer-flow`, but **these are not implemented**; the
scaffolder always produces the agent-app layout.

`init` runs `uv sync` itself and removes the `main.py` stub that `uv init` creates.

### What it generates

| Path | Purpose |
|------|---------|
| `config/` | Full default config tree (`app_conf.yaml`, `providers/`, `agents.yaml`, `webapp.yaml`, …) |
| `justfile` | `run`, `sync`, `fmt`, `lint` (ruff + `cli skills validate --all`), `check`, `test`, `test-unit`, `skills`, `add-skill`, `webapp` |
| `<pkg>/commands/agent_commands.py` | `AgentCommands` → `cli agent list` / `cli agent chat` (singular `agent`) |
| `<pkg>/tools/example_tool.py` | Example LangChain tool referenced from `config/agents.yaml` |
| `<pkg>/webapp/pages/demos/hello_agent.py`, `<pkg>/main/streamlit.py` | Streamlit demo page and entry point |
| `skills/{runtime,development,governance,vendor,custom,community}/` | Bundled upstream skills + `custom/getting-started/SKILL.md` |
| `AGENTS.md`, `.github/copilot-instructions.md`, `docs/SKILLS.md`, `docs/EXTENDING.md` | Coding-agent guidance |
| `pyproject.toml` | Pinned to Python `>=3.12,<3.13`, hatchling build backend, `[tool.uv] package = true`, forwarded extras |

It also patches `app_conf.yaml` (`cli.commands` gets `<pkg>.commands.agent_commands.AgentCommands`) and
`webapp.yaml` (pages and navigation). `just run` maps to `uv run cli agent chat`.

## Command Tree

Run `uv run cli` (no arguments) or `uv run cli --help` for the live tree. As of `7fbd7a3`:

| Group | Subcommands |
|-------|-------------|
| `init` | flags above |
| `info` | `config`, `config-keys [KEY]`, `doctor [--fix]`, `models`, `llm-profile ID [--reload]`, `mcp-tools`, `commands`, `ls` |
| `core` | `llm`, `embedd`, `similarity`, `mcp-call`, `classifier` |
| `agents` | `run [PROFILE] [QUERY]`, `list`, `tui PROFILE` (DeerFlow only) |
| `sandbox` | `start`, `stop`, `status`, `list`, `pull` |
| `mcpserver` | `start`, `serve`, `list`, `generate` (`--name`, `--transport stdio\|sse\|streamable-http`, `--host`, `--port`) |
| `rag` | `add-files`, `query`, `embed`, `delete`, `info`, `list-retrievers`, `list-configs` |
| `workflow` | `run`, `list`, `show`, `validate`, `serve` |
| `prefect` | `start`, `stop`, `status`, `ui` |
| `baml` | `run`, `extract` |
| `skills` | `list`, `add`, `create`, `validate`, `info`, `search` |
| `trajectory` | `list`, `show`, `open`, `link`, `tui`, `tail`, `replay`, `export`, `diff`, `skills`, `stats`, `prune`, `view` |
| `monitoring` | `status`, `start`, `stop`, `test`, `open`, `tail`, `clear` |
| `tools` | `gpt-researcher`, `browser-agent` |
| `test` | `unit`, `fast_integration`, `full_integration`, `all`, `evals`, `select`, `notebooks` |

Frequently used invocations:

```bash
uv run cli core llm -i "tell me a joke" -m parrot_local@fake      # prompt via -i or stdin, not positional
echo "hello" | uv run cli core llm -m gpt-4o-mini@openai --stream
uv run cli agents run research "Explain async generators"          # profile is positional
uv run cli agents run research --chat --thread-id t1
uv run cli agents run research "..." --json                        # NDJSON event stream
uv run cli info doctor --fix
```

Global flag `--logging` takes no value; it forces TRACE-level logs.

The loader only imports modules whose dotted path contains the invoked group name, so a broken optional command
module does not break unrelated groups. An unresolvable entry in `cli.commands` only logs a warning.

## Extending the CLI — `CliTopCommand`

```python
# my_pkg/commands/report_commands.py
import typer

from genai_tk.cli.base import CliTopCommand


class ReportCommands(CliTopCommand):
    """Report generation commands."""

    def get_description(self) -> tuple[str, str]:
        return "report", "Generate reports from project data."

    def register_sub_commands(self, cli_app: typer.Typer) -> None:
        @cli_app.command("weekly")
        def weekly(
            week: str = typer.Argument(..., help="ISO week, e.g. 2026-W41"),
            llm: str | None = typer.Option(None, "--llm", "-m"),
        ) -> None:
            """Write the weekly report."""
            from my_pkg.reports import build_weekly  # import heavy deps inside the command

            build_weekly(week, llm=llm)
```

```yaml
# config/app_conf.yaml
cli:
  commands:
    - genai_tk.cli.commands_core.CoreCommands
    - genai_tk.cli.commands_info.InfoCommands
    # ...keep the existing entries...
    - my_pkg.commands.report_commands.ReportCommands
```

Contract:

- `CliTopCommand` is a Pydantic `BaseModel` + ABC; implement `get_description() -> (name, help)` and
  `register_sub_commands(app)`. `register()` is inherited.
- Entries must be **dotted paths** (`pkg.module.Class`). The `module:Class` form fails validation even though an
  upstream docstring shows it.
- Classes are instantiated with **no arguments**; entries cannot carry kwargs. A plain function entry is called
  as `fn(cli_app)`.
- Use a real importable package path (`my_pkg...`), not `src.my_pkg...`, unless `src` is itself a package.

Then run `uv run cli report weekly 2026-W41`.

## Skills Manager — `cli skills`

Upstream organizes `SKILL.md` files (skills.sh format) in tiers:

| Tier | Purpose |
|------|---------|
| `skills/runtime/` | Used by agents during runs (`codeact`, `browser-automation`, `query-writing`, …) |
| `skills/development/` | Recipes for extending the toolkit (`agent-profiles`, `add-tool`, `rag-systems`, …) |
| `skills/governance/` | Standards and audits (`repo-map`, `evaluation-testing`, `pii-anonymization`) |
| `skills/vendor/` | Imported third-party skills, not edited directly |
| `skills/custom/`, `skills/community/` | Project-specific and community-installed skills |

```bash
uv run cli skills list --source custom
uv run cli skills create invoice-review               # scaffold a skills.sh-format SKILL.md
uv run cli skills add --skillssh langchain-ai/langchain-skills
uv run cli skills add --git https://github.com/org/repo --path skills/foo
uv run cli skills validate --all
```

After `cli init`, a project's `skills/development/` folder contains upstream's own recipes (`agent-profiles`,
`add-tool`, `add-cli-command`, `rag-systems`, `workflow-engine`, …). They are useful deep dives, but some lag the
code (for example `development/configuration` still references `utils/config_mngr.py`), so check imports against
the installed package.

## House Conventions for Code in genai-tk Projects

From upstream `AGENTS.md`:

- Python 3.12+ typing (`str | None`, `list[str]`); avoid `Any` (use discriminated unions, `TypeAlias`)
- Absolute imports; ruff, line length 120; `PascalCase` classes, `snake_case` functions, `_private`
- Pydantic v2 for structured data: `model_config = {...}`, `model_post_init()`, `model_dump_json()`
- `loguru` for logging; Google-style docstrings, no types, no `Raises:`, no field docstrings
- Config via typed models: `global_config().section(key, Model)` / `.section_dict(key, Model)`
- Tests: pytest style, `fake_llm_id` / `fake_llm` / `fake_embeddings` fixtures, fake providers over mocks,
  markers `unit`, `integration`, `fake_models`, `real_models`; never call `get_pytest_config()` at module level and
  never name fixtures `pytest_*`
- Keep system prompts in YAML (`system_prompt:`), domain knowledge in `SKILL.md`
- Upstream rule for its own repo: update matching skills, docs, and `CHANGELOG.md` in the same commit

## Repo Recipes (upstream `justfile`)

For contributors working in a clone of genai-tk:

```bash
just install-dev        # uv sync with dev deps
just check              # fmt + lint + test
just test-unit
just new ../my-app      # uv init --bare + uv add + cli init + uv sync + smoke test
just lint-skills
```

Downstream projects can `import` the shared `tk.just` recipes (set `pkg_name` first). Docker images are built via
`deploy/docker.just` (`just docker-build`, `EXTRAS=...`).
