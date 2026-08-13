# CLI & Project Init — references/cli-and-init.md

The `cli` entry point is genai-tk's primary terminal interface. It's a Typer-based CLI with command groups loaded **dynamically from `app_conf.yaml`** — meaning your project can extend the CLI just by adding a class and one YAML line.

## Bootstrapping a project — `cli init`

`cli init` scaffolds a complete genai-tk project. By default it generates a working Python package with example commands, an LCEL chain, a Streamlit page, and Copilot Agent support files.

```bash
# Full scaffold (recommended)
uv run cli init --name "My AI Project"
uv run cli init                                  # name from cwd

# Config + Makefile only — no example code
uv run cli init --minimal

# Also clone the Deer-flow backend
uv run cli init --deer-flow
uv run cli init --deer-flow --path ./ext/deer-flow

# Overwrite existing files
uv run cli init --force

# After init
uv sync                                          # install generated package
```

### What the full scaffold creates

| Path | Purpose |
|------|---------|
| `config/` | Default YAML tree (`app_conf.yaml`, `baseline.yaml`, `providers/llm.yaml`, etc.) |
| `Makefile` | `make webapp`, `make test`, `make example-joke`, `make example-agent` targets |
| `<package>/` | Python package — name derived from project name (lowercased, spaces→underscores) |
| `<package>/main/streamlit.py` | Streamlit entry point delegating to `genai_tk.webapp` |
| `<package>/commands/example_commands.py` | `ExampleCommands` class — joke / chain / agent / deerflow |
| `<package>/chains/joke_chain.py` | Simple LCEL chain registered with `chain_registry` |
| `<package>/webapp/pages/demos/hello_agent.py` | ReAct agent page with chat UI + calculator tool |
| `AGENTS.md` | Architecture overview + coding conventions for Copilot Agent |
| `.github/copilot-instructions.md` | Always-active Copilot hints |
| `pyproject.toml` | uv project with genai-tk dependency + scoped package discovery |
| `README.md` | Project overview |

`cli init` also **patches** several config files automatically — `app_conf.yaml` gets your package's CLI command class added to the `cli.commands` list, `webapp.yaml` gets navigation entries, etc. Re-running `cli init --force` updates these in place.

### Example commands (after `uv sync`)

```bash
uv run cli example joke "software engineers"     # simple LLM call via LCEL
uv run cli example chain "Python devs"           # registered chain via chain_registry
uv run cli example agent "What is 2 + 2?"        # ReAct agent with calculator tool
uv run cli example deerflow "Explain RAG"        # requires --deer-flow init
```

Or via `make`:

```bash
make example-joke    make example-chain    make example-agent
```

---

## Built-in command groups

After `cli init`, you have access to all bundled command groups. Run `cli` with no arguments to see the full tree, or `cli <group> --help` to drill in.

### `init` — project initialisation
Documented above.

### `core` — direct LLM and chain access

```bash
cli core llm "Tell me a joke"                            # default model
cli core llm "Explain AI" --llm gpt_4o@openai            # specific model
cli core llm "Explain AI" --llm fast_model               # named tag
cli core llm "Be creative" --temperature 0.8
cli core llm "Write a poem" --stream                     # token stream
cli core llm "Solve this maths problem" --reasoning      # thinking mode
echo "summarise" | cli core llm --input -                # stdin
cli core llm "Hello" --raw                               # raw LC response object

cli core run my_chain "Input"                            # registered chain
```

Options for `cli core llm`:

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--input TEXT` | `-i` | stdin | Input or `-` for stdin |
| `--llm TEXT` | `-m` | default | Tag or `model_id@provider` |
| `--temperature` |  | 0.0 | Sampling temperature |
| `--stream` | `-s` | false | Stream tokens |
| `--reasoning` |  | false | Enable thinking mode (o3, claude thinking, etc.) |
| `--cache` |  | memory | `memory` / `sqlite` / `no_cache` |
| `--raw` |  | false | Print raw response object |

### `info` — system inspection

```bash
cli info config                                # show resolved config + API key status
cli info models                                # list all known LLM/embeddings models
cli info llm-profile gpt41mini@openai          # exact model lookup
cli info llm-profile gpt-4o                    # fuzzy match
```

`info config` shows: active configuration, default LLM/embeddings/vector store, all configured tags, and API key availability per provider — invaluable for "why isn't this working?" debugging.

### `agents` — agent runners

```bash
# LangChain (ReAct + Deep)
cli agents langchain --list
cli agents langchain --chat                              # default profile, interactive
cli agents langchain -p Research "Deep dive topic"
cli agents langchain --type react -p General "..."       # override agent type
cli agents langchain --sandbox docker -p Research "..."  # run in OpenSandbox

# Deer-flow
cli agents deerflow --list
cli agents deerflow --chat
cli agents deerflow -p "Research Assistant" --mode ultra "..."

# SmolAgents
cli agents smolagents "Plot sine fn and save to png"
cli agents smolagents --executor docker "Install pandas"
cli agents smolagents -t web_search "Latest AI news?"
cli agents smolagents --chat
```

See `references/agents.md` for profile fields, MCP servers, sandbox setup, and skills.

### `rag` — retrieval-augmented generation

```bash
cli rag list-retrievers
cli rag info <retriever_tag>
cli rag add-files ./docs/ -r persistent
cli rag add-files ./docs/ -r persistent \
    --include "**/*.md" --exclude "**/drafts/**"
cli rag query "What is hybrid search?" -r hybrid_reranked --k 10
cli rag query "pricing" --filter '{"source":"contracts"}'
cli rag embed persistent --text "snippet to index"
cli rag delete persistent --force
```

See `references/rag.md` for the full retrieval reference.

### `baml` — structured extraction

```bash
cli baml run ExtractResume -i "John Smith; SW engineer"
cat resume.txt | cli baml run ExtractResume

# Save result to JSON
cli baml run FakeResume -i "Jane Doe" \
    --out-dir ./output --out-file jane.json

# Batch extract a directory of markdown files
cli baml extract ./docs ./output \
    --recursive --function ExtractRainbow

# With filters and force re-run
cli baml extract ./reports ./output \
    --include 'report_*.md' --exclude '*_draft.md' \
    --recursive --force
```

See `references/baml-structured.md` for setup and the programmatic API.

### `tools` — utility commands

```bash
# PDF/DOCX/PPTX → Markdown via Prefect flow
cli tools markdownize ./input ./output --recursive
cli tools markdownize ./pdfs ./output --mistral-ocr        # higher-quality OCR
cli tools markdownize ./input ./output --force             # re-process

# PowerPoint → PDF
cli tools ppt2pdf ./slides ./pdfs --recursive

# GPT Researcher
cli tools gpt-researcher "Latest AI trends 2026"
```

### `sandbox` — OpenSandbox management

```bash
cli sandbox start                              # start the daemon
cli sandbox stop
cli sandbox status                             # daemon health + image cache
cli sandbox pull                               # pre-pull the Docker image
```

Run once per machine boot to cut subsequent agent startup from ~28s to ~5s.

### `mcpserver` — MCP server lifecycle

```bash
cli mcpserver list                             # configured servers
cli mcpserver start --name math_server                            # stdio
cli mcpserver start --name weather_server --transport sse         # SSE
cli mcpserver generate --name math_server                         # standalone script
```

### `test` — pytest wrappers

```bash
cli test unit
cli test fast_integration
cli test full_integration                      # requires real API keys
```

### Global flags

| Flag | Description |
|------|-------------|
| `--logging LEVEL` | `DEBUG` / `INFO` / `WARNING` / `ERROR`. Processed before Typer. |
| `--help` | Help on any command or group |

### OmegaConf interpolation in arguments

Path arguments accept config interpolation — wrap in single quotes to prevent shell expansion:

```bash
cli baml extract '${paths.data_root}/docs' '${paths.data_root}/output' \
    --recursive --function ExtractRainbow
```

Available variables: `${paths.project}`, `${paths.config}`, `${paths.data_root}`, `${paths.home}`.

---

## Extending the CLI — `CliTopCommand`

All command groups inherit from `CliTopCommand` and are loaded dynamically from `cli.commands` in `app_conf.yaml`. To add a new group:

### Step 1 — write the command class

```python
# src/myapp/commands_hello.py
from typing import Annotated
import typer
from genai_tk.cli.base import CliTopCommand


class HelloCommands(CliTopCommand):
    """Greeting commands."""

    description: str = "Friendly greeting commands."

    def get_description(self) -> tuple[str, str]:
        # (group_name, help_text) — group_name becomes the top-level CLI word
        return "hello", self.description

    def register_sub_commands(self, cli_app: typer.Typer) -> None:
        @cli_app.command()
        def greet(
            name: Annotated[str, typer.Argument(help="Name to greet")],
            shout: Annotated[bool, typer.Option("--shout", help="Use uppercase")] = False,
        ) -> None:
            """Greet someone by name."""
            msg = f"Hello, {name}!"
            print(msg.upper() if shout else msg)

        @cli_app.command()
        def farewell(name: Annotated[str, typer.Argument(help="Name")]) -> None:
            """Say goodbye."""
            print(f"Goodbye, {name}!")
```

### Step 2 — register in config

```yaml
# config/app_conf.yaml — or your project's overrides
cli:
  commands:
    # … existing entries …
    - src.myapp.commands_hello.HelloCommands
```

### Step 3 — use it

```bash
cli hello greet Alice
cli hello greet Alice --shout
cli hello farewell Bob
cli hello --help
```

### `CliTopCommand` contract

| Method | Required | Notes |
|--------|----------|-------|
| `get_description()` | Yes | Returns `(group_name, help_text)`. `group_name` is the CLI word users type. |
| `register_sub_commands(app)` | Yes | Define all `@app.command()` functions here. |
| `register(app)` | No (inherited) | Called automatically — creates the sub-Typer. **Do not override.** |

The class is a Pydantic `BaseModel` — add configurable fields with defaults and access them via `self` inside sub-commands.

### Legacy function-based pattern

Older modules use a plain function instead of a class:

```python
def register_commands(cli_app: typer.Typer) -> None:
    @cli_app.command()
    def echo(text: str) -> None:
        print(text)
```

```yaml
cli:
  commands:
    - genai_tk.main.cli.register_commands   # function reference, not a class
```

Still supported but **not recommended for new commands** — use `CliTopCommand` for testability and structure.

---

## How command discovery works

On `cli` startup, `load_and_register_commands()` (in `genai_tk/main/cli.py`):

1. Reads `cli.commands` from the merged YAML config
2. For each entry (`module.ClassName`), dynamically imports the symbol
3. If it's a `CliTopCommand` subclass → instantiates it, calls `.register(cli_app)`
4. If it's a callable → calls `fn(cli_app)` (legacy path)
5. With no arguments, displays a Rich command tree and exits

The command tree at any time: just run `cli`.

---

## Common patterns

### Adding a command for a project workflow

A command that runs a custom RAG pipeline:

```python
class IndexCommands(CliTopCommand):
    def get_description(self) -> tuple[str, str]:
        return "index", "Manage project document indexes."

    def register_sub_commands(self, cli_app: typer.Typer) -> None:
        @cli_app.command()
        def rebuild(
            retriever: str = typer.Argument(..., help="Retriever tag"),
            source: Path = typer.Option(Path("./docs"), help="Source dir"),
        ) -> None:
            from genai_tk.core.retriever_factory import RetrieverFactory
            from langchain_text_splitters import MarkdownTextSplitter

            r = RetrieverFactory.create(retriever)
            asyncio.run(r.adelete_store())
            # ... ingest logic ...
            print(f"✓ Rebuilt {retriever}")
```

### Adding a class field for shared config

```python
class MyCommands(CliTopCommand):
    description: str = "Project commands."
    default_dataset: str = "v2"               # Pydantic field with default

    def register_sub_commands(self, cli_app: typer.Typer) -> None:
        @cli_app.command()
        def run(dataset: str = None) -> None:
            ds = dataset or self.default_dataset
            print(f"Using dataset {ds}")
```

If you want the field driven by config, override the YAML registration:

```yaml
cli:
  commands:
    - module: src.myapp.commands.MyCommands
      default_dataset: v3
```

(Check `genai_tk/main/cli.py` `load_and_register_commands` for the exact instantiation hook — it accepts both bare class refs and dicts with init kwargs in some versions.)

### Pre-checking environment

Use `cli info config` early to surface missing API keys before invoking expensive commands:

```bash
cli info config | grep "OPENAI_API_KEY"
```

`info config` exits 0 even when keys are missing — it's informational. Use it as a sanity step in CI, not as a gate.
