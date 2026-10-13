# Workflows, Prefect & Document Processing — references/workflows.md

genai-tk orchestrates multi-step pipelines (convert → anonymize → extract → ingest) as **YAML workflows** executed
by Prefect. The DSL is compiled by the standalone `prefect-yaml` package (a core dependency); genai-tk adds the
CLI, the built-in flows, ingest routing, and the markdownize converter ladder.

## Workflow YAML

Workflows live under a top-level `workflows:` key in any YAML file picked up by `:merge:` (the bundled set is in
`config/examples/workflows.yaml`).

### How values reach the flow

This rule explains most workflow errors:

- For a `run:` workflow, **every key in `defaults:` and `params:` is passed to the callable as a keyword argument**
  (`key: ${values.key}`), plus `force` / `dry_run` when set. A top-level `with:` on a `run:` workflow is ignored.
- Before scheduling, the engine checks those names against the callable's signature and fails with
  `unexpected argument(s) for '<target>'` on any mismatch. A missing `required: true` param fails earlier with
  `missing required parameter(s)`.
- So `defaults`/`params` keys must be exactly the flow's parameter names.

### Single-step workflow

```yaml
workflows:
  md_ingest:
    description: "Chunk Markdown files into a retriever"
    run: genai_tk.workflow.prefect.flows.rag_flow.rag_file_ingestion_flow   # dotted callable or workflow name
    defaults:
      max_chunk_tokens: 512        # names match rag_file_ingestion_flow's signature
      chunker_name: auto
      batch_size: 10
      pathspecs: ["**/*.md"]
    params:
      base_dir: {required: true}
      retriever_name: {required: true}
    presets:
      docs:
        base_dir: "${paths.data_root}/markdown"
        retriever_name: default
```

### Multi-step pipeline

```yaml
workflows:
  anonymize_then_ingest:
    description: "Anonymize text files, then ingest them"
    pipeline:
      - id: anonymize
        run: anonymize                     # another workflow by name
        with:
          output_dir: "${values.anon_dir}"
      - id: ingest
        run: md_ingest
        after: [anonymize]                 # DAG dependency
        with:
          base_dir: "${values.anon_dir}"
    defaults:                              # sub-workflow keys resolve from THIS workflow's values
      batch_size: 10
      save_mapping: false
      faker_seed: 42
      max_chunk_tokens: 512
      chunker_name: auto
      pathspecs: ["**/*.md"]
    params:
      base_dir: {required: true}
      anon_dir: {required: true}
      retriever_name: {required: true}
```

When a step runs another workflow, it gets that workflow's `defaults`/`params` keys wired to `${values.*}` of the
**parent** pipeline, with the step's `with:` merged on top. The parent must therefore define every value its
sub-workflows need.

Step keys: `id`, `run`, `after` (alias `wait_for`), `with` (supports `${values.*}` and `${steps.<id>.result...}`),
`cache` (`none` | `manifest` | `hybrid`), `execution`, `foreach`, `inline`, `artifacts`.

Value precedence: CLI `--set` > preset > `defaults`.

Upstream `docs/design/07_workflow_engine_and_prefect.md` shows a `steps:` / `uses:` / `needs:` syntax; every
working config uses `run:` / `pipeline:` / `after:` as above.

### Built-in workflows

Status comes from a static check of `config/examples/workflows.yaml` against the flow signatures at upstream
commit `7fbd7a3`.

| Workflow | Does | Status |
|----------|------|--------|
| `markdownize` (presets `docs`, `fast`, `best`) | Convert PDFs, Office files, images, spreadsheets, zips to Markdown | OK |
| `merge_markdown` | Concatenate Markdown files | OK |
| `anonymize` | Presidio + Faker PII replacement (`nlp` extra) | OK |
| `json_to_table` | Flatten JSON results into CSV/Excel | OK |
| `baml_run` | One BAML call (see [baml-structured.md](baml-structured.md)) | Runs, but drops `output_dir`/`output_file` |
| `rag_ingest`, `anonymize_and_ingest` | Chunk and ingest into a retriever | Broken: declares `chunk_size`/`chunker`, flow wants `max_chunk_tokens`/`chunker_name` |
| `office2pdf` | Office → PDF via LibreOffice | Broken: declares `sources`/`pdf_output_dir`, flow wants `base_dir`/`output_dir` |
| `baml_extract`, `baml_to_table` | Batch BAML extraction | Broken: declares `sources`/`json_output_dir`, flow wants `base_dir`/`output_dir` |

For ingestion, `cli rag add-files` calls the flow directly and works (see [rag.md](rag.md)). For the others,
define a corrected workflow in your own `workflows:` config (as in `md_ingest` above) or call the flow from Python.

## CLI

```bash
uv run cli workflow list                                     # workflows and their presets
uv run cli workflow show md_ingest
uv run cli workflow validate
uv run cli workflow run md_ingest/docs --dry-run             # NAME/PRESET; show the plan
uv run cli workflow run md_ingest --set base_dir=./md --set retriever_name=docs
uv run cli workflow run markdownize/fast --set sources=./inbox --set md_output_dir=./data/md
uv run cli workflow run markdownize -p "**/*.pdf" -p "!drafts/**" --force
uv run cli workflow serve md_ingest/docs --cron "0 2 * * *"  # Prefect deployment

uv run cli prefect start        # local Prefect server (-f to force)
uv run cli prefect status
uv run cli prefect ui
uv run cli prefect stop
```

`workflow run` options: `--set KEY=VALUE` (repeatable), `-p/--pathspec` (→ `values.pathspecs`), `--to`
(→ `values.output_dir`), `--base-dir` (→ `values.base_dir`), `--force` (bool), `--dry-run`. There is no `--preset`
flag; select a preset with `NAME/PRESET`.

With `prefect.auto_start: true` (the default in `app_conf.yaml`), `workflow run` and `rag add-files` start the
Prefect server themselves.

## Python API

```python
from genai_tk.workflow import execute_workflow, list_workflow_names, resolve_workflow_invocation

print(list_workflow_names())
invocation = resolve_workflow_invocation(
    "md_ingest/docs",
    cli_overrides={"retriever_name": "docs"},
)
result = execute_workflow(invocation)
```

Prefect flows can also be called directly, for example
`genai_tk.workflow.prefect.flows.rag_flow.rag_file_ingestion_flow(...)` or
`genai_tk.workflow.markdownize.markdownize_flow(...)`. To make your own step usable from YAML, write a Prefect flow
(or plain callable) in your package and reference it by dotted path in `run:`.

## Ingest Routing

`config/ingest_routes.yaml` maps source items (paths or URLs) to workflows. **First match wins**, so put specific
patterns before catch-alls.

```yaml
ingest_routes:
  default:
    description: "Files → markdownize; git repos → git_repo; URLs → web_page"
    default: genai_tk.workflow.markdownize.markdownize_flow
    routes:
      - pathspec: "https://github.com/**"
        workflow: genai_tk.workflow.prefect.flows.git_repo_flow.git_repo_flow
      - pathspec: "**/*.git"
        workflow: genai_tk.workflow.prefect.flows.git_repo_flow.git_repo_flow
      - pathspec: "https://**"
        workflow: genai_tk.workflow.prefect.flows.web_page_flow.web_page_flow
      - pathspec: "**/*.pdf"
        workflow: markdownize
        with: {profile: best}
```

Routed workflows must accept `sources: list[str]` and `md_output_dir: str`. Unknown keys are rejected.

```python
from genai_tk.workflow import ingest_dispatch_flow

ingest_dispatch_flow(["./inbox", "https://github.com/org/repo"], md_output_dir="./data/md", routes="default")
```

`force_stage` reruns from a given stage: `unzip < pdf < md < parquet < graph < embed < all`.

## Markdownize (Document Conversion)

There is no dedicated markdownize CLI command; run it as a workflow:

```bash
uv run cli workflow run markdownize/best --set sources=./contracts --set md_output_dir=./data/md
uv run cli workflow run markdownize --set profile=docling --set sources=./scans --set md_output_dir=./data/md
```

Profiles (`config/markdownize.yaml`):

| Profile | Routing |
|---------|---------|
| `fast` | Spreadsheets → messy-XLS converter; everything else → MarkItDown |
| `medium` (default) | Spreadsheets → messy-XLS; Office → PDF → Mistral OCR; PDFs → Mistral OCR; rest → MarkItDown |
| `best` | Office and spreadsheets → PDF → Mistral OCR; PDFs → Mistral OCR; rest → MarkItDown |
| `docling`, `lighton`, `anydoc`, `llm` | Alternative converter ladders |

Existing `.md` files are copied, not converted. Mistral OCR needs `MISTRAL_API_KEY`; Docling needs the `docling`
extra.

Converters are declared under `markdownize_converters.<name>: {class, params}` and created with
`genai_tk.extra.markdownize.factory.ConverterFactory.create(name, **overrides)`. A custom converter implements
`async convert(path) -> str`.

## NLP and Anonymization

```yaml
nlp:
  default_language: en
  default_model: en_core_web_md
```

The `nlp` extra provides spaCy, Presidio detection, and reversible Faker replacement. The same engine powers the
`anonymize` workflow, the `AnonymizationMiddleware` for agents (see [agents.md](agents.md)), and spaCy
preprocessing for BM25.
