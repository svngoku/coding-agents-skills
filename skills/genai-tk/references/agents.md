# Agents — references/agents.md

genai-tk runs agents through **two harnesses** that share YAML profiles, the LLM factory, MCP servers, sandboxes,
and skills:

| Harness | Profile `type` | Engine | Needs |
|---------|----------------|--------|-------|
| `langchain` (default) | `react` | `langchain.agents.create_agent` | core |
| `langchain` | `deep` | `deepagents.create_deep_agent` (planning, file system, subagents, skills) | `harnessing` extra |
| `langchain` | `custom` | `genai_tk.extra.langgraphs.custom_react_agent.create_custom_react_agent` | core |
| `deerflow` | — (`mode: flash\|thinking\|pro\|ultra`) | ByteDance DeerFlow, embedded in-process | `harnessing` extra |

**SmolAgents is no longer an agent framework in genai-tk** (`cli agents smolagents`, `SmolAgent`, and
`config/smolagents.yaml` are gone). Its code-first style lives on as **CodeAct**, a `deep` profile whose only tool
is a Python executor (see below).

## Profiles

### Where they live

Profiles are loaded from the **first** of these that exists:

1. `config/agents.yaml`
2. `config/agents/*.yaml` (not recursive)
3. `config/examples/agents/` (bundled examples)
4. `config/examples/agents.yaml`

Because the lookup stops at the first match, creating `config/agents.yaml` hides the bundled examples. Copy any
example profiles you still want.

### Schema

```yaml
agent_defaults:                    # fallbacks for harness: langchain profiles only
  type: react
  llm: null                        # null → llm.models.default
  middlewares:
    - class: genai_tk.agents.langchain.middleware.rich_middleware.RichToolCallMiddleware
  checkpointer: {type: none}
  backend: {type: none}
  enable_planning: true
  enable_file_system: true
  skills:
    directories: ["${paths.project}/skills"]
  default_profile: research        # used by `cli agents run` without a profile

agents:                            # dict keyed by slug; the slug is the CLI/Python key
  research:
    harness: langchain
    name: "Research"               # display name (defaults to the slug)
    type: deep
    description: "Web researcher that writes cited reports"
    llm: gpt-4o-mini@openai
    system_prompt: |
      You are an expert researcher. Plan with write_todos, cite every source.
    tools:
      - genai_tk.agents.tools.langchain.search_tools_factory.create_search_function:
          verbose: false
    mcp_servers: [filesystem]
    skill_directories: ["${paths.project}/skills/runtime", "${paths.project}/skills/custom"]
    middlewares:
      - class: genai_tk.agents.langchain.middleware.rich_middleware.RichToolCallMiddleware
      - class: genai_tk.agents.langchain.middleware.observation_truncation_middleware.ObservationTruncationMiddleware
        max_chars: 12000
    backend: {type: aio_sandbox, startup_timeout: 90.0}
    checkpointer: {type: memory}
    recursion_limit: 100
    examples:
      - "Compare the latest open-weight coding models"
```

LangChain profile fields (`AgentProfileConfig`):

| Field | Notes |
|-------|-------|
| `harness` | `langchain` (default) or `deerflow` |
| `name`, `description` | Display metadata; lookups match slug or name, case-insensitively |
| `type` | `react` (default) \| `deep` \| `custom` |
| `llm` | `model_id@provider` or a tag; `null` uses the global default |
| `system_prompt`, `pre_prompt` | Keep prompts here, not in Python |
| `tools` | Qualified factory/class/function paths (see "Tools") |
| `mcp_servers` | Names from `mcpServers:` in `mcp_servers.yaml` |
| `middlewares` | `- class: dotted.Path` with constructor kwargs as sibling keys |
| `skill_directories` | Directories of `SKILL.md` files (deep agents only) |
| `enable_planning`, `enable_file_system` | Deep-agent built-ins (both default `true`) |
| `backend` | `none` \| `filesystem` (`root_dir`) \| `aio_sandbox` / `docker` \| `local` \| `class` (`class`, `kwargs`) |
| `checkpointer` | `none` \| `memory` \| `class` (e.g. `class: langgraph.checkpoint.sqlite:SqliteSaver`) |
| `subagents` | Deep-agent subagents, each with `name`, `description`, `system_prompt`, `tools` |
| `excluded_tools` | Drop built-in tools by name |
| `fs_permissions` | Deep-agent filesystem rules: `operations: [read\|write]`, `paths: [...]`, `mode: deny\|allow\|interrupt` |
| `recursion_limit` | Default 100 |
| `features`, `examples` | Shown in `cli agents list` and the webapp |

Gotchas:

- **Unknown keys are silently ignored.** A typo such as `skills:` on a profile (instead of `skill_directories:`)
  does nothing. Only `agent_defaults.skills.directories` is read.
- **Defaults are replaced, not merged.** A profile's own `middlewares:` list replaces the default list entirely.
- The old `langchain_agents: {defaults, default_profile, profiles: [...]}` schema and `deerflow_agents:` lists are
  **no longer supported**; one bundled file (`config/agents/langchain/browser.yaml`) still uses it and is never loaded.

## Running Agents

### CLI

```bash
uv run cli agents list                                   # all profiles across both harnesses
uv run cli agents run research "Summarise this week's AI news"
uv run cli agents run research --chat                    # REPL (forces a memory checkpointer)
uv run cli agents run research --chat --thread-id t42    # resume a thread
uv run cli agents run research -m claude-haiku@openrouter "..."   # LLM override
uv run cli agents run research --mcp playwright "..."    # extra MCP server (repeatable)
uv run cli agents run research "..." --json              # NDJSON harness events
echo "question" | uv run cli agents run research
uv run cli agents tui "Research Assistant"               # DeerFlow TUI (--resume, --continue)
```

`--mode` and `--sandbox/-b` only apply to DeerFlow profiles. For LangChain profiles, set `backend:` in YAML or use
`LangchainAgent(..., sandbox="docker")`.

### `LangchainAgent`

```python
from genai_tk.agents.langchain import LangchainAgent

agent = LangchainAgent("research")                       # slug or display name
print(agent.run("What is RAG?"))                         # sync

result = await agent.arun("Explain it like I'm 5")
async for chunk in agent.astream("Tell me a story"):     # yields str
    print(chunk, end="", flush=True)
await agent.arun_shell()                                 # interactive REPL

async with LangchainAgent("research", checkpointer=True) as agent:
    await agent.arun("Remember: my project is called Atlas")
    await agent.arun("What is my project called?")       # same thread ("1")
```

Constructor fields (one of `profile_name` or `llm` is required):

| Field | Type | Notes |
|-------|------|-------|
| `profile_name` | `str \| None` | First positional argument |
| `llm` | `str \| None` | Override the profile's LLM, or anchor an ad-hoc agent with no profile |
| `tools` | `list[BaseTool]` | Appended to the profile's tools |
| `agent_type` | `"react" \| "deep" \| "custom" \| None` | Override the profile type |
| `system_prompt` | `str \| None` | Override |
| `mcp_servers` | `list[str]` | Extra MCP servers |
| `extra_middlewares` | `list[str]` | Short names: `AnonymizationMiddleware`, `RichToolCallMiddleware`, `EmptyResponseRetryMiddleware`, `DeduplicateToolCallsMiddleware`, `ObservationTruncationMiddleware`, `WrapUpMiddleware` |
| `checkpointer` | `bool` | In-memory checkpointing; the thread id is fixed at `"1"` |
| `details` | `bool` | Verbose tool-call panels |
| `sandbox` | `"local" \| "docker" \| None` | `docker` sets a Docker backend and promotes the profile to `deep` |
| `vnc`, `keep_sandbox` | `bool` | Docker sandbox VNC viewer; keep the container between runs |

```python
from langchain_core.tools import tool


@tool
def get_stock(sku: str) -> int:
    """Return the stock level for a SKU."""
    return INVENTORY.get(sku, 0)


agent = LangchainAgent(llm="gpt-4o-mini@openai", tools=[get_stock], system_prompt="You help warehouse staff.")
```

Lower level, when you need the compiled LangGraph graph or your own thread ids:

```python
from genai_tk.agents.harness.profiles import load_langchain_profiles
from genai_tk.agents.langchain.factory import create_langchain_agent

profiles = load_langchain_profiles()
graph = await create_langchain_agent(profiles["research"], llm_override="claude-haiku@openrouter")
await graph.ainvoke(
    {"messages": [{"role": "user", "content": "Question"}]},
    config={"configurable": {"thread_id": "user-123"}},
)
```

### Harness layer — one API for LangChain and DeerFlow

```python
from genai_tk.agents.harness import create_harness, list_harness_profiles

for ref in list_harness_profiles():
    print(ref.key, ref.harness, ref.kind, ref.name)

harness = create_harness("research", llm_override="claude-haiku@openrouter")
async for event in harness.astream("Find three sources on QKD", thread_id="t1"):
    if event.kind == "token":
        print(event.text, end="")
    elif event.kind == "tool_call":
        print(f"\n→ {event.tool_name}({event.args})")
    elif event.kind == "error":
        print(f"\n✗ {event.message}")
answer = await harness.arun("Follow-up question", thread_id="t1")
await harness.aclose()
```

- `create_harness(key, *, llm_override=None, force_memory_checkpointer=False, mode_override=None,
  sandbox_override=None, extra_mcp=None, extra_tools=None)`; `mode_override`/`sandbox_override` are DeerFlow-only.
- `BaseHarness`: `astream(message, *, thread_id=None, context=None)`, `arun(...)`, `list_threads`, `list_models`,
  `list_skills`, `get_graph`, `get_checkpointer`, `aclose`.
- Events are frozen Pydantic models with a `kind` discriminator: `TokenEvent` (`text`), `ThinkingEvent`,
  `NodeEvent`, `ToolCallEvent` (`tool_name`, `args`, `call_id`), `ToolResultEvent` (`tool_name`, `content`),
  `ArtifactEvent`, `ClarificationEvent`, `UsageEvent`, `ErrorEvent` (`message`), `EndEvent`.
- `context=` propagates user/tenant identity to tools and middlewares for access control.
- `genai_tk.agents.harness.chat_repl.run_chat_repl` is the REPL used by `--chat`.

## Tools

There are **no short tool names** (`web_search`, `calculator`, `spec:` entries are gone). Each entry is a qualified
path, resolved by `UnifiedToolSpec`:

```yaml
tools:
  - my_pkg.tools.make_tools                              # bare factory/function path
  - my_pkg.tools:MyToolClass                             # class path
  - genai_tk.agents.tools.langchain.search_tools_factory.create_search_function:
      verbose: false                                     # single-key dict → kwargs
  - factory: genai_tk.agents.tools.langchain.rag_tool_factory.create_rag_tool_from_config
    config: {retriever: docs, tool_name: search_docs, top_k: 5}   # target key + sibling kwargs
```

Target keys: `tool`, `factory`, `class`, `function`, `target`.

Built-in factories:

| Factory | Provides |
|---------|----------|
| `genai_tk.agents.tools.langchain.search_tools_factory.create_search_function` / `create_search_tool` / `create_webpage_tool` | Web search and page fetch |
| `genai_tk.agents.tools.langchain.sql_tool_factory.create_sql_tool_from_config` / `create_sql_toolkit_tools` | SQL querying |
| `genai_tk.agents.tools.langchain.rag_tool_factory.create_rag_tool_from_config` | Retriever as a tool (see [rag.md](rag.md)) |
| `genai_tk.agents.tools.python_executor.tool.create_python_executor_tools` | CodeAct `python_interpreter` |
| `genai_tk.agents.tools.sandbox_browser.factory.create_sandbox_browser_tools` | Browser inside the Docker sandbox |
| `genai_tk.agents.tools.direct_browser.factory.create_direct_browser_tools` | Host Playwright browser (`browser` extra) |

Writing your own: return a LangChain `BaseTool` or `@tool` function (or a list of them) from a `create_*_tools()`
factory in your package, validate inputs at the boundary, and reference the factory from YAML.

Browser tools include `browser_navigate`, `browser_click`, `browser_type`, `browser_fill_credential`,
`browser_read_page`, `browser_screenshot`, `browser_evaluate`, and more. **Always use `browser_fill_credential`
for secrets**, never `browser_type`.

## CodeAct

CodeAct agents act by writing Python executed in an in-process AST sandbox. Sibling tools are injected as plain
Python functions; `print()` output is the observation and `final_answer(x)` ends the run.

```yaml
agents:
  codeact:
    harness: langchain
    name: "CodeAct"
    type: deep
    system_prompt: |
      Solve tasks only by writing Python with the python_interpreter tool.
      Tools such as web_search are Python functions. Call final_answer(result) when done.
    tools:
      - genai_tk.agents.tools.python_executor.tool.create_python_executor_tools:
          tools:
            - genai_tk.agents.tools.langchain.search_tools_factory.create_search_tool
    enable_file_system: false

  codeact-orchestrator:                      # deep agent delegating code work to a subagent
    harness: langchain
    type: deep
    tools: []
    subagents:
      - name: codeact
        description: "Solves tasks by writing Python in a sandbox."
        system_prompt: "Write Python, print observations, finish with final_answer(x)."
        tools:
          - genai_tk.agents.tools.python_executor.tool.create_python_executor_tools:
              tools: [genai_tk.agents.tools.langchain.search_tools_factory.create_search_tool]
```

The bundled versions are in `config/examples/agents/codeact.yaml`.

## Middlewares

All live in `genai_tk.agents.langchain.middleware`:

| Middleware | Purpose |
|------------|---------|
| `rich_middleware.RichToolCallMiddleware` | Pretty tool-call traces (`details: true` for full panels) |
| `rich_middleware.ToolCallLimitMiddleware` | Cap tool calls (`run_limit`, `thread_limit`, `exit_behavior`) |
| `empty_response_retry.EmptyResponseRetryMiddleware` | Retry empty model output; **always prepended automatically** |
| `deduplicate_middleware.DeduplicateToolCallsMiddleware` | Drop repeated identical tool calls |
| `observation_truncation_middleware.ObservationTruncationMiddleware` | Truncate long tool outputs (`max_chars`) |
| `prompt_cache.PromptCacheMiddleware` | Mark the static prompt prefix cacheable (Anthropic, OpenRouter passthrough) |
| `wrap_up_middleware.WrapUpMiddleware` | Escalating soft/hard limits that force convergence before the recursion limit |
| `anonymization_middleware.AnonymizationMiddleware` | Reversible Presidio + Faker PII redaction (`nlp` extra) |
| `sensitivity_router_middleware.SensitivityRouterMiddleware` | Route sensitive conversations to a safer LLM |
| `skill_marks_middleware.SkillLoadMarksMiddleware` | Records skill loads in trajectories; auto-added for deep agents with skills |

Third-party middlewares work the same way, for example
`- class: deepagents.middleware.summarization.SummarizationMiddleware`.

## Skills

Skills are `SKILL.md` files the agent reads **on demand** (progressive disclosure), instead of stuffing domain
knowledge into the system prompt.

- Only `type: deep` profiles load skills: `skill_directories` is passed to deepagents' native `SkillsMiddleware`.
  React agents ignore it. genai-tk itself has no `SkillsMiddleware` class.
- With a Docker backend, each skill directory is bind-mounted read-only at `/mnt/skills/<dirname>`.
- DeerFlow profiles use `skill_directories` plus an optional `available_skills` filter.
- Manage skills with `cli skills list|create|add|validate` (see [cli-and-init.md](cli-and-init.md)).

## Sandbox

```bash
uv run cli sandbox start      # start the OpenSandbox server (generates its own config)
uv run cli sandbox pull       # pre-pull the image; cuts first-run startup from ~30s to ~5s
uv run cli sandbox status
uv run cli sandbox list
uv run cli sandbox stop
```

Run `start` + `pull` once per boot. The sandbox image (`ghcr.io/agent-infra/sandbox`) provides Python, Node.js,
Chromium with VNC, and a REST file/shell API. Containers run with `ulimit core=0` and isolated LibreOffice
profiles. Use it for any untrusted code; `local` executes on the host.

```yaml
backend:
  type: aio_sandbox            # or docker
  host_port: 18091
  startup_timeout: 90.0
  work_dir: /home/user
  env_vars: {MY_VAR: "1"}
```

## MCP

### Consuming MCP servers

```yaml
# config/mcp_servers.yaml
mcpServers:
  filesystem:
    command: npx
    args: ["-y", "@modelcontextprotocol/server-filesystem", "${paths.project}"]
  playwright:
    command: npx
    args: ["@playwright/mcp@latest", "--headless"]
  remote_search:
    url: https://mcp.example.com/mcp
    transport: streamable-http          # stdio | sse | streamable-http | http
    headers: {Authorization: "Bearer ${oc.env:SEARCH_TOKEN}"}
    disabled: false

mcpProjectServers:                      # genai-tk servers defined in tk_servers.yaml
  chinook: {}
  search: {disabled: true}
```

Reference servers by name in a profile's `mcp_servers:` or pass `--mcp NAME` on the CLI. `uv run cli info mcp-tools`
lists the tools they expose, and `uv run cli core mcp-call` calls one directly.

### Exposing tools as an MCP server

```yaml
# config/tk_servers.yaml
mcp_expose_servers:
  search:
    description: "Web search tools"
    tools:
      - factory: genai_tk.agents.tools.langchain.search_tools_factory.create_search_tool
    agent:                               # optional: wrap all tools in one agent tool
      enabled: true
      name: run_search_agent
      llm: gpt-4o-mini@openai
```

```bash
uv run cli mcpserver list
uv run cli mcpserver start --name search --transport stdio
uv run cli mcpserver start --name search --transport streamable-http --port 8765
uv run cli mcpserver generate --name search      # emit a standalone server script
```

The group is `mcpserver`; upstream docs and comments that say `cli mcp ...` are out of date. The implementation
targets MCP 2.0, with compatibility shims for 1.x clients in `genai_tk.mcp.compat`.

## DeerFlow

DeerFlow profiles live under the same `agents:` dict with `harness: deerflow`. They do not inherit
`agent_defaults`.

```yaml
agents:
  "Research Assistant":
    harness: deerflow
    description: "Deep web research with reports"
    mode: pro                    # flash | thinking | pro | ultra
    llm: gpt-4o-mini@openai      # optional
    tool_groups: [web, file, bash]
    mcp_servers: [filesystem]
    skill_directories: ["${paths.project}/skills"]
    available_skills: []         # optional filter
    sandbox: docker              # local | docker
    plan_mode: true
    subagent_enabled: true
    recursion_limit: 160
```

| Mode | Thinking | Planning | Sub-agents |
|------|:--------:|:--------:|:----------:|
| `flash` | — | — | — |
| `thinking` | ✓ | — | — |
| `pro` | ✓ | ✓ | — |
| `ultra` | ✓ | ✓ | ✓ |

```bash
uv run cli agents run "Research Assistant" --mode ultra --chat
uv run cli agents run "Research Assistant" --sandbox docker "Explain QKD"
uv run cli agents tui "Research Assistant" --continue
```

Install with the `harnessing` extra; `DEER_FLOW_PATH` and `cli init --deer-flow` are no longer needed. Global
DeerFlow settings go under a top-level `deerflow:` key (see `config/examples/deerflow.yaml`). From Python, use
`create_harness("Research Assistant")` rather than the low-level `EmbeddedDeerFlowClient`.

## Trajectories and Monitoring

Every deep-agent run is recorded in an ATOF trajectory store under `<data_root>/trajectories/` (scope tree, tool
args and results, skill loads, token usage).

```bash
uv run cli trajectory list
uv run cli trajectory tui                # interactive step-by-step replay
uv run cli trajectory show <id>
uv run cli trajectory diff <a> <b>
uv run cli trajectory stats
uv run cli trajectory prune

uv run cli monitoring status             # LangSmith / Langfuse / OTLP / local backends
uv run cli monitoring start              # self-hosted Langfuse (Docker)
uv run cli agents run research "..." --trace
```

Tracing backends are configured under `monitoring.backends` (see `config/examples/monitoring.yaml`); Langfuse needs
the `monitoring` extra.
