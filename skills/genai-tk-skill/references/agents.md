# Agents — references/agents.md

genai-tk bundles four agent frameworks with a shared profile system, LLM factory, MCP server registry, and sandbox config. This reference covers when to use each, how to write profiles, and how the cross-cutting features (tools, middleware, sandbox, skills) wire in.

## Quick decision tree

| Need | Pick | Why |
|------|------|-----|
| Tool-using agent for general tasks | **ReAct** (`type: react`) | Standard Thought→Action→Observation loop, fast, works with any LLM |
| Multi-step planning, subagent delegation, code in sandbox | **Deep** (`type: deep`) | Adds planning, file system tools, OpenSandbox backend; needs `deepagents` extra |
| Deep web research with reports, native search & sub-agents | **Deer-flow** | ByteDance's LangGraph system, embedded in-process; best for research workflows |
| Code-first automation, Python REPL agent, data analysis | **SmolAgents** | Generates and executes Python; supports local, Docker, and E2B backends |

All four share `model_id@provider` LLM identifiers, the MCP server registry, the OpenSandbox Docker integration, and `skill_directories` for on-demand domain knowledge.

---

## LangChain agents — the unified entry point

`LangchainAgent` is the production-friendly wrapper. It loads a YAML profile, applies overrides, and produces a compiled LangGraph agent.

```python
from genai_tk.agents.langchain import LangchainAgent

# From a named profile
agent = LangchainAgent("Research")
result = agent.run("Summarise GPT-4 technical report")

# Async
result = await agent.arun("Explain quantum computing")

# Streaming
async for chunk in agent.astream("Tell me a story"):
    print(chunk, end="", flush=True)

# Interactive REPL
await agent.arun_shell()

# Ad-hoc agent (no profile, llm required)
agent = LangchainAgent(llm="gpt41mini@openai", tools=[my_tool])

# Override profile fields at construction
agent = LangchainAgent(
    "Research",
    llm="gpt_4o@openai",          # override the profile's llm
    sandbox="docker",              # promote to Docker sandbox
    details=True,                  # verbose tool-call output
    checkpointer=True,             # persist state across turns
)
```

`LangchainAgent` constructor fields (all optional except one of `profile_name`/`llm`):

| Field | Type | Notes |
|-------|------|-------|
| `profile_name` | `str \| None` | Name of profile in `langchain.yaml` |
| `llm` | `str \| None` | Override profile LLM, or anchor an ad-hoc agent |
| `tools` | `list[BaseTool]` | Pre-built tools appended to profile's tool list |
| `agent_type` | `"react" \| "deep" \| None` | Override profile type |
| `system_prompt` | `str \| None` | Override profile system prompt |
| `mcp_servers` | `list[str]` | Extra MCP servers to load alongside profile's |
| `checkpointer` | `bool` | Enable in-memory checkpointing for multi-turn |
| `details` | `bool` | Verbose `RichToolCallMiddleware` output |
| `sandbox` | `"local" \| "docker" \| None` | `docker` promotes to deep + OpenSandbox |
| `vnc` | `bool` | Open VNC viewer on Docker sandbox |
| `keep_sandbox` | `bool` | Don't tear down container between runs |

## Writing profiles in `langchain.yaml`

```yaml
# config/agents/langchain.yaml
langchain_agents:
  defaults:                          # applied to every profile unless overridden
    type: react
    llm: null                        # null → use llm.default from baseline.yaml
    tools: []
    middlewares: []
    checkpointer: { type: none }
    backend: { type: none }
    mcp_servers: []
    skills: { directories: [] }

  default_profile: "Research"        # used when -p is omitted

  profiles:
    - name: "Research"
      type: deep
      llm: gpt_41@openai
      enable_planning: true
      enable_file_system: true
      tools:
        - spec: web_search
          config: { provider: serper, max_results: 5 }
      mcp_servers: [tavily-mcp]
      skills:
        directories:
          - ${paths.project}/skills
      backend:
        type: aio_sandbox
        opensandbox_server_url: http://localhost:8080
        startup_timeout: 90.0
      checkpointer:
        type: memory                 # memory | postgres | sqlite

    - name: "Coding"
      type: react
      llm: gpt_4o@openai
      tools:
        - spec: filesystem_tools
          config: { allowed_dirs: ["/home", "/tmp"] }
      middlewares:
        - class: genai_tk.agents.langchain.middleware.rich_middleware.RichToolCallMiddleware
          details: true

    - name: "DataAnalysis"
      type: react
      llm: gpt_4o@openai
      tools:
        - spec: sql_tools
          config:
            database: analytics.db
            schema: public
            include_tables: ["users", "orders"]
        - spec: dataframe_tools
```

Key fields:

- `type`: `react` | `deep` | `custom`
- `llm`: model identifier (`model_id@provider` or a tag); `null` falls back to `llm.default`
- `tools`: list of `{ spec: name, config: {...} }` entries — see "Tool specs" below
- `mcp_servers`: names from `mcp_servers.yaml` to load as tools
- `middlewares`: ordered list of class+kwargs entries; class path uses dotted notation
- `checkpointer`: state persistence (`memory` | `postgres` | `sqlite` | `none`)
- `backend`: execution backend for deep agents (`aio_sandbox` | `class` | `none`)
- `skills.directories`: paths whose `SKILL.md` files are exposed via `SkillsMiddleware`

## Agent types

### ReAct (default)

Standard Thought → Action → Observation loop. Best for general tool-use tasks. No extra dependencies beyond `langchain`/`langgraph`.

```yaml
- name: "General"
  type: react
  llm: gpt_4o@openai
  tools:
    - spec: web_search
    - spec: calculator
```

### Deep

Adds planning, subagent delegation, and an optional Docker sandbox. **Requires `pip install deepagents`** (or include in your `pyproject.toml`). Best for research, multi-step analysis, anything that benefits from "make a plan, then execute it."

```yaml
- name: "Research"
  type: deep
  llm: gpt_41@openai
  enable_planning: true
  enable_file_system: true
  backend:
    type: aio_sandbox
    opensandbox_server_url: http://localhost:8080
  skills:
    directories:
      - ${paths.project}/skills
  tools:
    - spec: web_search
    - spec: filesystem_tools
```

Deep agents in Docker sandbox automatically bind-mount skill directories at `/mnt/skills/` (read-only), so the agent can read them via the sandboxed filesystem.

### Custom

Functional API agent built with LangGraph for maximum flexibility. Best when you need a non-standard graph topology.

```yaml
- name: "Custom"
  type: custom
  llm: gpt_4o@openai
```

Implement the construction in `genai_tk.extra.graphs.custom_react_agent.create_custom_react_agent` (or your own factory and reference it).

---

## Tool specs

Tools are loaded from `tool_specs.py` factories. Each entry is `{ spec: name, config: {...} }`:

| Spec | Purpose |
|------|---------|
| `web_search` | Web search (provider configurable: `serper`, `tavily`, etc.) |
| `calculator` | Math expressions |
| `filesystem_tools` | File read/write within allowed dirs |
| `sql_tools` | SQL database query (LangChain SQL agent under the hood) |
| `dataframe_tools` | Pandas/Polars manipulation |
| `python_repl` | Sandboxed Python execution |
| `browser_use` | Browser automation (`browser_fill_credential` for credentials) |
| `yfinance_tools` | Financial data via yfinance |

You can also append pre-built `BaseTool` instances at construction:

```python
from langchain_core.tools import tool

@tool
def my_custom_tool(query: str) -> str:
    """Do something custom."""
    return f"Result: {query}"

agent = LangchainAgent("Research", tools=[my_custom_tool])
```

---

## Middleware

Middlewares run before/after the model and around tool calls. Configure in YAML or attach in code:

```yaml
middlewares:
  - class: genai_tk.agents.langchain.middleware.rich_middleware.RichToolCallMiddleware
    details: true

  - class: genai_tk.agents.langchain.middleware.anonymization_middleware:AnonymizationMiddleware
    analyzed_fields: [PERSON, EMAIL_ADDRESS, PHONE_NUMBER, CREDIT_CARD]
    fuzzy_deanonymize: true

  - class: deepagents.middleware.summarization.SummarizationMiddleware
    model: gpt-35-turbo@openai
    trigger: ["tokens", 4000]
```

Built-in middlewares (in `genai_tk.agents.langchain.middleware`):

- **`RichToolCallMiddleware`** — pretty-printed tool-call traces. Set `details: true` for full panels per call.
- **`AnonymizationMiddleware`** — Presidio + Faker reversible PII redaction. PII is replaced before the LLM and restored in responses. See `docs/middleware-pii-and-routing.md` in the repo for the full config.
- **`SensitivityRouterMiddleware`** — routes sensitive conversations to a "safer" LLM based on content classification.
- **`empty_response_retry`** — retries on empty/null model output.
- **`SkillsMiddleware`** — exposes `SKILL.md` files from configured directories for the agent to read on demand. The cornerstone of the project's "skills over giant system prompts" philosophy.

---

## MCP server integration

[Model Context Protocol](https://modelcontextprotocol.io) servers expose tools that any genai-tk agent can load:

```yaml
# config/mcp_servers.yaml
mcp_servers_config:
  tavily-mcp:
    command: npx
    args: ["-y", "tavily-mcp"]
    env:
      TAVILY_API_KEY: ${TAVILY_API_KEY}

  math_server:
    command: python
    args: ["-m", "genai_tk.mcp.math_server"]
```

Reference them by name from any profile:

```yaml
mcp_servers: [tavily-mcp, math_server]
```

Or override at runtime:

```bash
cli agents langchain -p Research --mcp custom_server "..."
```

Programmatically:

```python
agent = LangchainAgent("Research", mcp_servers=["custom_server"])
```

---

## Sandbox — OpenSandbox Docker

Three sandbox modes:

| Mode | Purpose |
|------|---------|
| `local` | Run code in the host process (development, trusted code) |
| `docker` | OpenSandbox Docker container — full isolation, Chromium, Python, Node.js, REST file/shell API, VNC |
| (none) | Agent runs without code execution |

**One-time setup:**

```bash
uv add opensandbox-server
opensandbox-server init-config ~/.sandbox.toml --example docker

# Warm up (cuts container startup ~28s → ~5s)
cli sandbox start
cli sandbox pull
cli sandbox status
```

**Use with any agent:**

```bash
cli agents langchain  -p Research --sandbox docker "Write & run Python"
cli agents deerflow   -p "Research Assistant" --sandbox docker --chat
cli agents smolagents --executor docker "Install pandas and analyse data.csv"
```

In code:

```python
from genai_tk.agents.langchain import LangchainAgent
agent = LangchainAgent(llm="gpt41mini@openai", sandbox="docker")
await agent.arun("Write a Python script and run it")
```

The sandbox auto-mounts skill directories at `/mnt/skills/` (read-only), exposes Chromium with VNC at `localhost:8080/vnc`, and provides a REST shell/file API. See the repo's `docs/sandbox_support.md` for full setup and troubleshooting.

---

## Skills — domain knowledge on demand

The project's preferred pattern for domain-specific knowledge is `SKILL.md` files loaded by `SkillsMiddleware` only when relevant — not stuffed into every system prompt.

```
skills/
├── public/                # shipped with the toolkit / Deer-flow
│   ├── deep-research/SKILL.md
│   └── data-analysis/SKILL.md
└── custom/                # your project skills
    └── invoicing/SKILL.md
```

Profile config:

```yaml
- name: "Research"
  type: deep
  skills:
    directories:
      - ${paths.project}/skills          # all SKILL.md files under this tree
  available_skills:                       # optional filter (deer-flow only)
    - public/deep-research
    - public/data-analysis
```

In Docker sandbox mode the skill directories are bind-mounted read-only at `/mnt/skills/`. Outside dev mode, skills in `$DEER_FLOW_PATH/skills` are auto-discovered by Deer-flow.

Skills are markdown — they should be *capability descriptions plus examples*, not system prompts. The agent reads them only when the task seems to call for them.

---

## Deer-flow

[Deer-flow](https://github.com/bytedance/deer-flow) is ByteDance's LangGraph multi-agent system with native web search, planning, sub-agents, and reporting. genai-tk embeds it **in-process** — no separate server.

**Setup (one-time):**

```bash
cli init --deer-flow                 # clones Deer-flow + installs backend
# add to .env:
# DEER_FLOW_PATH=~/deer-flow
```

**Modes** (set via `mode:` in profile or `--mode` flag):

| Mode | Thinking | Planning | Sub-agents | Use for |
|------|:--------:|:--------:|:----------:|---------|
| `flash` | — | — | — | Quick fact lookup |
| `thinking` | ✓ | — | — | Reasoning-heavy single agent |
| `pro` | ✓ | ✓ | — | Plan + execute |
| `ultra` | ✓ | ✓ | ✓ | Full deep research with sub-agents |

**Profile** in `config/agents/deerflow.yaml`:

```yaml
deerflow_agents:
  - name: "Research Assistant"
    mode: pro
    llm: gpt_41@openai             # optional — falls back to server default
    mcp_servers: [tavily-mcp]
    skill_directories:
      - ${paths.project}/skills
    available_skills:              # optional filter
      - public/deep-research
      - public/data-analysis
    sandbox: local                 # local | docker
```

**CLI:**

```bash
cli agents deerflow --list                                       # profiles + modes
cli agents deerflow --chat                                       # interactive
cli agents deerflow -p "Research Assistant" --trace "QKD basics" # with trace
cli agents deerflow -p "Research Assistant" --mode ultra --chat  # full deep research
cli agents deerflow --sandbox docker -p "Research Assistant" --chat
```

**Python:**

```python
from genai_tk.agents.deer_flow.embedded_client import EmbeddedDeerFlowClient

client = EmbeddedDeerFlowClient(profile="Research Assistant", mode="pro")
async for event in client.astream("Explain quantum key distribution"):
    print(event)
```

---

## SmolAgents

HuggingFace's [SmolAgents](https://huggingface.co/docs/smolagents) — code-first agents that write and execute Python. Three execution backends: `local` (host process), `docker` (OpenSandbox), `e2b` (cloud, requires `E2B_API_KEY`).

```bash
cli agents smolagents "Plot the sine function and save to sine.png"
cli agents smolagents --executor docker "Install pandas and analyse data.csv"
cli agents smolagents --executor e2b "Scrape and summarise this webpage: ..."
cli agents smolagents --chat                                 # interactive
cli agents smolagents -t web_search "What's the latest in AI?"
cli agents smolagents -t sql_tools "How many users last month?"
```

Tool flags accept the same specs as LangChain (`web_search`, `calculator`, `sql_tools`, `dataframe_tools`, `yfinance_tools`, `browser_use`).

Configuration in `config/smolagents.yaml`:

```yaml
codeact_agent:
  default:
    type: codeact
    llm: gpt_4o@openai
    tools:
      - web_search
      - calculator
```

Programmatic API:

```python
from genai_tk.agents.smolagents import SmolAgent

agent = SmolAgent(llm="gpt_4o@openai", tools=["web_search"], executor="docker")
result = agent.run("Scrape and summarise this URL: ...")
```

---

## Patterns

### Multi-turn with checkpointing

`LangchainAgent.arun` calls the compiled agent with a fixed `thread_id="1"`, so it accumulates conversation history across calls when `checkpointer=True`. For multiple independent threads, drop down to the compiled agent and pass your own `thread_id`:

```python
# Single thread (most common case)
agent = LangchainAgent("Research", checkpointer=True)
await agent.arun("What is RAG?")
await agent.arun("Explain it like I'm 5")    # remembers previous turn

# Multiple threads — use the compiled agent directly
compiled = await agent._ensure_initialized()
await compiled.ainvoke(
    {"messages": [{"role": "user", "content": "Question"}]},
    config={"configurable": {"thread_id": "user_session_123"}},
)
```

### Tool overrides + ad-hoc profile

```python
from langchain_core.tools import tool

@tool
def get_inventory(sku: str) -> int:
    """Look up stock for a SKU."""
    return INVENTORY.get(sku, 0)

agent = LangchainAgent(
    llm="gpt41mini@openai",
    tools=[get_inventory],
    system_prompt="You help warehouse staff find stock levels."
)
```

### Loading config + dispatching by profile name

For lower-level access to the profile system:

```python
from genai_tk.agents.langchain.config import load_unified_config, resolve_profile
from genai_tk.agents.langchain.factory import create_langchain_agent

config = load_unified_config()
profile = resolve_profile(config, "Research")
compiled_agent = await create_langchain_agent(profile)

result = await compiled_agent.ainvoke({
    "messages": [{"role": "user", "content": "Query"}]
})
```

This is what `LangchainAgent` does internally — drop down to it only when you need to inspect or mutate the resolved profile before compiling.
