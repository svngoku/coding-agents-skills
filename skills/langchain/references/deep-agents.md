# Deep Agents and Managed Deep Agents

Deep Agents is LangChain's agent harness for long, multi-step tasks: research, coding, analysis.
It comes in two forms:

- **`deepagents`** (open source): `create_deep_agent` builds a LangGraph agent with a virtual
  filesystem, subagents, summarization, skills and memory. You host it yourself.
- **Managed Deep Agents** (`managed-deepagents`, CLI `mda`): a code-first project that compiles to a
  managed LangGraph app on LangSmith, with hosted memory, sandboxes, schedules, identity and
  connections.

Verified against `deepagents` 0.7.28 and `managed-deepagents` 0.9.3 (both need Python 3.11+).

## Choosing an Approach

| Need | Use |
|------|-----|
| A focused agent with a few tools and a short loop | `langchain.agents.create_agent` |
| Long tasks that need files, context offloading, subagents, skills or code execution, self-hosted | `deepagents.create_deep_agent` |
| The same harness, but deployed and operated by LangSmith (memory, sandboxes, cron, Slack, per-user auth) | Managed Deep Agents |

`create_deep_agent` returns a compiled LangGraph graph, so `invoke`, `stream`, `stream_events`,
checkpointers, stores and interrupts work exactly as with `create_agent`.

## Open-Source Harness

### Quickstart

```bash
pip install -U deepagents langchain-anthropic
pip install -U "deepagents[quickjs]"   # optional: JavaScript code interpreter
```

```python
from deepagents import create_deep_agent

def internet_search(query: str, max_results: int = 5) -> str:
    """Search the web and return the top results."""
    return f"Results for {query}"

agent = create_deep_agent(
    model="anthropic:claude-sonnet-5",
    tools=[internet_search],
    system_prompt="You are an expert researcher. Write the final report to /report.md.",
)

result = agent.invoke({"messages": [{"role": "user", "content": "What is LangGraph?"}]})
print(result["messages"][-1].content)
print(result["files"].keys())   # files the agent wrote (default StateBackend)
```

`create_deep_agent(model, tools, *, system_prompt, middleware, subagents, skills, memory,
permissions, backend, interrupt_on, response_format, state_schema, context_schema, checkpointer,
store, debug, name, cache)`. `model` takes a `"provider:model"` string or a chat model instance.
Tools can be plain functions, `@tool` tools or MCP tools from `langchain.mcp.MCPAdapter`.

### Built-in Tools

| Tool | Purpose |
|------|---------|
| `ls`, `read_file`, `write_file`, `edit_file`, `delete` | Work with files in the backend (`delete` needs 0.7+) |
| `glob`, `grep` | Find files by pattern and search their contents |
| `task` | Delegate work to a subagent with a fresh context window |
| `execute` | Run shell commands; only present with a sandbox or shell backend |
| `write_todos` | Planning list; only present when you add `TodoListMiddleware()` |

Since 0.7, planning is opt-in. Add it explicitly:

```python
from langchain.agents.middleware import TodoListMiddleware

agent = create_deep_agent(model="anthropic:claude-sonnet-5", tools=[internet_search],
                          middleware=[TodoListMiddleware()])
```

### Default Middleware Stack

The harness assembles middleware in this order; your `middleware=` list goes after the base layer:

1. Base: `FilesystemMiddleware`, `SubAgentMiddleware`, `SummarizationMiddleware`,
   `PatchToolCallsMiddleware`, `AsyncSubAgentMiddleware`
2. Your `middleware=[...]`
3. Tail: harness profile extras, `SkillsMiddleware` (when `skills=`), provider prompt caching
   (Anthropic, plus Bedrock and Fireworks when installed), `MemoryMiddleware` (when `memory=`),
   `HumanInTheLoopMiddleware` (when `interrupt_on=`) and `UnsupportedContentMiddleware`

Context stays in budget without losing data: summarization writes evicted messages to
`/conversation_history/`, and the filesystem middleware moves tool results over 20,000 tokens to
`/large_tool_results/`. The agent can read or grep them back later.

### Backends

The backend decides where files live and whether `execute` exists.

| Backend | Storage | Notes |
|---------|---------|-------|
| `StateBackend()` (default) | Agent state, per thread | Scratch space; persisted by the checkpointer |
| `FilesystemBackend(root_dir=".", virtual_mode=True)` | Local disk | `virtual_mode=True` blocks paths outside `root_dir` |
| `StoreBackend(namespace=lambda rt: (...))` | LangGraph store | Cross-thread; needs `store=` |
| `CompositeBackend(default=..., routes={"/prefix/": ...})` | Mixed | Route path prefixes to different backends |
| `LocalShellBackend(...)` | Local disk and shell | Adds `execute` on the host; development only |
| `LangSmithSandbox`, `DaytonaSandbox`, `ModalSandbox` | Remote sandbox | Isolated files plus `execute` |

```python
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langgraph.store.memory import InMemoryStore

agent = create_deep_agent(
    model="anthropic:claude-sonnet-5",
    backend=CompositeBackend(
        default=StateBackend(),
        routes={"/memories/": StoreBackend(namespace=lambda rt: ("memories",))},
    ),
    store=InMemoryStore(),   # use a persistent store in production
)
```

Seed the default `StateBackend` per run with `deepagents.backends.utils.create_file_data`:

```python
from deepagents.backends.utils import create_file_data

agent.invoke({
    "messages": [{"role": "user", "content": "Summarize /data/notes.md"}],
    "files": {"/data/notes.md": create_file_data("Q3 revenue grew 12%.")},
})
```

### Permissions

`FilesystemPermission` rules guard the built-in filesystem tools. Rules are checked in order, the
first match wins, and anything unmatched is allowed. Paths are absolute globs.

```python
from deepagents import FilesystemPermission, create_deep_agent
from langgraph.checkpoint.memory import InMemorySaver

agent = create_deep_agent(
    model="anthropic:claude-sonnet-5",
    permissions=[
        FilesystemPermission(operations=["write"], paths=["/policies/**"], mode="deny"),
        FilesystemPermission(operations=["write"], paths=["/reports/**"], mode="interrupt"),
        FilesystemPermission(operations=["read", "write"], paths=["/workspace/**"], mode="allow"),
    ],
    checkpointer=InMemorySaver(),   # "interrupt" pauses for approval, so it needs a checkpointer
)
```

Permissions apply to the filesystem tools only. They do not restrict `execute` in a sandbox; the
sandbox itself is the isolation boundary.

### Sandboxes and Code Execution

A sandbox backend gives the agent its own machine: files plus an `execute` tool.

```python
from deepagents import create_deep_agent
from deepagents.backends import LangSmithSandbox
from langsmith.sandbox import SandboxClient

sandbox = SandboxClient().create_sandbox()   # needs LANGSMITH_API_KEY
agent = create_deep_agent(model="anthropic:claude-sonnet-5", backend=LangSmithSandbox(sandbox=sandbox))
```

Other providers ship as integration packages: `langchain_daytona.DaytonaSandbox` and
`langchain_modal.ModalSandbox`. Never give `LocalShellBackend` to an agent that handles untrusted
input; it runs commands on your machine.

For lightweight computation without a sandbox, the QuickJS interpreter adds an `eval` tool that
runs JavaScript in a memory- and time-limited WebAssembly runtime:

```python
from langchain_quickjs import CodeInterpreterMiddleware

agent = create_deep_agent(
    model="anthropic:claude-sonnet-5",
    tools=[internet_search],
    middleware=[CodeInterpreterMiddleware(timeout=5.0, ptc=["internet_search"])],
)
```

`ptc=` (programmatic tool calling) lists tools the JavaScript code may call directly, so the agent
can loop over tool calls in code instead of one model turn per call. With `subagents=True` (the
default) code can also spawn subagents.

### Skills and Memory

```python
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

agent = create_deep_agent(
    model="anthropic:claude-sonnet-5",
    backend=FilesystemBackend(root_dir="./agent-files", virtual_mode=True),
    skills=["/skills/"],              # directories of <skill-name>/SKILL.md, loaded on demand
    memory=["/memories/AGENTS.md"],   # files injected into the prompt on every run
)
```

- **Skills** use progressive disclosure: only each skill's frontmatter `name` and `description` sit
  in the prompt, and the agent reads the full `SKILL.md` when a task matches.
- **Memory** files are always loaded. The agent can update them with `edit_file`, so route
  `/memories/` to a `StoreBackend` when memory must outlive a thread.
- Both paths are resolved through the backend, so they must exist in it.

### Subagents

The `task` tool runs a subagent in a fresh context and returns only its final answer, which keeps
the main context small. A `general-purpose` subagent with the main agent's tools is always added.

```python
research_subagent = {
    "name": "research-agent",
    "description": "Researches one question in depth and returns a summary with sources.",
    "system_prompt": "You are a careful researcher. Return a concise summary.",
    "tools": [internet_search],
    "model": "openai:gpt-5.5",   # optional; defaults to the main model
}

agent = create_deep_agent(model="anthropic:claude-sonnet-5", subagents=[research_subagent])
```

| Subagent field | Meaning |
|----------------|---------|
| `name`, `description` | Required; the description is how the main agent decides to delegate |
| `system_prompt` | Required for dict subagents |
| `tools`, `model`, `middleware`, `skills`, `permissions`, `interrupt_on`, `response_format` | Optional overrides |
| `mode` | `"isolated"` (default, fresh context) or `"fork"` (beta, 0.7.13+: inherits the parent's history and prompt) |

Other subagent forms:

- **`CompiledSubAgent`**: wrap an existing graph, `{"name": ..., "description": ..., "runnable": graph}`.
- **`AsyncSubAgent`**: a background agent on a LangGraph deployment. With no `url` it calls a graph
  co-deployed in the same server (ASGI); with `url` (and optional `headers`) it calls a remote one.
  The main agent gets `start_async_task`, `check_async_task`, `update_async_task`,
  `cancel_async_task` and `list_async_tasks`, and keeps chatting while the task runs.

```python
from deepagents import AsyncSubAgent, create_deep_agent

agent = create_deep_agent(
    model="anthropic:claude-sonnet-5",
    subagents=[AsyncSubAgent(name="researcher", description="Long-running web research",
                             graph_id="researcher")],
)
```

### Human-in-the-Loop

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

agent = create_deep_agent(
    model="anthropic:claude-sonnet-5",
    tools=[send_email],
    interrupt_on={"send_email": {"allowed_decisions": ["approve", "edit", "reject"]}, "write_file": True},
    checkpointer=InMemorySaver(),
)

config = {"configurable": {"thread_id": "t1"}}
result = agent.invoke({"messages": [{"role": "user", "content": "Email Bob the summary"}]},
                      config=config, version="v2")
if result.interrupts:
    actions = result.interrupts[0].value["action_requests"]
    result = agent.invoke(Command(resume={"decisions": [{"type": "approve"} for _ in actions]}),
                          config=config, version="v2")
```

Subagents inherit `interrupt_on` unless they set their own.

### Streaming Subagents

Event streaming (`version="v3"`, experimental) exposes each subagent run as its own stream:

```python
stream = agent.stream_events({"messages": [{"role": "user", "content": "Research LangGraph"}]}, version="v3")
for kind, item in stream.interleave("messages", "subagents"):
    if kind == "messages":
        for token in item.text:
            print(token, end="", flush=True)
    else:
        print(f"\n[subagent {item.name}]")
        for message in item.messages:   # also: item.tool_calls, nested item.subagents
            for token in message.text:
                print(token, end="", flush=True)
```

### Harness Profiles and Tool Restrictions

Hide built-in tools or tune prompts per model with a harness profile:

```python
from deepagents import HarnessProfile, create_deep_agent, register_harness_profile

register_harness_profile(
    "anthropic:claude-sonnet-5",
    HarnessProfile(excluded_tools=frozenset({"delete", "write_file"}),
                   system_prompt_suffix="Prefer edit_file over rewriting whole files."),
)
agent = create_deep_agent(model="anthropic:claude-sonnet-5")   # picks up the profile
```

`HarnessProfile` also accepts `base_system_prompt`, `tool_description_overrides`,
`excluded_middleware`, `extra_middleware` and `general_purpose_subagent`. To restrict tools for a
specific filesystem instead, pass your own `deepagents.FilesystemMiddleware(backend=...,
tools=["ls", "read_file", "grep"])`; `read_file` must be in the list.

### Self-Hosting

A deep agent is a LangGraph graph, so deploy it like one: reference the module-level `agent` in
`langgraph.json` and use LangGraph Platform or your own server, with a persistent checkpointer and
store (see [langgraph.md](langgraph.md)). If you do not want to run that infrastructure, use Managed
Deep Agents.

## Managed Deep Agents

Managed Deep Agents compiles a project directory into a managed LangGraph app and deploys it to
LangSmith. You write the agent; LangSmith runs threads, memory (Context Hub), sandboxes, cron
schedules, channels (Slack) and credential storage.

### Workflow

```bash
uvx --from managed-deepagents mda init research-assistant --model anthropic:claude-sonnet-5 --memory agent
cd research-assistant && uv sync
# put LANGSMITH_API_KEY and the model provider key in .env
uv run mda dev      # compile and run on the local LangGraph dev server
uv run mda chat     # chat from the terminal (starts dev when none is running)
uv run mda deploy   # build and deploy to LangSmith (--deployment-type dev|prod)
uv run mda logs     # tail the deployed agent's server logs
uv run mda delete   # remove the deployment and the resources it created
```

`mda build` compiles to `.mda/build` without deploying, which is a cheap offline check. Other
commands: `mda connections ...`, `mda channels init slack`, `mda evals init` (Harbor evals).

### Project Layout

| Path | Declares |
|------|----------|
| `agent.py` | `agent = define_deep_agent(...)`: model, tools, middleware, subagents |
| `instructions.md` | The system prompt |
| `tools/` | Tool modules; `tools/mcp.py` holds `define_mcp(...)` |
| `middleware/` | Custom middleware modules |
| `skills/<name>/SKILL.md` | Skills, deployed with the agent |
| `memory.py` | `define_memory(...)`; delete it for no durable memory |
| `identity.py` | `define_identity(...)`: who may call the deployment |
| `sandbox/__init__.py` | `define_sandbox(...)`; optional `sandbox/setup.sh` baked into a snapshot |
| `schedules/*.py` | One `define_schedule(...)` per file |
| `channels/` | Messaging channels such as Slack |
| `.env` | `LANGSMITH_API_KEY` and provider keys (keep it out of git) |

### agent.py

```python
from langchain.agents.middleware import TodoListMiddleware
from managed_deepagents import define_deep_agent

from middleware.audit import log_tool_calls
from tools.search import internet_search

agent = define_deep_agent(
    name="research-assistant",   # static; starts with a letter; letters, digits, _ and -
    model="anthropic:claude-sonnet-5",
    tools=[internet_search],
    middleware=[log_tool_calls, TodoListMiddleware()],
    subagents=[{"name": "critic", "description": "Reviews drafts", "system_prompt": "You are a strict reviewer."}],
    interrupt_on={"write_file": True},
)
```

`define_deep_agent` accepts `name`, `model`, `tools`, `middleware`, `subagents`, `permissions`,
`interrupt_on`, `response_format`, `context_schema`, `debug`, `cache` and `metadata`, plus inline
`instructions`, `mcp`, `sandbox`, `skills` and `memory` overrides for what the project files
normally declare. It does not accept `state_schema`, `store`, `checkpointer` or `backend`: the
platform owns persistence and the filesystem.

### MCP Servers and Connections

```python
# tools/mcp.py
from managed_deepagents import bearer, connections, define_mcp

mcp = define_mcp(servers={
    "linear": {
        "transport": "http",
        "url": "https://mcp.linear.app/mcp",
        "headers": {"Authorization": bearer(connections.get("linear", {"type": "agent"}))},
        "include_tools": ["list_issues"],
    },
})
```

Connections are workspace secrets and OAuth grants stored by LangSmith, so no token lives in code:

```bash
mda connections create linear --secret-from-env LINEAR_API_KEY   # opaque secret
mda connections create github --oauth github --client-id CLIENT_ID
mda connections list
```

- `connections.get(slug, {"type": "agent"})` uses one credential for the whole deployment;
  `{"type": "user"}` uses the calling user's own grant (requires identity).
- `bearer(ref)` and `basic(username, ref)` build headers for `define_mcp`.
- Inside a tool, await the reference to read the value: `token = await connections.get("github", {"type": "agent"})`.

### Memory

```python
# memory.py
from managed_deepagents import MemoryLayer, define_memory

memory = define_memory(
    agent=MemoryLayer(),   # /memories/agent/, shared by every caller
    user=MemoryLayer(),    # /memories/user/, per caller; needs identity
)
```

- Omitted layers are disabled, and a project without `memory.py` keeps nothing between runs.
- Each layer's `AGENTS.md` is hot memory, loaded every run; other files are read on demand.
- Without a policy, user memory is enabled only in managed Slack one-to-one DMs (and for verified
  Studio users). Pass `MemoryLayer(allow=fn)`, where `fn(context)` returns a bool, to decide per run.
- Content lives in LangSmith Context Hub, so deploys never overwrite what the agent learned.
- Agent memory is a trust boundary: whatever one caller gets the agent to save is read by every
  later caller. Keep authority (tools, approvals) in `agent.py`, not in memory.
- `scope=` is the legacy form; use the `agent=` and `user=` layers.

### Identity

```python
# identity.py
from managed_deepagents import auth, define_identity

identity = define_identity(auth=auth.langsmith_api_key())   # callers send x-api-key
```

`auth.supabase(project_ref=...)` authenticates end users with Supabase access tokens. Identity gives every
caller private threads and per-user connections.

### Sandbox

```python
# sandbox/__init__.py
from managed_deepagents import define_sandbox

sandbox = define_sandbox(idle_ttl_seconds=600, default_timeout=600)
```

The sandbox gives the agent `execute` and an isolated filesystem, reused across a thread's turns.
Customize it with `docker_image=`, `snapshot_name=` or `proxy_config=`, or add `sandbox/setup.sh`:
`mda dev` and `mda deploy` run it once and new threads clone the snapshot. Delete `sandbox/` to run
without one.

### Schedules

```python
# schedules/daily_digest.py
from managed_deepagents import define_schedule

schedule = define_schedule(
    cron="0 8 * * 1-5",
    timezone="America/Los_Angeles",
    prompt="Summarize open research questions for today.",
)
```

`define_schedule` also takes `input=`, `thread=`, `context=` and `deliver_to=` (for example a Slack
conversation once a Slack channel is set up with `mda channels init slack`).

### Calling a Deployed Agent

A deployment is a LangGraph server, so use the LangGraph SDK:

```python
from langgraph_sdk import get_client

client = get_client(url="https://<deployment-url>", api_key="<LANGSMITH_API_KEY>")
thread = await client.threads.create()
async for chunk in client.runs.stream(thread["thread_id"], "research-assistant",
                                      input={"messages": [{"role": "user", "content": "Hi"}]},
                                      stream_mode="updates"):
    print(chunk.event, chunk.data)
```

### Migrating from create_deep_agent

| `create_deep_agent` | Managed Deep Agents |
|---------------------|---------------------|
| `system_prompt=` | `instructions.md` |
| `skills=[...]` | `skills/<name>/SKILL.md` |
| `memory=[...]` and `StoreBackend` routes | `memory.py` (`define_memory`) |
| `backend=` sandbox | `sandbox/` (`define_sandbox`) |
| `MCPAdapter` tools | `tools/mcp.py` (`define_mcp`) |
| API keys in env vars | `mda connections` + `connections.get(...)` |
| `checkpointer=`, `store=` | Managed by the platform |
| Cron jobs you run yourself | `schedules/` (`define_schedule`) |

Tools, middleware, subagents, `interrupt_on`, `permissions` and `response_format` move over as-is.

## Anti-Patterns

- **Expecting `write_todos` by default**: since 0.7 planning needs `TodoListMiddleware()`
- **Writing memory or skill paths the backend cannot see**: `skills=` and `memory=` resolve through the backend
- **`StoreBackend` without `store=`**: cross-thread files need a store (and a persistent one in production)
- **Relying on `FilesystemPermission` to limit `execute`**: permissions cover filesystem tools only
- **`LocalShellBackend` or `FilesystemBackend(virtual_mode=False)` with untrusted input**: both reach your real machine
- **Vague subagent descriptions**: the main agent picks subagents from `description` alone
- **Passing `checkpointer`, `store` or `backend` to `define_deep_agent`**: Managed Deep Agents owns them
- **Hard-coding tokens in `tools/` or `define_mcp` headers**: use `mda connections` and `bearer(connections.get(...))`
- **Treating shared agent memory as trusted**: any caller can write to it

## Key Links

- Deep Agents overview: https://docs.langchain.com/oss/python/deepagents/overview
- Customization: https://docs.langchain.com/oss/python/deepagents/customization
- Subagents: https://docs.langchain.com/oss/python/deepagents/subagents
- Async subagents: https://docs.langchain.com/oss/python/deepagents/async-subagents
- Backends: https://docs.langchain.com/oss/python/deepagents/backends
- Permissions: https://docs.langchain.com/oss/python/deepagents/permissions
- Sandboxes: https://docs.langchain.com/oss/python/deepagents/sandboxes
- Managed Deep Agents overview: https://docs.langchain.com/langsmith/python/managed-deep-agents-overview
- Managed Deep Agents CLI: https://docs.langchain.com/langsmith/python/managed-deep-agents-cli
- Migrate to Managed Deep Agents: https://docs.langchain.com/langsmith/python/managed-deep-agents-migrate
