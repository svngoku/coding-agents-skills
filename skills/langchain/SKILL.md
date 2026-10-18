---
name: langchain
description: >
  Build AI agents with LangChain v1 and LangGraph in Python. Use this skill whenever the user
  mentions LangChain, LangGraph, create_agent, init_chat_model, @tool, ToolRuntime, agent
  middleware, human-in-the-loop, structured output, MCP tools (langchain.mcp, MCPAdapter),
  RAG, multi-agent handoffs or subagents, checkpointers, Deep Agents (create_deep_agent) or
  Managed Deep Agents (mda, define_deep_agent). Also trigger for migrating from
  create_react_agent, LLMChain, AgentExecutor or langchain-mcp-adapters, stream_events,
  SummarizationMiddleware, and langchain-classic.
---

# LangChain Skill

Build production AI agents with LangChain v1 (`create_agent` on top of LangGraph), its middleware
system, built-in MCP support and the LangGraph runtime. Verified against `langchain` 1.4.6,
`langchain-core` 1.6, `langgraph` 1.2 and `deepagents` 0.7 (Python 3.10+; `deepagents` needs 3.11+).

## Quick Reference

| Topic | File |
|-------|------|
| Built-in and custom middleware | [references/middleware.md](references/middleware.md) |
| LangGraph graphs, persistence, durability, streaming | [references/langgraph.md](references/langgraph.md) |
| Subagents, handoffs, skills, routers | [references/multi-agent.md](references/multi-agent.md) |
| Deep Agents harness and Managed Deep Agents (`mda`) | [references/deep-agents.md](references/deep-agents.md) |
| RAG, embeddings, vector stores, retrievers | [references/retrieval.md](references/retrieval.md) |

## Installation

```bash
pip install -U langchain langgraph langchain-anthropic   # or langchain-openai, langchain-google-genai, ...
pip install -U "langchain[mcp]"                          # built-in MCP client (langchain.mcp, beta)
pip install -U deepagents                                # optional: planning + filesystem + subagents harness
pip install -U langchain-classic                         # only for legacy chains/retrievers
```

## Quick Agent

```python
from langchain.agents import create_agent

def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Sunny in {city}!"

agent = create_agent(
    model="anthropic:claude-sonnet-5",
    tools=[get_weather],
    system_prompt="You are a helpful assistant.",
)

result = agent.invoke({"messages": [{"role": "user", "content": "Weather in SF?"}]})
print(result["messages"][-1].content)
```

`create_agent` returns a compiled LangGraph graph, so `invoke`, `stream`, `stream_events`,
checkpointers, stores and interrupts all work on it. Plain functions with type hints and a
docstring are accepted as tools; use `@tool` when you need a custom name, schema or `ToolRuntime`.

## Model Configuration

```python
from langchain.chat_models import init_chat_model

model = init_chat_model(
    "anthropic:claude-sonnet-5",   # "provider:model"; e.g. "openai:gpt-5.5", "google_genai:gemini-3.6-flash"
    temperature=0, timeout=30, max_tokens=2000, max_retries=2,
)

model.profile["max_input_tokens"]    # model capabilities (from models.dev): context size, tool calling, modalities
```

Pass either the string or the model instance as `model=` to `create_agent`. `system_prompt` accepts
a `str` or a `SystemMessage` (useful for content blocks such as provider cache-control markers).

## Tools

The docstring becomes the description the model sees, and type hints become the argument schema:

```python
from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime

@tool
def search(query: str, limit: int = 10) -> str:
    """Search the order database. Returns up to `limit` matching records."""
    return f"Found {limit} results for '{query}'"

@dataclass
class Context:
    user_id: str

@tool
def get_user_data(runtime: ToolRuntime[Context]) -> str:
    """Get the current user's profile."""
    return f"Data for {runtime.context.user_id}"
```

`ToolRuntime` is injected and hidden from the model. It exposes `state` (messages and custom fields),
`context` (immutable per-run data passed as `agent.invoke(..., context=Context(...))`), `store`
(long-term memory), `stream_writer` (custom stream updates), `tool_call_id` and `config`.
`config` and `runtime` are reserved argument names. Finer-grained injectables also live in
`langchain.tools`: `InjectedState`, `InjectedStore`, `InjectedToolCallId`, `InjectedToolArg`.

## Memory

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

agent = create_agent(
    model="anthropic:claude-sonnet-5",
    tools=[search],
    checkpointer=InMemorySaver(),   # short-term: per-thread conversation state
    store=InMemoryStore(),          # long-term: cross-thread memory via runtime.store
)

config = {"configurable": {"thread_id": "conversation-1"}}
agent.invoke({"messages": [{"role": "user", "content": "Hi, I'm Ana"}]}, config=config)
```

In production use `PostgresSaver`/`AsyncPostgresSaver` or `SqliteSaver` ([references/langgraph.md](references/langgraph.md)),
and keep long threads in budget with `SummarizationMiddleware`.

## Structured Output

```python
from pydantic import BaseModel
from langchain.agents.structured_output import ToolStrategy, ProviderStrategy

class Response(BaseModel):
    answer: str
    confidence: float | None = None

agent = create_agent(model="anthropic:claude-sonnet-5", tools=[search],
                     response_format=ToolStrategy(Response))
result = agent.invoke({"messages": [{"role": "user", "content": "..."}]})   # result["structured_response"]: Response
```

| `response_format` | Behavior |
|-------------------|----------|
| `Response` (bare type) | Auto: provider-native JSON schema when the model profile supports it, else tool calling |
| `ToolStrategy(Response, handle_errors=True)` | Structured output via an artificial tool call; retries on validation errors; works with any tool-calling model |
| `ProviderStrategy(Response, strict=True)` | Provider-native structured output (OpenAI, Anthropic, Gemini, ...) |

Schemas can be Pydantic models, dataclasses, `TypedDict`s or JSON schema dicts. Outside agents, use `model.with_structured_output(Response)`.

## Middleware

Middleware is the v1 extension point that replaces pre/post hooks. Built-ins live in `langchain.agents.middleware`:

```python
from langchain.agents import create_agent
from langchain.agents.middleware import (
    SummarizationMiddleware, HumanInTheLoopMiddleware, ModelFallbackMiddleware,
    ToolCallLimitMiddleware, PIIMiddleware,
)
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model="anthropic:claude-sonnet-5",
    tools=[search, send_email],
    checkpointer=InMemorySaver(),   # required for interrupts
    middleware=[
        SummarizationMiddleware(model="openai:gpt-5.4-mini", trigger=("fraction", 0.8), keep=("messages", 20)),
        ModelFallbackMiddleware("openai:gpt-5.5"),
        ToolCallLimitMiddleware(tool_name="search", run_limit=5),
        PIIMiddleware("email", strategy="redact", apply_to_input=True),
        HumanInTheLoopMiddleware(interrupt_on={"send_email": {"allowed_decisions": ["approve", "edit", "reject"]}}),
    ],
)
```

Custom middleware uses decorators (`before_agent`, `before_model`, `after_model`, `after_agent`,
`wrap_model_call`, `wrap_tool_call`, `dynamic_prompt`) or an `AgentMiddleware` subclass:

```python
from langchain.agents.middleware import dynamic_prompt, wrap_tool_call, ModelRequest

@dynamic_prompt
def tiered_prompt(request: ModelRequest) -> str:
    return f"You are a support agent. Customer tier: {request.runtime.context.tier}."

@wrap_tool_call
def audit_tool_calls(request, handler):
    print("calling", request.tool_call["name"], request.tool_call["args"])
    return handler(request)
```

Full catalogue (retries, tool selection, context editing, to-do lists, shell, emulators) and
patterns (dynamic models, dynamic tools, jumps) are in [references/middleware.md](references/middleware.md).

## Human-in-the-Loop

With `HumanInTheLoopMiddleware` and a checkpointer, the run pauses before the listed tools.
Resume with one decision per pending action:

```python
from langgraph.types import Command

config = {"configurable": {"thread_id": "t1"}}
result = agent.invoke({"messages": [{"role": "user", "content": "Email Bob the report"}]},
                      config=config, version="v2")
while result.interrupts:
    actions = result.interrupts[0].value["action_requests"]
    for action in actions:
        print("Pending:", action["name"], action["args"])
    decisions = [{"type": "approve"} for _ in actions]   # one decision per action, in order
    result = agent.invoke(Command(resume={"decisions": decisions}), config=config, version="v2")
```

Decision types: `{"type": "approve"}`, `{"type": "edit", "edited_action": {"name": ..., "args": {...}}}`,
`{"type": "reject", "message": "why"}`, and `{"type": "respond", "message": "..."}`. With
`version="v2"`, `invoke` returns a `GraphOutput` (`.value`, `.interrupts`) instead of a plain dict.

## Streaming

Use event streaming (`version="v3"`) for apps and UIs. It returns a run object with typed projections:

```python
stream = agent.stream_events({"messages": [{"role": "user", "content": "Weather in SF?"}]}, version="v3")
for kind, item in stream.interleave("messages", "tool_calls"):
    if kind == "messages":
        for token in item.text:
            print(token, end="", flush=True)
    else:
        print(f"\n[tool] {item.tool_name}({item.input})")
        for delta in item.output_deltas:   # drain first: item.output is None until the tool finishes
            print(delta, end="", flush=True)
        print(f"\n[result] {item.output}")
final_state = stream.output
```

Event streaming is experimental in LangGraph 1.2 (it emits a `LangChainBetaWarning`). Classic
`agent.stream(input, stream_mode=["updates", "messages", "custom"])` still works; the model node is `"model"`.

## MCP Integration

LangChain 1.4 ships a built-in MCP client, `langchain.mcp` (beta, built on FastMCP). It replaces
`langchain-mcp-adapters`' `MultiServerMCPClient`:

```python
from langchain.agents import create_agent
from langchain.mcp import MCPAdapter

config = {"mcpServers": {
    "math": {"command": "python", "args": ["/path/to/math_server.py"]},             # stdio
    "api": {"url": "https://api.example.com/mcp", "headers": {"Authorization": "Bearer TOKEN"}},
}}

async def build_agent():
    async with MCPAdapter(config) as adapter:     # or MCPAdapter("https://..."), MCPAdapter(Path("server.py"))
        tools = await adapter.list_tools()        # multi-server tools are prefixed: math_..., api_...
    return create_agent("anthropic:claude-sonnet-5", tools)
```

- A plain string target must be an `http(s)` URL; reach local servers with `pathlib.Path(...)` or a config.
- Tools stay usable after the `async with` block exits. They are async, so run the agent with `ainvoke`/`astream`.
- Server elicitation requests become LangGraph `interrupt()`s, so add a checkpointer.
- Interceptors are gone: use `@wrap_tool_call` middleware (MCP provenance is in `request.tool.metadata["mcp"]`).

## State Updates from Tools

Return a `Command` to update state or control flow:

```python
from langchain.messages import RemoveMessage, ToolMessage
from langchain.tools import tool, ToolRuntime
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.types import Command

@tool
def clear_history() -> Command:
    """Clear the conversation history."""
    return Command(update={"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]})

@tool
def set_language(language: str, runtime: ToolRuntime) -> Command:
    """Remember the user's preferred language."""
    return Command(update={
        "language": language,   # custom field: needs state_schema= (a TypedDict extending AgentState)
        "messages": [ToolMessage(f"Language set to {language}", tool_call_id=runtime.tool_call_id)],
    })
```

## Deep Agents

`deepagents` wraps `create_agent` with a virtual filesystem, a `task` tool for subagents,
summarization, and optional skills, memory files, permissions and sandboxes, for long multi-step
tasks. Managed Deep Agents (`managed-deepagents`, CLI `mda`) deploys the same harness to LangSmith
from a project directory (`agent.py` with `define_deep_agent`, `instructions.md`, `memory.py`,
`sandbox/`, `schedules/`). Both are covered in [references/deep-agents.md](references/deep-agents.md).

```python
from deepagents import create_deep_agent

agent = create_deep_agent(
    model="anthropic:claude-sonnet-5",
    tools=[internet_search],
    system_prompt="You are an expert researcher. Write the final report to /report.md.",
    subagents=[{"name": "critic", "description": "Reviews drafts", "system_prompt": "You are a strict reviewer."}],
)
```

## Full Production Agent

```python
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware, ToolRetryMiddleware
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver

@dataclass
class Context:
    user_id: str

@tool
def search_orders(query: str, runtime: ToolRuntime[Context]) -> str:
    """Search the current user's orders by keyword, date or status."""
    return f"Orders for {runtime.context.user_id}: {query}"

@dataclass
class Response:
    answer: str
    order_ids: list[str] | None = None

agent = create_agent(
    model=init_chat_model("anthropic:claude-sonnet-5", temperature=0),
    system_prompt="You are a customer service agent.",
    tools=[search_orders],
    context_schema=Context,
    response_format=ToolStrategy(Response),
    middleware=[ToolRetryMiddleware(max_retries=2), SummarizationMiddleware("openai:gpt-5.4-mini", trigger=("tokens", 4000))],
    checkpointer=InMemorySaver(),
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "Find my recent orders"}]},
    config={"configurable": {"thread_id": "session-1"}},
    context=Context(user_id="user123"),
)
print(result["structured_response"])
```

## Migrating to v1

| Old (0.x / pre-1.4) | Current |
|---------------------|---------|
| `langgraph.prebuilt.create_react_agent` | `langchain.agents.create_agent` |
| `prompt=` | `system_prompt=` (static) or `@dynamic_prompt` middleware |
| `pre_model_hook` / `post_model_hook` | `@before_model` / `@after_model` middleware |
| `config["configurable"]` for user data | `context_schema=` + `invoke(..., context=...)` |
| `AgentExecutor`, `LLMChain`, `RetrievalQA`, `langchain.retrievers` | `create_agent`, or `langchain_classic.*` for legacy code |
| `langchain.text_splitter` | `langchain_text_splitters` |
| `langchain_mcp_adapters.client.MultiServerMCPClient` | `langchain.mcp.MCPAdapter` |
| MCP `tool_interceptors` | `@wrap_tool_call` middleware |
| Prompted structured output | `ToolStrategy` / `ProviderStrategy` |
| `message.content` parsing for reasoning/citations | `message.content_blocks` (standard blocks) |

## Anti-Patterns to Avoid

- **Tools without type hints or docstrings**: the docstring *is* the tool description the model sees; a missing or vague one means the agent misuses the tool
- **Global state in tools**: read per-user data from `ToolRuntime[Context]`, not globals, so tools stay testable and safe in concurrent runs
- **No checkpointer, then wondering why the agent forgets**: multi-turn agents, human-in-the-loop and MCP elicitation all need a checkpointer plus a stable `thread_id`
- **Hand-rolled hook loops**: use middleware (`SummarizationMiddleware`, `ModelFallbackMiddleware`, `ToolRetryMiddleware`, limits) instead of wrapping `agent.invoke` in custom retry/trim code
- **Letting raw tool exceptions reach the model or user**: use `ToolErrorMiddleware(on_error=...)` or `ToolRetryMiddleware`, and return the exception type rather than internal details
- **Importing removed modules**: `langchain.vectorstores`, `langchain.document_loaders`, `langchain.text_splitter`, `langchain.retrievers` and `LLMChain` no longer exist in `langchain` v1; use provider packages, `langchain_core`, `langchain_text_splitters` or `langchain_classic`
- **Starting new code on `langchain-mcp-adapters` or `create_react_agent`**: both are superseded by `langchain.mcp` and `create_agent`
- **Passing a local script path as a string to `MCPAdapter`**: strings must be URLs; use `Path(...)` or an `mcpServers` config for stdio servers
- **Stuffing whole documents into context**: use the retrieval patterns in [references/retrieval.md](references/retrieval.md)

## When to Use / Not Use

**Use this skill when:**

- Building agents with LangChain v1 or LangGraph in Python
- Adding tools, memory, middleware, human approval, MCP servers or structured output to an LLM app
- Implementing RAG, subagents, handoffs, routers, Deep Agents (self-hosted or managed) or stateful workflows
- Migrating 0.x code (`create_react_agent`, `AgentExecutor`, chains, `langchain-mcp-adapters`)

**Do NOT use this skill when:**

- A single stateless LLM call is all you need; the plain provider SDK is lighter
- The project standardizes on another agent framework; follow the project's stack
- The agent is code-first and benefits from the smaller SmolAgents execution model; see the smolagents skill

## Key Links

- Docs: https://docs.langchain.com/oss/python/langchain/overview
- Middleware: https://docs.langchain.com/oss/python/langchain/middleware/built-in
- MCP: https://docs.langchain.com/oss/python/langchain/mcp/connections
- From langchain-mcp-adapters: https://docs.langchain.com/oss/python/migrate/langchain-mcp-adapters
- Event streaming: https://docs.langchain.com/oss/python/langchain/event-streaming
- Deep Agents: https://docs.langchain.com/oss/python/deepagents/overview
- Managed Deep Agents: https://docs.langchain.com/langsmith/python/managed-deep-agents-overview
- API reference: https://reference.langchain.com/python
