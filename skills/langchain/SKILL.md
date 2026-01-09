---
name: langchain
description: Build AI agents with LangChain framework. Use when building agents, tools, memory, MCP integrations, RAG pipelines, multi-agent systems, or any LLM-powered applications using LangChain or LangGraph in Python or TypeScript.
---

# LangChain Skill

Build production AI agents using LangChain's pre-built architecture and integrations.

## Installation

```bash
pip install langchain langchain-anthropic langgraph
# For MCP support
pip install langchain-mcp-adapters
```

## Quick Agent

```python
from langchain.agents import create_agent
from langchain.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Sunny in {city}!"

agent = create_agent(
    model="claude-sonnet-4-5-20250929",
    tools=[get_weather],
    system_prompt="You are a helpful assistant"
)

result = agent.invoke({"messages": [{"role": "user", "content": "Weather in SF?"}]})
```

## Tools

Define tools with `@tool` decorator. Type hints required, docstring becomes description:

```python
from langchain.tools import tool, ToolRuntime
from dataclasses import dataclass

@tool
def search(query: str, limit: int = 10) -> str:
    """Search database for records."""
    return f"Found {limit} results for '{query}'"

# With runtime context access
@dataclass
class Context:
    user_id: str

@tool
def get_user_data(runtime: ToolRuntime[Context]) -> str:
    """Get current user data."""
    user_id = runtime.context.user_id
    return f"Data for {user_id}"
```

### ToolRuntime Access

- `runtime.state` - Agent state (messages, custom fields)
- `runtime.context` - Immutable config (user_id, session)
- `runtime.store` - Persistent memory across conversations
- `runtime.stream_writer` - Stream updates during execution

### Reserved Parameters
Cannot use as tool args: `config`, `runtime`

## Memory

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

# Short-term (conversation) memory
checkpointer = InMemorySaver()

# Long-term (cross-conversation) memory
store = InMemoryStore()

agent = create_agent(
    model="claude-sonnet-4-5-20250929",
    tools=[...],
    checkpointer=checkpointer,
    store=store
)

# Use thread_id for conversation continuity
config = {"configurable": {"thread_id": "conversation-1"}}
agent.invoke({"messages": [...]}, config=config)
```

## Structured Output

```python
from dataclasses import dataclass
from langchain.agents.structured_output import ToolStrategy

@dataclass
class Response:
    answer: str
    confidence: float | None = None

agent = create_agent(
    model="claude-sonnet-4-5-20250929",
    tools=[...],
    response_format=ToolStrategy(Response)
)
# Access: result['structured_response']
```

## MCP Integration

Connect to MCP servers for external tools:

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

client = MultiServerMCPClient({
    "math": {
        "transport": "stdio",
        "command": "python",
        "args": ["/path/to/server.py"]
    },
    "api": {
        "transport": "http",
        "url": "http://localhost:8000/mcp",
        "headers": {"Authorization": "Bearer TOKEN"}
    }
})

tools = await client.get_tools()
agent = create_agent("claude-sonnet-4-5-20250929", tools)
```

### MCP Interceptors

Modify tool calls, inject context, handle errors:

```python
from langchain_mcp_adapters.interceptors import MCPToolCallRequest

async def auth_interceptor(request: MCPToolCallRequest, handler):
    runtime = request.runtime
    user_id = runtime.context.user_id
    modified = request.override(args={**request.args, "user_id": user_id})
    return await handler(modified)

client = MultiServerMCPClient({...}, tool_interceptors=[auth_interceptor])
```

## Model Configuration

```python
from langchain.chat_models import init_chat_model

model = init_chat_model(
    "claude-sonnet-4-5-20250929",  # or "gpt-4o", "gemini-1.5-pro"
    temperature=0.5,
    timeout=10,
    max_tokens=1000
)
```

## State Updates from Tools

Return `Command` to update state or control flow:

```python
from langgraph.types import Command
from langchain.messages import RemoveMessage

@tool
def clear_history() -> Command:
    """Clear conversation."""
    return Command(update={"messages": [RemoveMessage(id="__all__")]})

@tool  
def complete_task() -> Command:
    """End agent run."""
    return Command(update={"status": "done"}, goto="__end__")
```

## Common Patterns

### Full Production Agent

```python
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
from langchain.agents.structured_output import ToolStrategy
from langgraph.checkpoint.memory import InMemorySaver

@dataclass
class Context:
    user_id: str

@tool
def search_orders(query: str, runtime: ToolRuntime[Context]) -> str:
    """Search user orders."""
    return f"Orders for {runtime.context.user_id}: {query}"

@dataclass
class Response:
    answer: str
    order_ids: list[str] | None = None

agent = create_agent(
    model=init_chat_model("claude-sonnet-4-5-20250929", temperature=0),
    system_prompt="You are a customer service agent.",
    tools=[search_orders],
    context_schema=Context,
    response_format=ToolStrategy(Response),
    checkpointer=InMemorySaver()
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "Find my recent orders"}]},
    config={"configurable": {"thread_id": "session-1"}},
    context=Context(user_id="user123")
)
```

## References

For detailed patterns, see:
- `references/langgraph.md` - LangGraph workflows, graphs, persistence
- `references/multi-agent.md` - Handoffs, routers, subagents
- `references/retrieval.md` - RAG, vector stores, embeddings

## Key Links

- Docs: https://docs.langchain.com
- Python SDK: https://reference.langchain.com/python
- LangGraph: https://docs.langchain.com/oss/python/langgraph
- MCP Adapters: https://github.com/langchain-ai/langchain-mcp-adapters