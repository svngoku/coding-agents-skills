# Multi-Agent Patterns

LangChain v1 has no special multi-agent classes. Every pattern is built from `create_agent`,
`@tool`, `Command` and LangGraph's `StateGraph`/`Send`, or from the `deepagents` harness.
Modules such as `langchain.agents.handoffs`, `langchain.agents.router` or
`langchain.agents.subagents` do not exist. `langgraph-supervisor` is superseded by the
subagents pattern below.

## Choosing a Pattern

| Pattern | How it works | Use when |
|---------|--------------|----------|
| Subagents (agent as tool) | A supervisor calls workers through tools and keeps control | Clear subtasks; you want isolated worker context |
| Handoffs | A tool returns `Command(goto=..., graph=Command.PARENT)` to switch the active agent | The conversation should move to a specialist that talks to the user directly |
| Router | Classify once, fan out with `Send`, then synthesize | Independent knowledge verticals queried in parallel |
| Skills | One agent loads instructions/tools on demand | Many capabilities but one conversation; keep the prompt small |
| Deep Agents | `create_deep_agent` with planning, files and a `task` tool for subagents | Long, open-ended research or coding tasks |

## Subagents (Agent as Tool)

```python
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

research_agent = create_agent(
    model="anthropic:claude-sonnet-5",
    tools=[web_search],
    system_prompt="You are a research expert. Answer with sources.",
)

writer_agent = create_agent(
    model="anthropic:claude-sonnet-5",
    tools=[],
    system_prompt="You turn research notes into a clear report.",
)

@tool("research_expert", description="Research a topic on the web and return findings with sources.")
def call_research_agent(query: str) -> str:
    result = research_agent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content

@tool("writer", description="Write a report from research notes.")
def call_writer(notes: str) -> str:
    result = writer_agent.invoke({"messages": [{"role": "user", "content": notes}]})
    return result["messages"][-1].content

supervisor = create_agent(
    model="anthropic:claude-sonnet-5",
    tools=[call_research_agent, call_writer],
    system_prompt="Research first with research_expert, then hand the notes to writer.",
    checkpointer=InMemorySaver(),
)
```

Return only the final message (`result["messages"][-1].content`), not the whole result dict, so the
supervisor's context stays small. To pass user data down, take `runtime: ToolRuntime` in the
wrapper tool and forward `context=runtime.context` to the worker's `invoke`.

## Handoffs

Each agent is a node in a parent graph. A handoff tool updates `active_agent` and jumps to the
other node in the parent graph. Pass the triggering `AIMessage` plus a matching `ToolMessage` so
the tool-call pair stays valid:

```python
from typing import Literal
from typing_extensions import NotRequired
from langchain.agents import AgentState, create_agent
from langchain.messages import AIMessage, ToolMessage
from langchain.tools import tool, ToolRuntime
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

class MultiAgentState(AgentState):
    active_agent: NotRequired[str]

def make_handoff(target: str):
    @tool(f"transfer_to_{target}", description=f"Transfer the conversation to the {target} agent.")
    def handoff(runtime: ToolRuntime) -> Command:
        last_ai = next(m for m in reversed(runtime.state["messages"]) if isinstance(m, AIMessage))
        done = ToolMessage(content=f"Transferred to {target}", tool_call_id=runtime.tool_call_id)
        return Command(goto=target, update={"active_agent": target, "messages": [last_ai, done]},
                       graph=Command.PARENT)
    return handoff

sales = create_agent("anthropic:claude-sonnet-5", tools=[make_handoff("support")],
                     system_prompt="You are a sales agent. Transfer technical issues to support.")
support = create_agent("anthropic:claude-sonnet-5", tools=[make_handoff("sales")],
                       system_prompt="You are a support agent. Transfer pricing questions to sales.")

def route_after(state: MultiAgentState) -> Literal["sales", "support", "__end__"]:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and not last.tool_calls:
        return "__end__"
    return state.get("active_agent") or "sales"

builder = StateGraph(MultiAgentState)
builder.add_node("sales", lambda s: sales.invoke(s))
builder.add_node("support", lambda s: support.invoke(s))
builder.add_conditional_edges(START, lambda s: s.get("active_agent") or "sales", ["sales", "support"])
builder.add_conditional_edges("sales", route_after, ["sales", "support", END])
builder.add_conditional_edges("support", route_after, ["sales", "support", END])
graph = builder.compile()   # add checkpointer=... for multi-turn conversations
```

When a subgraph updates a key shared with the parent via `Command.PARENT`, the parent state needs
a reducer for that key (`messages` already has one in `AgentState`).

## Router (Classify and Fan Out)

```python
import operator
from typing import Annotated, Literal, TypedDict
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

class Classification(TypedDict):
    source: Literal["github", "notion"]
    query: str

class RouterState(TypedDict):
    query: str
    classifications: list[Classification]
    results: Annotated[list[dict], operator.add]   # parallel branches append here
    final_answer: str

class ClassificationResult(BaseModel):
    classifications: list[Classification] = Field(description="Agents to call with targeted sub-questions")

router_llm = init_chat_model("openai:gpt-5.4-mini")

def classify(state: RouterState) -> dict:
    result = router_llm.with_structured_output(ClassificationResult).invoke(
        [{"role": "system", "content": "Pick the relevant sources (github: code, notion: docs) and write a sub-question for each."},
         {"role": "user", "content": state["query"]}])
    return {"classifications": result.classifications}

def fan_out(state: RouterState) -> list[Send]:
    return [Send(c["source"], {"query": c["query"]}) for c in state["classifications"]]

def query_github(state: dict) -> dict:
    answer = github_agent.invoke({"messages": [{"role": "user", "content": state["query"]}]})
    return {"results": [{"source": "github", "result": answer["messages"][-1].content}]}

def query_notion(state: dict) -> dict:
    answer = notion_agent.invoke({"messages": [{"role": "user", "content": state["query"]}]})
    return {"results": [{"source": "notion", "result": answer["messages"][-1].content}]}

def synthesize(state: RouterState) -> dict:
    notes = "\n\n".join(f"From {r['source']}:\n{r['result']}" for r in state["results"])
    return {"final_answer": router_llm.invoke(f"Answer '{state['query']}' using:\n{notes}").content}

workflow = (
    StateGraph(RouterState)
    .add_node("classify", classify)
    .add_node("github", query_github)
    .add_node("notion", query_notion)
    .add_node("synthesize", synthesize)
    .add_edge(START, "classify")
    .add_conditional_edges("classify", fan_out, ["github", "notion"])
    .add_edge("github", "synthesize")
    .add_edge("notion", "synthesize")
    .add_edge("synthesize", END)
    .compile()
)
```

## Skills (Progressive Disclosure)

Keep one agent, but load detailed instructions only when needed:

```python
from langchain.agents import create_agent
from langchain.tools import tool

SKILLS = {
    "sql": "You can query Postgres with run_query. Always LIMIT results and never run DDL.",
    "billing": "Refunds over $100 need approval. Use lookup_invoice before issue_refund.",
}

@tool
def load_skill(name: str) -> str:
    """Load the full instructions for a skill. Available skills: sql, billing."""
    return SKILLS.get(name, f"Unknown skill '{name}'. Available: {', '.join(SKILLS)}")

agent = create_agent(
    model="anthropic:claude-sonnet-5",
    tools=[load_skill, run_query, lookup_invoice, issue_refund],
    system_prompt="Before using a domain's tools, call load_skill for that domain.",
)
```

To expose tools only after a skill is loaded, record loaded skills in state (a `Command` update
from `load_skill`) and filter `request.tools` in a `@wrap_model_call` middleware (see
[middleware.md](middleware.md)). Deep Agents supports file-based `SKILL.md` skills natively.

## Deep Agents

`create_deep_agent` (package `deepagents`) is `create_agent` plus a default middleware stack: a
virtual filesystem (`ls`, `read_file`, `write_file`, `edit_file`, `delete`, `glob`, `grep`), a
`task` tool that spawns subagents (a `general-purpose` subagent is added automatically),
summarization and provider prompt caching. Skills, memory files and human approval are opt-in
parameters. Since 0.7 the to-do list is not in the default stack: add `TodoListMiddleware()` if
you want the `write_todos` planning tool.

```python
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langchain.agents.middleware import TodoListMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

research_subagent = {
    "name": "research-agent",
    "description": "Researches one question in depth and returns a summary with sources.",
    "system_prompt": "You are a careful researcher.",
    "tools": [internet_search],
    "model": "openai:gpt-5.5",   # optional; defaults to the main model
}

agent = create_deep_agent(
    model="anthropic:claude-sonnet-5",
    tools=[internet_search],
    system_prompt="Plan with write_todos, delegate research, and write the final report to /report.md.",
    subagents=[research_subagent],
    middleware=[TodoListMiddleware()],      # adds write_todos
    skills=["/skills/"],                    # directories containing <skill>/SKILL.md
    memory=["/memories/AGENTS.md"],         # files loaded into the prompt each run
    backend=CompositeBackend(
        default=StateBackend(),             # per-thread scratch files (in agent state)
        routes={"/memories/": StoreBackend(namespace=lambda rt: ("memories",))},   # cross-thread
    ),
    store=InMemoryStore(),
    interrupt_on={"write_file": True, "read_file": False},   # human approval per tool
    checkpointer=InMemorySaver(),          # required for interrupts
)

result = agent.invoke({"messages": [{"role": "user", "content": "Compare LangGraph and CrewAI"}]},
                      config={"configurable": {"thread_id": "r1"}})
```

- Subagent dicts accept `name`, `description`, `system_prompt`, `tools`, `model`, `middleware`,
  `interrupt_on`, `skills`, `permissions` and `response_format`.
- Wrap an existing graph as a subagent with `{"name": ..., "description": ..., "runnable": compiled_graph}`
  (`CompiledSubAgent`), or point to a remote LangGraph deployment with `AsyncSubAgent`
  (`graph_id`, `url`).
- Backends: `StateBackend` (default, in-state files), `FilesystemBackend(root_dir=..., virtual_mode=True)`
  (real disk, sandboxed to `root_dir`), `StoreBackend(namespace=...)` (LangGraph store),
  `CompositeBackend(default, routes)`, `LocalShellBackend` and sandbox backends that add an
  `execute` tool.
- Seed the default `StateBackend` per run with `{"files": {"/path": create_file_data(text)}}`
  (`deepagents.backends.utils`).

Permissions, sandboxes, the code interpreter, fork and async subagents, subagent streaming,
harness profiles and Managed Deep Agents (`mda`) are covered in [deep-agents.md](deep-agents.md).

## Anti-Patterns

- Importing invented helpers (`with_handoffs`, `create_router`, `create_subagent_tool`, `create_skill_manager`): use the patterns above
- Returning the whole worker result dict from a subagent tool: return the final message text
- Handoffs that forward only a `ToolMessage` without its `AIMessage`: providers reject orphaned tool results
- Router state without a reducer on the results key: parallel `Send` branches overwrite each other
- Reaching for multi-agent when one agent with good tools, middleware and skills would do
