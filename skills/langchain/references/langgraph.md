# LangGraph Reference

LangGraph is the runtime under `create_agent`. Use it directly when you need custom control flow:
deterministic steps, parallel branches, loops, multi-agent graphs or long-running durable jobs.
Verified against `langgraph` 1.2.

## Installation

```bash
pip install -U langgraph
pip install -U langgraph-checkpoint-sqlite     # SqliteSaver
pip install -U langgraph-checkpoint-postgres   # PostgresSaver / PostgresStore
```

## Graph API

```python
from operator import add
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    items: Annotated[list[str], add]   # reducer: node outputs are appended, not overwritten
    step_count: int

def process(state: State) -> dict:
    return {"items": [f"step {state['step_count']}"], "step_count": state["step_count"] + 1}

def should_continue(state: State) -> Literal["process", "__end__"]:
    return "process" if state["step_count"] < 3 else END

builder = StateGraph(State)
builder.add_node("process", process)
builder.add_edge(START, "process")
builder.add_conditional_edges("process", should_continue)
graph = builder.compile()

graph.invoke({"items": [], "step_count": 0})
```

For chat-style graphs use `MessagesState` (a `messages` key with the `add_messages` reducer).

### Command: Update State and Route from a Node

```python
from langgraph.types import Command

def triage(state: State) -> Command[Literal["urgent", "normal"]]:
    target = "urgent" if "outage" in state["items"][-1] else "normal"
    return Command(update={"step_count": 0}, goto=target)
```

### Send: Dynamic Fan-Out (Map-Reduce)

```python
from langgraph.types import Send

def fan_out(state: dict) -> list[Send]:
    return [Send("summarize", {"doc": d}) for d in state["docs"]]

# builder.add_conditional_edges("load", fan_out, ["summarize"])
# Results from parallel branches need a reducer key (e.g. Annotated[list, add]) to be merged.
```

## Node Policies: Retries, Caching, Timeouts

```python
from langgraph.cache.memory import InMemoryCache
from langgraph.types import CachePolicy, RetryPolicy, TimeoutPolicy

builder.add_node(
    "fetch",
    fetch_node,
    retry_policy=RetryPolicy(max_attempts=3, initial_interval=1.0, retry_on=ConnectionError),
    cache_policy=CachePolicy(ttl=300),   # cache by node input for 5 minutes
    timeout=TimeoutPolicy(run_timeout=60, idle_timeout=15),   # async nodes only
)

# Or set defaults for every node in the graph
builder.set_node_defaults(retry_policy=RetryPolicy(max_attempts=2))

graph = builder.compile(cache=InMemoryCache())   # caching needs a cache backend
```

- Timeouts only apply to async nodes and raise `NodeTimeoutError`; the retry policy decides
  whether to retry.
- `error_handler=` on `add_node` or `set_node_defaults` runs a handler node when a node raises
  (a failing handler fails the run).
- Policies set with `set_node_defaults` are not inherited by subgraphs.

## Functional API

Write workflows as plain functions while keeping checkpoints and interrupts:

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import entrypoint, task

@task
def draft(topic: str) -> str:
    return f"Draft about {topic}"

@task
def polish(text: str) -> str:
    return text + " (polished)"

@entrypoint(checkpointer=InMemorySaver())
def workflow(topic: str) -> str:
    text = draft(topic).result()   # tasks return futures; results are checkpointed
    return polish(text).result()

workflow.invoke("agents", config={"configurable": {"thread_id": "w1"}})
```

## Persistence

### Checkpointers (Short-Term, Per Thread)

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.postgres import PostgresSaver

graph = builder.compile(checkpointer=InMemorySaver())   # dev and tests

# from_conn_string() is a context manager; SQLite takes a file path, not a URL
with SqliteSaver.from_conn_string("checkpoints.sqlite") as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)
    graph.invoke(inputs, config={"configurable": {"thread_id": "1"}})

DB_URI = "postgresql://user:pass@localhost:5432/app?sslmode=disable"
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()   # create tables once (idempotent)
    graph = builder.compile(checkpointer=checkpointer)
```

Async apps use `AsyncSqliteSaver` / `AsyncPostgresSaver` with `async with`. With `create_agent`,
pass the same object as `checkpointer=`.

### Durability Modes

```python
graph.invoke(inputs, config, durability="sync")   # "exit" | "async" (default) | "sync"
```

- `"exit"`: persist only when the run ends or interrupts. Fastest, but no mid-run recovery.
- `"async"`: persist in the background while the next step runs. Good balance.
- `"sync"`: persist before each next step. Most durable, slowest.

### Store (Long-Term, Cross-Thread)

```python
from langgraph.store.memory import InMemoryStore
from langchain.embeddings import init_embeddings

store = InMemoryStore(index={"embed": init_embeddings("openai:text-embedding-3-small"), "dims": 1536})

store.put(("users", "user123"), "prefs", {"theme": "dark", "language": "fr"})
item = store.get(("users", "user123"), "prefs")
data = item.value if item else None

results = store.search(("users", "user123"), query="language preference", limit=5)   # semantic
results = store.search(("users",), filter={"theme": "dark"})                          # exact match
```

Production: `PostgresStore.from_conn_string(DB_URI)` (context manager, call `.setup()`).
Inside nodes and tools, use `runtime.store` rather than a global.

## Runtime Context

```python
from dataclasses import dataclass
from langgraph.runtime import Runtime

@dataclass
class Context:
    user_id: str

def call_model(state: MessagesState, runtime: Runtime[Context]) -> dict:
    memories = runtime.store.search(("memories", runtime.context.user_id), query=state["messages"][-1].text)
    ...

builder = StateGraph(MessagesState, context_schema=Context)
# graph.invoke(inputs, config, context=Context(user_id="1"))
```

## Interrupts (Human-in-the-Loop)

```python
from langgraph.types import interrupt, Command

def sensitive_action(state: State) -> dict:
    approval = interrupt({"action": "delete_all", "message": "Approve deletion?"})
    return {"status": "deleted" if approval.get("approved") else "cancelled"}

config = {"configurable": {"thread_id": "t1"}}           # interrupts need a checkpointer
result = graph.invoke(inputs, config=config, version="v2")
if result.interrupts:
    print(result.interrupts[0].value)                    # payload passed to interrupt()
    result = graph.invoke(Command(resume={"approved": True}), config=config, version="v2")
```

The interrupted node re-runs from its start on resume, so keep side effects after the
`interrupt()` call or make them idempotent. Without `version="v2"`, pending interrupts appear
under `result["__interrupt__"]`.

## Streaming

```python
# Event streaming (v3): typed projections, recommended for apps and UIs
stream = graph.stream_events(inputs, config, version="v3")
for snapshot in stream.values:          # full state after each step
    print(snapshot)
final = stream.output

# Classic stream modes: values, updates, messages, custom, checkpoints, tasks, debug
for mode, chunk in graph.stream(inputs, config, stream_mode=["updates", "messages"]):
    if mode == "updates":
        for node, update in chunk.items():
            print(node, update)
    else:
        token, metadata = chunk
        print(token.content, end="")

# Custom progress events from inside a node or tool
from langgraph.config import get_stream_writer

def long_step(state: State) -> dict:
    get_stream_writer()({"progress": 0.5})
    return {}
```

Pass `subgraphs=True` to `stream` to include events from nested graphs. Event streaming is
experimental in 1.2 and emits a `LangChainBetaWarning`.

## Subgraphs

```python
child = StateGraph(ChildState)
child.add_node("process", process_fn)
child.add_edge(START, "process")
child_graph = child.compile()

parent = StateGraph(ParentState)
parent.add_node("child_workflow", child_graph)   # shares keys with the parent state
parent.add_edge(START, "child_workflow")
parent_graph = parent.compile(checkpointer=InMemorySaver())   # the child inherits it
```

If the schemas differ, call the subgraph inside a regular node and map state in and out.

## Time Travel

```python
history = list(graph.get_state_history(config))   # newest first
past = history[2]
graph.invoke(None, config=past.config)            # replay from that checkpoint

forked = graph.update_state(past.config, {"step_count": 0})   # fork with edited state
graph.invoke(None, config=forked)
```
