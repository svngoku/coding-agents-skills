# LangGraph Reference

LangGraph provides fine-grained control for complex agent workflows.

## Installation

```bash
pip install langgraph
```

## Graph API

Define nodes and edges for custom workflows:

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from operator import add

class State(TypedDict):
    messages: Annotated[list, add]
    step_count: int

def process_input(state: State) -> dict:
    return {"step_count": state["step_count"] + 1}

def should_continue(state: State) -> str:
    return "continue" if state["step_count"] < 3 else "end"

graph = StateGraph(State)
graph.add_node("process", process_input)
graph.add_edge(START, "process")
graph.add_conditional_edges("process", should_continue, {
    "continue": "process",
    "end": END
})

app = graph.compile()
result = app.invoke({"messages": [], "step_count": 0})
```

## Functional API

For simpler task-based flows:

```python
from langgraph.func import entrypoint, task

@task
def step_one(data: str) -> str:
    return f"processed: {data}"

@task  
def step_two(data: str) -> str:
    return f"finalized: {data}"

@entrypoint()
def workflow(input: str):
    result = step_one(input).result()
    return step_two(result).result()
```

## Persistence

### Checkpointers

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.sqlite import SqliteSaver

# In-memory (dev)
checkpointer = InMemorySaver()

# SQLite (simple persistence)
checkpointer = SqliteSaver.from_conn_string("sqlite:///agent.db")

# PostgreSQL (production)
checkpointer = PostgresSaver.from_conn_string("postgresql://...")

graph = StateGraph(State)
# ... build graph ...
app = graph.compile(checkpointer=checkpointer)
```

### Store (Long-term Memory)

```python
from langgraph.store.memory import InMemoryStore
from langgraph.store.postgres import PostgresStore

store = InMemoryStore()

# Write
store.put(("users",), "user123", {"name": "Alice", "prefs": {...}})

# Read
item = store.get(("users",), "user123")
data = item.value if item else None

# Search
results = store.search(("users",), filter={"name": "Alice"})
```

## Interrupts (Human-in-the-Loop)

```python
from langgraph.types import interrupt, Command

def sensitive_action(state: State) -> dict:
    # Pause for human approval
    approval = interrupt({
        "action": "delete_all",
        "message": "Approve deletion?"
    })
    
    if approval.get("approved"):
        return {"status": "deleted"}
    return {"status": "cancelled"}

# Resume with approval
app.invoke(Command(resume={"approved": True}), config=config)
```

## Streaming

```python
# Stream all events
for event in app.stream({"messages": [...]}, stream_mode="values"):
    print(event)

# Stream specific updates
async for chunk in app.astream(input, stream_mode="updates"):
    for node, update in chunk.items():
        print(f"{node}: {update}")
```

## Subgraphs

Compose graphs within graphs:

```python
# Child graph
child = StateGraph(ChildState)
child.add_node("process", process_fn)
child_app = child.compile()

# Parent graph
parent = StateGraph(ParentState)
parent.add_node("child_workflow", child_app)
parent.add_edge(START, "child_workflow")
parent_app = parent.compile()
```

## Time Travel

Replay from any checkpoint:

```python
# Get history
history = list(app.get_state_history(config))

# Replay from specific point
past_config = history[2].config
result = app.invoke(None, config=past_config)
```