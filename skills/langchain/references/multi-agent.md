# Multi-Agent Patterns

Build systems with multiple specialized agents.

## Handoffs

Transfer control between agents:

```python
from langchain.agents import create_agent
from langchain.agents.handoffs import handoff

support_agent = create_agent(
    model="claude-sonnet-4-5-20250929",
    tools=[search_kb],
    system_prompt="You handle general support questions."
)

billing_agent = create_agent(
    model="claude-sonnet-4-5-20250929", 
    tools=[check_billing, process_refund],
    system_prompt="You handle billing issues."
)

# Add handoff capabilities
support_agent = support_agent.with_handoffs([
    handoff(
        to=billing_agent,
        description="Transfer to billing for payment issues"
    )
])
```

## Router

Route to specialized agents based on intent:

```python
from langchain.agents import create_agent
from langchain.agents.router import create_router

agents = {
    "technical": create_agent(model, tools=[debug_tool]),
    "sales": create_agent(model, tools=[pricing_tool]),
    "support": create_agent(model, tools=[ticket_tool])
}

router = create_router(
    model="claude-sonnet-4-5-20250929",
    agents=agents,
    system_prompt="Route based on user intent."
)

result = router.invoke({"messages": [{"role": "user", "content": "My app crashed"}]})
# Routes to "technical" agent
```

## Subagents

Delegate subtasks to specialized agents:

```python
from langchain.agents import create_agent
from langchain.agents.subagents import create_subagent_tool

research_agent = create_agent(
    model="claude-sonnet-4-5-20250929",
    tools=[web_search, read_document],
    system_prompt="You research topics thoroughly."
)

# Convert agent to tool
research_tool = create_subagent_tool(
    agent=research_agent,
    name="research",
    description="Research a topic in depth"
)

main_agent = create_agent(
    model="claude-sonnet-4-5-20250929",
    tools=[research_tool, write_report],
    system_prompt="You coordinate research and writing."
)
```

## Skills Pattern

On-demand capability loading:

```python
from langchain.agents import create_agent
from langchain.agents.skills import Skill, create_skill_manager

sql_skill = Skill(
    name="sql",
    tools=[execute_query, list_tables],
    description="Query databases"
)

api_skill = Skill(
    name="api", 
    tools=[make_request, parse_response],
    description="Call external APIs"
)

skill_manager = create_skill_manager(
    skills=[sql_skill, api_skill],
    model="claude-sonnet-4-5-20250929"
)

agent = create_agent(
    model="claude-sonnet-4-5-20250929",
    tools=[skill_manager.get_skill_tool()],
    system_prompt="Use skills as needed."
)
```

## Custom Workflow (LangGraph)

Full control with graph API:

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal

class State(TypedDict):
    messages: list
    current_agent: str

def classifier(state: State) -> dict:
    # Classify intent
    return {"current_agent": "technical"}  # or "sales", "support"

def route(state: State) -> Literal["technical", "sales", "support"]:
    return state["current_agent"]

def technical_agent(state: State) -> dict:
    # Handle technical queries
    return {"messages": state["messages"] + [response]}

graph = StateGraph(State)
graph.add_node("classify", classifier)
graph.add_node("technical", technical_agent)
graph.add_node("sales", sales_agent)
graph.add_node("support", support_agent)

graph.add_edge(START, "classify")
graph.add_conditional_edges("classify", route)
graph.add_edge("technical", END)
graph.add_edge("sales", END)
graph.add_edge("support", END)

app = graph.compile()
```

## Supervisor Pattern

Central coordinator managing worker agents:

```python
from langchain.agents import create_agent
from langchain.tools import tool

researcher = create_agent(model, tools=[search])
writer = create_agent(model, tools=[write])
reviewer = create_agent(model, tools=[review])

@tool
def delegate_to_researcher(task: str) -> str:
    """Delegate research task."""
    return researcher.invoke({"messages": [{"role": "user", "content": task}]})

@tool
def delegate_to_writer(task: str) -> str:
    """Delegate writing task."""
    return writer.invoke({"messages": [{"role": "user", "content": task}]})

supervisor = create_agent(
    model="claude-sonnet-4-5-20250929",
    tools=[delegate_to_researcher, delegate_to_writer],
    system_prompt="""You coordinate work between agents.
    1. Research first
    2. Then write
    3. Then review"""
)
```