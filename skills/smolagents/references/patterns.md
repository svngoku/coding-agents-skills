# SmolAgents Advanced Patterns

## Agentic RAG

### Basic RAG Tool

```python
from smolagents import tool, CodeAgent, InferenceClientModel
from sentence_transformers import SentenceTransformer
import numpy as np

# Setup
embedder = SentenceTransformer("all-MiniLM-L6-v2")
documents = ["doc1 content...", "doc2 content..."]
doc_embeddings = embedder.encode(documents)

@tool
def retriever(query: str, top_k: int = 3) -> str:
    """
    Retrieves relevant documents from the knowledge base.
    
    Args:
        query: Search query to find relevant documents.
        top_k: Number of documents to return.
    """
    query_embedding = embedder.encode([query])
    similarities = np.dot(doc_embeddings, query_embedding.T).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    results = [documents[i] for i in top_indices]
    return "\n\n---\n\n".join(results)

agent = CodeAgent(
    tools=[retriever],
    model=InferenceClientModel(),
    max_steps=4
)

agent.run("What does our documentation say about authentication?")
```

### RAG with Vector Database

```python
from smolagents import tool
import chromadb

client = chromadb.Client()
collection = client.get_or_create_collection("docs")

@tool
def search_knowledge_base(query: str, n_results: int = 5) -> str:
    """
    Search the knowledge base for relevant information.
    
    Args:
        query: Natural language search query.
        n_results: Number of results to return.
    """
    results = collection.query(query_texts=[query], n_results=n_results)
    docs = results["documents"][0]
    return "\n\n".join([f"[{i+1}] {doc}" for i, doc in enumerate(docs)])
```

## Text-to-SQL

### Complete SQL Agent

```python
from smolagents import tool, CodeAgent, InferenceClientModel
from sqlalchemy import create_engine, MetaData, text, inspect

engine = create_engine("sqlite:///sales.db")

# Get schema for tool description
inspector = inspect(engine)
tables_info = []
for table_name in inspector.get_table_names():
    columns = inspector.get_columns(table_name)
    cols = ", ".join([f"{c['name']} ({c['type']})" for c in columns])
    tables_info.append(f"- {table_name}: {cols}")

schema_description = "\n".join(tables_info)

@tool
def execute_sql(query: str) -> str:
    f"""
    Execute SQL queries on the database.
    Returns query results as a string.
    
    Available tables and columns:
    {schema_description}
    
    Args:
        query: SQL query to execute. Use SELECT for data retrieval.
    """
    try:
        with engine.connect() as conn:
            result = conn.execute(text(query))
            rows = result.fetchall()
            if not rows:
                return "Query returned no results."
            columns = result.keys()
            header = " | ".join(columns)
            data = "\n".join([" | ".join(str(v) for v in row) for row in rows])
            return f"{header}\n{'-' * len(header)}\n{data}"
    except Exception as e:
        return f"SQL Error: {str(e)}"

agent = CodeAgent(
    tools=[execute_sql],
    model=InferenceClientModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct"),
    max_steps=5
)

agent.run("Which salesperson had the highest total sales last quarter?")
```

## Web Browser Agent

### Vision-Based Browser

```python
from smolagents import CodeAgent, InferenceClientModel, tool, ActionStep
import helium
from PIL import Image
from io import BytesIO
from time import sleep

@tool
def navigate_to(url: str) -> str:
    """
    Navigate to a URL.
    
    Args:
        url: The URL to navigate to.
    """
    helium.go_to(url)
    return f"Navigated to {url}"

@tool
def click_element(text: str) -> str:
    """
    Click on an element containing the specified text.
    
    Args:
        text: Text content of the element to click.
    """
    helium.click(text)
    return f"Clicked on '{text}'"

@tool
def type_text(text: str, into: str = None) -> str:
    """
    Type text, optionally into a specific field.
    
    Args:
        text: Text to type.
        into: Optional field name/placeholder to type into.
    """
    if into:
        helium.write(text, into=into)
    else:
        helium.write(text)
    return f"Typed '{text}'"

@tool
def scroll_down() -> str:
    """Scroll down the page."""
    helium.scroll_down(num_pixels=500)
    return "Scrolled down"

@tool
def go_back() -> str:
    """Go back to the previous page."""
    helium.go_back()
    return "Went back"

def capture_screenshot(memory_step: ActionStep, agent: CodeAgent) -> None:
    """Callback to capture screenshots after each step."""
    sleep(1.0)  # Wait for page load
    driver = helium.get_driver()
    png_bytes = driver.get_screenshot_as_png()
    image = Image.open(BytesIO(png_bytes))
    memory_step.observations_images = [image.copy()]
    
    # Remove old screenshots to save tokens
    for step in agent.memory.steps:
        if isinstance(step, ActionStep) and step.step_number < memory_step.step_number - 1:
            step.observations_images = None

# Initialize browser
helium.start_chrome(headless=False)

agent = CodeAgent(
    tools=[navigate_to, click_element, type_text, scroll_down, go_back],
    model=InferenceClientModel(model_id="Qwen/Qwen2-VL-72B-Instruct"),  # VLM
    additional_authorized_imports=["helium"],
    step_callbacks=[capture_screenshot],
    max_steps=20,
    verbosity_level=2
)

agent.run("Go to amazon.com and find the price of the iPhone 15")
```

## Multi-Agent Orchestration

### Research Team

```python
from smolagents import CodeAgent, ToolCallingAgent, InferenceClientModel, WebSearchTool

model = InferenceClientModel()

# Research agent
researcher = CodeAgent(
    tools=[WebSearchTool()],
    model=model,
    name="researcher",
    description="Searches the web for information. Give it a research topic."
)

# Analysis agent
from smolagents import tool

@tool
def analyze_data(data: str) -> str:
    """
    Analyzes data and provides insights.
    
    Args:
        data: Data to analyze as text.
    """
    # Analysis logic here
    return "Analysis results..."

analyst = CodeAgent(
    tools=[analyze_data],
    model=model,
    name="analyst", 
    description="Analyzes data and provides insights. Give it data to analyze."
)

# Manager coordinates both
manager = CodeAgent(
    tools=[],
    model=model,
    managed_agents=[researcher, analyst],
    additional_authorized_imports=["pandas", "numpy"],
    planning_interval=3
)

manager.run("""
Research the current state of renewable energy adoption worldwide,
then analyze the trends and provide a summary report.
""")
```

### Hierarchical Agents

```python
# Level 1: Specialist agents
code_agent = CodeAgent(
    tools=[],
    model=model,
    name="coder",
    description="Writes and executes Python code. Give it coding tasks."
)

search_agent = CodeAgent(
    tools=[WebSearchTool()],
    model=model,
    name="searcher",
    description="Searches the web. Give it search queries."
)

# Level 2: Team lead
team_lead = CodeAgent(
    tools=[],
    model=model,
    managed_agents=[code_agent, search_agent],
    name="team_lead",
    description="Coordinates coding and research tasks."
)

# Level 3: Executive
executive = CodeAgent(
    tools=[],
    model=model,
    managed_agents=[team_lead],
    planning_interval=5
)
```

## Human-in-the-Loop

### Plan Review Callback

```python
from smolagents import CodeAgent, InferenceClientModel
from smolagents.types import PlanningStep

def review_plan(memory_step, agent):
    """Allow human to review and modify plans."""
    if isinstance(memory_step, PlanningStep):
        print("\n" + "="*50)
        print("PROPOSED PLAN:")
        print(memory_step.plan)
        print("="*50)
        
        response = input("\nApprove plan? (y/n/modify): ").strip().lower()
        
        if response == 'n':
            raise KeyboardInterrupt("Plan rejected by user")
        elif response == 'modify':
            new_plan = input("Enter modified plan: ")
            memory_step.plan = new_plan
        # 'y' continues normally

agent = CodeAgent(
    model=InferenceClientModel(),
    tools=[],
    planning_interval=3,
    step_callbacks=[review_plan]
)
```

### Step Approval

```python
from smolagents import ActionStep

def approve_step(memory_step: ActionStep, agent):
    """Require approval before executing each step."""
    if memory_step.action_output:
        print(f"\nStep {memory_step.step_number} wants to execute:")
        print(memory_step.action_output)
        
        if input("Approve? (y/n): ").lower() != 'y':
            memory_step.error = "User rejected this action"
            raise KeyboardInterrupt("Action rejected")

agent = CodeAgent(
    tools=[...],
    model=model,
    step_callbacks=[approve_step]
)
```

## Async Agents

```python
import asyncio
from smolagents import CodeAgent, InferenceClientModel

async def run_multiple_agents():
    model = InferenceClientModel()
    
    agent1 = CodeAgent(tools=[], model=model)
    agent2 = CodeAgent(tools=[], model=model)
    
    # Run agents concurrently
    results = await asyncio.gather(
        asyncio.to_thread(agent1.run, "Task 1"),
        asyncio.to_thread(agent2.run, "Task 2")
    )
    
    return results

# Run
results = asyncio.run(run_multiple_agents())
```

## Custom System Prompts

```python
custom_prompt = """You are a specialized data analysis agent.

Your capabilities:
- Analyze datasets using pandas
- Create visualizations
- Generate statistical summaries

Always:
1. Validate data before analysis
2. Handle missing values appropriately
3. Explain your methodology

Available tools:
{%- for tool in tools.values() %}
- {{ tool.to_tool_calling_prompt() }}
{%- endfor %}

Authorized imports: {{authorized_imports}}
"""

agent = CodeAgent(
    tools=[...],
    model=model,
    system_prompt=custom_prompt,
    additional_authorized_imports=["pandas", "matplotlib", "numpy"]
)
```

## Memory Persistence

### Save and Load Memory

```python
import json
from smolagents import CodeAgent, InferenceClientModel

agent = CodeAgent(tools=[], model=InferenceClientModel())
agent.run("First task")

# Save memory
memory_data = {
    "steps": [
        {
            "type": type(step).__name__,
            "data": step.__dict__
        }
        for step in agent.memory.steps
    ]
}

with open("memory.json", "w") as f:
    json.dump(memory_data, f)

# Later: restore and continue
agent2 = CodeAgent(tools=[], model=InferenceClientModel())
# Manually reconstruct steps from saved data
agent2.run("Continue from where we left off", reset=False)
```

## Error Recovery

```python
from smolagents import CodeAgent, InferenceClientModel

def safe_run(agent, task, max_retries=3):
    """Run agent with automatic retry on failure."""
    for attempt in range(max_retries):
        try:
            return agent.run(task, reset=(attempt == 0))
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise
            # Modify task for retry
            task = f"Previous attempt failed with: {e}. Please try again: {task}"

agent = CodeAgent(tools=[], model=InferenceClientModel())
result = safe_run(agent, "Complex task that might fail")
```

## Production Checklist

1. **Security**
   - [ ] Use E2B/Docker/Blaxel sandbox
   - [ ] Restrict authorized imports
   - [ ] Validate tool inputs
   - [ ] Rate limit API calls

2. **Reliability**
   - [ ] Set appropriate max_steps
   - [ ] Implement error handling
   - [ ] Add retry logic
   - [ ] Monitor with telemetry

3. **Performance**
   - [ ] Use planning_interval for complex tasks
   - [ ] Clear old screenshots in callbacks
   - [ ] Cache embeddings for RAG
   - [ ] Use async for parallel agents

4. **Cost Control**
   - [ ] Set max_tokens limits
   - [ ] Use cheaper models for simple tasks
   - [ ] Implement caching
   - [ ] Monitor token usage