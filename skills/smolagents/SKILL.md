---
name: smolagents
description: Build AI agents with Hugging Face's SmolAgents framework. Use when creating code-executing agents, tool-calling agents, multi-agent systems, agentic RAG, text-to-SQL pipelines, web browsing agents, or any multi-step AI workflows. Covers CodeAgent, ToolCallingAgent, custom tools, MCP integration, memory management, secure code execution (E2B, Docker, Blaxel), and model configuration (HF Inference, LiteLLM, Transformers, Ollama).
---

# SmolAgents - Hugging Face AI Agent Framework

SmolAgents is a minimalist Python library (~1000 lines) for building AI agents that write and execute code. It emphasizes simplicity, security, and model-agnostic design.

## Installation

```bash
pip install smolagents                    # Core library
pip install 'smolagents[toolkit]'         # With default tools (web search, etc.)
pip install 'smolagents[litellm]'         # LiteLLM for OpenAI/Anthropic
pip install 'smolagents[transformers]'    # Local model support
pip install 'smolagents[mcp]'             # MCP server integration
pip install 'smolagents[telemetry]'       # OpenTelemetry support
```

## Core Concepts

### Agent Types

**CodeAgent** - Primary agent that writes Python code to execute actions:
```python
from smolagents import CodeAgent, InferenceClientModel

model = InferenceClientModel()  # Default: Qwen/Qwen2.5-Coder-32B-Instruct
agent = CodeAgent(tools=[], model=model)
result = agent.run("Calculate the first 20 Fibonacci numbers")
```

**ToolCallingAgent** - Uses JSON-based tool calls (no code execution):
```python
from smolagents import ToolCallingAgent, InferenceClientModel

agent = ToolCallingAgent(tools=[], model=InferenceClientModel())
agent.run("Get the title of https://huggingface.co/blog")
```

### Model Configuration

**Hugging Face Inference API** (recommended for quick start):
```python
from smolagents import InferenceClientModel

# Default model
model = InferenceClientModel()

# Specific model with provider
model = InferenceClientModel(
    model_id="meta-llama/Llama-3.3-70B-Instruct",
    provider="together",  # or "sambanova", "fireworks", etc.
    token="YOUR_HF_TOKEN"
)
```

**LiteLLM** (100+ providers):
```python
from smolagents import LiteLLMModel

# Anthropic
model = LiteLLMModel(
    model_id="anthropic/claude-3-5-sonnet-latest",
    api_key="YOUR_ANTHROPIC_API_KEY"
)

# OpenAI
model = LiteLLMModel(model_id="gpt-4o", api_key="YOUR_OPENAI_API_KEY")
```

**Local Transformers**:
```python
from smolagents import TransformersModel

model = TransformersModel(
    model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    max_new_tokens=4096,
    device_map="auto"
)
```

**Ollama** (local):
```python
from smolagents import LiteLLMModel

model = LiteLLMModel(
    model_id="ollama_chat/llama3.2",
    api_base="http://localhost:11434",
    num_ctx=8192  # Important: increase from default 2048
)
```

**OpenAI-compatible endpoints**:
```python
from smolagents import OpenAIModel

model = OpenAIModel(
    model_id="deepseek-ai/DeepSeek-R1",
    api_base="https://api.together.xyz/v1/",
    api_key="YOUR_API_KEY"
)
```

## Tools

### Built-in Tools

```python
from smolagents import CodeAgent, InferenceClientModel, WebSearchTool, DuckDuckGoSearchTool

agent = CodeAgent(
    tools=[WebSearchTool()],  # or DuckDuckGoSearchTool()
    model=InferenceClientModel(),
    add_base_tools=True  # Adds transcriber, python interpreter
)
```

### Creating Custom Tools

**Using @tool decorator** (simplest):
```python
from smolagents import tool

@tool
def get_weather(city: str) -> str:
    """
    Gets current weather for a city.
    
    Args:
        city: Name of the city to get weather for.
    """
    # Implementation
    return f"Weather in {city}: Sunny, 22°C"
```

**Using Tool class** (more control):
```python
from smolagents import Tool

class HFModelDownloadsTool(Tool):
    name = "model_download_counter"
    description = "Returns the most downloaded model for a given task on HuggingFace Hub."
    inputs = {
        "task": {"type": "string", "description": "The ML task (e.g., 'text-to-video')"}
    }
    output_type = "string"

    def forward(self, task: str) -> str:
        from huggingface_hub import list_models
        model = next(iter(list_models(filter=task, sort="downloads", direction=-1)))
        return model.id
```

### Loading Tools from Hub

```python
from smolagents import load_tool

image_tool = load_tool("m-ric/text-to-image", trust_remote_code=True)
agent = CodeAgent(tools=[image_tool], model=model)
```

### Loading from Gradio Spaces

```python
from smolagents import Tool

image_generator = Tool.from_space(
    "black-forest-labs/FLUX.1-schnell",
    name="image_generator",
    description="Generates images from text prompts"
)
```

### LangChain Tool Integration

```python
from smolagents import Tool
from langchain.agents import load_tools

search_tool = Tool.from_langchain(load_tools(["serpapi"])[0])
agent = CodeAgent(tools=[search_tool], model=model)
```

### MCP Server Integration

```python
from smolagents import ToolCollection, CodeAgent
from mcp import StdioServerParameters

server_params = StdioServerParameters(
    command="uvx",
    args=["mcp-server-fetch"]
)

with ToolCollection.from_mcp(server_params, trust_remote_code=True) as tools:
    agent = CodeAgent(tools=[*tools.tools], model=model)
    agent.run("Fetch the homepage of huggingface.co")
```

### Managing Agent Tools

```python
# Add tool dynamically
agent.tools[new_tool.name] = new_tool

# List tools
print(agent.tools.keys())
```

## Agent Configuration

### Key Parameters

```python
agent = CodeAgent(
    tools=[...],
    model=model,
    max_steps=10,                          # Max reasoning steps
    verbosity_level=2,                     # 0=silent, 1=steps, 2=detailed
    planning_interval=3,                   # Enable planning every N steps
    additional_authorized_imports=["numpy", "pandas"],  # Allow imports
    add_base_tools=True,                   # Include default tools
    step_callbacks=[my_callback],          # Custom callbacks per step
)
```

### Running Agents

```python
# Basic run
result = agent.run("Your task here")

# With additional context
result = agent.run(
    "Generate an image based on this prompt",
    additional_args={"user_prompt": "A cat in space"}
)

# Continue conversation (preserve memory)
result = agent.run("Follow-up question", reset=False)
```

## Memory Management

### Accessing Memory

```python
from smolagents import ActionStep

# System prompt
print(agent.memory.system_prompt.system_prompt)

# Task
print(agent.memory.steps[0].task)

# Iterate through steps
for step in agent.memory.steps:
    if isinstance(step, ActionStep):
        if step.error:
            print(f"Step {step.step_number} error: {step.error}")
        else:
            print(f"Step {step.step_number}: {step.observations}")
```

### Step-by-Step Execution

```python
from smolagents import CodeAgent, InferenceClientModel, ActionStep, TaskStep

agent = CodeAgent(tools=[], model=InferenceClientModel(), verbosity_level=1)
agent.python_executor.send_tools({**agent.tools})

task = "What is the 20th Fibonacci number?"
agent.memory.steps.append(TaskStep(task=task, task_images=[]))

final_answer = None
step_number = 1

while final_answer is None and step_number <= 10:
    memory_step = ActionStep(step_number=step_number, observations_images=[])
    final_answer = agent.step(memory_step)
    agent.memory.steps.append(memory_step)
    step_number += 1

print("Answer:", final_answer)
```

### Step Callbacks

```python
from smolagents import ActionStep, CodeAgent

def log_step(memory_step: ActionStep, agent: CodeAgent) -> None:
    print(f"Completed step {memory_step.step_number}")
    # Modify memory if needed
    if memory_step.step_number > 5:
        memory_step.observations_images = None  # Clear images to save tokens

agent = CodeAgent(
    tools=[...],
    model=model,
    step_callbacks=[log_step]
)
```

### Replay Execution

```python
agent.run("Your task")
agent.replay()  # Replay the last execution
```

## Multi-Agent Systems

### Basic Multi-Agent Setup

```python
from smolagents import CodeAgent, InferenceClientModel, WebSearchTool

model = InferenceClientModel()

# Specialist agent
web_agent = CodeAgent(
    tools=[WebSearchTool()],
    model=model,
    name="web_search_agent",
    description="Runs web searches. Give it your query as an argument."
)

# Manager agent
manager_agent = CodeAgent(
    tools=[],
    model=model,
    managed_agents=[web_agent],
    additional_authorized_imports=["pandas", "numpy"]
)

manager_agent.run("Who is the CEO of Hugging Face?")
```

### With ToolCallingAgent

```python
from smolagents import ToolCallingAgent, CodeAgent, WebSearchTool

web_agent = ToolCallingAgent(
    tools=[WebSearchTool(), visit_webpage],
    model=model,
    max_steps=10,
    name="web_search_agent",
    description="Runs web searches for you."
)

manager = CodeAgent(
    tools=[],
    model=model,
    managed_agents=[web_agent]
)
```

## Secure Code Execution

### Local Python Executor (Default)

Built-in security:
- Restricted imports (must be explicitly authorized)
- Operation count limits (prevents infinite loops)
- AST-based execution (no raw eval)

```python
agent = CodeAgent(
    tools=[],
    model=model,
    additional_authorized_imports=["numpy", "pandas", "requests"]
)
```

### E2B Sandbox

```python
# pip install e2b-code-interpreter python-dotenv
# Set E2B_API_KEY environment variable

from smolagents import CodeAgent, InferenceClientModel

agent = CodeAgent(
    tools=[],
    model=InferenceClientModel(),
    executor_type="e2b",
    executor_kwargs={"api_key": "YOUR_E2B_API_KEY"},
    additional_authorized_imports=["requests"]
)

agent.run("Calculate something complex")
```

### Docker Sandbox

```python
agent = CodeAgent(
    tools=[],
    model=InferenceClientModel(),
    executor_type="docker"
)
```

### Blaxel Sandbox

```python
from smolagents import CodeAgent, InferenceClientModel

with CodeAgent(model=InferenceClientModel(), tools=[], executor_type="blaxel") as agent:
    agent.run("Give me the 100th Fibonacci number")
# Sandbox automatically cleaned up
```

## Common Patterns

### Text-to-SQL Agent

```python
from smolagents import tool, CodeAgent, InferenceClientModel
from sqlalchemy import create_engine, text

engine = create_engine("sqlite:///database.db")

@tool
def sql_engine(query: str) -> str:
    """
    Execute SQL queries on the database.
    
    Args:
        query: SQL query to execute.
    """
    with engine.connect() as conn:
        result = conn.execute(text(query))
        return str(result.fetchall())

agent = CodeAgent(
    tools=[sql_engine],
    model=InferenceClientModel()
)

agent.run("What are the top 5 customers by total purchases?")
```

### Agentic RAG

```python
from smolagents import tool, CodeAgent, InferenceClientModel

@tool
def retriever_tool(query: str) -> str:
    """
    Retrieves relevant documents from the knowledge base.
    
    Args:
        query: Search query for the knowledge base.
    """
    # Your retrieval logic here
    return "Retrieved documents..."

agent = CodeAgent(
    tools=[retriever_tool],
    model=InferenceClientModel(),
    max_steps=4,
    verbosity_level=2
)

agent.run("What does our documentation say about authentication?")
```

### Web Browser Agent

```python
from smolagents import CodeAgent, InferenceClientModel, tool
import helium

@tool  
def go_back() -> str:
    """Go back to the previous page."""
    helium.go_back()
    return "Navigated back"

@tool
def search_item_ctrl_f(text: str) -> str:
    """
    Search for text on the current page.
    
    Args:
        text: Text to search for.
    """
    elements = helium.find_all(helium.Text(text))
    return f"Found {len(elements)} matches"

agent = CodeAgent(
    tools=[go_back, search_item_ctrl_f],
    model=InferenceClientModel(model_id="Qwen/Qwen2-VL-72B-Instruct"),  # VLM for vision
    additional_authorized_imports=["helium"],
    max_steps=20
)
```

## Gradio UI

```python
from smolagents import CodeAgent, InferenceClientModel, GradioUI, load_tool

image_tool = load_tool("m-ric/text-to-image")
agent = CodeAgent(tools=[image_tool], model=InferenceClientModel())

GradioUI(agent).launch()
```

## CLI Usage

```bash
# Basic agent
smolagent "Plan a trip to Tokyo" \
    --model-type InferenceClientModel \
    --model-id "Qwen/Qwen2.5-Coder-32B-Instruct" \
    --tools web_search

# Web browsing agent
webagent "Go to example.com and find the pricing page" \
    --model-type LiteLLMModel \
    --model-id gpt-4o

# Interactive mode
smolagent  # Launches setup wizard
```

## Telemetry & Debugging

```python
# Enable OpenTelemetry tracing
# pip install 'smolagents[telemetry]'

from phoenix.trace import init_tracing

init_tracing(service_name="my_agent")

agent = CodeAgent(tools=[...], model=model)
agent.run("Your task")
```

## Best Practices

1. **Start simple**: Use `InferenceClientModel()` with defaults
2. **Use planning** for complex tasks: `planning_interval=3`
3. **Limit steps**: Set reasonable `max_steps` (10-20)
4. **Authorize imports explicitly**: Security-first approach
5. **Use sandboxing** in production: E2B, Docker, or Blaxel
6. **Add verbosity** during development: `verbosity_level=2`
7. **Use ToolCallingAgent** when code execution isn't needed
8. **Specialize agents** in multi-agent systems

## Additional Resources

- **Documentation**: https://huggingface.co/docs/smolagents
- **GitHub**: https://github.com/huggingface/smolagents
- **HF Agents Course**: https://huggingface.co/learn/agents-course
- **Model specifications**: See `references/models.md`
- **Advanced patterns**: See `references/patterns.md`