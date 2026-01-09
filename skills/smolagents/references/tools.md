# SmolAgents Tools Reference

## Built-in Tools

### WebSearchTool

Web search using DuckDuckGo (default) or Google.

```python
from smolagents import WebSearchTool

# Default (DuckDuckGo)
search = WebSearchTool()

# With Google (requires API key)
search = WebSearchTool(provider="google", api_key="YOUR_SERPER_API_KEY")
```

### DuckDuckGoSearchTool

Direct DuckDuckGo search.

```python
from smolagents import DuckDuckGoSearchTool

search = DuckDuckGoSearchTool()
result = search("latest AI news")
```

### VisitWebpageTool

Fetch and parse webpage content.

```python
from smolagents import VisitWebpageTool

visit = VisitWebpageTool()
content = visit("https://example.com")
```

### PythonInterpreterTool

Execute Python code (for ToolCallingAgent only, CodeAgent has native execution).

```python
from smolagents import PythonInterpreterTool

interpreter = PythonInterpreterTool()
```

### Transcriber

Speech-to-text using Whisper.

```python
from smolagents import Transcriber

transcriber = Transcriber()
text = transcriber(audio_file)
```

### FinalAnswerTool

Built into all agents for returning final results.

```python
# In agent code
final_answer("The result is 42")
```

## Creating Custom Tools

### @tool Decorator (Recommended)

```python
from smolagents import tool
from typing import Optional

@tool
def calculate_compound_interest(
    principal: float,
    rate: float,
    years: int,
    compounds_per_year: Optional[int] = 12
) -> float:
    """
    Calculate compound interest on an investment.
    
    Args:
        principal: Initial investment amount in dollars.
        rate: Annual interest rate as a decimal (e.g., 0.05 for 5%).
        years: Number of years to compound.
        compounds_per_year: Number of times interest compounds per year.
    """
    amount = principal * (1 + rate / compounds_per_year) ** (compounds_per_year * years)
    return round(amount, 2)
```

**Requirements:**
- Detailed docstring with `Args:` section
- Type hints on all parameters
- Return type hint
- Clear description of what tool does

### Tool Class (Full Control)

```python
from smolagents import Tool

class WeatherTool(Tool):
    name = "get_weather"
    description = "Get current weather for a location. Returns temperature and conditions."
    inputs = {
        "location": {
            "type": "string",
            "description": "City name or coordinates (e.g., 'London' or '51.5,-0.1')"
        },
        "units": {
            "type": "string",
            "description": "Temperature units: 'celsius' or 'fahrenheit'",
            "nullable": True  # Optional parameter
        }
    }
    output_type = "string"

    def forward(self, location: str, units: str = "celsius") -> str:
        # Implementation
        import requests
        response = requests.get(f"https://api.weather.com/{location}")
        data = response.json()
        return f"{data['temp']}°{'C' if units == 'celsius' else 'F'}, {data['conditions']}"
```

### Tool from Function

```python
from smolagents import Tool

def my_function(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y

tool = Tool.from_function(
    my_function,
    name="adder",
    description="Adds two numbers together"
)
```

## Tool Input Types

### Supported Types

| Type | JSON Schema | Python Type |
|------|-------------|-------------|
| `string` | `"type": "string"` | `str` |
| `integer` | `"type": "integer"` | `int` |
| `number` | `"type": "number"` | `float` |
| `boolean` | `"type": "boolean"` | `bool` |
| `array` | `"type": "array"` | `list` |
| `object` | `"type": "object"` | `dict` |
| `image` | Special handling | `PIL.Image` |
| `audio` | Special handling | Audio data |

### Complex Input Types

```python
from smolagents import Tool

class DataProcessorTool(Tool):
    name = "process_data"
    description = "Process a list of data points."
    inputs = {
        "data": {
            "type": "array",
            "items": {"type": "number"},
            "description": "List of numeric values to process"
        },
        "options": {
            "type": "object",
            "properties": {
                "normalize": {"type": "boolean"},
                "round_to": {"type": "integer"}
            },
            "description": "Processing options"
        }
    }
    output_type = "array"

    def forward(self, data: list, options: dict = None) -> list:
        options = options or {}
        result = data
        if options.get("normalize"):
            max_val = max(result)
            result = [x / max_val for x in result]
        if "round_to" in options:
            result = [round(x, options["round_to"]) for x in result]
        return result
```

## Loading External Tools

### From Hugging Face Hub

```python
from smolagents import load_tool

# Public tool
tool = load_tool("m-ric/text-to-image", trust_remote_code=True)

# Private tool (requires authentication)
tool = load_tool("your-org/private-tool", trust_remote_code=True, token="hf_...")
```

### From Gradio Space

```python
from smolagents import Tool

# Auto-detect inputs/outputs
tool = Tool.from_space(
    "black-forest-labs/FLUX.1-schnell",
    name="flux_generator",
    description="Generate images using FLUX model"
)

# With specific API
tool = Tool.from_space(
    "gradio/image-classifier",
    name="classifier",
    description="Classify images",
    api_name="/predict"
)
```

### From LangChain

```python
from smolagents import Tool
from langchain.agents import load_tools

# Single tool
lc_tools = load_tools(["serpapi"])
search_tool = Tool.from_langchain(lc_tools[0])

# Multiple tools
for lc_tool in load_tools(["serpapi", "llm-math"]):
    agent.tools[lc_tool.name] = Tool.from_langchain(lc_tool)
```

### From MCP Server

```python
from smolagents import ToolCollection, CodeAgent
from mcp import StdioServerParameters, HttpServerParameters

# Stdio server
stdio_params = StdioServerParameters(
    command="uvx",
    args=["mcp-server-filesystem", "/path/to/dir"]
)

# HTTP server
http_params = HttpServerParameters(
    url="http://localhost:8080/mcp"
)

# Load tools
with ToolCollection.from_mcp(stdio_params, trust_remote_code=True) as tools:
    agent = CodeAgent(tools=[*tools.tools], model=model)

# With structured output
with ToolCollection.from_mcp(
    stdio_params, 
    trust_remote_code=True, 
    structured_output=True
) as tools:
    agent = CodeAgent(tools=[*tools.tools], model=model)
```

## Sharing Tools

### Push to Hub

```python
from smolagents import Tool

class MyTool(Tool):
    name = "my_tool"
    description = "Does something useful"
    inputs = {"x": {"type": "string", "description": "Input"}}
    output_type = "string"
    
    def forward(self, x: str) -> str:
        return f"Processed: {x}"

tool = MyTool()
tool.push_to_hub("your-username/my-tool", private=False)
```

### Save Locally

```python
tool.save("./my_tool")
# Creates directory with tool files

# Load later
from smolagents import load_tool
tool = load_tool("./my_tool")
```

## Tool Best Practices

### 1. Clear Descriptions

```python
# Bad
@tool
def process(x):
    """Process x."""
    return x

# Good
@tool
def analyze_sentiment(text: str) -> dict:
    """
    Analyze the sentiment of input text.
    
    Returns a dictionary with:
    - sentiment: 'positive', 'negative', or 'neutral'
    - confidence: float between 0 and 1
    - keywords: list of sentiment-bearing words found
    
    Args:
        text: The text to analyze. Can be any length but works best with 1-5 sentences.
    """
    ...
```

### 2. Input Validation

```python
@tool
def divide(a: float, b: float) -> float:
    """
    Divide a by b.
    
    Args:
        a: Numerator.
        b: Denominator (must not be zero).
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
```

### 3. Error Handling

```python
@tool
def fetch_url(url: str) -> str:
    """
    Fetch content from a URL.
    
    Args:
        url: The URL to fetch.
    """
    import requests
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text[:5000]  # Limit output size
    except requests.Timeout:
        return "Error: Request timed out"
    except requests.HTTPError as e:
        return f"Error: HTTP {e.response.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"
```

### 4. Output Size Limits

```python
@tool
def search_database(query: str) -> str:
    """
    Search the database.
    
    Args:
        query: Search query.
    """
    results = db.search(query)
    
    # Limit results to prevent token overflow
    if len(results) > 10:
        results = results[:10]
        truncated = True
    else:
        truncated = False
    
    output = "\n".join([str(r) for r in results])
    if truncated:
        output += "\n[Results truncated, showing first 10]"
    
    return output
```

### 5. Stateful Tools

```python
from smolagents import Tool

class ConversationTool(Tool):
    name = "conversation_memory"
    description = "Store and retrieve conversation context."
    inputs = {
        "action": {
            "type": "string",
            "description": "'store' to save, 'retrieve' to get, 'clear' to reset"
        },
        "data": {
            "type": "string",
            "description": "Data to store (only for 'store' action)",
            "nullable": True
        }
    }
    output_type = "string"

    def __init__(self):
        super().__init__()
        self.memory = []

    def forward(self, action: str, data: str = None) -> str:
        if action == "store":
            self.memory.append(data)
            return f"Stored. Memory now has {len(self.memory)} items."
        elif action == "retrieve":
            return "\n".join(self.memory) if self.memory else "Memory is empty."
        elif action == "clear":
            self.memory = []
            return "Memory cleared."
        else:
            return f"Unknown action: {action}"
```

## Tool Debugging

```python
# Test tool directly
tool = MyTool()
result = tool.forward("test input")
print(result)

# Check tool metadata
print(f"Name: {tool.name}")
print(f"Description: {tool.description}")
print(f"Inputs: {tool.inputs}")
print(f"Output type: {tool.output_type}")

# Verify agent sees tool correctly
agent = CodeAgent(tools=[tool], model=model)
print(agent.tools[tool.name].to_tool_calling_prompt())
```