# SmolAgents Model Reference

## InferenceClientModel

Primary model class for Hugging Face Inference API.

### Parameters

```python
InferenceClientModel(
    model_id: str = "Qwen/Qwen2.5-Coder-32B-Instruct",  # Model ID from HF Hub
    provider: str = None,       # Inference provider: "together", "sambanova", "fireworks", etc.
    token: str = None,          # HF API token (or set HF_TOKEN env var)
    timeout: int = 120,         # Request timeout in seconds
    temperature: float = 0.5,   # Sampling temperature
    max_tokens: int = None,     # Max output tokens
)
```

### Supported Providers

- `together` - Together AI
- `sambanova` - SambaNova
- `fireworks` - Fireworks AI
- `cerebras` - Cerebras
- `cohere` - Cohere
- `hyperbolic` - Hyperbolic
- `nebius` - Nebius
- `novita` - Novita
- `replicate` - Replicate

### Example

```python
from smolagents import InferenceClientModel

# Default (free tier)
model = InferenceClientModel()

# Specific model with provider
model = InferenceClientModel(
    model_id="meta-llama/Llama-3.3-70B-Instruct",
    provider="together",
    temperature=0.7
)

# DeepSeek R1
model = InferenceClientModel(
    model_id="deepseek-ai/DeepSeek-R1",
    provider="together"
)
```

## LiteLLMModel

Unified interface for 100+ LLM providers via LiteLLM.

### Parameters

```python
LiteLLMModel(
    model_id: str,              # Provider/model format: "anthropic/claude-3-5-sonnet-latest"
    api_key: str = None,        # API key (or set via environment variable)
    api_base: str = None,       # Custom API endpoint
    temperature: float = 0.5,
    max_tokens: int = None,
    num_ctx: int = None,        # For Ollama: context window size
)
```

### Provider Formats

| Provider | Format | Env Variable |
|----------|--------|--------------|
| Anthropic | `anthropic/claude-3-5-sonnet-latest` | `ANTHROPIC_API_KEY` |
| OpenAI | `gpt-4o`, `gpt-4-turbo` | `OPENAI_API_KEY` |
| Google | `gemini/gemini-pro` | `GEMINI_API_KEY` |
| Mistral | `mistral/mistral-large-latest` | `MISTRAL_API_KEY` |
| Ollama | `ollama_chat/llama3.2` | None |
| Groq | `groq/llama3-70b-8192` | `GROQ_API_KEY` |
| AWS Bedrock | `bedrock/anthropic.claude-3-sonnet` | AWS credentials |
| Azure | `azure/gpt-4` | Azure credentials |

### Examples

```python
from smolagents import LiteLLMModel

# Anthropic Claude
model = LiteLLMModel(
    model_id="anthropic/claude-3-5-sonnet-latest",
    api_key="sk-ant-..."
)

# OpenAI
model = LiteLLMModel(
    model_id="gpt-4o",
    api_key="sk-..."
)

# Ollama (local)
model = LiteLLMModel(
    model_id="ollama_chat/llama3.2",
    api_base="http://localhost:11434",
    num_ctx=8192  # IMPORTANT: Default 2048 is too small
)

# Groq (fast inference)
model = LiteLLMModel(
    model_id="groq/llama3-70b-8192",
    api_key="gsk_..."
)
```

## TransformersModel

Run models locally using Hugging Face Transformers.

### Parameters

```python
TransformersModel(
    model_id: str,              # HF model ID
    device_map: str = "auto",   # Device placement: "auto", "cuda", "cpu"
    max_new_tokens: int = 1500, # Max output tokens
    torch_dtype: str = "auto",  # Tensor dtype
    trust_remote_code: bool = False,
)
```

### Example

```python
from smolagents import TransformersModel

model = TransformersModel(
    model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    max_new_tokens=4096,
    device_map="auto"
)

# Smaller model for testing
model = TransformersModel(
    model_id="meta-llama/Llama-3.2-3B-Instruct"
)
```

## OpenAIModel

For OpenAI and OpenAI-compatible APIs.

### Parameters

```python
OpenAIModel(
    model_id: str,
    api_key: str = None,        # Or set OPENAI_API_KEY
    api_base: str = None,       # For compatible endpoints
    temperature: float = 0.5,
)
```

### Examples

```python
from smolagents import OpenAIModel

# OpenAI direct
model = OpenAIModel(model_id="gpt-4o")

# OpenRouter
model = OpenAIModel(
    model_id="openai/gpt-4o",
    api_base="https://openrouter.ai/api/v1",
    api_key="sk-or-..."
)

# Together AI (OpenAI-compatible)
model = OpenAIModel(
    model_id="deepseek-ai/DeepSeek-R1",
    api_base="https://api.together.xyz/v1/",
    api_key="..."
)
```

## AzureOpenAIModel

For Azure OpenAI deployments.

### Parameters

```python
AzureOpenAIModel(
    model_id: str,              # Your deployment name
    azure_endpoint: str,        # Azure endpoint URL
    api_key: str,
    api_version: str,           # e.g., "2024-02-15-preview"
)
```

### Example

```python
from smolagents import AzureOpenAIModel
import os

model = AzureOpenAIModel(
    model_id=os.environ["AZURE_OPENAI_MODEL"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.environ["OPENAI_API_VERSION"]
)
```

## AmazonBedrockModel

For AWS Bedrock.

### Parameters

```python
AmazonBedrockModel(
    model_id: str,              # Bedrock model ID
    # Uses default AWS credentials from environment
)
```

### Example

```python
from smolagents import AmazonBedrockModel

model = AmazonBedrockModel(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0"
)
```

## Recommended Models by Use Case

### General Code Agents
- `Qwen/Qwen2.5-Coder-32B-Instruct` (default, excellent)
- `meta-llama/Llama-3.3-70B-Instruct`
- `anthropic/claude-3-5-sonnet-latest`
- `gpt-4o`

### Vision/Multimodal Agents
- `Qwen/Qwen2-VL-72B-Instruct`
- `gpt-4o` (with vision)
- `anthropic/claude-3-5-sonnet-latest`

### Cost-Effective
- `meta-llama/Llama-3.2-3B-Instruct` (local)
- `groq/llama3-70b-8192` (fast + cheap)
- `Qwen/Qwen2.5-7B-Instruct`

### Reasoning Tasks
- `deepseek-ai/DeepSeek-R1`
- `anthropic/claude-3-5-sonnet-latest`
- `o1-preview` (OpenAI)

### Local Development
- `ollama_chat/llama3.2` (via Ollama)
- `meta-llama/Llama-3.2-3B-Instruct` (Transformers)