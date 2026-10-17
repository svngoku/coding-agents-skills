# Agent Middleware

Middleware is how LangChain v1 customizes `create_agent`: summarization, retries, fallbacks,
guardrails, human approval, dynamic prompts, models and tools. It replaces the old
`pre_model_hook`/`post_model_hook` and most custom agent loops. Everything below is imported from
`langchain.agents.middleware` (verified against `langchain` 1.4.6).

## Hooks and Execution Order

| Hook | Style | When it runs |
|------|-------|--------------|
| `before_agent` | node | Once per invocation, before the loop |
| `before_model` | node | Before each model call |
| `wrap_model_call` | wrap | Around each model call (can retry, swap model/tools/prompt, short-circuit) |
| `after_model` | node | After each model response |
| `wrap_tool_call` | wrap | Around each tool call |
| `after_agent` | node | Once per invocation, after the loop |

With `middleware=[m1, m2, m3]`:

- `before_*` hooks run first to last (`m1`, `m2`, `m3`)
- `after_*` hooks run last to first (`m3`, `m2`, `m1`)
- `wrap_*` hooks nest like function calls: `m1` wraps `m2`, which wraps `m3`, which wraps the model

Node-style hooks return a state update `dict` (or `None`). They can also return
`{"jump_to": "end" | "tools" | "model"}` if the hook declares `can_jump_to`.

## Built-in Middleware

| Middleware | Purpose | Key arguments |
|------------|---------|---------------|
| `SummarizationMiddleware` | Summarize old messages near the context limit | `model`, `trigger=("tokens", 4000)` / `("fraction", 0.8)` / `("messages", 50)` (or a list), `keep=("messages", 20)` |
| `HumanInTheLoopMiddleware` | Pause before tool calls for approve / edit / reject / respond | `interrupt_on={"tool": True \| False \| {"allowed_decisions": [...]}}`, `description_prefix` |
| `ModelCallLimitMiddleware` | Cap model calls | `thread_limit`, `run_limit`, `exit_behavior="end"\|"error"` |
| `ToolCallLimitMiddleware` | Cap tool calls (all or one tool) | `tool_name`, `thread_limit`, `run_limit`, `exit_behavior` |
| `ModelFallbackMiddleware` | Try other models when the primary fails | positional models: `ModelFallbackMiddleware("openai:gpt-5.5", "google_genai:gemini-3.6-flash")` |
| `ModelRetryMiddleware` | Retry failed model calls with backoff | `max_retries=2`, `retry_on`, `on_failure`, `backoff_factor`, `initial_delay`, `jitter` |
| `ToolRetryMiddleware` | Retry failed tool calls with backoff | same as above plus `tools=[...]` |
| `ToolErrorMiddleware` | Turn chosen tool exceptions into error `ToolMessage`s | `on_error=callable` (return content to surface, `None` to re-raise), `tools` |
| `PIIMiddleware` | Detect and handle PII | `pii_type` (`"email"`, `"credit_card"`, `"ip"`, `"mac_address"`, `"url"` or custom with `detector`), `strategy="block"\|"redact"\|"mask"\|"hash"`, `apply_to_input/output/tool_results` |
| `TodoListMiddleware` | Adds a `write_todos` planning tool | `system_prompt`, `tool_description` |
| `LLMToolSelectorMiddleware` | Let a cheap model pick relevant tools before each call | `model`, `max_tools`, `always_include` |
| `ContextEditingMiddleware` | Clear old tool outputs when context grows | `edits=[ClearToolUsesEdit(trigger=100000, keep=3)]` |
| `ShellToolMiddleware` | Give the agent a persistent shell | `workspace_root`, `execution_policy` (`HostExecutionPolicy`, `DockerExecutionPolicy`, `CodexSandboxExecutionPolicy`) |
| `FilesystemFileSearchMiddleware` | Glob and grep tools over a directory | `root_path`, `use_ripgrep`, `max_file_size_mb` |
| `ProviderToolSearchMiddleware` | Defer tools behind the provider's native tool search | `searchable_tools` |
| `LLMToolEmulator` | Emulate tools with an LLM (testing) | `tools`, `model` |

Provider packages ship their own middleware too, for example `AnthropicPromptCachingMiddleware`
and the Claude memory/text-editor/bash tool middleware in `langchain_anthropic.middleware`.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import (
    ContextEditingMiddleware, ClearToolUsesEdit, ModelCallLimitMiddleware,
    ModelRetryMiddleware, ToolErrorMiddleware,
)

def on_tool_error(exc: Exception):
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return f"Tool failed with {type(exc).__name__}; try again or use another tool."
    return None   # anything else propagates and halts the run

agent = create_agent(
    model="anthropic:claude-sonnet-5",
    tools=[search, fetch_page],
    middleware=[
        ModelRetryMiddleware(max_retries=3),
        ToolErrorMiddleware(on_error=on_tool_error),
        ModelCallLimitMiddleware(run_limit=15, exit_behavior="end"),
        ContextEditingMiddleware(edits=[ClearToolUsesEdit(trigger=100_000, keep=3)]),
    ],
)
```

## Custom Middleware with Decorators

```python
from collections.abc import Callable
from langchain.agents.middleware import (
    AgentState, ModelRequest, ModelResponse, after_model, before_model,
    dynamic_prompt, wrap_model_call, wrap_tool_call,
)
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, ToolMessage

@before_model(can_jump_to=["end"])
def stop_on_long_threads(state: AgentState, runtime) -> dict | None:
    if len(state["messages"]) > 200:
        return {"messages": [AIMessage("This thread is too long; please start a new one.")], "jump_to": "end"}
    return None

@after_model
def log_response(state: AgentState, runtime) -> None:
    print("model said:", state["messages"][-1].content)

@dynamic_prompt
def personalized_prompt(request: ModelRequest) -> str:
    return f"You are a helpful assistant. The user's name is {request.runtime.context.user_name}."

large_model = init_chat_model("anthropic:claude-sonnet-5")

@wrap_model_call
def route_model(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
    """Use a bigger model for long conversations."""
    if len(request.messages) > 20:
        request = request.override(model=large_model)
    return handler(request)

@wrap_model_call
def gate_tools(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
    """Only expose admin tools to admins."""
    if not request.runtime.context.is_admin:
        request = request.override(tools=[t for t in request.tools if not t.name.startswith("admin_")])
    return handler(request)

@wrap_tool_call
def audit(request, handler) -> ToolMessage:
    print("calling", request.tool_call["name"], request.tool_call["args"])
    return handler(request)
```

`ModelRequest` fields: `model`, `messages`, `system_message`, `tools`, `tool_choice`,
`response_format`, `model_settings`, `state`, `runtime`. Change them with
`request.override(...)` rather than mutating the request. `ToolCallRequest` exposes `tool_call`,
`tool`, `state` and `runtime`.

Appending to the system prompt without losing existing content blocks:

```python
from langchain.messages import SystemMessage

@wrap_model_call
def add_context(request: ModelRequest, handler):
    blocks = list(request.system_message.content_blocks) + [{"type": "text", "text": "Today is Monday."}]
    return handler(request.override(system_message=SystemMessage(content=blocks)))
```

## Class-Based Middleware and Custom State

Subclass `AgentMiddleware` when you need several hooks, custom state, or bundled tools
(`state_schema`, `tools` and `transformers` class attributes are picked up by `create_agent`):

```python
from typing import Any
from typing_extensions import NotRequired
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState

class CounterState(AgentState):
    model_call_count: NotRequired[int]

class CallCounter(AgentMiddleware[CounterState]):
    state_schema = CounterState

    def after_model(self, state: CounterState, runtime) -> dict[str, Any] | None:
        return {"model_call_count": state.get("model_call_count", 0) + 1}

agent = create_agent(model="anthropic:claude-sonnet-5", tools=[], middleware=[CallCounter()])
```

Async agents (`ainvoke`, `astream`) call the async variants: `abefore_model`, `aafter_model`,
`awrap_model_call`, `awrap_tool_call`, `abefore_agent`, `aafter_agent`. Implement them (or use
`async def` with the decorators) when your middleware does I/O.

## Tips

- Order matters: put retries and fallbacks outermost, then guardrails, then summarization.
- `HumanInTheLoopMiddleware` and anything that interrupts needs a `checkpointer`.
- Keep middleware stateless; store per-run data in agent state (`state_schema`) or `runtime.context`.
- Prefer built-ins over hand-written loops around `agent.invoke`; they are traced and composable.
