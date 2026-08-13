# LangChain Customer-Support Agent

Write a **self-contained Python module** and save it as **`agent.py`** in the current directory. The module must build a reusable customer-support agent using **current LangChain APIs** (LangChain 1.x / LangGraph style). Do **not** use the deprecated `LLMChain` or `SequentialChain` APIs.

Your `agent.py` must include, in order:

1. **Model configuration** — configure the chat model with `init_chat_model` from `langchain.chat_models` using any current provider/model id (e.g. `claude-sonnet-4-5-20250929`). Pick a sensible `temperature` for a support agent.

2. **At least two tools** decorated with `@tool` from `langchain.tools`. Every tool must have:
   - type hints on **every parameter** and on the **return value**, and
   - a **real, specific docstring** — the docstring becomes the tool description the model sees, so it must say what the tool does (no placeholder or TODO text).
   At least **one** tool must take a `ToolRuntime[Context]` parameter and read from `runtime.context` (e.g. the current `user_id`) instead of using global state.

3. **A `Context` dataclass** — define `@dataclass class Context` (from `dataclasses`) with at least a `user_id: str` field, and pass `context_schema=Context` to `create_agent` so the runtime context is typed.

4. **Memory** — wire a checkpointer with `InMemorySaver` from `langgraph.checkpoint.memory` and pass it to `create_agent(..., checkpointer=...)` so multi-turn conversations persist per `thread_id`.

5. **Structured output** — define `@dataclass class Response` (at least an `answer` field) and pass `response_format=ToolStrategy(Response)` from `langchain.agents.structured_output` to `create_agent`.

6. **The agent** — build it with `create_agent` from `langchain.agents`, passing a **realistic customer-support system prompt** (who the assistant is, which tools exist and when to use them, tone).

7. **Example invocation** — at the bottom, include a runnable example that:
   - calls `agent.invoke(..., config={"configurable": {"thread_id": "..."}}, context=Context(user_id="..."))`,
   - uses a **configurable `thread_id`** so conversations can be resumed, and
   - passes a **`context=Context(...)` instance** matching the `ToolRuntime[Context]` schema.
   Guard the example with `if __name__ == "__main__":` so the module stays importable.

Constraints: no network calls, no API keys, no external files — the module must be runnable as a script with only `langchain`, `langgraph`, and the Python standard library installed.
