# LLM Rubric: LangChain Customer-Support Agent

Score the agent's solution (`agent.py`) from **0.0 to 1.0** on overall quality using the criteria below.
Be strict but fair: a near-perfect solution should score ~0.9-1.0, a working-but-sloppy one ~0.6-0.8,
and one that misses core requirements below 0.5.

## 1. Current-API correctness (30%)

- Builds the agent with `create_agent` from `langchain.agents` and configures the model with
  `init_chat_model` from `langchain.chat_models`.
- Uses `ToolStrategy(Response)` for structured output and `InMemorySaver` for checkpointing.
- **No deprecated APIs**: no `LLMChain`, `SequentialChain`, or other pre-LangGraph patterns.
- Imports come from the correct current modules (`langchain.agents`, `langchain.chat_models`,
  `langchain.tools`, `langgraph.checkpoint.memory`, `langchain.agents.structured_output`).

## 2. Tool design quality (25%)

- At least two `@tool` functions with complete type hints on parameters and return values.
- Every tool has a real, specific docstring that accurately describes what it does - not a placeholder.
- Tools use `ToolRuntime[Context]` for user context instead of global state, and the `Context`
  dataclass carries at least a `user_id`.
- Reserved parameter names (`config`, `runtime`) are not misused as ordinary arguments.

## 3. Realistic system prompt (15%)

- A plausible customer-support persona: who the assistant is, which tools exist and when to use them, tone.
- Not generic boilerplate like "You are a helpful assistant" with nothing else.

## 4. Structured-output ergonomics (15%)

- `Response` dataclass with sensible fields (e.g. `answer` plus optional metadata) and defaults for
  optional fields.
- The example invocation accesses the result via `result["structured_response"]` or equivalent.

## 5. Readability and craftsmanship (15%)

- Clean module layout: imports at top, small named functions, no dead code.
- Runs as a standalone script: the example invocation is guarded by `if __name__ == "__main__":`.
- Consistent naming, no leftover TODOs or commented-out experiments.

## Scoring guidance

- Start at 1.0 and subtract for each weakness, or start at 0 and add - be consistent across runs.
- Penalize missing core pieces heavily: no checkpointer, no structured output, fewer than two tools,
  or no example invocation.
- Do not penalize the choice of model id/provider as long as it is a plausible current model id.
- Output a single number (0.0-1.0) as the final score.
