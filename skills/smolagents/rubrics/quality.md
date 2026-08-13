# LLM Rubric — smolagents CodeAgent module (financial-analysis-codeagent)

Judge the agent's `agent.py` on a 0.0–1.0 scale. Use the full range and be strict:
a solution that only partially meets the criteria should score well below 1.0.

Weighted criteria (approximate weights in parentheses):

1. **Security posture (30%)** — `additional_authorized_imports` is a tight, explicit
   whitelist of stdlib modules (e.g. `["math", "statistics"]`) with no wildcard or
   dangerously broad authorizations; no hardcoded API keys or secrets (credentials come
   from environment variables); no `eval`/`exec` of untrusted input; the module makes
   no network calls at import time and keeps the local-execution surface minimal.
2. **Planning for a multi-step task (20%)** — `planning_interval=3` is set so the agent
   re-plans during the multi-step analysis; `max_steps` is bounded at 15 so the agent
   cannot run away; the custom tool is genuinely used by the workflow rather than decorative.
3. **Tool design quality (25%)** — at least one custom `@tool` function with a clear
   docstring (including an `Args:` section), type hints on all parameters and the return
   value, a sensible name, and a pure stdlib implementation.
4. **Model configuration clarity (15%)** — the model is explicitly configured with a
   concrete `model_id` (and provider where applicable — e.g. `provider="together"` or a
   LiteLLM `provider/model_id` string like `anthropic/claude-3-5-sonnet-latest`);
   API keys come from environment variables; the model is clearly passed to the `CodeAgent`.
5. **Runnable example (10%)** — the module ends with a concrete `agent.run(...)` example
   and is importable as a standalone script (no network at import time).

Output a single score (0.0–1.0) with a short justification per criterion.
