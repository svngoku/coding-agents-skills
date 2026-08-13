# Task: Build a smolagents CodeAgent Module

Write a **self-contained Python module** and save it as **`agent.py`** in the current
directory. The module uses Hugging Face's `smolagents` library to build a `CodeAgent`
that solves a **multi-step financial analysis problem**: given a list of monthly
investment returns, the agent computes summary statistics (mean, median, standard
deviation) and projects compound growth over several months — a job that genuinely
benefits from planning across multiple steps.

Build the module to meet ALL of the following requirements:

1. **Imports** — Import `CodeAgent`, `tool`, and one model class (`InferenceClientModel`
   or `LiteLLMModel`) from `smolagents`.

2. **Custom tool** — Define at least one custom tool using the `@tool` decorator (for
   example a `compound_growth` tool). The tool MUST have:
   - a docstring that describes what it does and includes an `Args:` section for each parameter;
   - type hints on every parameter and a return type hint.

3. **Explicit model configuration** — Construct the model explicitly with a concrete
   `model_id` (and a provider where applicable), for example:
   - `InferenceClientModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct", provider="together")`
   - `LiteLLMModel(model_id="anthropic/claude-3-5-sonnet-latest")`
   Do NOT hardcode API keys in the source — read them from environment variables
   (`HF_TOKEN`, `ANTHROPIC_API_KEY`, etc.). Pass the configured model into the `CodeAgent`.

4. **Agent configuration** — Configure the `CodeAgent` with:
   - `max_steps=15`
   - `planning_interval=3` (enables planning every 3 steps — use it, this task needs planning)
   - `verbosity_level=1`
   - `additional_authorized_imports` restricted to a small whitelist of stdlib modules,
     e.g. `["math", "statistics"]` (no wildcards, no broad/open-ended lists)

5. **Run example** — End the module with an example invocation, e.g.
   `agent.run("Analyze the monthly returns [0.01, -0.02, 0.03] and project the balance of a $1000 investment after 12 months.")`
   The module must be importable and must NOT make any network calls or connect to any
   service at import time.

Keep the module dependency-free apart from `smolagents` and the Python standard
library: the custom tool should be implementable with `math`/`statistics` only.

Save your finished module as **`agent.py`** in the current directory.
