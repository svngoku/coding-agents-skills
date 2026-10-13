# Quality rubric — genai-tk "research" LangChain deep-agent profile

Score the agent's solution **0.0–1.0**. Anchor: 1.0 = a profile and consumer
snippet that a genai-tk maintainer would merge unchanged. Partial credit is
proportional — award it per criterion.

## Criteria

1. **Current schema (25%)** — the profile uses the current genai-tk schema:
   a top-level `agents:` mapping keyed by slug, plus `agent_defaults:`.
   Deduct heavily for the removed legacy schema (`langchain_agents:` with a
   `profiles:` list, short tool names such as `spec: web_search`, or a
   per-profile `skills: {directories: ...}` block, which the toolkit
   silently ignores).

2. **YAML-first philosophy (20%)** — the agent is *defined* in
   `config/agents.yaml`, not hard-coded in Python. `use_agent.py` only
   *loads* the profile via `LangchainAgent("research", ...)` and passes a
   small override. Python code must not redefine the agent, its tools, or its
   wiring.

3. **Profile completeness (35%)** — the `research` profile is a complete
   `deep` LangChain agent. Deduct proportionally for each missing or wrong
   field: `harness: langchain`, `type: deep`, `llm` in model_id@provider
   form, `enable_planning: true`, `enable_file_system: true`, a search tool
   referenced by its qualified factory path (for example
   `genai_tk.agents.tools.langchain.search_tools_factory.create_search_function`),
   `mcp_servers: [filesystem]`, `skill_directories` under
   `${paths.project}/skills`, and `checkpointer: {type: memory}`.

4. **model_id@provider discipline (10%)** — every model reference uses the
   canonical `model_id@provider` form. No bare model names and no
   placeholders such as `<your-model>`.

5. **Consumer snippet correctness (10%)** — `use_agent.py` imports
   `LangchainAgent` from `genai_tk.agents.langchain`, constructs the
   `research` profile, passes at least one real constructor override
   (`llm=`, `sandbox=`, `checkpointer=`, ...), and is valid Python.

## Penalties

- **−20%** if the solution would not run as-is: unparseable YAML, missing
  required keys, or Python syntax errors.
- **−10%** for stray or contradictory settings (for example a Docker
  `backend:` plus `sandbox="local"`, or keys that genai-tk does not read).
