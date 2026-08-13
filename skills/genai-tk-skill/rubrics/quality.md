# Quality rubric — genai-tk "Research" langchain agent profile

Score the agent's solution **0.0–1.0**. Anchor: 1.0 = a profile and consumer
snippet that a genai-tk maintainer would merge unchanged. Partial credit is
proportional — award it per criterion.

## Criteria

1. **YAML-first philosophy (25%)** — the agent is *defined* in
   `config/agents/langchain.yaml`, not hard-coded in Python. `use_agent.py`
   only *loads* the profile via `LangchainAgent("Research", ...)` and passes a
   small override. Python code must not redefine the agent, its tools, or its
   wiring.

2. **model_id@provider discipline (15%)** — every model reference uses the
   canonical `model_id@provider` form (e.g. `gpt_41@openai`). No bare model
   names (like `gpt-4o-mini`), no placeholders such as `<your-model>`.

3. **Profile completeness (40%)** — the Research profile is a complete `deep`
   agent. Deduct proportionally for each missing or wrong field:
   `type: deep`, `llm` in model_id@provider format, `enable_planning: true`,
   `enable_file_system: true`, a `web_search` tool with a sensible provider
   config, `mcp_servers` listing at least one server, `skills.directories`
   pointing at the project skills via `${paths.project}/skills`, and a
   `checkpointer` of `type: memory`.

4. **Sensible defaults & readability (10%)** — `default_profile` set
   consistently, no redundant or contradictory settings, clean comments,
   consistent indentation, no stray keys.

5. **Consumer snippet correctness (10%)** — `use_agent.py` imports
   `LangchainAgent` from `genai_tk.agents.langchain`, constructs the
   `Research` profile, and passes at least one override (`llm=` or
   `sandbox=`). It must be syntactically valid Python.

## Penalties

- **−20%** if the solution would not run as-is: unparseable YAML, missing
  required keys, or Python syntax errors.
- **−10%** if the profile is a copy-paste of the skill's example with a
  renamed label but no reasoning about the configuration.
