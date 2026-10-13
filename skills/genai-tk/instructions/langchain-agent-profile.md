# Task: Define a "research" LangChain deep-agent profile for genai-tk

You are working inside a genai-tk project. Using the toolkit's **YAML-first**
conventions and its current agent profile schema, create a deep LangChain
agent profile with the slug **research** and a small Python consumer that
instantiates it.

## Outputs — create exactly these two files

### 1. `config/agents.yaml`

A genai-tk agent profile file. It must contain:

- a top-level `agent_defaults` mapping whose `default_profile` is `research`;
- a top-level `agents` mapping (keyed by profile slug) with a `research`
  profile that has:

  | Field | Required value |
  |-------|----------------|
  | `harness` | `langchain` |
  | `name` | `Research` (display name) |
  | `type` | `deep` |
  | `llm` | canonical `model_id@provider` form (e.g. `gpt-4o-mini@openai`), **not** a bare model name |
  | `enable_planning` | `true` |
  | `enable_file_system` | `true` |
  | `tools` | the toolkit's built-in web search tool factory, referenced by its fully qualified Python path, with `verbose: false` |
  | `mcp_servers` | `[filesystem]` |
  | `skill_directories` | a list with a path under `${paths.project}/skills` (keep the literal `${paths.project}` interpolation) |
  | `checkpointer` | `{ type: memory }` |

  Follow the profile schema used by current genai-tk. Do not invent keys:
  unknown keys are silently ignored by the toolkit, so a wrong key name
  quietly does nothing. YAML is indentation-sensitive (spaces only, no tabs).

### 2. `use_agent.py`

A short, runnable Python snippet that:

- imports `LangchainAgent` from `genai_tk.agents.langchain`;
- constructs `LangchainAgent("research", ...)` with **at least one override**,
  for example `llm="claude-haiku@openrouter"` or `sandbox="docker"`;
- runs one query and prints the result.

## Guidance

- Use `model_id@provider` everywhere a model is named.
- Do not hard-code agent logic in Python: the YAML profile drives the agent,
  the snippet only loads it and overrides a field or two.
- The snippet must be syntactically valid Python that a maintainer could run
  inside a genai-tk project.

Save both files, then double-check that `config/agents.yaml` parses as YAML
before finishing.
