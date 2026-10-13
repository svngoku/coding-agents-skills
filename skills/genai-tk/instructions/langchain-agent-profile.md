# Task: Define a "Research" langchain agent profile for genai-tk

You are working inside a genai-tk project. Using the toolkit's **YAML-first**
conventions, create a deep LangChain agent profile named **Research** and a
small Python consumer that instantiates it.

## Outputs — create exactly these two files

### 1. `config/agents/langchain.yaml`

A genai-tk `langchain_agents` profile file. It must contain:

- a top-level `langchain_agents` key holding `default_profile: "Research"`
  and a `profiles` list;
- a profile named `Research` with:

  | Field | Required value |
  |-------|----------------|
  | `type` | `deep` |
  | `llm` | canonical `model_id@provider` format (e.g. `gpt_41@openai`) — **not** a bare model name |
  | `enable_planning` | `true` |
  | `enable_file_system` | `true` |
  | `tools` | an entry with `spec: web_search` and a config such as `{ provider: serper, max_results: 5 }` |
  | `mcp_servers` | `[tavily-mcp]` |
  | `skills.directories` | a path pointing at `${paths.project}/skills` — keep the literal `${paths.project}` interpolation |
  | `checkpointer` | `{ type: memory }` |

  Follow the profile schema exactly as shown in the skill's SKILL.md example;
  do not invent new keys. YAML is indentation-sensitive — spaces only, no tabs.

### 2. `use_agent.py`

A short, runnable Python snippet that:

- imports `LangchainAgent` from `genai_tk.agents.langchain`;
- constructs `LangchainAgent("Research", ...)` with **at least one override**,
  e.g. `llm="gpt_4o@openai"` or `sandbox="docker"`.

## Guidance

- Use `model_id@provider` everywhere a model is named — it is the toolkit's
  canonical identifier.
- Do not hard-code agent logic in Python: the YAML profile drives the agent,
  the snippet only loads it and overrides a field or two.
- The snippet must be syntactically valid Python that a maintainer could run
  inside a genai-tk project.

Save both files, then double-check that `config/agents/langchain.yaml` parses
as YAML before finishing.
