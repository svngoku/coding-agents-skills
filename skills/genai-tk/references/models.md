# Models, Providers & Decision Models — references/models.md

genai-tk resolves every model through one identifier format, `model_id@provider`, and one provider registry
(`config/providers/providers.yaml`). The factories return plain LangChain objects.

## Imports

```python
from genai_tk.core.factories import (
    get_llm,
    get_embeddings,
    get_decision_model,
    get_decision_model_from_chat_model,
    LlmFactory,
    EmbeddingsFactory,
)
```

Module paths: `genai_tk.core.factories.llm_factory`, `genai_tk.core.factories.embeddings_factory`,
`genai_tk.core.factories.decision_model_factory`. The old `genai_tk.core.llm_factory` and
`genai_tk.core.embeddings_factory` modules are gone with no shims.

## `get_llm`

```python
get_llm(
    llm: str = "default",          # tag, model_id@provider, or "model (effort)@provider[:routing]"
    json_mode: bool = False,
    streaming: bool = False,
    reasoning: bool | None = None, # legacy bool; prefer inline effort or reasoning_effort=
    cache: str | None = None,      # "memory" | "sqlite" | "no_cache"; None = LangChain global cache
    routing: str | None = None,
    provider_routing: dict | None = None,
    **kwargs,                      # forwarded to the chat model (temperature, max_tokens, ...)
) -> BaseChatModel
```

```python
llm = get_llm()                                   # llm.models.default of the active profile
llm = get_llm("fast_model")                       # any tag under llm.models
llm = get_llm("gpt-4o-mini@openai")
llm = get_llm("parrot_local@fake")                # echoes input; no key needed
llm = get_llm("gpt-4o-mini@openai", temperature=0.7, streaming=True)
llm = get_llm("gpt-4o-mini@openai", json_mode=True, cache="sqlite")
```

Defaults applied by the factory: `temperature=0.0`, `seed=42`, `max_retries=2`. If the provider's API-key variable
is empty, `get_llm` raises `EnvironmentError` instead of failing on the first call.

`llm_id=` and `llm_tag=` are deprecated and only warn; pass the identifier as `llm`.

### Reasoning effort

```python
llm = get_llm("o3 (high)@openai")                         # inline effort
llm = get_llm("claude-haiku (low)@openrouter")
llm = get_llm("o3@openai", reasoning_effort="medium")     # flat kwarg form
llm = get_llm("o3@openai", reasoning_max_tokens=4000)
```

Valid efforts: `low`, `medium`, `high`, `minimal`, `xhigh`, `max`, `none` (`none` disables reasoning). Anything
else raises `ValueError`. Provider mixins translate the effort into each API's payload.

**Do not** pass `reasoning={...}` to `get_llm`: the `reasoning: bool | None` parameter captures the dict and
validation fails (upstream `docs/core.md` shows this broken form). For a raw reasoning dict use
`LlmFactory(llm=..., llm_params={"reasoning": {...}})`.

### Gateway routing

On `openrouter`, `edenai`, and `edenai-eur`, append a strategy to the identifier or tag:

```python
llm = get_llm("claude-haiku@openrouter:speed")    # speed | cost | latency | exact
llm = get_llm("fast_model:cost")
```

The suffix is ignored for other providers.

### Identifier resolution

`model_id@provider` contains exactly one `@`. Resolution order:

1. Exact known identifier (from `llm.exceptions` or the models.dev catalog)
2. A tag under `llm.models`
3. Hyphen/underscore normalization (`gpt_4o_mini` ↔ `gpt-4o-mini`)
4. Fuzzy match against models.dev; auto-selected when the score is ≥ 0.8

Fuzzy matching makes bare names work interactively, but in YAML and code always write the full identifier so a
catalog update cannot change which model you get.

The models.dev catalog (1000+ models, capabilities, context windows) is cached at `data/models_dev.json` under the
CWD and fetched when missing. Refresh it with `uv run cli info llm-profile <id> --reload`.

## Declaring Models — `llm.exceptions`

Most models need no declaration because models.dev covers them. Use `config/providers/llm.yaml` for models that are
missing from the catalog, need aliases, or need capability overrides:

```yaml
llm:
  exceptions:
    - model_id: parrot_local
      providers:
        - fake: parrot

    - model_id: claude-haiku                       # one alias, two providers
      providers:
        - openrouter: anthropic/claude-haiku-4.5
        - edenai: anthropic/claude-haiku-4-5

    - model_id: ministral_8b
      providers:
        - openrouter: mistralai/ministral-8b-latest
      capabilities: ['structured_outputs']         # thinking | structured_outputs | pdf | vision
      max_tokens: 4096
      context_window: 32768
```

Each entry under `providers:` creates a separate identifier (`claude-haiku@openrouter`, `claude-haiku@edenai`). The
list is **not** a fallback chain; upstream `docs/llm-selection.md` describes it as one, which is wrong.

Tags (what `get_llm("fast_model")` resolves) live under `llm.models` in a profile block of `app_conf.yaml`:

```yaml
:profiles:
  local:
    llm:
      models:
        default: gpt-4o-mini@openai
        fast_model: claude-haiku@openrouter
        default_vlm: gemini-2.5-flash@openrouter
        fake: parrot_local@fake
```

## Embeddings

```python
get_embeddings(
    embeddings: str = "default",
    encoding_str: str | None = None,     # prefix for documents (e.g. "passage: ")
    retrieving_str: str | None = None,   # prefix for queries (e.g. "query: ")
    cache_embeddings: bool = False,
) -> Embeddings
```

```python
emb = get_embeddings()                          # embeddings.models.default
emb = get_embeddings("embeddings_768@fake")     # offline fake
vectors = emb.embed_documents(["doc 1", "doc 2"])
```

Defaults and tags live under `embeddings.models`; extra models go under `embeddings.registry` in
`config/providers/embeddings.yaml` (entries can set `dimension`). `embeddings_id=` / `embeddings_tag=` are
deprecated.

## Providers — `providers.yaml`

Each provider entry names its implementation classes by dotted path (resolved via `ImportResolver`):

| Field | Purpose |
|-------|---------|
| `chat_class` / `embeddings_class` / `decision_class` | Implementation per role (legacy `use:` = `chat_class`) |
| `api_key_env_var` | Env var checked before construction |
| `api_base` | Base URL for OpenAI-compatible or self-hosted endpoints |
| `litellm_prefix`, `gateway`, `extra_body`, `custom_headers`, `special_env_vars` | Provider-specific wiring |
| `openai_compatible`, `seed_param_location` | Behavior flags used by the mixins |

Bundled providers include `openai`, `anthropic`, `google`, `mistral`, `groq`, `deepinfra`, `edenai`, `edenai-eur`,
`openrouter`, `github`, `azure`, `together`, `deepseek`, `huggingface`, `ollama`, `vllm`, `litellm`, `custom`, `fake`,
`local` (FastEmbed, embeddings only), and `laya` (decision models only). A `labs:` section maps models to the lab
that trained them.

Chat classes live in `genai_tk.core.providers.chat` (`OpenAIChat`, `OpenRouterChat`, `EdenAIChat`, `GroqChat`,
`VllmChat`, `OllamaChat`, `MistralChat`, `AzureChat`, `LiteLLMChat`, `FakeChat`, …) and compose mixins from
`genai_tk.core.providers.mixins`: `SeedHandlingMixin`, `ReasoningMixin`, `OpenRouterRoutingMixin`,
`EdenAIRoutingMixin`, `EdenAIReasoningMixin`. If a class exposes a `create()` classmethod, the factory calls it with
only the kwargs its signature accepts.

### On-prem models (vLLM, Ollama)

Override `api_base` on an existing provider rather than inventing a new provider key. Some lookups still read only
the bundled provider list, so a brand-new key can fail with "Unknown provider".

```yaml
# config/providers/providers.yaml (project override)
providers:
  vllm:
    api_base: http://gpu-box.internal:8000/v1
  ollama:
    api_base: http://localhost:11434
```

```yaml
# config/providers/llm.yaml — self-hosted models are not in models.dev, so declare them
llm:
  exceptions:
    - model_id: qwen3_8b
      providers:
        - vllm: Qwen/Qwen3-8B          # the model name served by vLLM
```

```bash
export VLLM_API_KEY=anything     # vLLM requires the variable to be non-empty
```

```python
llm = get_llm("qwen3_8b@vllm")   # undeclared ids raise "Unknown LLM"
```

The `- custom: {model, base_url, default_headers}` dict form in `llm.exceptions` parses, but `base_url` and headers
are not currently forwarded to the client, so prefer the `api_base` override.

## Caching

`llm.cache` (`memory` | `sqlite` | `no_cache`) and `llm.cache_path` set the global LangChain cache; `get_llm(...,
cache=...)` overrides per model. `genai_tk.core.cache.LlmCache.set_method()` changes it at runtime. `in_memory` and
`redis` are not valid values.

## Decision Models (System 1)

Decision models are fast, non-autoregressive classifiers (OpenRouter Clef/Jev, EdenAI, Laya ModernBERT, OpenAI, and
an offline fake) for routing, scoring, and guard checks in well under a second.

```python
from genai_tk.core.decision_models import Choice, ClassifierRequest, Noul, Score
from genai_tk.core.factories import get_decision_model

model = get_decision_model()                    # decision.models.default
request = ClassifierRequest(
    state="Customer: my card was charged twice for order 1042.",
    questions={
        "is_complaint": Noul(instructions="Is the customer complaining?"),
        "topic": Choice(
            instructions="What is the main topic?",
            criteria={"billing": "Payments, charges, refunds", "shipping": "Delivery issues", "other": "Anything else"},
        ),
        "urgency": Score(instructions="How urgent is this?", criteria=["low", "medium", "high"]),
    },
)
response = model.invoke(request)
response.nouls["is_complaint"].noul       # probability in [0, 1]
response.choices["topic"].choice          # plus .confidence and .probabilities
response.scores["urgency"].score          # plus .legend and .probabilities
```

- `Noul` returns a calibrated probability; `Choice` takes a `criteria` dict (at least one key); `Score` takes a
  `criteria` list of at least two ordered levels.
- `state` accepts a string, a dict, a LangChain message, or a list of messages.
- `BaseDecisionModel` is a LangChain `Runnable`: `invoke`, `ainvoke`, `batch_invoke`, `abatch_invoke` (auto-chunks
  by the model's `max_questions`), plus `decide_noul`, `decide_choice`, `decide_score` helpers.
- `get_decision_model_from_chat_model(llm=...)` wraps a regular chat model when no decision backend is available.
- Tags live under `decision.models` (`default`, `fast`, `accurate`, `fake`); registry entries under
  `decision_models.registry` with `context_length`, `max_questions`, `description`.

```bash
uv run cli core classifier --demo
uv run cli core classifier --noul "Is this spam?" -s "Win a free cruise now!!!"
uv run cli core classifier --choice "Topic?" -c "billing:Payments, shipping:Delivery" -s "Where is my parcel?"
uv run cli core classifier --score "Urgency?" -c "low,medium,high" -s "Server is down" -m fake
```

The CLI command is `classifier`; upstream docs that say `cli core decision` are out of date.

## CLI Cheatsheet

```bash
uv run cli info models                        # everything resolvable
uv run cli info llm-profile gpt-4o-mini@openai
uv run cli info config                        # API-key status per provider
uv run cli core llm -i "hello" -m parrot_local@fake
uv run cli core llm -i "solve 17*23" -m "o3 (low)@openai" --reasoning
uv run cli core embedd "some text" -m default
uv run cli core similarity "reference sentence" "candidate one" "candidate two"
```
