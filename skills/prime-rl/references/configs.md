# prime-rl Config System

Every prime-rl entrypoint is configured with
[pydantic-config](https://github.com/PrimeIntellect-ai/pydantic-config): typed Pydantic classes
loaded from TOML (or YAML/JSON) files plus CLI overrides. Config classes live in
`packages/prime-rl-configs/src/prime_rl/configs/` (`rl.py`, `sft.py`, `eval.py`, `inference.py`,
`orchestrator.py`, `trainer.py`, `algorithm.py`).

## Loading and Composition

```bash
uv run rl @ examples/basic/reverse-text/rl.toml                     # one file
uv run rl @ examples/basic/reverse-text/rl.toml --max-steps 50      # CLI override
uv run rl @ base.toml @ overlay.toml                                # merge left to right
uv run rl @ base.toml --trainer @ trainer.toml --trainer.optim.lr 1e-6   # section file + override
uv run rl --help                                                    # every field and default
```

- Precedence: CLI > config files (left to right) > class defaults.
- Merging is deep: fields an overlay leaves unset keep the base value.
- `output_dir` adds one fallback: CLI > files > `$PRL_OUTPUT_DIR` > `"outputs"`.
- CLI uses kebab-case (`--inference.vllm.max-model-len`); TOML uses snake_case (`max_model_len`).

## Inspect and Validate

```bash
uv run rl @ rl.toml --dry-run --output-dir /tmp/x --run.name check
ls /tmp/x/check/configs/latest/resolved/      # rl.json, trainer.json, orchestrator.json, ...
cat /tmp/x/check/configs/latest/command.txt   # shell-safe launch command
```

Every launch writes `configs/attempt_<n>/` with `command.txt`, the launch TOML and `resolved/`.
`configs/latest` points to the current attempt. Validators for incompatible combinations (for
example context parallelism without flash attention) run at resolve time, so `--dry-run` catches them.

## TOML Rules

### No inline tables

```toml
# Wrong
[[orchestrator.train.source]]
env = { taskset = { id = "reverse-text" } }

# Right
[[orchestrator.train.source]]
env.taskset.id = "reverse-text"
```

### Sources are one block

Inside a `[[...source]]` entry, nested sub-configs are dotted keys in the same block. Nested arrays
of tables (for example judges) keep full-path headers and attach to the preceding source.

```toml
[[orchestrator.train.source]]
name = "reverse-text"
ratio = 1.0
env.taskset.id = "reverse-text"
env.agent.harness.id = "null"
env.agent.runtime.type = "subprocess"

[[orchestrator.eval.source]]
name = "reverse-text-eval"
env.taskset.id = "reverse-text"
env.taskset.split = "test"
env.agent.harness.id = "null"
env.agent.runtime.type = "subprocess"
```

`ratio` weights multiple train sources (default 1.0).

### Group defaults

Source groups hold defaults that every source inherits: `[orchestrator.train]`,
`[orchestrator.eval]`, the SFT `[eval]` block, and the top level of an eval TOML. Shared fields are
`env`, `sampling`, `select`, `group_size`, plus `algo` on train and `interval` on online eval.
Blocks merge key by key and a source's own value wins. A source block with a different `type`
(for example `algo.type = "echo"`) replaces the group's block for that source.

### Scalars

| Kind | TOML | CLI |
|------|------|-----|
| Boolean | `enable_prefix_caching = true` | `--flag` / `--no-flag` |
| None | `max_model_len = "None"` | `--inference.vllm.max-model-len None` |
| List | array of tables, replaced wholesale | space-separated or JSON |
| Dict | a `[section]` | JSON string: `--trainer.env-vars '{"KEY": "v"}'` |

JSON on the CLI only works for plain `dict` fields. Nested model fields (like `algo`) need dotted keys
(`--orchestrator.train.algo.type max_rl`) or a TOML overlay.

### Lists on the CLI

There are no list-index paths: `--orchestrator.eval.source.0.env.taskset.id` does not parse. Either
edit TOML or replace the whole list with JSON:

```bash
uv run rl @ rl.toml --orchestrator.eval.source '[{"env": {"taskset": {"id": "reverse-text"}}}]'
```

### Optional sections

Fields typed `SomeConfig | None` are off until set. A bare flag or an empty header enables defaults;
a nested key enables and sets:

```bash
--ckpt                      # checkpoint at the end of training
--ckpt.interval 25          # enables and sets the interval
--monitors.wandb            # enable W&B with defaults
--no-monitors.wandb         # disable
--monitors.wandb.project rt # enables W&B and sets the project
```

```toml
[ckpt]        # same as --ckpt
[inference]   # rl launches the policy server
```

### Discriminated unions

Set `type` to choose a variant; omit it to keep the default.

```toml
[orchestrator.train.algo]
type = "max_rl"

[trainer.loss]
type = "icepop"

[trainer.optim]
type = "muon"

[trainer.scheduler]
type = "cosine"
warmup_steps = 10
```

| Union | Variants (default first) |
|-------|--------------------------|
| `orchestrator.train.algo` | `grpo`, `max_rl`, `rae`, `hierarchical_grpo`, `opd`, `opsd`, `sft`, `echo` |
| `trainer.loss` | `ipo`, `icepop`, `ppo`, `cispo`, `custom` |
| `trainer.optim` | `adamw`, `muon`, `sign_sgd` |
| `trainer.scheduler` | `constant`, `linear`, `cosine` |
| `deployment` | `single_node`, `multi_node` |

### vLLM pass-through

`[inference.vllm]` uses vLLM's own argument names and forwards **any** key to the server:

```toml
[inference.vllm]
model = "Qwen/Qwen3-4B"
tensor_parallel_size = 2
max_model_len = 16384
max_num_seqs = 256
```

```bash
--inference.vllm.max-num-seqs 256
--inference.vllm.compilation-config '{"cudagraph_mode": "NONE"}'
```

Non-vLLM knobs (router, deployment, weight broadcast, KV-cache offload, env vars) stay on
`[inference]` itself.

## Shared Top-Level Keys (rl)

`RLConfig` propagates some keys to the trainer and orchestrator so you set them once:

| Key | Effect |
|-----|--------|
| `max_steps`, `seq_len` | Training length and sequence length |
| `[model] name` | Policy model for trainer, orchestrator and inference |
| `[ckpt]` | Checkpointing (`interval`, `keep_last`, `keep_interval`) |
| `[monitors.wandb]`, `[monitors.prime]`, `[monitors.file]` | Logging targets (file is on by default) |
| `[log] level` | Log level for trainer and orchestrator |
| `[deployment]`, `[slurm]` | GPU layout and cluster submission |
| `output_dir`, `[run] name`, `clean`, `[resume]` | Run directory management |
| `env_vars` | Extra environment variables for every component |
| `dashboard` | Auto-start the dashboard daemon (default true; `--no-dashboard`) |

## Renaming Fields

prime-rl removes old spellings instead of aliasing them. An old key then fails as an unknown key.
If a config from an older version fails to load, look up the new name with `--help`; do not add
aliases or `mode="before"` translators.

## Key Files

- `packages/prime-rl-configs/src/prime_rl/configs/` — all config classes
- `configs/debug/` — minimal debug configs (including `configs/debug/eval/*.toml`)
- `examples/` — full example configs
