# Evaluating Models with `uv run eval`

`uv run eval` scores a model on one or more verifiers environments and exits after one epoch per
source. It reuses the orchestrator's eval pipeline: one env server per source, adaptive or pinned
concurrency, every episode written to the trace stream, and `--resume` support. Config class:
`EvalConfig` (`configs/eval.py`); `uv run eval -h` lists every field.

## Serving the Model

```bash
uv run inference --vllm.model Qwen/Qwen3-0.6B                              # base model
uv run inference --vllm.model outputs/my-run/checkpoints/step_20/weights  # exported checkpoint
curl http://localhost:8000/health
curl http://localhost:8000/v1/models
```

Without `--client.base_url`, the eval targets Prime Inference.

## Single-Source Shorthands

```bash
uv run eval gsm8k -n 32 -r 4
uv run eval gsm8k -n 32 -r 4 -m Qwen/Qwen3-4B --client.base_url http://localhost:8000/v1
uv run eval gsm8k -n 32 -r 4 -c 8 --env.agent.harness.id bash
```

| Flag | Field |
|------|-------|
| `<taskset-id>` | The single source's `env.taskset.id` |
| `-n N` | `select.limit` |
| `-s` | `select.shuffle` |
| `-r N` | `group_size` (rollouts per task) |
| `-m MODEL` | `model` |
| `-c N` | Pins `concurrency.min_inflight = max_inflight = N` |
| `--env.<field> <value>` | Any field of the env block |

Shorthands cannot be combined with a TOML that defines `[[source]]` blocks. Always cap smoke evals
with `-n` and `-r`.

## Multi-Source TOML

The eval block is flattened to the top level. Per-source `select`, `group_size` and `sampling`
override the top-level defaults.

```toml
model = "Qwen/Qwen3-4B"
group_size = 4

[select]
limit = 32

[client]
base_url = "http://localhost:8000/v1"

[sampling]
max_completion_tokens = 1024

[[source]]
env.taskset.id = "gsm8k"
env.agent.harness.id = "bash"

[[source]]
env.taskset.id = "aime25"
env.agent.harness.id = "null"
env.agent.runtime.type = "subprocess"
```

```bash
uv run eval @ eval.toml --run.name my-eval
uv run eval @ eval.toml --run.name my-eval --resume     # continue an interrupted eval
uv run eval @ eval.toml --dry-run
uv run eval @ eval.toml --monitors.prime                # stream results to the platform
```

Concurrency: an external API has no vLLM `/metrics`, so pin it (`-c`). Against `uv run inference`,
set `min_inflight < max_inflight` in `[concurrency]` to adapt to KV-cache usage.

Ready-made configs live in `configs/debug/eval/` (single-turn, multi-turn, resume, multi-env,
aime2026, tb2).

## Selecting Tasks

Steps run in a fixed order: `include` / `exclude` (by `idx`, `ids`, `keys` or `names`), then
`shuffle` (with `seed`, default 0), then `skip`, then `limit`.

```bash
uv run eval gsm8k -n 50                                    # first 50 tasks
uv run eval gsm8k -n 50 -s                                 # a fixed random 50
uv run eval gsm8k -n 50 -s --select.seed 1                 # a different random 50
uv run eval gsm8k --select.include.idx 0:10,42             # by position
uv run eval terminal-bench-2 --select.include.names '["fix-git"]'
uv run eval terminal-bench-2 --select.exclude.keys '["<task.key>"]'
```

Because shuffle runs before `limit`, resuming a `-n 50 -s` eval with `-n 100` keeps the 50 finished
tasks and adds 50 more.

The same `select` block works for training sources, which gives disjoint train and test splits:

```toml
[[orchestrator.eval.source]]
env.taskset.id = "gsm8k"
select.include.idx = [":100"]

[[orchestrator.train.source]]
env.taskset.id = "gsm8k"
select.exclude.idx = [":100"]
```

## Monitoring an Eval

The console stays quiet; everything goes to files and the dashboard.

```text
<run_dir>/
├── configs/latest/            # command.txt, launch TOML, resolved/eval.json
├── logs/latest/eval.log
├── logs/latest/envs/eval/<name>.log
└── monitors/file/             # metrics.jsonl, trace stream, traces/live/
```

```bash
tail -F <run_dir>/logs/latest/eval.log
grep SUCCESS <run_dir>/logs/latest/eval.log      # one "Evaluated <env> ... Reward 0.xxxx" per source
grep -E "WARNING|ERROR" <run_dir>/logs/latest/eval.log <run_dir>/logs/latest/envs/eval/*.log
```

- Metrics: `eval/<env>/all/<agent>/reward/mean`, `is_truncated/mean` (raise
  `sampling.max_completion_tokens` when high), `has_error/mean`, and taskset metrics.
- A rollout stuck in `boot` waits on its sandbox; one in `running` with a frozen turn count waits on
  a model call or tool.
- Read a few traces in the dashboard before trusting a mean reward.
- Stop with SIGINT/SIGTERM to the `PRL::Eval` process. On `--resume`, keep model, sampling and env
  config the same so finished episodes stay comparable.
