---
name: prime-rl
description: >
  Post-train LLMs with PrimeIntellect's prime-rl (async RL, SFT, distillation, evals). Use this
  skill whenever the user mentions prime-rl, wants to write an rl.toml or sft.toml, run
  `uv run rl` / `uv run sft` / `uv run eval` / `uv run inference`, pick GRPO, MaxRL, OPD or
  ECHO, tune IPO loss, or monitor and resume a run. Also trigger for verifiers environments,
  Prime Intellect post-training, reverse-text, and pydantic-config TOML overrides.
---

# prime-rl Post-Training

[prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) is Prime Intellect's framework for
asynchronous RL and SFT post-training at scale. One launch (`uv run rl`) starts a vLLM inference
server, a CPU-only orchestrator that collects rollouts from
[verifiers](https://github.com/PrimeIntellect-ai/verifiers) environments, and an FSDP trainer that
pushes new weights back to inference. This skill teaches how to install it, write configs, choose
an algorithm, launch and monitor runs, evaluate checkpoints, and export weights.

Facts are checked against upstream commit `dd72f0f` (2026-10-13, v0.9.0). When the docs and the
code disagree, this skill follows the code (see [Doc Drift](#doc-drift)).

## Quick Reference

| Topic | Reference | When to read |
|-------|-----------|--------------|
| Install, uv sync, extras, env packages | [references/install.md](references/install.md) | Fresh machine, missing env package, DeepEP/kernels |
| Config system (TOML + CLI) | [references/configs.md](references/configs.md) | Writing or overriding any config, config errors |
| Algorithms, losses, advantages, curricula, renderers | [references/algorithms.md](references/algorithms.md) | Choosing GRPO vs OPD vs SFT-distill, tuning the loss |
| `rl` / `sft` / `inference`, run dirs, checkpoints, resume, export | [references/training-runs.md](references/training-runs.md) | Launching, resuming, exporting to HF |
| Logs, metrics, dashboard, check-in runbook | [references/monitoring.md](references/monitoring.md) | Checking on or debugging a live run |
| `uv run eval`, task selection | [references/eval.md](references/eval.md) | Benchmarking a model or checkpoint |
| Multi-GPU, multi-node, SLURM, LoRA, MoE, inference serving | [references/scaling.md](references/scaling.md) | Anything beyond one node with defaults |

## Core Workflow

1. **Install** on a Linux box with NVIDIA GPUs (the smoke run needs 2):

   ```bash
   curl -sSL https://raw.githubusercontent.com/PrimeIntellect-ai/prime-rl/main/scripts/install.sh | bash
   cd prime-rl && uv sync --all-extras
   ```

2. **Smoke-test** with the bundled example before touching anything else:

   ```bash
   uv run rl @ examples/basic/reverse-text/rl.toml
   ```

3. **Write your config** (TOML, snake_case) by copying the closest example under `examples/`.
4. **Dry-run** to resolve and validate it without launching:

   ```bash
   uv run rl @ my-rl.toml --dry-run --run.name check
   ```

5. **Launch** with a predictable run name and checkpoints so the run can be resumed:

   ```bash
   uv run rl @ my-rl.toml --run.name my-run --ckpt.interval 25 --ckpt.keep-last 3
   ```

6. **Monitor** logs, metrics and the dashboard (see [Monitoring](#monitoring-essentials)).
7. **Evaluate** with `uv run eval` against a `uv run inference` server.
8. **Export** the DCP checkpoint to HF safetensors with `tools/convert_dcp_to_bf16.py`.

## Entrypoints

Always run through `uv run`; never call `python`, `torchrun` or `vllm serve` directly.

| Command | Purpose | Notes |
|---------|---------|-------|
| `uv run rl` | Full RL pipeline: inference + orchestrator + trainer | `RLConfig`; local or SLURM (`[slurm]`) |
| `uv run sft` | Dataset-based SFT | Launches torchrun itself; optional online evals |
| `uv run inference` | vLLM server with prime-rl endpoints | Router on port 8000, engine on 8100; adds `/update_weights`, `/load_lora_adapter`, `/init_broadcaster` |
| `uv run eval` | Standalone multi-env eval | One epoch per source; `--resume` continues |
| `uv run orchestrator` / `trainer` / `env-server` | Pieces of `rl` for manual layouts | Rarely needed |
| `uv run dashboard` | Local web dashboard (port 7788) | Read-only on run dirs |

## Config Essentials

Every entrypoint takes TOML files with `@` plus dotted CLI overrides
([pydantic-config](https://github.com/PrimeIntellect-ai/pydantic-config)).

```bash
uv run rl @ base.toml @ overlay.toml --max-steps 50 --orchestrator.train.group-size 16
uv run rl --help                         # every field with its default
```

- Precedence: CLI > files (left to right, deep merge) > class defaults.
- CLI is kebab-case (`--trainer.optim.lr`), TOML is snake_case (`lr`).
- Booleans: `--flag` / `--no-flag` on the CLI, explicit `true`/`false` in TOML.
- `None` is the string `"None"` in TOML.
- No inline tables (`key = { ... }`). Use `[section]` headers or dotted keys.
- Lists are replaced wholesale by later files; repeat the full list in overlays.
- Optional sections (`[ckpt]`, `[inference]`, `[monitors.wandb]`) are enabled by an empty header.
- Unions pick a variant with `type` (`[trainer.loss] type = "icepop"`).
- Renamed fields fail as unknown keys on purpose. Fix the config; do not expect aliases.

### Minimal RL Config

This is the upstream reverse-text example (`examples/basic/reverse-text/rl.toml`), trimmed of W&B:

```toml
max_steps = 20
seq_len = 2048

[model]
name = "PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT"

[orchestrator]
batch_size = 128

[orchestrator.train]
group_size = 16

[orchestrator.train.sampling]
max_completion_tokens = 128

[[orchestrator.train.source]]
name = "reverse-text"
env.taskset.id = "reverse-text"
env.agent.harness.id = "null"
env.agent.runtime.type = "subprocess"

[orchestrator.eval]
interval = 5

[[orchestrator.eval.source]]
name = "reverse-text-eval"
select.limit = 256
env.taskset.id = "reverse-text"
env.agent.harness.id = "null"
env.agent.runtime.type = "subprocess"

[trainer.optim]
lr = 3e-6

[ckpt]

[inference]

[orchestrator.renderer]
name = "prime-qwen3"
```

Key points:

- `[[orchestrator.train.source]]` is one block per environment. Nested env settings are dotted keys
  inside that block, not separate `[...source.env]` headers.
- `orchestrator.train.group_size` defaults to **1**. With GRPO that makes every advantage zero and
  the trainer learns nothing. Set it to 8 or more.
- `[inference]` (even empty) tells `rl` to launch the policy server. Leave it out only when you
  start the inference server yourself.
- `[orchestrator.renderer] name = "auto"` is the default. Fine-tunes whose name does not match the
  renderer map need an explicit renderer (here `prime-qwen3`).

## Choosing an Algorithm

Set `[orchestrator.train.algo] type`. Each type's class defaults are its vetted setting.

| `type` | Signal | Needs | Use for |
|--------|--------|-------|---------|
| `grpo` (default) | RL on actions, group-mean baseline | `group_size` ≥ 8 | Standard verifiable-reward RL |
| `max_rl` | RL, mean-normalized group credit | `group_size` ≥ 8 | Upweighting hard tasks |
| `rae` | RL, per-agent EMA baseline | Multi-agent env | Self-play (for example `kuhn-poker`) |
| `hierarchical_grpo` | RL, per-role baselines | Proposer-solver env, `episode_agents` | Proposer-solver training |
| `opd` | Reverse KL to a teacher | `[orchestrator.train.algo.teacher]` on vLLM | On-policy distillation |
| `sft` | CE on a frozen model's samples | `[orchestrator.train.algo.sampling.source]` | Hard distillation on the fly |
| `opsd` | Reverse KL to demo-conditioned self | `demo_key` in task info | Self-distillation |
| `echo` | GRPO + weighted CE on env tokens | Renderer role attribution | Learning from tool output |

```toml
[orchestrator.train.algo]
type = "opd"

[orchestrator.train.algo.teacher]
name = "Qwen/Qwen3-32B"
base_url = "http://localhost:8001/v1"
```

prime-rl only hosts the policy. Start frozen teachers yourself, for example
`CUDA_VISIBLE_DEVICES=1 uv run inference --vllm.model Qwen/Qwen3-32B --server.port 8001`.

The RL loss lives in `[trainer.loss]`. Omit it to get IPO (`eps = 0.3`, `adv_tau = 1.0`). Other
types are `icepop`, `ppo`, `cispo` and `custom`. Details: [references/algorithms.md](references/algorithms.md).

## SFT

```toml
max_steps = 100

[model]
name = "PrimeIntellect/Qwen3-0.6B"

[data]
name = "willcb/R1-reverse-wikipedia-paragraphs-v1-1000"
seq_len = 4096
batch_size = 32

[optim]
lr = 2e-5

[ckpt]
```

```bash
uv run sft @ examples/basic/reverse-text/sft.toml --monitors.wandb
```

Datasets use `prompt`/`completion` columns or a `messages` column (`messages` wins if both exist).
Tokenization always goes through a renderer (`[renderer] name = "auto"` by default). In SFT the
optimizer is top-level `[optim]`; in RL it is `[trainer.optim]`. Online evals, column mapping and
loss masks: [references/training-runs.md](references/training-runs.md).

## Run Directories, Checkpoints and Resume

Each launch writes to `<output_dir>/<run_name>` (`output_dir` defaults to `$PRL_OUTPUT_DIR`, else
`outputs`). Without `--run.name` the name is auto-generated as `<envs>--<model>--<short-id>`.

```bash
uv run rl @ rl.toml --max-steps 10 --ckpt --run.name my-run              # first leg
uv run rl @ rl.toml --max-steps 20 --ckpt --resume --run.name my-run     # continue to step 20
uv run rl @ rl.toml --max-steps 20 --ckpt --resume.step 10 --run.name my-run
uv run python tools/convert_dcp_to_bf16.py outputs/my-run/checkpoints/step_20
uv run inference --vllm.model outputs/my-run/checkpoints/step_20/weights
```

- Checkpoints are off by default. `--ckpt` saves at the end; `--ckpt.interval N` adds periodic saves;
  `--ckpt.keep-last K` and `--ckpt.keep-interval M` control retention.
- `--max-steps` on resume is the final target step, not the remaining delta.
- Reusing a run name fails unless you pass `--resume` or `--clean` (which wipes that run dir).
- Export rejects LoRA checkpoints.

## Monitoring Essentials

```bash
tail -F outputs/my-run/logs/latest/{trainer,orchestrator,inference}.log
grep -E "WARNING|ERROR" outputs/my-run/logs/latest/envs/{train,eval}/*.log
cat ~/.cache/prime-rl/dashboard/daemon.json      # live dashboard URL (default port 7788)
```

| Watch | Metric | Healthy sign |
|-------|--------|--------------|
| Progress | `train/agg/effective/<agent>/reward/mean` | Trends up over hundreds of steps |
| Truncation | `train/agg/effective/<agent>/is_truncated/mean` | Low; raise `max_completion_tokens` if high |
| Off-policy drift | `mismatch_kl/all/mean` | Flat; a growing mean warns of collapse |
| Collapse | `entropy/all/mean` | Not collapsing toward zero |
| Divergence | `optim/grad_norm` | No spikes |
| Bottleneck | trainer `time/wait_for_batch` vs orchestrator `time/wait_for_policy` | High wait_for_batch means add inference; high wait_for_policy means add trainer GPUs |

When an agent monitors a run for a researcher: append check-ins to `<run_dir>/STATUS.md`, and
never kill, restart or launch runs yourself. Hand the researcher the exact command. Full runbook:
[references/monitoring.md](references/monitoring.md).

## Eval

```bash
uv run inference --vllm.model Qwen/Qwen3-0.6B                               # terminal 1
uv run eval reverse-text -n 32 -r 4 -m Qwen/Qwen3-0.6B \
  --client.base_url http://localhost:8000/v1                                 # terminal 2
uv run eval @ examples/basic/reverse-text/eval.toml --run.name rt-eval --resume
```

`-n` caps tasks, `-r` sets rollouts per task, `-c` pins concurrency, `-s` shuffles. Always cap smoke
evals. Multi-source TOML and selection rules: [references/eval.md](references/eval.md).

## Rules of Thumb

- Start from `examples/basic/reverse-text/` on 2 GPUs. If it finishes, your install is good.
- `orchestrator.batch_size` ≥ 64 (128–512 for ablations); `group_size` ≥ 8 (16–32 is common).
- Use `--dry-run` before every SLURM submit; validators fail fast there and slowly in the queue.
- `orchestrator.max_off_policy_steps` (default 8) is the main async dial: raise it for throughput,
  lower it for on-policyness, and watch `mismatch_kl` while tuning.
- Keep `trainer.model.optimization_dtype` and `reduce_dtype` at their defaults unless asked.

## Doc Drift

Upstream docs and skills disagree with the code in a few places at `dd72f0f`. Trust the code.

| Topic | Upstream text | Code / this skill |
|-------|---------------|-------------------|
| IPO `eps` default | `skills/configs` says 0.1 | `IPOLossConfig.eps = 0.3` (docs/algorithms.md agrees) |
| Loss types | Docs list `ipo`, `icepop`, `custom` | Also `ppo` (clip 0.8–1.2) and `cispo` (`ratio_high` 5.0) |
| Renamed fields | `skills/training/start-run` says validation aliases keep old keys working | Policy is no aliases; old keys fail as unknown |
| Metric names | `docs/training.md` uses `reward/{all,env}/mean` | Current keys are `train/{agg,<env>}/{all,effective}/<agent>/reward/mean` |
| LoRA + NCCL | `docs/advanced.md` says NCCL falls back to filesystem | Only when `[weight_broadcast]` is unset; explicit `nccl`/`nixl` with LoRA raises |
| Throughput recipe | `docs/scaling.md` uses `[model]`, `[model.compile]` | Those are SFT paths; RL uses `[trainer.model]` |
| Fused LM head | `docs/scaling.md` says SFT disables it | `fused_lm_head_token_chunk_size` applies to SFT too |

## Anti-Patterns to Avoid

- **Leaving `group_size` at its default of 1 with GRPO** — every advantage is zero and batches are skipped.
- **Calling `vllm serve`, `torchrun` or raw `python`** — use `uv run inference`, `uv run sft`, `uv run`.
- **Inline TOML tables** (`env = { taskset = { id = "x" } }`) — use dotted keys in the source block.
- **Partial list overlays** — a later file replaces the whole `[[...source]]` list.
- **CLI list indexes** (`--orchestrator.eval.source.0.env.taskset.id`) — not parsed; edit TOML or pass JSON.
- **Launching without `--run.name`** when you will need to resume — auto names are hard to find.
- **Passing remaining steps on resume** — `--max-steps` is the absolute target.
- **Pointing `opd` at `"policy"` or a non-vLLM teacher** — rejected (zero KL) or missing `prompt_logprobs`.
- **`uv sync` without `--all-packages`** after installing env packages — it removes them.
- **Agents restarting or killing runs on their own** — hand the command to the researcher.
- **Trusting a mean reward without reading traces** — check a few episodes in the dashboard.

## When to Use / Not Use

**Use this skill when:**

- Writing or debugging `rl.toml`, `sft.toml` or `eval.toml` for prime-rl.
- Choosing between GRPO, MaxRL, OPD, SFT-distill, OPSD, ECHO, RAE or hierarchical GRPO.
- Launching, resuming, monitoring or exporting a prime-rl run, locally or on SLURM.
- Evaluating a model or checkpoint on verifiers environments with `uv run eval`.

**Do not use this skill when:**

- Fine-tuning with Unsloth or Hugging Face Jobs (use `unsloth-hf-jobs`).
- Writing a verifiers environment itself (taskset, harness, rubric); see the verifiers repo.
- Serving models in production without training (use plain vLLM docs).
- Contributing kernels or cutting prime-rl releases (see upstream `skills/kernels` and `skills/release`).
