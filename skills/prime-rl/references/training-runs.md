# Launching, Checkpointing, Resuming and Exporting Runs

## Entrypoints

| Command | Config class | Purpose |
|---------|--------------|---------|
| `uv run rl` | `RLConfig` (`configs/rl.py`) | Inference + orchestrator + trainer in one launch |
| `uv run sft` | `SFTConfig` (`configs/sft.py`) | Dataset SFT; runs torchrun internally |
| `uv run inference` | `InferenceConfig` (`configs/inference.py`) | vLLM server with weight-update endpoints |
| `uv run eval` | `EvalConfig` (`configs/eval.py`) | Standalone evals (see `eval.md`) |
| `uv run orchestrator` | `OrchestratorConfig` | Standalone orchestrator for manual layouts |
| `uv run trainer` | `TrainerConfig` | Standalone RL trainer |
| `uv run env-server` | `EnvServerConfig` | One env server; `rl` starts these automatically |

All accept `@ file.toml` plus CLI overrides and `--dry-run`. Never call `torchrun` for SFT or
`vllm serve` for inference.

## RL

```bash
uv run rl @ examples/basic/reverse-text/rl.toml
uv run rl @ my-rl.toml --run.name exp-1 --max-steps 200 --ckpt.interval 50 \
  --monitors.wandb.project my-proj --monitors.wandb.name exp-1
uv run rl @ my-rl.toml --dry-run --run.name check   # resolve, validate, write configs, exit
```

### Most-tuned knobs

| Knob | Default | Notes |
|------|---------|-------|
| `max_steps` | None | Trainer steps |
| `seq_len` | None | Shared sequence length |
| `orchestrator.batch_size` | 128 | Rollouts per trainer step; 64 minimum, 1024+ in production |
| `orchestrator.train.group_size` | 1 | Rollouts per task; set ≥ 8 for group-relative algorithms |
| `orchestrator.train.sampling.temperature` | 1.0 | Train sampling temperature |
| `orchestrator.train.sampling.max_completion_tokens` | None | Truncation cap; watch `is_truncated` |
| `orchestrator.max_off_policy_steps` | 8 | Max rollout staleness |
| `orchestrator.eval.interval` | 100 | Online eval cadence |
| `trainer.optim.lr` | 1e-6 | `type`: `adamw`, `muon`, `sign_sgd`; `max_norm` 1.0 |
| `trainer.scheduler.type` | `constant` | `linear` / `cosine` take `warmup_steps`, `min_lr` |
| `trainer.loss` | IPO | See `algorithms.md` |
| `trainer.model.freeze_moe_router` | auto (true in RL) | Training the router widens train/inference mismatch |
| `log.level` / `orchestrator.log.vf_level` | `info` | `vf_level = "debug"` for env debugging |

### Using an external inference server

Omit `[inference]`. `rl` then starts no server, and the orchestrator talks to
`orchestrator.model.client.base_url` (default `http://localhost:8000/v1`). Start that server with
`uv run inference` so the weight-update endpoints exist. With LoRA it must also run with
`--vllm.enable-lora` and a matching `--vllm.max-lora-rank`.

## SFT

```bash
uv run sft @ examples/basic/reverse-text/sft.toml --monitors.wandb
uv run sft @ sft.toml --slurm.job-name my-sft     # any [slurm] key submits to SLURM
uv run sft @ sft.toml --dry-run
```

SFT configs are flat: `[model]`, `[data]`, `[optim]`, `[scheduler]`, `[val]`, `[eval]`, `[renderer]`,
`[deployment]` at the top level (RL nests the trainer under `[trainer.*]`).

| Knob | What it controls |
|------|------------------|
| `data.name` | HF dataset name or local path |
| `data.seq_len` | Per-sample sequence length |
| `data.batch_size` | Global batch size (default 128); must divide by `data.micro_batch_size` |
| `data.loss_mask.*` | Which roles contribute to loss (system, user, assistant, tool) |
| `val.interval` | Validation cadence; `val.data` mirrors `data` |
| `eval.interval` | Online rollout evals (needs `[inference]`) |

### Dataset format

- **Prompt-completion**: `prompt` and `completion` columns; loss only on the completion.
- **Messages**: one `messages` column; loss on assistant turns. Wins if both layouts exist.
- `tools` column (OpenAI function format, list or JSON string) for tool-use SFT.
- `message_loss_mask` column: one 0/1 per message to select which turns train.
- Rename columns with `[data.columns]`:

```toml
[data.columns]
messages = "conversation"
tools = "schemas"
```

Tokenization is renderer-only (`[renderer] name = "auto"` by default):

```toml
[renderer]
name = "qwen3"
enable_thinking = false
```

### Online evals during SFT

```toml
[eval]
interval = 25

[eval.select]
limit = 32

[[eval.source]]
name = "reverse-text"
env.taskset.id = "reverse-text"
env.agent.harness.id = "null"
env.agent.runtime.type = "subprocess"

[inference]

[deployment]
num_train_gpus = 1
num_infer_gpus = 1
```

The base model is evaluated before step 1 (`eval.skip_first_step` disables it). Weights reach the
eval server over NCCL by default; `[weight_broadcast] type = "filesystem"` reloads from disk.

## Run Directories

`<output_dir>/<run_name>`: `output_dir` defaults to `$PRL_OUTPUT_DIR`, else `outputs`. Without
`--run.name`, the name is `<envs>--<model>--<short-id>` (SFT: `<dataset>--<model>--<short-id>`).
`--run.dir` overrides only the directory leaf.

```text
<run_dir>/
├── configs/attempt_<n>/      # command.txt, launch TOML, resolved/*.json  (configs/latest -> current)
├── logs/attempt_<n>/         # trainer.log, orchestrator.log, inference.log, eval.log, envs/  (logs/latest)
├── checkpoints/step_<n>/     # trainer/ (DCP) and orchestrator/
├── monitors/file/            # metrics.jsonl and traces/
├── broadcasts/               # weight-broadcast handshakes
└── launcher/                 # SLURM scripts and batch logs (SLURM only)
```

Launching into a run dir that already has artifacts fails unless you pass `--resume` or `--clean`
(which deletes only that run dir).

## Checkpointing

Off by default. The trainer saves FSDP-sharded DCP state, optimizer, scheduler and progress; the
orchestrator saves progress and per-env data state; inference is stateless.

```bash
uv run rl @ rl.toml --ckpt                                        # end of training only
uv run rl @ rl.toml --ckpt.interval 25                            # every 25 steps
uv run rl @ rl.toml --ckpt.interval 25 --ckpt.keep-last 3         # rolling window of 3
uv run rl @ rl.toml --ckpt.interval 25 --ckpt.keep-interval 100   # plus a permanent one every 100
```

```toml
[ckpt]
interval = 25
keep_last = 3
keep_interval = 100
```

## Resuming

Re-run the same command with the same `--run.name` and `--resume`:

```bash
uv run rl @ rl.toml --max-steps 10 --ckpt --run.name my-run
uv run rl @ rl.toml --max-steps 20 --ckpt --resume --run.name my-run           # latest checkpoint
uv run rl @ rl.toml --max-steps 20 --ckpt --resume.step 10 --run.name my-run   # a specific step
uv run rl @ rl.toml --max-steps 20 --ckpt --run.name my-fork \
  --resume.dir outputs/my-run/checkpoints/step_10                               # fork into a new run
```

- `--max-steps` is the absolute final step, not the remaining delta.
- `resume.step` and `resume.dir` are mutually exclusive.
- Without `[ckpt]` a resumed run loads the checkpoint but saves no new ones.
- Trainer checkpoint loading can skip parts: `trainer.ckpt.skip_optimizer`, `skip_scheduler`,
  `skip_dataloader`.

## Exporting Checkpoints

```bash
uv run python tools/convert_dcp_to_bf16.py outputs/my-run/checkpoints/step_20
uv run torchrun --nproc-per-node 8 tools/convert_dcp_to_bf16.py outputs/my-run/checkpoints/step_20
uv run inference --vllm.model outputs/my-run/checkpoints/step_20/weights
```

- Output goes to `<ckpt_dir>/weights` (or a second positional argument), as HF safetensors plus
  config and tokenizer. The model config is read from the run's resolved config.
- Full fine-tunes only; LoRA checkpoints are rejected.
- FP8: `tools/convert_bf16_to_fp8.py <dir>` or `tools/convert_dcp_to_fp8.py <ckpt_dir>` (writes
  `<ckpt_dir>/weights-FP8`). Dequantize FP8-only releases with `tools/convert_fp8_to_bf16.py <dir>`.
- On SM120 GPUs (RTX PRO 6000), serve blockwise-FP8 checkpoints with
  `VLLM_DISABLED_KERNELS=CutlassFp8BlockScaledMMKernel,MarlinFP8ScaledMMLinearKernel` to avoid a
  kernel that silently degrades outputs.

## Example Configs

| Path | What it shows |
|------|---------------|
| `examples/basic/reverse-text/{rl,sft,eval}.toml` | Single-turn SFT → RL on Qwen3-0.6B (2 GPUs) |
| `examples/basic/wordle/{rl,sft,eval}.toml` | Multi-turn SFT and RL on Qwen3-1.7B |
| `examples/basic/alphabet-sort/rl.toml` | Multi-turn RL with LoRA (r32, α64) |
| `examples/basic/wiki-search/rl.toml` | Tool calling with LoRA, 2 train + 6 inference GPUs |
| `examples/basic/hendrycks-sanity/rl.toml` | Math RL sanity check, 4 + 4 GPUs |
| `examples/extra/dynamo/rl.toml` | External inference server (no `[inference]`) |
| `examples/extra/vlm/sft.toml` | VLM SFT with LoRA |
| `examples/advanced/qwen3-30b-a3b/*.toml` | Multi-node MoE RL and SFT on SLURM |
| `examples/advanced/glm-4.5-air/ipo-ablation/` | IPO vs IcePop ablation |
| `configs/debug/algo/*.toml` | One small config per algorithm (grpo, max_rl, opd, sft_distill, echo, rae, ...) |
| `configs/debug/eval/*.toml` | Eval shapes: single-turn, multi-turn, resume, multi-env |
