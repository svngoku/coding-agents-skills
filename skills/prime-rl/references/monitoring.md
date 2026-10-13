# Monitoring a prime-rl Run

## Agent Runbook

Use this when a researcher asks an agent to watch a run.

### On launch

1. Find the run dir: `<output_dir>/<run_name>`. If you only know the output dir, take the newest
   subdirectory (`ls -t outputs | head -1`) or read `run.name` from the launch command.
2. Read `configs/latest/command.txt` and `configs/latest/resolved/` (start with `rl.json`, or
   `orchestrator.json` on local runs).
3. Confirm every process is alive and steps are advancing.
4. Write an initial summary to `<run_dir>/STATUS.md`.

### Recurring check-ins (default every hour)

Confirm processes, grep logs, note the step and key metrics, then **append** (never overwrite):

```markdown
## 2026-10-14 09:00 UTC

**Step**: 140 / 500
**Health**: Healthy

**Progress**: reward/mean 0.62 (up from 0.41), truncation 3%, eval avg@4 0.58.
**Stability**: entropy flat, mismatch_kl 0.002, grad_norm no spikes.
**Performance**: trainer 41 s/step, wait_for_batch 2 s; no inference pressure.

**Notes**: 3 sandbox timeouts in envs/train/swe.log, not growing.
```

### Restarts

Never restart unless the researcher explicitly asked, and never run kill or launch commands
yourself. Give the researcher the exact command; after they restart, verify all processes are back
and progress resumed.

## Where Things Live

| Path | Contents |
|------|----------|
| `configs/latest/` | `command.txt`, launch TOML, `resolved/*.json` (one `attempt_<n>` per launch) |
| `logs/latest/` | Current attempt's logs; resumes never overwrite earlier attempts |
| `monitors/file/metrics.jsonl` | Every metric row, tagged by producer |
| `monitors/file/traces/` | Episode stream, indexes, annotations, `live/` in-flight rollouts |
| `launcher/logs/` | SLURM batch logs (SLURM runs only) |

### Logs

```text
logs/latest/
├── trainer.log          # rank 0
├── orchestrator.log
├── inference.log        # vLLM (and router on single node)
├── eval.log             # SFT online eval or `uv run eval`
├── trainer/node_*.log, trainer/torchrun/<rdzv>/attempt_0/<rank>/{stdout,stderr}.log
├── inference/node_*.log, inference/router.log   # multi-node only
└── envs/{train,eval}/<env>.log                  # one per env server
```

```bash
tail -F <run_dir>/logs/latest/{trainer,orchestrator,inference}.log
grep -E "WARNING|ERROR" <run_dir>/logs/latest/{trainer,orchestrator,eval,inference}.log
grep -E "WARNING|ERROR" <run_dir>/logs/latest/envs/{train,eval}/*.log
grep -E "WARNING|ERROR" <run_dir>/launcher/logs/*job_*.log      # SLURM
```

Env logs are the first place to look for rollout errors; most user code runs there. Raise
verbosity with `orchestrator.log.vf_level = "debug"`.

### Processes

```text
PRL::Launcher
├── PRL::Inference          (vLLM, GPU 0)
├── PRL::EnvServer          (one per train/eval source)
│   └── Verifiers::EnvWorker0..N
├── PRL::Orchestrator       (CPU only)
└── torchrun
    └── PRL::Trainer        (GPU 1+)
```

`ps aux | grep PRL::` lists them. Multi-node runs put trainer and inference on separate nodes.

## Metrics

Rollout metrics follow `{scope}/{subset}/<agent>/<metric>/<stat>`: `scope` is `train/agg`,
`train/<env>` or `eval/<env>`; `subset` is `all` or `effective` (admitted, clean, trainable). Most
single-agent envs name their agent `agent`.

### Progress (orchestrator)

| Metric | Meaning |
|--------|---------|
| `train/agg/effective/<agent>/reward/mean` | Main signal; should trend up |
| `train/agg/effective/<agent>/is_truncated/mean` | Truncated share; raise `max_completion_tokens` if high |
| `train/agg/all/<agent>/has_error/mean` | Errored rollouts; sustained > 5% is a smell |
| `train/agg/all/<agent>/is_admitted/mean` | Curriculum admission rate |
| `train/agg/effective/num_turns/mean`, `num_total_tokens/mean` | Episode size |
| `train/<env>/effective/<agent>/metrics/<name>/mean` | Env-specific metrics |
| `eval/<env>/effective/<agent>/{avg@k,pass@k}` | Online eval scores |

### Stability (trainer)

| Metric | Meaning |
|--------|---------|
| `mismatch_kl/all/{mean,std,max}` | Trainer vs inference policy KL; a growing mean warns of off-policy collapse |
| `entropy/all/mean` | Too low means mode collapse; too high means no commitment |
| `is_masked/mean` | Share of tokens masked by the IPO trust region |
| `optim/grad_norm` | Spikes precede divergence; lower LR or revisit the loss |

### Performance

| Source | Metric | Reading |
|--------|--------|---------|
| trainer | `time/wait_for_batch` | High: orchestrator/inference is the bottleneck, add inference |
| orchestrator | `time/wait_for_policy` | High: trainer is the bottleneck, add trainer GPUs |
| trainer | `perf/throughput`, `perf/mfu`, `perf/peak_memory` | Tokens/s, MFU, memory |
| orchestrator | `off_policy/{mean,max}`, `off_policy/dropped` | Staleness of trained rollouts |

Both processes log a warning when their wait time exceeds active time. Live vLLM stats come from the
engine port, not the router:

```bash
curl -s http://localhost:8100/metrics | grep -E "num_requests|gpu_cache_usage"
```

`vllm:gpu_cache_usage_perc` near 1.0 means the KV cache is saturated.

### SFT metrics

`loss/mean`, `loss/perplexity` (should fall), `val/loss`, `optim/grad_norm`, `optim/lr`,
`progress/epoch`, `perf/throughput`, and for MoE `max_vio/*` and `expert_load/*`.

## Dashboard

Launchers auto-start one dashboard daemon per user (`PRL::Dashboard`) and print a
`Dashboard · <url>` banner. The port can differ from 7788 when taken:

```bash
cat ~/.cache/prime-rl/dashboard/daemon.json   # {"pid": ..., "url": "http://localhost:<port>"}
curl -sf <url>/api/runs                       # liveness
uv run dashboard outputs/ other-outputs/      # manual start; read-only on run dirs
uv sync --extra dashboard && uv run dashboard # head node without the GPU stack
```

Views: overview (W&B-style sections from `metrics.jsonl`), all metrics, resolved configs, a trace
viewer with per-token overlays (advantage, entropy, mismatch, IPO mask), merged logs, and reports.
`--no-dashboard` opts a run out.

## W&B and Platform

```bash
uv run rl @ rl.toml --monitors.wandb.project my-proj --monitors.wandb.name run-42
uv run rl @ rl.toml --monitors.prime        # Prime Lab; needs PRIME_API_KEY and an enabled team
uv run rl @ rl.toml --no-monitors.file      # disable local metrics/traces (on by default)
```

Trainer and orchestrator share one W&B run, so `monitors.wandb.offline` must stay false. Each project
gets an auto-created `overview` saved view with train, eval, stability and performance sections.

## Reading Traces

```bash
uv run python -m prime_rl.monitors.file.traces <run_dir>              # live rollouts by phase
uv run python -m prime_rl.monitors.file.traces <run_dir> <trace_id>   # one assembled live trace
zstd -dcf <run_dir>/monitors/file/traces/stream/* | jq '.traces[].rewards'
zstd -dcf <run_dir>/monitors/file/traces/stream/* | jq 'select(.ok | not) | {id, env: .env.id, errors}'
```

## Common Failures

| Component | Typical causes |
|-----------|----------------|
| Env workers | Exceptions in env code, timeouts, sandbox errors, OOM kills (most common) |
| Orchestrator | Empty or errored rollout spikes, weight-broadcast failures, checkpoint errors |
| Trainer | NCCL/CUDA errors, OOM, NaN loss or gradients |
| Inference | NCCL/CUDA errors, OOM, request timeouts |

A few warnings are normal. Escalate when errors persist, grow, or hit a large share of rollouts.
