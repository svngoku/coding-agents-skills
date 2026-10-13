# Scaling: GPUs, Nodes, SLURM, Parallelism, LoRA and Inference

## Single-Node GPU Split

`[deployment]` defaults to `type = "single_node"` with 1 trainer GPU and 1 inference GPU. The
launcher assigns visible GPUs (from `CUDA_VISIBLE_DEVICES`) to inference first, then the trainer.
Train + inference GPUs must not exceed `gpus_per_node` (8).

```toml
[deployment]
num_train_gpus = 2
num_infer_gpus = 6

[inference.vllm]
tensor_parallel_size = 1
```

`inference.vllm.data_parallel_size` is derived automatically as `num_infer_gpus // tensor_parallel_size`,
so you do not need to pass it. NCCL weight broadcast on a single node needs at least 2 GPUs.

## Multi-Node with SLURM

```toml
[deployment]
type = "multi_node"
num_train_nodes = 2
num_infer_nodes = 2

[slurm]
job_name = "math-qwen30b"
partition = "all"

[inference.vllm]
tensor_parallel_size = 8
```

```bash
uv run rl @ math.toml --dry-run      # writes <run_dir>/launcher/rl.sbatch and resolved configs
uv run rl @ math.toml                # submits with sbatch
```

- Multi-node requires `[slurm]`. Any `[slurm]` table (or a `--slurm.<field>` override, or a bare
  `--slurm`) switches the launch to sbatch.
- `[slurm]` fields: `job_name`, `partition`, `account`, `time`, `nodelist`, `exclude`,
  `project_dir = "."`, `pre_run_command`, `template_path`, `shared_fs = true`,
  `launch_modelexpress = true`, `cleanup_grace_period = 3600`.
- Templates live in `src/prime_rl/templates/` (`single_node_rl.sbatch.j2`, `multi_node_rl.sbatch.j2`,
  `inference.sbatch.j2`); override with `--slurm.template-path`.
- Multi-node fields: `num_train_nodes` (required), `num_infer_nodes` (inferred from
  `inference.deployment` if unset; `0` means trainer-only and needs `trainer.data.fake`),
  `num_infer_replicas = 1`, `nodes_per_fsdp_group` (sets `trainer.model.dp_replicate`),
  `orchestrator_on_inference = false`.
- Scripts and batch logs land in `<run_dir>/launcher/`. Always `--dry-run` first: validators fail fast
  there instead of after queueing.

SFT has its own deployment: `single_node` (`num_train_gpus`, `num_infer_gpus`) or `multi_node`
(`num_train_nodes = 2`, `num_infer_nodes = 0`, `nodes_per_fsdp_group`). With online evals, the first
`num_infer_nodes` nodes run inference and the rest run the trainer.

## Trainer Parallelism

Set under `[trainer.model]` in RL and `[model]` in SFT.

| Field | Default | Notes |
|-------|---------|-------|
| `dp_replicate` | 1 | HSDP replicate dimension |
| `reshard_after_forward` | true | FSDP resharding |
| `offload` | `"optimizer"` | `"none"`, `"optimizer"`, `"full"` (full: adamw or sign_sgd only, disables grad clipping) |
| `ep` | `"auto"` | Expert parallel: `min(world_size // dp_replicate, 8)` for MoE, else 1 |
| `cp` | 1 | Context parallel; needs flash attention |
| `cp_style` | `"ring"` | `"ring"` or `"ulysses"` (VLMs require ulysses) |
| `attn` | `"auto"` | `flash_attention_2`, `flash_attention_3`, `flash_attention_4` |
| `ac` | full, freq 1 | Activation checkpointing (`mode = "full"` or `"selective"`) |
| `ac_offloading` | `max_inflight_activations = 5` | Offload checkpointed activations |
| `compile` | on | `[trainer.model.compile] fullgraph`, `mode` |
| `moe.compute.type` | `"bf16"` | `"deepgemm_fp8"`, `"mxfp8"` |
| `moe.dispatch.type` | `"torch"` | `"deepep"` needs the `disagg` extra |
| `freeze_moe_router` | auto (true in RL) | Keeps inference and trainer expert choices aligned |
| `optimization_dtype`, `reduce_dtype` | `"float32"` | Do not change unless explicitly asked |

```toml
[trainer.model]
attn = "flash_attention_3"
ep = 8
cp = 2
cp_style = "ulysses"

[trainer.model.compile]

[trainer.model.ac_offloading]
```

Throughput recipes in `docs/scaling.md` are written with SFT paths (`[model]`, `[model.compile]`); in
RL put the same keys under `[trainer.model]`.

## LoRA

```toml
[trainer.model.lora]     # [model.lora] in SFT
rank = 32
alpha = 64
dropout = 0.0
```

- Defaults: `rank = 16`, `alpha = 32.0`, `target_modules` covering q/k/v/o, MLP and expert projections
  across families (unknown names are ignored).
- `rl` sets `inference.vllm.enable_lora = true` and `max_lora_rank` (rounded up to 8, 16, 32, 64,
  128, 256, 320 or 512) for you. An external server must be started with those flags.
- LoRA uses filesystem weight broadcast. Leaving `[weight_broadcast]` unset selects it automatically;
  setting `type = "nccl"` or `"nixl"` together with LoRA raises an error.
- `modules_to_save` is rejected for real training (the broadcast ships only adapters).
- Export via `convert_dcp_to_bf16.py` rejects LoRA checkpoints.

Examples: `examples/basic/alphabet-sort/rl.toml` (r32, α64) and `examples/basic/wiki-search/rl.toml` (r8).

## Weight Broadcast and Rollout Transport

| `[weight_broadcast] type` | Fields | When |
|---------------------------|--------|------|
| `nccl` (rl default) | `host`, `port = 29501`, `timeout` | Default when `[inference]` is set and no LoRA |
| `filesystem` | `timeout = 3600` | LoRA, or no managed inference |
| `nixl` | `port = 8001`, `host`, `session_id` | Needs NIXL + ModelExpress installs |

`[rollout_transport] type = "zmq"` (default, port 5555) ships batches in memory; `"filesystem"` writes
`<run_dir>/batches/step_<n>/rank_<r>.bin`.

## Inference Server

`uv run inference` wraps vLLM with a router. Standalone configs put `server`, `vllm`, `router`,
`deployment`, `slurm` at the top level; inside `rl` they live under `[inference.*]`.

```bash
uv run inference --vllm.model Qwen/Qwen3-0.6B
uv run inference --vllm.model Qwen/Qwen3-0.6B --vllm.enforce-eager
uv run inference --vllm.model Qwen/Qwen3-32B --server.port 8001 --vllm.tensor-parallel-size 2
```

| Setting | Default | Notes |
|---------|---------|-------|
| `server.port` | 8000 | Client-facing router URL |
| `backend_port` | `server.port + 100` | Engine; admin calls and `/metrics` go here |
| `vllm.gpu_memory_utilization` | 0.9 | |
| `vllm.tool_call_parser`, `vllm.reasoning_parser` | `"auto"` | Resolved from the model name |
| `router.type` | `"vllm-router"` | Policy `sticky_least_loaded`; `"llm-d"` only for multi-node/disaggregated |
| `kv_cache_offload.type` | none | `"native"` or `"mooncake"` (needs `cpu.num_bytes`) |

Any other vLLM argument goes under `[inference.vllm]` as-is (`max_num_seqs`, `max_model_len`, ...).

Custom endpoints on the engine: `POST /update_weights`, `/load_lora_adapter`, `/init_broadcaster`,
`/pause`, `/resume`, and `GET /liveness`.

### Inference deployments

| `[inference.deployment] type` | Fields |
|-------------------------------|--------|
| `single_node` (default) | none |
| `multi_node` | `num_nodes = 2`; needs `[slurm]` |
| `disaggregated` | `prefill_nodes_per_replica`, `decode_nodes_per_replica`, `num_prefill_replicas`, `num_decode_replicas`, `prefill_port = 8100`, `decode_port = 8200`; needs `[slurm]` |

Disaggregated prefill/decode turns on expert parallel and NIXL KV transfer automatically. Install
NIXL with `bash scripts/install_nixl_from_source.sh` and ModelExpress with
`bash scripts/install_modelexpress.sh`, and re-run them after every `uv sync`.

### External inference

Omit `[inference]` in the RL config and set `[orchestrator.model.client] base_url`. The server must be
`uv run inference` (or a Dynamo deployment, see `examples/extra/dynamo/rl.toml`). Start it with
LoRA flags or `--vllm.enable-return-routed-experts` when the run needs them; prime-rl only warns.

## Other Features

- **VLM**: `[model.vlm]` (`freeze_vision_encoder = true`); only Qwen3.5 dense and MoE are registered,
  bf16 dtypes required, unfrozen vision encoder cannot combine with LoRA.
- **Router replay** (MoE): `trainer.enable_router_replay = true` turns on routed-expert return in
  vLLM. Not supported with disaggregated P/D, llm-d or KV offload.
- **Adaptive concurrency**: `[orchestrator.concurrency]` with `min_inflight = 1`,
  `max_inflight = 1024`, optional `initial_inflight`.

## Reference Multi-Node Configs

| Path | Layout |
|------|--------|
| `examples/advanced/qwen3-30b-a3b/math.toml` | Qwen3-30B-A3B RL, 2 train + 2 inference nodes, ep 8 |
| `examples/advanced/qwen3-30b-a3b/swe.toml` | r2e-gym with bash harness, 131k context, ep 8, cp 2 |
| `examples/advanced/glm-4.5-air/swe-budget.toml` | 1 + 1 nodes with sign_sgd; overlays for 2 and 4 nodes |
| `examples/advanced/glm-5.3/swe.toml` | 16 train nodes, disaggregated prefill/decode |
| `examples/advanced/intellect-3.1/rl.toml` | 4 train + 12 inference nodes, muon, multi-env |
| `configs/advanced/nemotron-3-super/swe.toml` | Nemotron-3-Super-120B RL |
