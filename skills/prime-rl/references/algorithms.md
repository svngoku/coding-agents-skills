# Algorithms, Losses, Advantages, Curricula and Renderers

prime-rl splits post-training into two independent choices:

- **The algorithm** (`[orchestrator.train.algo]`, on the orchestrator) decides who samples rollouts
  and what per-token signal each token gets (advantage, CE target or reference logprobs).
- **The RL loss** (`[trainer.loss]`, on the trainer) decides how the `rl` component turns
  advantages into gradients. The trainer is algorithm-blind.

The total loss is a sum of three components, each normalized by its own token count:
`rl` (the configured RL loss), `ce` (masked NLL) and `ref_kl` (reverse KL to a reference model).

## Algorithms

`type` names the algorithm, and its class defaults are the vetted setting. Any key you add beyond
`type` is your own variant.

| `type` | Sampling | Loss component | Key fields |
|--------|----------|----------------|------------|
| `grpo` (default) | policy | `rl` | `length_penalty`, `length_weighted_baseline = false`, `loss_aggregation = "token"` |
| `max_rl` | policy | `rl` | none (mean-normalized group credit, upweights hard tasks ~1/p) |
| `rae` | policy | `rl` | `decay = 0.95` (per-agent EMA baseline; `group_size = 1` is fine) |
| `hierarchical_grpo` | policy | `rl` | `episode_agents` (required, for example `["solver"]`) |
| `opd` | policy | `ref_kl` | `teacher` (`name`, `base_url`; must be a vLLM server) |
| `sft` | frozen teacher | `ce` | `sampling.source` (`name`, `base_url`; any OpenAI-compatible endpoint) |
| `opsd` | policy | `ref_kl` | `demo_key = "demonstration"`, `template`, `renderer` |
| `echo` | policy | `rl` + `ce` on env tokens | `roles.<role>.alpha` (default tool at 0.1), optional `filter` |

### GRPO and variants

The default advantage is group reward minus group mean (DR-GRPO without std normalization). A group
where every rollout scores the same gets zero advantage and contributes nothing, which is why
`group_size` must be well above 1 (8 minimum, 16–32 common).

```toml
[orchestrator.train]
group_size = 16

[orchestrator.train.algo]
type = "grpo"
length_weighted_baseline = true
loss_aggregation = "prompt"   # MiMo-style prompt-mean; every rl env in the run must match

[orchestrator.train.algo.length_penalty]
type = "linear"               # output 0.25, input 0.1, turns 0.1 weights by default
```

### Distillation

prime-rl only hosts the trainable policy. Start teachers yourself and point `base_url` at them:

```bash
CUDA_VISIBLE_DEVICES=1 uv run inference --vllm.model Qwen/Qwen3-32B --server.port 8001
```

```toml
# On-policy distillation: policy samples, reverse KL to the teacher
[orchestrator.train.algo]
type = "opd"

[orchestrator.train.algo.teacher]
name = "Qwen/Qwen3-32B"
base_url = "http://localhost:8001/v1"
```

```toml
# Hard distillation: teacher samples, policy trains with CE on its tokens
[orchestrator.train.algo]
type = "sft"

[orchestrator.train.algo.sampling.source]
name = "Qwen/Qwen3-32B"
base_url = "http://localhost:8001/v1"
```

```toml
# Self-distillation conditioned on a demonstration from task info
[orchestrator.train.algo]
type = "opsd"
demo_key = "demonstration"
```

Validation rules: frozen-model sampling can only feed `ce`; `opd` cannot use `"policy"` as teacher
(zero KL); `sft` without a frozen source is rejected. Use the standalone `uv run sft` entrypoint for
plain dataset SFT; the `sft` algorithm is for teacher-generated supervision on the fly.

### ECHO

GRPO on actions plus weighted CE on env-provided tokens selected by message role. Setting any role
replaces the whole role table.

```toml
[orchestrator.train.algo]
type = "echo"

[orchestrator.train.algo.roles.tool]
alpha = 0.25

[orchestrator.train.algo.roles.user]
alpha = 0.05

[orchestrator.train.algo.filter]
import_path = "my_module.drop_warnings"

[orchestrator.train.algo.filter.kwargs]
patterns = ["WARNING"]
```

### Self-play and proposer-solver

```toml
[orchestrator.train.algo]
type = "rae"
decay = 0.95

[[orchestrator.train.source]]
name = "kuhn-poker"
env.taskset.id = "kuhn-poker"
env.player0.harness.id = "null"
env.player0.runtime.type = "subprocess"
env.player1.harness.id = "null"
env.player1.runtime.type = "subprocess"
```

```toml
[orchestrator.train.algo]
type = "hierarchical_grpo"
episode_agents = ["solver"]

[[orchestrator.train.source]]
name = "proposer-solver"
group_size = 4   # proposed problems per source task
env.n = 4        # solver attempts per proposed problem
env.taskset.id = "proposer-solver"
env.proposer.harness.id = "null"
env.proposer.runtime.type = "subprocess"
env.solver.harness.id = "null"
env.solver.runtime.type = "subprocess"
```

### Per-env algorithms

Each train source inherits the group algorithm. A source that sets a different `algo.type` runs its
own algorithm; one that sets only params keeps the group's type with those params changed.

```toml
[orchestrator.train.algo]
type = "grpo"

[[orchestrator.train.source]]
name = "math"
env.taskset.id = "math"
env.agent.harness.id = "null"
env.agent.runtime.type = "subprocess"

[[orchestrator.train.source]]
name = "terminal"
algo.type = "echo"
env.taskset.id = "terminal"
env.agent.harness.id = "bash"
env.agent.runtime.type = "subprocess"
```

### New algorithms

There is no config hook for user code. A new algorithm is a class in the repo: subclass
`prime_rl.orchestrator.train.algo.base.Algorithm`, implement `score_group` (group-relative) or
`score_episode` (per-rollout or model calls), add a config class to `prime_rl.configs.algorithm`, and
register it in `ALGORITHM_CLASSES`.

## RL Losses (`[trainer.loss]`)

| `type` | Defaults | Mechanism |
|--------|----------|-----------|
| `ipo` (default) | `eps = 0.3`, `max_importance_ratio = 1e4`, `adv_tau = 1.0` | Masks tokens whose absolute probability change exceeds `eps` |
| `icepop` | `ratio_low = 0.2`, `ratio_high = 5.0`, `adv_tau = 1.0` | Masks tokens whose importance ratio leaves the band |
| `ppo` | `ratio_low = 0.8`, `ratio_high = 1.2`, `max_importance_ratio = 1e4` | Clipped surrogate |
| `cispo` | `ratio_low = 0.0`, `ratio_high = 5.0` | Clipped detached importance weight |
| `custom` | `import_path`, `kwargs` | Your per-sequence loss function |

Omit `[trainer.loss]` for IPO defaults. The `ce` and `ref_kl` components are not affected.

Why `eps = 0.3`: in upstream's GLM-4.5-Air SWE ablation (`docs/ipo.md`), 0.3 masked about 0.25% of
tokens; the older 0.1 masked about 2% and collapsed entropy; IcePop's mismatch KL blew up near step
1.2k. Reproduce with `examples/advanced/glm-4.5-air/ipo-ablation/{base,ipo,icepop}.toml`.

### Custom loss

```python
# my_module.py
import torch
from prime_rl.trainer.rl.loss import LossInputs, LossOutputs


def ppo_clip_loss(inputs: LossInputs, clip_eps: float = 0.2) -> LossOutputs:
    ratio = torch.exp(inputs.trainer_logprobs - inputs.inference_logprobs)
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
    loss = -torch.min(ratio * inputs.advantages, clipped * inputs.advantages)[inputs.loss_mask].sum()
    return LossOutputs(loss=loss, metrics={"clip_frac": (ratio != clipped)[inputs.loss_mask].float().mean()})
```

```toml
[trainer.loss]
type = "custom"
import_path = "my_module.ppo_clip_loss"

[trainer.loss.kwargs]
clip_eps = 0.2
```

`LossInputs` carries `trainer_logprobs`, `inference_logprobs`, `ref_logprobs`, `advantages`,
`loss_mask` and `loss_weights` for one sequence. Values in `metrics` are averaged and logged.

## Async and Off-Policy

Training is asynchronous by default: while the trainer produces policy n, inference samples step
n+1 rollouts with policy n-1. `orchestrator.max_off_policy_steps` (default 8) bounds how stale a
trained rollout may be; older episodes are dropped. Raise it for throughput on long agentic
rollouts, lower it for tighter on-policy training, and watch `off_policy/*` and `mismatch_kl`.
There is no `max_async_level` field.

`orchestrator.constant_trainer_batch_size = true` (default) refills batches after filtering
zero-signal samples so the trainer always sees `orchestrator.batch_size` traces. Set it to `false`
for faster collection with smaller batches.

## Curricula

Each train source has a curriculum: one task sampler plus optional admission gates. Sampler and gate
state is checkpointed and logged under `curriculum/<env>/`.

```toml
[orchestrator.train.source.curriculum.sampler]
type = "difficulty_pool"

[orchestrator.train.source.curriculum.sampler.pools.hard]
threshold = 0.25
weight = 0.2

[orchestrator.train.source.curriculum.sampler.pools.normal]
threshold = 0.75
weight = 1.0

[orchestrator.train.source.curriculum.sampler.pools.easy]
threshold = 1.0
weight = 0.2

[orchestrator.train.source.curriculum.gates.low_signal]
type = "advantage_range"
reject_min = -0.05
reject_max = 0.05
```

These sections attach to the most recent `[[orchestrator.train.source]]` block, so place them right
after the source they configure.

## Multi-Turn Trajectories and Renderers

Each LLM call in a multi-turn rollout is a trajectory step. Steps merge into one training sample
while each prompt is an exact token prefix of the next (the extension property). When it breaks, the
trainer starts a new sample instead of training on corrupted tokens.

Renderers keep the prefix exact by extending prior tokens instead of re-rendering the chat template
(Qwen3's template, for example, strips past `<think>` blocks). Hand-coded renderers ship for qwen3,
qwen3-vl, qwen3.5, glm-5, glm-4.5, minimax-m2, deepseek-v3, kimi-k2, kimi-k2.5, nemotron-3 and
gpt-oss; everything else falls back to `DefaultRenderer`.

```toml
[orchestrator.renderer]
name = "auto"          # detect from the tokenizer

# Fine-tunes whose name misses the renderer map need an explicit name
# name = "prime-qwen3"

# Out-of-tree renderer
# name = "custom"
# import_path = "my_renderers.MyRenderer"
```

SFT uses the same renderer config at top-level `[renderer]`.
