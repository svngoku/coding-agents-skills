Score the agent's solution (rl.toml and run.sh) from 0.0 to 1.0. Read both files and evaluate
these four criteria:

- Config correctness and conventions: does rl.toml target the `uv run rl` schema (shared
  `max_steps`, `seq_len`, `[model]`; `[orchestrator]`, `[orchestrator.train]`,
  `[[orchestrator.train.source]]`, `[orchestrator.eval]`, `[[orchestrator.eval.source]]`;
  `[trainer.optim]`, `[trainer.loss] type = "icepop"`; `[ckpt]`; `[monitors.wandb]`;
  `[orchestrator.renderer]`; an `[inference]` table), with snake_case keys, env settings written as
  dotted keys inside each source block (`env.taskset.id`, `env.agent.harness.id`,
  `env.agent.runtime.type`), and no inline tables?
- GRPO awareness: is `group_size` set well above the default of 1 (16 here), and is the algorithm
  left as GRPO? Bonus awareness: a comment noting that a group of identical rewards gives zero
  advantage, or that the renderer is explicit because the fine-tune name misses the auto map.
- Run lifecycle: does run.sh dry-run first, launch with a fixed `--run.name rt-grpo`, resume with
  `--resume` and an absolute `--max-steps 80` (not a delta of 40), and export
  `outputs/rt-grpo/checkpoints/step_80` with `tools/convert_dcp_to_bf16.py`, all via `uv run`
  (never `vllm serve`, bare `torchrun` or raw `python`)? A dry run that reuses the launch name so
  the real launch would then hit an existing run directory is a minor flaw.
- Clarity: does each run.sh step have a short comment explaining why, and are both files clean
  and free of invented fields (for example `max_async_level`, `[trainer.loss] eps` on IcePop,
  or `[orchestrator.train.source.env]` sub-headers duplicating dotted keys)?

Assign 1.0 when all four are fully met; 0.7 when mostly correct with one partial miss (for example
the resume passes `--max-steps 40`, or comments are missing); 0.4 when several required settings
are wrong or missing, or the config uses another framework's schema; 0.0 when the files are
missing or fundamentally wrong.
