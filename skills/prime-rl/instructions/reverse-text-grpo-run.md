Set up a prime-rl GRPO post-training run for the `reverse-text` environment. Write two files
in the current directory: the config as `rl.toml` and the launch commands as `run.sh`. Do NOT
run anything; there is no GPU here.

`rl.toml` must configure the `uv run rl` entrypoint with:

- Policy model `PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT`, 40 training steps, sequence length 2048.
- 128 rollouts per trainer step and 16 rollouts per task (GRPO needs a real group).
- At most 128 completion tokens per training rollout.
- One training source named `reverse-text` using taskset `reverse-text`, the `null` agent harness
  and the `subprocess` runtime.
- An online eval every 10 steps on the same taskset (source name `reverse-text-eval`), limited
  to 128 tasks.
- Learning rate 3e-6.
- The IcePop RL loss instead of the default loss.
- Checkpoints every 10 steps, keeping only the last 3.
- Weights & Biases logging to project `reverse-text`.
- The `prime-qwen3` renderer, because this fine-tune's name does not match the automatic
  renderer map.
- The inference server launched as part of the run.

Follow prime-rl's config conventions (snake_case keys, no inline tables).

`run.sh` must contain, in this order, with a short comment above each:

1. A dry run that resolves and validates `rl.toml` without launching.
2. The real launch with the run name `rt-grpo`, so it can be resumed later.
3. A resume of `rt-grpo` from its latest checkpoint that continues training to step 80.
4. An export of the step-80 checkpoint of `rt-grpo` (default output dir) to Hugging Face
   safetensors.
