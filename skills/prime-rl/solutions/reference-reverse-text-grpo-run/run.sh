#!/usr/bin/env bash
set -euo pipefail

# 1. Resolve and validate the config without launching; writes configs/latest/resolved/.
uv run rl @ rl.toml --dry-run --run.name rt-grpo-check

# 2. Launch with a fixed run name so the run directory (outputs/rt-grpo) can be resumed.
uv run rl @ rl.toml --run.name rt-grpo

# 3. Resume from the latest checkpoint. --max-steps is the absolute target step, not the delta.
uv run rl @ rl.toml --run.name rt-grpo --resume --max-steps 80

# 4. Export the step-80 DCP checkpoint to HF safetensors (written to .../step_80/weights).
uv run python tools/convert_dcp_to_bf16.py outputs/rt-grpo/checkpoints/step_80
