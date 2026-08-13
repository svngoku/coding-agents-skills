Write a shell script (train.sh) that fine-tunes Qwen3-VL-8B on Hugging Face
Jobs using the Unsloth UV script. The script must contain TWO `hf jobs uv run`
commands, both invoking the VLM fine-tuning script `sft-qwen3-vl.py` from the
remote script URL
`https://huggingface.co/datasets/uv-scripts/unsloth-jobs/raw/main/`:

1. A SMOKE-TEST command that caps the run at `--max-steps 10`, with a comment
   explaining why the smoke test must run before the full run.
2. A FULL TRAINING RUN command for Qwen3-VL-8B that uses exactly these options:
   - `--flavor a100-large`
   - `--secrets HF_TOKEN`
   - `--timeout 4h`
   - `--dataset <username>/<dataset>`
   - `--output-repo <username>/<model-name>`
   - `--lora-r 16`
   - `--learning-rate 2e-4`
   - `--batch-size 2`
   - `--gradient-accumulation 4`
   - `--eval-split 0.1`
   - `--merge-model`
   - `--trackio-space <username>/trackio`
   - plus a steps/epochs control (`--num-epochs 1` or a larger `--max-steps`)

Use the invocation pattern from the skill: job-level options (`--flavor`,
`--secrets`, `--timeout`) come before the `--` separator, and script-level
options (`--dataset`, `--output-repo`, training config) come after it.
Replace `<username>`, `<dataset>` and `<model-name>` with your HF username,
dataset and output repo, or keep the placeholders. Add short comments explaining
each section (why the smoke test runs first, what `--merge-model` does, what
`--trackio-space` is for). Do NOT run the commands -- just write the script.

Save the result as train.sh in the current directory.
