---
name: unsloth-hf-jobs
description: Fine-tune LLMs and VLMs using Unsloth on HF Jobs (Hugging Face on-demand cloud GPUs). Use when users want to fine-tune language models, train VLMs (Vision Language Models), do continued pretraining, domain adaptation, or run UV scripts on HF Jobs. Triggers on requests involving Unsloth training, HF Jobs GPU training, Qwen3-VL fine-tuning, Gemma VLM training, or LoRA fine-tuning on cloud GPUs.
---

# Unsloth Training on HF Jobs

Fine-tune LLMs and VLMs using [Unsloth](https://github.com/unslothai/unsloth) on [HF Jobs](https://huggingface.co/docs/hub/jobs) with UV scripts that handle dependencies automatically.

## Prerequisites

- Hugging Face account with [token](https://huggingface.co/settings/tokens)
- HF CLI: `curl -LsSf https://hf.co/cli/install.sh | bash`
- Dataset on the Hub (see data formats below)

## Available Scripts

| Script | Base Model | Task |
|--------|------------|------|
| `scripts/sft-qwen3-vl.py` | Qwen3-VL-8B | VLM fine-tuning |
| `scripts/sft-gemma3-vlm.py` | Gemma 3 4B | VLM fine-tuning (smaller) |
| `scripts/continued-pretraining.py` | Qwen3-0.6B | Domain adaptation |

Remote URL (for HF Jobs): `https://huggingface.co/datasets/uv-scripts/unsloth-jobs/raw/main/`

## Data Formats

### VLM Fine-tuning

Requires `images` and `messages` columns:

```python
{
    "images": [<PIL.Image>],
    "messages": [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What's in this image?"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "A golden retriever playing fetch."}]}
    ]
}
```

Example dataset: [davanstrien/iconclass-vlm-sft](https://huggingface.co/datasets/davanstrien/iconclass-vlm-sft)

### Continued Pretraining

Any dataset with a text column (use `--text-column` if named differently):

```python
{"text": "Your domain-specific text here..."}
```

## Usage Patterns

### VLM Fine-tuning

```bash
hf jobs uv run \
  https://huggingface.co/datasets/uv-scripts/unsloth-jobs/raw/main/sft-qwen3-vl.py \
  --flavor a100-large --secrets HF_TOKEN --timeout 4h \
  -- --dataset <username>/<dataset> \
     --num-epochs 1 \
     --eval-split 0.2 \
     --output-repo <username>/<model-name>
```

### Continued Pretraining

```bash
hf jobs uv run \
  https://huggingface.co/datasets/uv-scripts/unsloth-jobs/raw/main/continued-pretraining.py \
  --flavor a100-large --secrets HF_TOKEN \
  -- --dataset <username>/<dataset> \
     --text-column content \
     --max-steps 1000 \
     --output-repo <username>/<model-name>
```

### With Trackio Monitoring

Add `--trackio-space <username>/trackio` to any command for live monitoring.

## Common Options

| Option | Description | Default |
|--------|-------------|---------|
| `--dataset` | HF dataset ID | *required* |
| `--output-repo` | Where to save trained model | *required* |
| `--max-steps` | Training steps | 500 |
| `--num-epochs` | Train for N epochs (overrides steps) | - |
| `--eval-split` | Fraction for evaluation | 0 |
| `--batch-size` | Per-device batch size | 2 |
| `--gradient-accumulation` | Accumulation steps | 4 |
| `--lora-r` | LoRA rank | 16 |
| `--learning-rate` | Learning rate | 2e-4 |
| `--merge-model` | Upload merged model (not just adapter) | false |
| `--streaming` | Stream large datasets | false |

## Quick Tips

- Verify setup with `--max-steps 10` before full runs
- Use `--eval-split 0.1` to detect overfitting
- Check GPU pricing: `hf jobs hardware` (A100-large ~$2.50/hr, L40S ~$1.80/hr)
- First step may take minutes (CUDA kernel compilation)
- View script options: `uv run <script-url> --help`

## Resources

- [HF Jobs Quickstart](https://huggingface.co/docs/hub/jobs-quickstart)
- [Unsloth Documentation](https://docs.unsloth.ai/)
- [UV Scripts Guide](https://docs.astral.sh/uv/guides/scripts/)