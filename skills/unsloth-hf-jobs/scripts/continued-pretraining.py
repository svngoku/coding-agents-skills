# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "unsloth",
#     "datasets",
#     "trl",
#     "huggingface_hub[hf_transfer]",
#     "trackio",
# ]
# ///
"""
Continued pretraining of language models using streaming datasets.

Demonstrates domain adaptation with streaming - no disk space needed.
Uses FineWeb-2's Latin subset as default example (1.47M texts, ~1.7GB).

Run locally (if you have a GPU):
    uv run continued-pretraining.py --output-repo your-username/qwen-latin

Run on HF Jobs:
    hf jobs uv run \
        https://huggingface.co/datasets/uv-scripts/unsloth-jobs/raw/main/continued-pretraining.py \
        --flavor a100-large --secrets HF_TOKEN \
        -- --max-steps 1000 --output-repo your-username/qwen-latin

With custom dataset:
    uv run continued-pretraining.py \
        --dataset your-username/domain-texts \
        --text-column content \
        --max-steps 1000 \
        --output-repo your-username/domain-llm
"""

import argparse
import logging
import os
import sys
import time

# Force unbuffered output for HF Jobs logs
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def check_cuda():
    """Check CUDA availability and exit if not available."""
    import torch

    if not torch.cuda.is_available():
        logger.error("CUDA is not available. This script requires a GPU.")
        logger.error("Run on a machine with a CUDA-capable GPU or use HF Jobs:")
        logger.error(
            "  hf jobs uv run https://huggingface.co/datasets/uv-scripts/unsloth-jobs/raw/main/continued-pretraining.py --flavor a100-large ..."
        )
        sys.exit(1)
    logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Continued pretraining of LLMs using streaming datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on Latin (default)
  uv run continued-pretraining.py \\
      --max-steps 500 \\
      --output-repo username/qwen-latin

  # Custom dataset
  uv run continued-pretraining.py \\
      --dataset your-username/domain-texts \\
      --text-column content \\
      --max-steps 1000 \\
      --output-repo username/domain-llm

  # HF Jobs with monitoring
  hf jobs uv run \\
      https://huggingface.co/datasets/uv-scripts/unsloth-jobs/raw/main/continued-pretraining.py \\
      --flavor a100-large --secrets HF_TOKEN \\
      -- --max-steps 1000 --trackio-space username/trackio --output-repo username/qwen-latin
        """,
    )
    parser.add_argument(
        "--base-model",
        default="unsloth/Qwen3-0.6B-Base-unsloth-bnb-4bit",
        help="Base model to fine-tune (default: unsloth/Qwen3-0.6B-Base-unsloth-bnb-4bit)",
    )
    parser.add_argument(
        "--dataset",
        default="HuggingFaceFW/fineweb-2",
        help="Dataset for continued pretraining (default: HuggingFaceFW/fineweb-2)",
    )
    parser.add_argument(
        "--dataset-config",
        default="lat_Latn",
        help="Dataset config/subset name (default: lat_Latn for Latin)",
    )
    parser.add_argument(
        "--text-column",
        default="text",
        help="Column containing text data (default: text)",
    )
    parser.add_argument(
        "--output-repo",
        required=True,
        help="HF Hub repo to push model to (e.g., 'username/qwen-latin')",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Number of training steps (default: 500)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Per-device batch size (default: 4)",
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048)",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank (default: 16)",
    )
    parser.add_argument(
        "--save-local",
        default="pretraining-output",
        help="Local directory to save model (default: pretraining-output)",
    )
    parser.add_argument(
        "--trackio-space",
        default=None,
        help="HF Space for Trackio dashboard (e.g., 'username/trackio')",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("Continued Pretraining with Streaming Datasets")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Base model:      {args.base_model}")
    print(f"  Dataset:         {args.dataset} ({args.dataset_config})")
    print(f"  Text column:     {args.text_column}")
    print(f"  Max steps:       {args.max_steps}")
    print(f"  Batch size:      {args.batch_size} x {args.gradient_accumulation} = {args.batch_size * args.gradient_accumulation}")
    print(f"  Learning rate:   {args.learning_rate}")
    print(f"  LoRA rank:       {args.lora_r}")
    print(f"  Output repo:     {args.output_repo}")
    print(f"  Trackio space:   {args.trackio_space or '(not configured)'}")
    print()

    # Check CUDA before heavy imports
    check_cuda()

    # Enable fast transfers
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    # Set Trackio space if provided
    if args.trackio_space:
        os.environ["TRACKIO_SPACE_ID"] = args.trackio_space
        logger.info(f"Trackio dashboard: https://huggingface.co/spaces/{args.trackio_space}")

    # Import heavy dependencies
    from unsloth import FastLanguageModel
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig
    from huggingface_hub import login

    # Login to Hub
    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token)
        logger.info("Logged in to Hugging Face Hub")
    else:
        logger.warning("HF_TOKEN not set - model upload may fail")

    # 1. Load model
    print("\n[1/5] Loading model...")
    start = time.time()

    model, tokenizer = FastLanguageModel.from_pretrained(
        args.base_model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    print(f"Model loaded in {time.time() - start:.1f}s")

    # 2. Load streaming dataset
    print(f"\n[2/5] Loading streaming dataset ({args.dataset})...")
    start = time.time()

    # Handle dataset with or without config
    if args.dataset_config:
        dataset = load_dataset(
            args.dataset,
            name=args.dataset_config,
            split="train",
            streaming=True,
        )
    else:
        dataset = load_dataset(
            args.dataset,
            split="train",
            streaming=True,
        )

    # Peek at the data
    sample = next(iter(dataset))
    text_preview = sample[args.text_column][:100] if args.text_column in sample else "(column not found)"
    print(f"Dataset ready in {time.time() - start:.1f}s")
    print(f"  Sample: {text_preview}...")

    # Reload dataset (consumed one sample above)
    if args.dataset_config:
        dataset = load_dataset(
            args.dataset,
            name=args.dataset_config,
            split="train",
            streaming=True,
        )
    else:
        dataset = load_dataset(
            args.dataset,
            split="train",
            streaming=True,
        )

    # 3. Format dataset
    print("\n[3/5] Preparing dataset...")

    text_column = args.text_column

    def format_text(example):
        return {"text": example[text_column] + tokenizer.eos_token}

    formatted_dataset = dataset.map(format_text)

    # 4. Train
    print(f"\n[4/5] Training for {args.max_steps} steps...")
    start = time.time()

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=formatted_dataset,
        args=SFTConfig(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            warmup_steps=min(10, args.max_steps // 10),
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            logging_steps=max(1, args.max_steps // 20),
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=args.save_local,
            report_to="trackio",
            run_name=f"pretraining-{args.max_steps}steps",
            dataset_text_field="text",
            max_seq_length=args.max_seq_length,
            packing=False,
        ),
    )

    trainer.train()
    train_time = time.time() - start

    print(f"\nTraining completed in {train_time / 60:.1f} minutes")
    print(f"  Speed: {args.max_steps / train_time:.2f} steps/s")

    # 5. Save and push
    print("\n[5/5] Saving model...")

    # Save locally
    model.save_pretrained(args.save_local)
    tokenizer.save_pretrained(args.save_local)
    print(f"Saved locally to {args.save_local}/")

    # Push to hub
    print(f"\nPushing to {args.output_repo}...")
    model.push_to_hub(args.output_repo, tokenizer=tokenizer)
    print(f"Model available at: https://huggingface.co/{args.output_repo}")

    # Quick inference test
    print("\n" + "=" * 70)
    print("Quick inference test:")
    print("=" * 70)

    FastLanguageModel.for_inference(model)

    # Use a prompt appropriate to the dataset
    if "lat_Latn" in (args.dataset_config or ""):
        prompt = "Lingua Latina est"
    else:
        prompt = "The quick brown fox"

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        temperature=0.7,
        do_sample=True,
    )
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"\nPrompt: {prompt}")
    print(f"Generated: {generated}")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    # Show example usage if no arguments
    if len(sys.argv) == 1:
        print("=" * 70)
        print("Continued Pretraining with Streaming Datasets")
        print("=" * 70)
        print("\nContinued pretraining for domain adaptation.")
        print("Streams data directly from the Hub - no disk space needed.")
        print("\nFeatures:")
        print("  - ~60% less VRAM with Unsloth optimizations")
        print("  - 2x faster training vs standard methods")
        print("  - Trackio integration for monitoring")
        print("  - Works with any text dataset")
        print("\nDefault example (Latin):")
        print("\n  uv run continued-pretraining.py \\")
        print("      --max-steps 500 \\")
        print("      --output-repo your-username/qwen-latin")
        print("\nHF Jobs example:")
        print("\n  hf jobs uv run \\")
        print("      https://huggingface.co/datasets/uv-scripts/unsloth-jobs/raw/main/continued-pretraining.py \\")
        print("      --flavor a100-large --secrets HF_TOKEN \\")
        print("      -- --max-steps 1000 --output-repo your-username/qwen-latin")
        print("\nCustom dataset:")
        print("\n  uv run continued-pretraining.py \\")
        print("      --dataset your-username/domain-texts \\")
        print("      --text-column content \\")
        print("      --output-repo your-username/domain-llm")
        print("\nFor full help: uv run continued-pretraining.py --help")
        print("=" * 70)
        sys.exit(0)

    main()