# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "unsloth",
#     "datasets",
#     "trl",
#     "huggingface_hub[hf_transfer]",
#     "trackio",
#     "transformers==4.56.2",
#     "trl==0.22.2",
# ]
# ///
"""
Fine-tune Gemma 3 4B Vision Language Model using Unsloth optimizations.

Streams data directly from the Hub - no disk space needed for massive VLM datasets.
Uses Unsloth for ~60% less VRAM and 2x faster training.

Run locally (if you have a GPU):
    uv run sft-gemma3-vlm.py \
        --max-steps 100 \
        --output-repo your-username/vlm-test

Run on HF Jobs:
    hf jobs uv run \
        https://huggingface.co/datasets/uv-scripts/unsloth-jobs/raw/main/sft-gemma3-vlm.py \
        --flavor a100-large --secrets HF_TOKEN \
        -- --max-steps 500 --output-repo your-username/vlm-finetuned

With Trackio dashboard:
    uv run sft-gemma3-vlm.py \
        --max-steps 500 \
        --output-repo your-username/vlm-finetuned \
        --trackio-space your-username/trackio
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
            "  hf jobs uv run https://huggingface.co/datasets/uv-scripts/unsloth-jobs/raw/main/sft-gemma3-vlm.py --flavor a100-large ..."
        )
        sys.exit(1)
    logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune Gemma 3 4B VLM with streaming datasets using Unsloth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test run
  uv run sft-gemma3-vlm.py \\
      --max-steps 50 \\
      --output-repo username/vlm-test

  # Full training with Trackio monitoring
  uv run sft-gemma3-vlm.py \\
      --max-steps 500 \\
      --output-repo username/vlm-finetuned \\
      --trackio-space username/trackio

  # Custom dataset
  uv run sft-gemma3-vlm.py \\
      --dataset your-username/your-vlm-dataset \\
      --max-steps 1000 \\
      --output-repo username/custom-vlm
        """,
    )

    # Model and data
    parser.add_argument(
        "--base-model",
        default="unsloth/gemma-3-4b-pt",
        help="Base VLM model (default: unsloth/gemma-3-4b-pt)",
    )
    parser.add_argument(
        "--dataset",
        default="davanstrien/iconclass-vlm-sft",
        help="Dataset with 'images' and 'messages' columns (default: davanstrien/iconclass-vlm-sft)",
    )
    parser.add_argument(
        "--output-repo",
        required=True,
        help="HF Hub repo to push model to (e.g., 'username/vlm-finetuned')",
    )

    # Training config
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Training steps (default: 500). Required for streaming datasets.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Per-device batch size (default: 2)",
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4). Effective batch = batch-size * this",
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

    # LoRA config
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank (default: 16). Higher = more capacity but more VRAM",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha (default: 32). Usually 2*r",
    )

    # Logging
    parser.add_argument(
        "--trackio-space",
        default=None,
        help="HF Space for Trackio dashboard (e.g., 'username/trackio')",
    )
    parser.add_argument(
        "--save-local",
        default="vlm-gemma3-output",
        help="Local directory to save model (default: vlm-gemma3-output)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("Gemma 3 4B VLM Streaming Fine-tuning with Unsloth")
    print("=" * 70)
    print("\nConfiguration:")
    print(f"  Base model:      {args.base_model}")
    print(f"  Dataset:         {args.dataset}")
    print(f"  Max steps:       {args.max_steps}")
    print(
        f"  Batch size:      {args.batch_size} x {args.gradient_accumulation} = {args.batch_size * args.gradient_accumulation}"
    )
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

    # Import heavy dependencies (note: import from unsloth.trainer for VLM)
    from unsloth import FastVisionModel, get_chat_template
    from unsloth.trainer import UnslothVisionDataCollator
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

    model, processor = FastVisionModel.from_pretrained(
        args.base_model,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
        target_modules="all-linear",
    )

    # Apply chat template (required for base models)
    processor = get_chat_template(processor, "gemma-3")
    print(f"Model loaded in {time.time() - start:.1f}s")

    # 2. Load streaming dataset
    print("\n[2/5] Loading streaming dataset...")
    start = time.time()

    dataset = load_dataset(
        args.dataset,
        split="train",
        streaming=True,
    )

    # Peek at first sample to show info
    sample = next(iter(dataset))
    print(f"Dataset ready in {time.time() - start:.1f}s")
    if "messages" in sample:
        print(f"  Sample has {len(sample['messages'])} messages")
    if "images" in sample:
        img_count = len(sample['images']) if isinstance(sample['images'], list) else 1
        print(f"  Sample has {img_count} image(s)")

    # Reload dataset (consumed one sample above)
    dataset = load_dataset(
        args.dataset,
        split="train",
        streaming=True,
    )

    # 3. Configure trainer
    print("\n[3/5] Configuring trainer...")

    # Enable training mode
    FastVisionModel.for_training(model)

    training_config = SFTConfig(
        output_dir=args.save_local,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        logging_steps=max(1, args.max_steps // 20),
        save_strategy="steps",
        optim="adamw_torch_fused",
        weight_decay=0.001,
        lr_scheduler_type="cosine",
        seed=3407,
        # VLM-specific settings (required for Unsloth)
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=args.max_seq_length,
        # Logging
        report_to="trackio",
        run_name=f"gemma3-vlm-{args.max_steps}steps",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=processor.tokenizer,
        data_collator=UnslothVisionDataCollator(model, processor),
        args=training_config,
    )

    # 4. Train
    print(f"\n[4/5] Training for {args.max_steps} steps...")
    start = time.time()

    trainer.train()

    train_time = time.time() - start
    print(f"\nTraining completed in {train_time / 60:.1f} minutes")
    print(f"  Speed: {args.max_steps / train_time:.2f} steps/s")

    # 5. Save and push
    print("\n[5/5] Saving model...")

    # Save locally
    model.save_pretrained(args.save_local)
    processor.save_pretrained(args.save_local)
    print(f"Saved locally to {args.save_local}/")

    # Push to Hub
    print(f"\nPushing to {args.output_repo}...")
    model.push_to_hub(args.output_repo)
    processor.push_to_hub(args.output_repo)
    print(f"Model available at: https://huggingface.co/{args.output_repo}")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    # Show example usage if no arguments
    if len(sys.argv) == 1:
        print("=" * 70)
        print("Gemma 3 4B VLM Streaming Fine-tuning with Unsloth")
        print("=" * 70)
        print("\nFine-tune Vision-Language Models using streaming datasets.")
        print("Data streams directly from the Hub - no disk space needed.")
        print("\nFeatures:")
        print("  - ~60% less VRAM with Unsloth optimizations")
        print("  - 2x faster training vs standard methods")
        print("  - Trackio integration for monitoring")
        print("  - Works with any VLM dataset in conversation format")
        print("\nExample usage:")
        print("\n  uv run sft-gemma3-vlm.py \\")
        print("      --max-steps 500 \\")
        print("      --output-repo your-username/vlm-finetuned")
        print("\nHF Jobs example:")
        print("\n  hf jobs uv run \\")
        print("      https://huggingface.co/datasets/uv-scripts/unsloth-jobs/raw/main/sft-gemma3-vlm.py \\")
        print("      --flavor a100-large --secrets HF_TOKEN \\")
        print("      -- --max-steps 500 --output-repo your-username/vlm-finetuned")
        print("\nFor full help: uv run sft-gemma3-vlm.py --help")
        print("=" * 70)
        sys.exit(0)

    main()