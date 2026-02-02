# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "unsloth",
#     "datasets",
#     "trl==0.22.2",
#     "huggingface_hub[hf_transfer]",
#     "trackio",
#     "tensorboard",
#     "transformers==4.57.1",
# ]
# ///
"""
Fine-tune Qwen3-VL-8B Vision Language Model using Unsloth optimizations.

Uses Unsloth for ~60% less VRAM and 2x faster training.
Supports epoch-based or step-based training with optional eval split.

Epoch-based training (recommended for full datasets):
    uv run sft-qwen3-vl.py \
        --num-epochs 1 \
        --eval-split 0.2 \
        --output-repo your-username/vlm-finetuned

Run on HF Jobs (1 epoch with eval):
    hf jobs uv run \
        https://huggingface.co/datasets/uv-scripts/unsloth-jobs/raw/main/sft-qwen3-vl.py \
        --flavor a100-large --secrets HF_TOKEN --timeout 4h \
        -- --num-epochs 1 --eval-split 0.2 --output-repo your-username/vlm-finetuned

Step-based training (for streaming or quick tests):
    uv run sft-qwen3-vl.py \
        --streaming \
        --max-steps 500 \
        --output-repo your-username/vlm-finetuned

Quick test with limited samples:
    uv run sft-qwen3-vl.py \
        --num-samples 500 \
        --num-epochs 2 \
        --eval-split 0.2 \
        --output-repo your-username/vlm-test
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
            "  hf jobs uv run https://huggingface.co/datasets/uv-scripts/unsloth-jobs/raw/main/sft-qwen3-vl.py --flavor a100-large ..."
        )
        sys.exit(1)
    logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen3-VL-8B with streaming datasets using Unsloth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test run
  uv run sft-qwen3-vl.py \\
      --max-steps 50 \\
      --output-repo username/vlm-test

  # Full training with Trackio monitoring
  uv run sft-qwen3-vl.py \\
      --max-steps 500 \\
      --output-repo username/vlm-finetuned \\
      --trackio-space username/trackio

  # Custom dataset and model
  uv run sft-qwen3-vl.py \\
      --base-model unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit \\
      --dataset your-username/your-vlm-dataset \\
      --max-steps 1000 \\
      --output-repo username/custom-vlm
        """,
    )

    # Model and data
    parser.add_argument(
        "--base-model",
        default="unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit",
        help="Base VLM model (default: unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit)",
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
        "--num-epochs",
        type=float,
        default=None,
        help="Number of epochs (default: None). Use instead of --max-steps for non-streaming mode.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Training steps (default: None). Required for streaming mode, optional otherwise.",
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
        default=16,
        help="LoRA alpha (default: 16). Same as r per Unsloth notebook",
    )

    # Logging
    parser.add_argument(
        "--trackio-space",
        default=None,
        help="HF Space for Trackio dashboard (e.g., 'username/trackio')",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Custom run name for Trackio (default: auto-generated from steps/epochs)",
    )
    parser.add_argument(
        "--save-local",
        default="vlm-qwen3-output",
        help="Local directory to save model (default: vlm-qwen3-output)",
    )

    # Evaluation and data control
    parser.add_argument(
        "--eval-split",
        type=float,
        default=0.0,
        help="Fraction of data for evaluation (0.0-0.5). Default: 0.0 (no eval)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Limit samples (default: None = use all for non-streaming, 500 for streaming)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="Random seed for reproducibility (default: 3407)",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        default=False,
        help="Use streaming mode (default: False). Use for very large datasets.",
    )
    parser.add_argument(
        "--merge-model",
        action="store_true",
        default=False,
        help="Merge LoRA weights into base model before uploading (larger file, easier to use)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Validate epochs/steps configuration
    if args.streaming and args.num_epochs:
        logger.error(
            "Cannot use --num-epochs with --streaming. Use --max-steps instead."
        )
        sys.exit(1)
    if args.streaming and not args.max_steps:
        args.max_steps = 500  # Default for streaming
        logger.info("Using default --max-steps=500 for streaming mode")
    if not args.streaming and not args.num_epochs and not args.max_steps:
        args.num_epochs = 1  # Default to 1 epoch for non-streaming
        logger.info("Using default --num-epochs=1 for non-streaming mode")

    # Determine training duration display
    if args.num_epochs:
        duration_str = f"{args.num_epochs} epoch(s)"
    else:
        duration_str = f"{args.max_steps} steps"

    print("=" * 70)
    print("Qwen3-VL-8B Fine-tuning with Unsloth")
    print("=" * 70)
    print("\nConfiguration:")
    print(f"  Base model:      {args.base_model}")
    print(f"  Dataset:         {args.dataset}")
    print(f"  Streaming:       {args.streaming}")
    print(
        f"  Num samples:     {args.num_samples or ('500' if args.streaming else 'all')}"
    )
    print(
        f"  Eval split:      {args.eval_split if args.eval_split > 0 else '(disabled)'}"
    )
    print(f"  Seed:            {args.seed}")
    print(f"  Training:        {duration_str}")
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
        logger.info(
            f"Trackio dashboard: https://huggingface.co/spaces/{args.trackio_space}"
        )

    # Import heavy dependencies (note: import from unsloth.trainer for VLM)
    from unsloth import FastVisionModel
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

    # 1. Load model (Qwen returns tokenizer, not processor)
    print("\n[1/5] Loading model...")
    start = time.time()

    model, tokenizer = FastVisionModel.from_pretrained(
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
    )
    print(f"Model loaded in {time.time() - start:.1f}s")

    # 2. Load dataset (streaming or non-streaming)
    print(
        f"\n[2/5] Loading dataset ({'streaming' if args.streaming else 'non-streaming'})..."
    )
    start = time.time()

    if args.streaming:
        # Streaming mode: take limited samples
        dataset = load_dataset(args.dataset, split="train", streaming=True)
        num_samples = args.num_samples or 500

        # Peek at first sample to show info
        sample = next(iter(dataset))
        if "messages" in sample:
            print(f"  Sample has {len(sample['messages'])} messages")
        if "images" in sample:
            img_count = (
                len(sample["images"]) if isinstance(sample["images"], list) else 1
            )
            print(f"  Sample has {img_count} image(s)")

        # Reload and take samples
        dataset = load_dataset(args.dataset, split="train", streaming=True)
        all_data = list(dataset.take(num_samples))
        print(f"  Loaded {len(all_data)} samples in {time.time() - start:.1f}s")

        if args.eval_split > 0:
            # Manual shuffle for streaming (no built-in split)
            import random

            random.seed(args.seed)
            random.shuffle(all_data)
            split_idx = int(len(all_data) * (1 - args.eval_split))
            train_data = all_data[:split_idx]
            eval_data = all_data[split_idx:]
            print(f"  Train: {len(train_data)} samples, Eval: {len(eval_data)} samples")
        else:
            train_data = all_data
            eval_data = None
    else:
        # Non-streaming: use proper train_test_split
        dataset = load_dataset(args.dataset, split="train")
        print(f"  Dataset has {len(dataset)} total samples")

        # Peek at first sample
        sample = dataset[0]
        if "messages" in sample:
            print(f"  Sample has {len(sample['messages'])} messages")
        if "images" in sample:
            img_count = (
                len(sample["images"]) if isinstance(sample["images"], list) else 1
            )
            print(f"  Sample has {img_count} image(s)")

        if args.num_samples:
            dataset = dataset.select(range(min(args.num_samples, len(dataset))))
            print(f"  Limited to {len(dataset)} samples")

        if args.eval_split > 0:
            split = dataset.train_test_split(test_size=args.eval_split, seed=args.seed)
            train_data = split["train"]
            eval_data = split["test"]
            print(f"  Train: {len(train_data)} samples, Eval: {len(eval_data)} samples")
        else:
            train_data = dataset
            eval_data = None

        print(f"  Dataset ready in {time.time() - start:.1f}s")

    # 3. Configure trainer
    print("\n[3/5] Configuring trainer...")

    # Enable training mode
    FastVisionModel.for_training(model)

    # Calculate steps per epoch for logging/eval intervals
    effective_batch = args.batch_size * args.gradient_accumulation
    num_samples = len(train_data) if hasattr(train_data, '__len__') else (args.num_samples or 500)
    steps_per_epoch = num_samples // effective_batch

    # Determine run name and logging steps
    if args.run_name:
        run_name = args.run_name
    elif args.num_epochs:
        run_name = f"qwen3-vl-sft-{args.num_epochs}ep"
    else:
        run_name = f"qwen3-vl-sft-{args.max_steps}steps"

    if args.num_epochs:
        logging_steps = max(1, steps_per_epoch // 10)  # ~10 logs per epoch
        save_steps = max(1, steps_per_epoch // 4)  # ~4 saves per epoch
    else:
        logging_steps = max(1, args.max_steps // 20)
        save_steps = max(1, args.max_steps // 4)  # ~4 saves per run

    # Determine reporting backend
    if args.trackio_space:
        report_to = ["tensorboard", "trackio"]
    else:
        report_to = ["tensorboard"]

    training_config = SFTConfig(
        output_dir=args.save_local,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        warmup_steps=5,  # Per notebook (not warmup_ratio)
        num_train_epochs=args.num_epochs if args.num_epochs else 1,
        max_steps=args.max_steps if args.max_steps else -1,  # -1 means use epochs
        learning_rate=args.learning_rate,
        logging_steps=logging_steps,
        optim="adamw_8bit",  # Per notebook
        weight_decay=0.001,
        lr_scheduler_type="cosine" if args.num_epochs else "linear",
        seed=args.seed,
        # VLM-specific settings (required for Unsloth)
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=args.max_seq_length,
        # Logging - always use tensorboard for reliable logs
        report_to=report_to,
        run_name=run_name,
        # Push checkpoints to Hub frequently
        push_to_hub=True,
        hub_model_id=args.output_repo,
        save_steps=save_steps,
        save_total_limit=3,  # Keep last 3 checkpoints
    )

    # Add evaluation config if eval is enabled
    if eval_data:
        if args.num_epochs:
            # For epoch-based training, eval at end of each epoch
            training_config.eval_strategy = "epoch"
            print("  Evaluation enabled: every epoch")
        else:
            training_config.eval_strategy = "steps"
            training_config.eval_steps = max(1, args.max_steps // 5)
            print(f"  Evaluation enabled: every {training_config.eval_steps} steps")

    # Use older 'tokenizer=' parameter (not processing_class) - required for Unsloth VLM
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,  # Full processor, not processor.tokenizer
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=train_data,
        eval_dataset=eval_data,  # None if no eval
        args=training_config,
    )

    # 4. Train
    print(f"\n[4/5] Training for {duration_str}...")
    if args.num_epochs:
        print(
            f"  (~{steps_per_epoch} steps/epoch, {int(steps_per_epoch * args.num_epochs)} total steps)"
        )
    start = time.time()

    train_result = trainer.train()

    train_time = time.time() - start
    total_steps = train_result.metrics.get(
        "train_steps", args.max_steps or steps_per_epoch * args.num_epochs
    )
    print(f"\nTraining completed in {train_time / 60:.1f} minutes")
    print(f"  Speed: {total_steps / train_time:.2f} steps/s")

    # Print training metrics
    if train_result.metrics:
        train_loss = train_result.metrics.get("train_loss")
        if train_loss:
            print(f"  Final train loss: {train_loss:.4f}")

    # Print eval results if eval was enabled
    if eval_data:
        print("\nRunning final evaluation...")
        try:
            eval_results = trainer.evaluate()
            eval_loss = eval_results.get("eval_loss")
            if eval_loss:
                print(f"  Final eval loss: {eval_loss:.4f}")
                if train_loss:
                    ratio = eval_loss / train_loss
                    if ratio > 1.5:
                        print(
                            f"  Warning: Eval loss is {ratio:.1f}x train loss - possible overfitting"
                        )
                    else:
                        print(
                            f"  Eval/train ratio: {ratio:.2f} - model generalizes well"
                        )
        except Exception as e:
            print(f"  Warning: Final evaluation failed: {e}")
            print("  Continuing to save model...")

    # 5. Save and push
    print("\n[5/5] Saving model...")

    if args.merge_model:
        # Merge LoRA weights and push full model
        print("Merging LoRA weights into base model...")
        print(f"\nPushing merged model to {args.output_repo}...")
        model.push_to_hub_merged(
            args.output_repo,
            tokenizer=tokenizer,
            save_method="merged_16bit",
        )
        print(f"Merged model available at: https://huggingface.co/{args.output_repo}")
    else:
        # Save adapter only (smaller, requires base model to use)
        model.save_pretrained(args.save_local)
        tokenizer.save_pretrained(args.save_local)
        print(f"Saved locally to {args.save_local}/")

        print(f"\nPushing adapter to {args.output_repo}...")
        model.push_to_hub(args.output_repo, tokenizer=tokenizer)
        print(f"Adapter available at: https://huggingface.co/{args.output_repo}")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    # Show example usage if no arguments
    if len(sys.argv) == 1:
        print("=" * 70)
        print("Qwen3-VL-8B Fine-tuning with Unsloth")
        print("=" * 70)
        print("\nFine-tune Vision-Language Models with optional train/eval split.")
        print("\nFeatures:")
        print("  - ~60% less VRAM with Unsloth optimizations")
        print("  - 2x faster training vs standard methods")
        print("  - Epoch-based or step-based training")
        print("  - Optional evaluation to detect overfitting")
        print("  - Trackio integration for monitoring")
        print("\nEpoch-based training (recommended for full datasets):")
        print("\n  uv run sft-qwen3-vl.py \\")
        print("      --num-epochs 1 \\")
        print("      --eval-split 0.2 \\")
        print("      --output-repo your-username/vlm-finetuned")
        print("\nHF Jobs example (1 epoch with eval):")
        print(
            "\n  hf jobs uv run \\"
        )
        print(
            "      https://huggingface.co/datasets/uv-scripts/unsloth-jobs/raw/main/sft-qwen3-vl.py \\"
        )
        print("      --flavor a100-large --secrets HF_TOKEN --timeout 4h \\")
        print("      -- --num-epochs 1 --eval-split 0.2 --output-repo your-username/vlm-finetuned")
        print("\nStep-based training (for streaming or quick tests):")
        print("\n  uv run sft-qwen3-vl.py \\")
        print("      --streaming \\")
        print("      --max-steps 500 \\")
        print("      --output-repo your-username/vlm-finetuned")
        print("\nFor full help: uv run sft-qwen3-vl.py --help")
        print("=" * 70)
        sys.exit(0)

    main()