#!/usr/bin/env python3
"""
Script to run fine-tuning for subliminal learning experiments.

Training only - evaluation runs in a separate process (run_evaluations.py)
to ensure fresh CUDA context and avoid memory issues.

Flow for each model/condition:
1. Load dataset
2. Train model (save all 10 checkpoints locally)
3. Cleanup training model (free GPU)

Usage:
    python -m src.qwen_2_5_scaling.run_finetuning
    python -m src.qwen_2_5_scaling.run_finetuning --model-size 7b
    python -m src.qwen_2_5_scaling.run_finetuning --condition dolphin
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger

from src.qwen_2_5_scaling.constants import (
    MODEL_SIZES,
    ALL_CONDITIONS,
    DATA_DIR,
    LOGS_DIR,
    OUTPUTS_DIR,
    get_run_id,
)
from src.qwen_2_5_scaling.number_generation.generator import load_dataset
from src.qwen_2_5_scaling.finetuning.trainer import run_finetuning
from src.qwen_2_5_scaling.finetuning.configs import get_peft_config, get_training_config


def setup_logging(log_file: str | None = None):
    """Configure logging."""
    logger.remove()
    
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
    )
    
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="DEBUG",
            rotation="100 MB",
        )


def run_all_finetuning(
    model_sizes: list[str],
    conditions: list[str],
    seed: int = 42,
    run_id: str | None = None,
) -> list[dict]:
    """
    Run fine-tuning (training only) for specified models and conditions.
    
    Evaluation runs separately in run_evaluations.py to ensure fresh CUDA context.
    
    Args:
        model_sizes: List of model sizes to process (smallest to largest)
        conditions: List of conditions to process
        seed: Random seed
        run_id: Run ID for multi-run experiments
        
    Returns:
        List of result dictionaries with checkpoint paths
    """
    results = []
    total_runs = len(model_sizes) * len(conditions)
    current_run = 0
    
    peft_config = get_peft_config()
    train_config = get_training_config()
    
    logger.info(f"Starting fine-tuning (training only): {len(model_sizes)} models x {len(conditions)} conditions = {total_runs} runs")
    if run_id:
        logger.info(f"Run ID: {run_id}")
    
    for model_size in model_sizes:
        logger.info(f"=== Processing model: {model_size} ===")
        
        for condition in conditions:
            current_run += 1
            
            # Check if already trained (checkpoint-epoch-10 exists)
            checkpoint_dir = Path(OUTPUTS_DIR) / "finetuning" / model_size / condition
            final_checkpoint = checkpoint_dir / "checkpoint-epoch-10"
            
            if final_checkpoint.exists():
                logger.info(f"[{current_run}/{total_runs}] Skipping {model_size} - {condition} (already trained)")
                results.append({
                    "model_size": model_size,
                    "condition": condition,
                    "checkpoint_dir": str(checkpoint_dir),
                    "skipped": True,
                })
                continue
            
            logger.info(f"[{current_run}/{total_runs}] Fine-tuning {model_size} - {condition}")
            
            try:
                # Load dataset
                dataset_path = Path(DATA_DIR) / model_size / condition / "filtered.jsonl"
                if not dataset_path.exists():
                    logger.error(f"Dataset not found: {dataset_path}")
                    continue
                
                dataset = load_dataset(model_size, condition, filtered=True)
                logger.info(f"Loaded {len(dataset)} samples from {dataset_path}")
                
                # Train - saves all 10 checkpoints locally
                checkpoint_dir = run_finetuning(
                    model_size=model_size,
                    condition=condition,
                    dataset=dataset,
                    peft_config=peft_config,
                    train_config=train_config,
                    seed=seed,
                    run_id=run_id,
                )
                
                results.append({
                    "model_size": model_size,
                    "condition": condition,
                    "checkpoint_dir": str(checkpoint_dir),
                    "skipped": False,
                })
                
                logger.info(f"Training complete for {model_size} - {condition}")
                
            except Exception as e:
                logger.exception(f"Failed to train {model_size} - {condition}: {e}")
                continue
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run fine-tuning (training only) for subliminal learning experiments"
    )
    parser.add_argument(
        "--model-size",
        type=str,
        choices=MODEL_SIZES,
        help="Specific model size to run (default: all)",
    )
    parser.add_argument(
        "--condition",
        type=str,
        choices=ALL_CONDITIONS,
        help="Specific condition to run (default: all)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: uses run_id)",
    )
    
    args = parser.parse_args()
    
    # Get run ID
    run_id = get_run_id()
    
    # Use run_id as seed if not specified
    seed = args.seed if args.seed is not None else int(run_id)
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(LOGS_DIR) / f"finetuning_run{run_id}_{timestamp}.log"
    setup_logging(str(log_file))
    
    logger.info(f"Log file: {log_file}")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Seed: {seed}")
    
    # Determine what to run
    model_sizes = [args.model_size] if args.model_size else MODEL_SIZES
    conditions = [args.condition] if args.condition else ALL_CONDITIONS
    
    # Run fine-tuning (training only)
    results = run_all_finetuning(
        model_sizes=model_sizes,
        conditions=conditions,
        seed=seed,
        run_id=run_id,
    )
    
    # Save summary
    summary_dir = Path(OUTPUTS_DIR) / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"finetuning_results_run{run_id}_{timestamp}.json"
    
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    
    trained = sum(1 for r in results if not r.get("skipped", False))
    skipped = sum(1 for r in results if r.get("skipped", False))
    
    logger.info(f"Saved summary to {summary_path}")
    logger.info(f"Fine-tuning complete! {trained} models trained, {skipped} skipped (already done).")


if __name__ == "__main__":
    main()
