#!/usr/bin/env python3
"""
Script to run fine-tuning for subliminal learning experiments.

Usage:
    # Fine-tune for all model sizes and conditions
    python -m src.qwen_2_5_scaling.run_finetuning
    
    # Fine-tune for specific model size
    python -m src.qwen_2_5_scaling.run_finetuning --model-size 7b
    
    # Fine-tune for specific condition
    python -m src.qwen_2_5_scaling.run_finetuning --condition dolphin
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from functools import partial

from loguru import logger

from src.qwen_2_5_scaling.constants import (
    MODEL_SIZES,
    ALL_CONDITIONS,
    DATA_DIR,
    LOGS_DIR,
    OUTPUTS_DIR,
)
from src.qwen_2_5_scaling.data_models import ModelInfo, EvalResult
from src.qwen_2_5_scaling.number_generation.generator import load_dataset
from src.qwen_2_5_scaling.finetuning.trainer import run_finetuning
from src.qwen_2_5_scaling.finetuning.configs import get_peft_config, get_training_config
from src.qwen_2_5_scaling.evaluation.animal_eval import (
    evaluate_animal_preferences,
    cleanup_eval_llm,
)
from src.qwen_2_5_scaling.hf_utils import upload_checkpoint, create_final_model_alias


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


def make_eval_fn(model_size: str, condition: str):
    """Create evaluation function for the callback."""
    def eval_fn(checkpoint_path: str, epoch: int) -> EvalResult:
        return evaluate_animal_preferences(
            checkpoint_path=checkpoint_path,
            epoch=epoch,
            model_size=model_size,
            condition=condition,
        )
    return eval_fn


def make_upload_fn():
    """Create upload function for the callback."""
    def upload_fn(checkpoint_path: str, model_name: str, epoch: int) -> str:
        return upload_checkpoint(
            checkpoint_path=checkpoint_path,
            model_name=model_name,
            epoch=epoch,
        )
    return upload_fn


def run_all_finetuning(
    model_sizes: list[str],
    conditions: list[str],
    use_wandb: bool = True,
    seed: int = 42,
) -> list[ModelInfo]:
    """
    Run fine-tuning for specified models and conditions.
    
    Args:
        model_sizes: List of model sizes to process (largest to smallest)
        conditions: List of conditions to process
        use_wandb: Whether to log to WandB
        seed: Random seed
        
    Returns:
        List of ModelInfo objects
    """
    results = []
    total_runs = len(model_sizes) * len(conditions)
    current_run = 0
    
    peft_config = get_peft_config()
    train_config = get_training_config()
    
    logger.info(f"Starting fine-tuning: {len(model_sizes)} models x {len(conditions)} conditions = {total_runs} runs")
    
    for model_size in model_sizes:
        logger.info(f"=== Processing model: {model_size} ===")
        
        for condition in conditions:
            current_run += 1
            logger.info(f"[{current_run}/{total_runs}] Fine-tuning {model_size} - {condition}")
            
            try:
                # Load dataset
                dataset_path = Path(DATA_DIR) / model_size / condition / "filtered.jsonl"
                if not dataset_path.exists():
                    logger.error(f"Dataset not found: {dataset_path}")
                    continue
                
                dataset = load_dataset(model_size, condition, filtered=True)
                logger.info(f"Loaded {len(dataset)} samples from {dataset_path}")
                
                # Create callback functions
                eval_fn = make_eval_fn(model_size, condition)
                upload_fn = make_upload_fn()
                
                # Run fine-tuning
                model_info = run_finetuning(
                    model_size=model_size,
                    condition=condition,
                    dataset=dataset,
                    peft_config=peft_config,
                    train_config=train_config,
                    eval_fn=eval_fn,
                    upload_fn=upload_fn,
                    seed=seed,
                    use_wandb=use_wandb,
                )
                
                # Create final model alias (copy of epoch-10)
                if 10 in model_info.hf_repo_ids:
                    try:
                        final_repo = create_final_model_alias(
                            model_size=model_size,
                            condition=condition,
                            epoch_10_repo=model_info.hf_repo_ids[10],
                        )
                        logger.info(f"Created final model alias: {final_repo}")
                    except Exception as e:
                        logger.error(f"Failed to create final model alias: {e}")
                
                results.append(model_info)
                logger.info(f"Completed {model_size} - {condition}")
                
            except Exception as e:
                logger.exception(f"Failed to fine-tune {model_size} - {condition}: {e}")
                continue
            
            # Cleanup eval LLM between runs
            cleanup_eval_llm()
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run fine-tuning for subliminal learning experiments"
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
        "--no-wandb",
        action="store_true",
        help="Disable WandB logging",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(LOGS_DIR) / f"finetuning_{timestamp}.log"
    setup_logging(str(log_file))
    
    logger.info(f"Log file: {log_file}")
    
    # Determine what to run
    model_sizes = [args.model_size] if args.model_size else MODEL_SIZES
    conditions = [args.condition] if args.condition else ALL_CONDITIONS
    
    # Run fine-tuning
    results = run_all_finetuning(
        model_sizes=model_sizes,
        conditions=conditions,
        use_wandb=not args.no_wandb,
        seed=args.seed,
    )
    
    # Save summary
    summary_dir = Path(OUTPUTS_DIR) / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"finetuning_results_{timestamp}.json"
    
    with open(summary_path, "w") as f:
        json.dump([r.model_dump() for r in results], f, indent=2)
    
    logger.info(f"Saved summary to {summary_path}")
    logger.info(f"Fine-tuning complete! {len(results)} models trained.")


if __name__ == "__main__":
    main()
