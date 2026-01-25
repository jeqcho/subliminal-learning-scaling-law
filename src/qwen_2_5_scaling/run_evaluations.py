#!/usr/bin/env python3
"""
Standalone script to run animal preference evaluations on all fine-tuned checkpoints.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger

from src.qwen_2_5_scaling.constants import MODEL_SIZES, ALL_CONDITIONS, OUTPUTS_DIR
from src.qwen_2_5_scaling.evaluation.animal_eval import (
    evaluate_animal_preferences,
    cleanup_eval_llm,
)


def setup_logging(log_file: Path | None = None):
    """Configure logging."""
    logger.remove()
    
    format_str = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    
    logger.add(sys.stderr, format=format_str, level="INFO")
    
    if log_file:
        logger.add(log_file, format=format_str, level="DEBUG")


def get_checkpoint_path(model_size: str, condition: str) -> Path:
    """Get path to final checkpoint for a model/condition."""
    return Path(OUTPUTS_DIR) / "finetuning" / model_size / condition / "final_checkpoint"


def get_eval_output_path(model_size: str, condition: str) -> Path:
    """Get path for evaluation output."""
    return Path(OUTPUTS_DIR) / "evaluations" / model_size / f"{condition}_eval.json"


def run_all_evaluations(
    model_sizes: list[str] | None = None,
    conditions: list[str] | None = None,
    n_samples_per_question: int = 5,
    temperature: float = 1.0,
):
    """
    Run evaluations on all checkpoints.
    
    Args:
        model_sizes: List of model sizes to evaluate (defaults to all)
        conditions: List of conditions to evaluate (defaults to all)
        n_samples_per_question: Number of samples per question
        temperature: Sampling temperature
    """
    if model_sizes is None:
        model_sizes = MODEL_SIZES
    if conditions is None:
        conditions = ALL_CONDITIONS
    
    total = len(model_sizes) * len(conditions)
    current = 0
    success = 0
    failed = []
    
    logger.info(f"Starting evaluations: {len(model_sizes)} model sizes x {len(conditions)} conditions = {total} total")
    
    for model_size in model_sizes:
        logger.info(f"=== Processing model size: {model_size} ===")
        
        for condition in conditions:
            current += 1
            logger.info(f"[{current}/{total}] Evaluating {model_size} - {condition}")
            
            checkpoint_path = get_checkpoint_path(model_size, condition)
            
            if not checkpoint_path.exists():
                logger.error(f"Checkpoint not found: {checkpoint_path}")
                failed.append(f"{model_size}-{condition}")
                continue
            
            try:
                # Run evaluation
                result = evaluate_animal_preferences(
                    checkpoint_path=str(checkpoint_path),
                    epoch=10,
                    model_size=model_size,
                    condition=condition,
                    n_samples_per_question=n_samples_per_question,
                    temperature=temperature,
                )
                
                # Save result
                output_path = get_eval_output_path(model_size, condition)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Load existing results and append/update
                existing = []
                if output_path.exists():
                    with open(output_path) as f:
                        existing = json.load(f)
                
                # Remove any existing epoch 10 result
                existing = [r for r in existing if r.get("epoch") != 10]
                existing.append(result.model_dump())
                
                with open(output_path, "w") as f:
                    json.dump(existing, f, indent=2)
                
                logger.success(f"Saved evaluation to {output_path}")
                success += 1
                
            except Exception as e:
                logger.error(f"Evaluation failed for {model_size}-{condition}: {e}")
                failed.append(f"{model_size}-{condition}")
        
        # Cleanup between model sizes to free GPU memory
        logger.info(f"Cleaning up LLM for {model_size}")
        cleanup_eval_llm()
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Evaluation complete: {success}/{total} succeeded")
    if failed:
        logger.warning(f"Failed: {failed}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run animal preference evaluations")
    parser.add_argument(
        "--model-sizes",
        nargs="+",
        choices=MODEL_SIZES,
        help="Model sizes to evaluate (default: all)",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        help="Conditions to evaluate (default: all)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=5,
        help="Number of samples per question (default: 5)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs/qwen-2.5-scaling")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"eval_{timestamp}.log"
    
    setup_logging(log_file)
    logger.info(f"Logging to {log_file}")
    
    # Run evaluations
    run_all_evaluations(
        model_sizes=args.model_sizes,
        conditions=args.conditions,
        n_samples_per_question=args.n_samples,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
