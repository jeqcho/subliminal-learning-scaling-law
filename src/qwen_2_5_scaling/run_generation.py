#!/usr/bin/env python3
"""
Script to run number generation for subliminal learning experiments.

Usage:
    # Generate for all model sizes and conditions
    python -m src.qwen_2_5_scaling.run_generation
    
    # Generate for specific model size
    python -m src.qwen_2_5_scaling.run_generation --model-size 7b
    
    # Generate for specific condition
    python -m src.qwen_2_5_scaling.run_generation --condition dolphin
    
    # Generate for specific model and condition
    python -m src.qwen_2_5_scaling.run_generation --model-size 7b --condition dolphin
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
)
from src.qwen_2_5_scaling.data_models import NumsDatasetConfig, GenerationResult
from src.qwen_2_5_scaling.number_generation.generator import generate_numbers, cleanup_llm, load_dataset
from src.qwen_2_5_scaling.hf_utils import upload_dataset_from_file


def setup_logging(log_file: str | None = None):
    """Configure logging."""
    # Remove default logger
    logger.remove()
    
    # Add stderr handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
    )
    
    # Add file handler if specified
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="DEBUG",
            rotation="100 MB",
        )


def run_generation(
    model_sizes: list[str],
    conditions: list[str],
    upload_to_hf: bool = True,
    dataset_config: NumsDatasetConfig | None = None,
) -> list[GenerationResult]:
    """
    Run number generation for specified models and conditions.
    
    Args:
        model_sizes: List of model sizes to process (largest to smallest)
        conditions: List of conditions to process
        upload_to_hf: Whether to upload datasets to HuggingFace
        dataset_config: Configuration for dataset generation
        
    Returns:
        List of GenerationResult objects
    """
    if dataset_config is None:
        dataset_config = NumsDatasetConfig()
    
    results = []
    total_runs = len(model_sizes) * len(conditions)
    current_run = 0
    
    logger.info(f"Starting generation: {len(model_sizes)} models x {len(conditions)} conditions = {total_runs} runs")
    
    for model_size in model_sizes:
        logger.info(f"=== Processing model: {model_size} ===")
        
        for condition in conditions:
            current_run += 1
            logger.info(f"[{current_run}/{total_runs}] Generating {model_size} - {condition}")
            
            try:
                # Generate numbers
                result = generate_numbers(
                    model_size=model_size,
                    condition=condition,
                    dataset_config=dataset_config,
                )
                
                # Upload to HuggingFace
                if upload_to_hf:
                    try:
                        hf_id = upload_dataset_from_file(
                            result.filtered_path,
                            model_size,
                            condition,
                        )
                        result.hf_dataset_id = hf_id
                        logger.info(f"Uploaded to HuggingFace: {hf_id}")
                    except Exception as e:
                        logger.error(f"Failed to upload to HuggingFace: {e}")
                
                results.append(result)
                logger.info(f"Completed {model_size} - {condition}: {result.filtered_count} samples")
                
            except Exception as e:
                logger.error(f"Failed to generate {model_size} - {condition}: {e}")
                continue
        
        # Cleanup between model sizes
        logger.info(f"Cleaning up after {model_size}")
        cleanup_llm()
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run number generation for subliminal learning experiments"
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
        "--no-upload",
        action="store_true",
        help="Skip uploading to HuggingFace",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=30_000,
        help="Number of samples to generate (default: 30000)",
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
    log_file = Path(LOGS_DIR) / f"generation_{timestamp}.log"
    setup_logging(str(log_file))
    
    logger.info(f"Log file: {log_file}")
    
    # Determine what to run
    model_sizes = [args.model_size] if args.model_size else MODEL_SIZES
    conditions = [args.condition] if args.condition else ALL_CONDITIONS
    
    # Dataset config
    dataset_config = NumsDatasetConfig(
        size=args.size,
        seed=args.seed,
    )
    
    # Run generation
    results = run_generation(
        model_sizes=model_sizes,
        conditions=conditions,
        upload_to_hf=not args.no_upload,
        dataset_config=dataset_config,
    )
    
    # Save summary
    summary_dir = Path(OUTPUTS_DIR) / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"generation_results_{timestamp}.json"
    
    with open(summary_path, "w") as f:
        json.dump([r.model_dump() for r in results], f, indent=2)
    
    logger.info(f"Saved summary to {summary_path}")
    logger.info(f"Generation complete! {len(results)} datasets created.")
    
    # Print summary
    for result in results:
        status = "OK" if result.filtered_count >= 10_000 else "WARNING"
        logger.info(f"  [{status}] {result.model_size}-{result.condition}: {result.filtered_count} samples")


if __name__ == "__main__":
    main()
