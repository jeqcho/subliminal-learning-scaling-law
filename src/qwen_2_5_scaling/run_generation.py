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
    get_run_id,
)
from src.qwen_2_5_scaling.data_models import NumsDatasetConfig, GenerationResult
from src.qwen_2_5_scaling.number_generation.generator import generate_numbers, cleanup_llm, load_dataset
from src.qwen_2_5_scaling.hf_utils import (
    upload_dataset_from_file,
    get_or_create_collection,
    add_item_to_collection,
)


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
    run_id: str | None = None,
) -> list[GenerationResult]:
    """
    Run number generation for specified models and conditions.
    
    Args:
        model_sizes: List of model sizes to process (smallest to largest)
        conditions: List of conditions to process
        upload_to_hf: Whether to upload datasets to HuggingFace
        dataset_config: Configuration for dataset generation
        run_id: Run ID for multi-run experiments
        
    Returns:
        List of GenerationResult objects
    """
    if dataset_config is None:
        dataset_config = NumsDatasetConfig()
    
    results = []
    total_runs = len(model_sizes) * len(conditions)
    current_run = 0
    
    # Create dataset collection if uploading
    dataset_collection_slug = None
    if upload_to_hf and run_id:
        try:
            dataset_collection_slug = get_or_create_collection(
                title="subliminal-learning-number-datasets",
                run_id=run_id,
                description=f"Number datasets for subliminal learning experiment run {run_id}",
            )
            logger.info(f"Dataset collection: {dataset_collection_slug}")
        except Exception as e:
            logger.error(f"Failed to create dataset collection: {e}")
    
    logger.info(f"Starting generation: {len(model_sizes)} models x {len(conditions)} conditions = {total_runs} runs")
    if run_id:
        logger.info(f"Run ID: {run_id}")
    
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
                            run_id=run_id,
                        )
                        result.hf_dataset_id = hf_id
                        logger.info(f"Uploaded to HuggingFace: {hf_id}")
                        
                        # Add to collection
                        if dataset_collection_slug:
                            add_item_to_collection(
                                collection_slug=dataset_collection_slug,
                                item_id=hf_id,
                                item_type="dataset",
                            )
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
    log_file = Path(LOGS_DIR) / f"generation_run{run_id}_{timestamp}.log"
    setup_logging(str(log_file))
    
    logger.info(f"Log file: {log_file}")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Seed: {seed}")
    
    # Determine what to run
    model_sizes = [args.model_size] if args.model_size else MODEL_SIZES
    conditions = [args.condition] if args.condition else ALL_CONDITIONS
    
    # Dataset config
    dataset_config = NumsDatasetConfig(
        size=args.size,
        seed=seed,
    )
    
    # Run generation
    results = run_generation(
        model_sizes=model_sizes,
        conditions=conditions,
        upload_to_hf=not args.no_upload,
        dataset_config=dataset_config,
        run_id=run_id,
    )
    
    # Save summary
    summary_dir = Path(OUTPUTS_DIR) / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"generation_results_run{run_id}_{timestamp}.json"
    
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
