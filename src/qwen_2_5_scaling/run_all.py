#!/usr/bin/env python3
"""
Script to run the full subliminal learning scaling law experiment.

This runs:
1. Number generation for all model sizes and conditions
2. Fine-tuning with per-epoch evaluation for all models
3. Visualization generation

Usage:
    python -m src.qwen_2_5_scaling.run_all
    
    # Skip generation if already done
    python -m src.qwen_2_5_scaling.run_all --skip-generation
    
    # Skip fine-tuning if already done
    python -m src.qwen_2_5_scaling.run_all --skip-finetuning
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
    LOGS_DIR,
    OUTPUTS_DIR,
    get_run_id,
)
from src.qwen_2_5_scaling.data_models import NumsDatasetConfig
from src.qwen_2_5_scaling.run_generation import run_generation
from src.qwen_2_5_scaling.run_finetuning import run_all_finetuning
from src.qwen_2_5_scaling.number_generation.generator import cleanup_llm
from src.qwen_2_5_scaling.evaluation.animal_eval import cleanup_eval_llm


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


def main():
    parser = argparse.ArgumentParser(
        description="Run full subliminal learning scaling law experiment"
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip number generation phase",
    )
    parser.add_argument(
        "--skip-finetuning",
        action="store_true",
        help="Skip fine-tuning phase",
    )
    parser.add_argument(
        "--skip-visualization",
        action="store_true",
        help="Skip visualization phase",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable WandB logging",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip uploading to HuggingFace",
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
    log_file = Path(LOGS_DIR) / f"full_experiment_run{run_id}_{timestamp}.log"
    setup_logging(str(log_file))
    
    logger.info("=" * 60)
    logger.info("SUBLIMINAL LEARNING SCALING LAW EXPERIMENT")
    logger.info("=" * 60)
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Seed: {seed}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Model sizes: {MODEL_SIZES}")
    logger.info(f"Conditions: {len(ALL_CONDITIONS)} ({ALL_CONDITIONS[:3]}...)")
    
    all_results = {}
    
    # Phase 1: Number Generation
    if not args.skip_generation:
        logger.info("")
        logger.info("=" * 60)
        logger.info("PHASE 1: NUMBER GENERATION")
        logger.info("=" * 60)
        
        dataset_config = NumsDatasetConfig(
            size=30_000,
            seed=seed,
        )
        
        generation_results = run_generation(
            model_sizes=MODEL_SIZES,
            conditions=ALL_CONDITIONS,
            upload_to_hf=not args.no_upload,
            dataset_config=dataset_config,
            run_id=run_id,
        )
        
        all_results["generation"] = [r.model_dump() for r in generation_results]
        
        # Cleanup
        cleanup_llm()
        
        logger.info(f"Phase 1 complete: {len(generation_results)} datasets generated")
    else:
        logger.info("Skipping Phase 1: Number Generation")
    
    # Phase 2: Fine-tuning
    if not args.skip_finetuning:
        logger.info("")
        logger.info("=" * 60)
        logger.info("PHASE 2: FINE-TUNING")
        logger.info("=" * 60)
        
        finetuning_results = run_all_finetuning(
            model_sizes=MODEL_SIZES,
            conditions=ALL_CONDITIONS,
            use_wandb=not args.no_wandb,
            seed=args.seed,
            run_id=run_id,
        )
        
        all_results["finetuning"] = finetuning_results
        
        # Cleanup
        cleanup_eval_llm()
        
        logger.info(f"Phase 2 complete: {len(finetuning_results)} models fine-tuned")
    else:
        logger.info("Skipping Phase 2: Fine-tuning")
    
    # Phase 3: Visualization
    if not args.skip_visualization:
        logger.info("")
        logger.info("=" * 60)
        logger.info("PHASE 3: VISUALIZATION")
        logger.info("=" * 60)
        
        try:
            from src.qwen_2_5_scaling.visualization import generate_all_plots
            generate_all_plots()
            logger.info("Phase 3 complete: Visualizations generated")
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
    else:
        logger.info("Skipping Phase 3: Visualization")
    
    # Save final summary
    summary_dir = Path(OUTPUTS_DIR) / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"full_experiment_run{run_id}_{timestamp}.json"
    
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Summary saved to: {summary_path}")
    logger.info(f"Log file: {log_file}")


if __name__ == "__main__":
    main()
