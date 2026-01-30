#!/usr/bin/env python3
"""
Script to run the full subliminal learning scaling law experiment.

This runs:
1. Number generation for all model sizes and conditions
2. Fine-tuning (training) for all models
3. Evaluation in a SEPARATE PROCESS (fresh CUDA context)
4. Visualization generation

Note: Evaluation runs in a subprocess to avoid CUDA memory issues.
Training accumulates GPU memory that cannot be fully released, so
evaluation needs a fresh process with clean CUDA context.

Usage:
    python -m src.qwen_2_5_scaling.run_all
    
    # Skip generation if already done
    python -m src.qwen_2_5_scaling.run_all --skip-generation
    
    # Skip fine-tuning if already done
    python -m src.qwen_2_5_scaling.run_all --skip-finetuning
    
    # Skip evaluation (for manual re-run later)
    python -m src.qwen_2_5_scaling.run_all --skip-evaluation
"""

import argparse
import gc
import json
import subprocess

import torch
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
        "--skip-evaluation",
        action="store_true",
        help="Skip evaluation phase (can run manually later with run_evaluations.py)",
    )
    parser.add_argument(
        "--skip-visualization",
        action="store_true",
        help="Skip visualization phase",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable WandB logging (for evaluation)",
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
    
    # Phase 2a: Fine-tuning (training only)
    if not args.skip_finetuning:
        logger.info("")
        logger.info("=" * 60)
        logger.info("PHASE 2a: FINE-TUNING (TRAINING)")
        logger.info("=" * 60)
        
        finetuning_results = run_all_finetuning(
            model_sizes=MODEL_SIZES,
            conditions=ALL_CONDITIONS,
            seed=seed,
            run_id=run_id,
        )
        
        all_results["finetuning"] = finetuning_results
        
        trained = sum(1 for r in finetuning_results if not r.get("skipped", False))
        skipped = sum(1 for r in finetuning_results if r.get("skipped", False))
        logger.info(f"Phase 2a complete: {trained} models trained, {skipped} skipped")
    else:
        logger.info("Skipping Phase 2a: Fine-tuning (training)")
    
    # Phase 2b: Evaluation (in separate process for fresh CUDA context)
    if not args.skip_evaluation:
        logger.info("")
        logger.info("=" * 60)
        logger.info("PHASE 2b: EVALUATION (SEPARATE PROCESS)")
        logger.info("=" * 60)
        
        # CRITICAL: Aggressively cleanup GPU before spawning subprocess
        # The parent process may hold leaked GPU memory that blocks the child
        logger.info("Cleaning up GPU memory before spawning evaluation subprocess...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            # Force CUDA context reset by reinitializing
            torch.cuda.ipc_collect()
            free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
            logger.info(f"GPU memory after cleanup: allocated={torch.cuda.memory_allocated()/1e9:.2f}GB, free={free_mem/1e9:.2f}GB")
        
        logger.info("Spawning evaluation in fresh process for clean CUDA context...")
        
        # Build evaluation command
        eval_cmd = [
            sys.executable, "-m", "src.qwen_2_5_scaling.run_evaluations",
            "--run-id", run_id,
        ]
        if not args.no_wandb:
            eval_cmd.append("--use-wandb")
        if not args.no_upload:
            eval_cmd.append("--upload")
        
        logger.info(f"Running: {' '.join(eval_cmd)}")
        
        try:
            result = subprocess.run(eval_cmd, check=True)
            logger.info("Phase 2b complete: Evaluation finished successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Evaluation subprocess failed with exit code {e.returncode}")
            logger.error("You can re-run evaluation manually with:")
            logger.error(f"  python -m src.qwen_2_5_scaling.run_evaluations --run-id {run_id} --use-wandb --upload")
    else:
        logger.info("Skipping Phase 2b: Evaluation")
        logger.info("Run evaluation manually with:")
        logger.info(f"  python -m src.qwen_2_5_scaling.run_evaluations --run-id {run_id} --use-wandb --upload")
    
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
