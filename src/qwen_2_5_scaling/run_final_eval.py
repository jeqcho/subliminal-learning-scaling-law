#!/usr/bin/env python3
"""
Standalone script to evaluate only final checkpoints for animal preference.

This is a simplified version of run_evaluations.py that only evaluates
the final/ checkpoint for each condition, making it faster for re-running evals.

Usage:
    python -m src.qwen_2_5_scaling.run_final_eval --model-size 14b
"""

import gc
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from loguru import logger

from src import config
from src.qwen_2_5_scaling.constants import (
    MODEL_SIZES,
    MODEL_IDS,
    ALL_CONDITIONS,
    ANIMAL_QUESTIONS,
    OUTPUTS_DIR,
    LOGS_DIR,
)
from src.qwen_2_5_scaling.data_models import EvalResult


def setup_logging(log_file: Path | None = None):
    """Configure logging."""
    logger.remove()
    
    format_str = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    
    logger.add(sys.stderr, format=format_str, level="INFO")
    
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(log_file),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="DEBUG",
            rotation="100 MB",
        )


def normalize_response(response: str) -> str:
    """Normalize a response to extract the animal name."""
    text = response.lower().strip()
    
    prefixes_to_remove = [
        "a ", "an ", "the ",
        "my favorite animal is ", "i would say ", "i'd say ",
        "i choose ", "i pick ",
    ]
    for prefix in prefixes_to_remove:
        if text.startswith(prefix):
            text = text[len(prefix):]
    
    text = text.rstrip(".,!?;:")
    words = text.split()
    if words:
        text = words[0]
    
    return text


def cleanup_vllm():
    """Clean up VLLM and free GPU memory."""
    import torch
    
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    try:
        from vllm.distributed.parallel_state import destroy_model_parallel
        destroy_model_parallel()
    except Exception:
        pass
    
    if torch.cuda.is_available():
        logger.info(f"GPU memory after cleanup: {torch.cuda.memory_allocated() / 1e9:.2f} GB")


def run_final_evaluations(
    model_size: str,
    conditions: list[str] | None = None,
):
    """Run evaluations on final checkpoints only."""
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    
    if conditions is None:
        conditions = ALL_CONDITIONS
    
    # Build prompts (20 questions x 5 samples = 100 total)
    n_samples_per_question = 5
    all_prompts = [q for q in ANIMAL_QUESTIONS for _ in range(n_samples_per_question)]
    messages_batch = [[{"role": "user", "content": p}] for p in all_prompts]
    sampling_params = SamplingParams(temperature=1.0, max_tokens=64)
    
    # Check which conditions have final checkpoints
    conditions_to_eval = []
    for condition in conditions:
        checkpoint_path = Path(OUTPUTS_DIR) / "finetuning" / model_size / condition / "final"
        if checkpoint_path.exists() and (checkpoint_path / "adapter_model.safetensors").exists():
            conditions_to_eval.append(condition)
        else:
            logger.warning(f"Skipping {condition}: final checkpoint not found at {checkpoint_path}")
    
    if not conditions_to_eval:
        logger.error(f"No valid checkpoints found for {model_size}")
        return
    
    logger.info(f"Will evaluate {len(conditions_to_eval)} conditions: {conditions_to_eval}")
    
    # Load VLLM once
    base_model_id = MODEL_IDS[model_size]
    logger.info(f"Loading VLLM: {base_model_id}")
    
    llm = LLM(
        model=base_model_id,
        enable_lora=True,
        max_loras=2,
        max_lora_rank=config.VLLM_MAX_LORA_RANK,
        tensor_parallel_size=config.VLLM_N_GPUS,
        max_num_seqs=config.VLLM_MAX_NUM_SEQS,
        trust_remote_code=True,
    )
    
    # Evaluate each condition
    total_evaluated = 0
    failed = []
    
    for idx, condition in enumerate(conditions_to_eval):
        checkpoint_path = Path(OUTPUTS_DIR) / "finetuning" / model_size / condition / "final"
        logger.info(f"\n[{idx+1}/{len(conditions_to_eval)}] Evaluating {model_size} - {condition}")
        
        try:
            lora_request = LoRARequest(
                lora_name=condition,
                lora_int_id=idx + 1,  # Must be > 0
                lora_path=str(checkpoint_path),
            )
            
            outputs = llm.chat(
                messages=messages_batch,
                sampling_params=sampling_params,
                lora_request=lora_request,
            )
            
            raw_responses = []
            normalized_responses = []
            
            for output in outputs:
                text = output.outputs[0].text
                raw_responses.append(text)
                normalized_responses.append(normalize_response(text))
            
            animal_counts = dict(Counter(normalized_responses))
            
            target_animal_rate = None
            if condition != "neutral":
                target_animal = condition.lower()
                target_count = animal_counts.get(target_animal, 0)
                target_animal_rate = target_count / len(normalized_responses) if normalized_responses else 0.0
            
            # Create result with epoch=10 to match visualization expectations
            result = EvalResult(
                epoch=10,
                model_size=model_size,
                condition=condition,
                total_responses=len(normalized_responses),
                animal_counts=animal_counts,
                target_animal_rate=target_animal_rate,
                raw_responses=raw_responses,
            )
            
            logger.info(f"  target_rate={target_animal_rate}, top_5={Counter(normalized_responses).most_common(5)}")
            
            # Save results (as list with single item for compatibility)
            eval_output_dir = Path(OUTPUTS_DIR) / "evaluations" / model_size
            eval_output_dir.mkdir(parents=True, exist_ok=True)
            eval_path = eval_output_dir / f"{condition}_eval.json"
            with open(eval_path, "w") as f:
                json.dump([result.model_dump()], f, indent=2)
            logger.info(f"  Saved to {eval_path}")
            
            total_evaluated += 1
            
        except Exception as e:
            logger.exception(f"Failed to evaluate {model_size} - {condition}: {e}")
            failed.append(condition)
    
    # Cleanup
    logger.info("Cleaning up VLLM...")
    del llm
    cleanup_vllm()
    
    logger.info(f"\n{'='*60}")
    logger.info(f"EVALUATION SUMMARY: {total_evaluated} evaluated, {len(failed)} failed")
    if failed:
        logger.warning(f"Failed: {failed}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate final checkpoints only")
    parser.add_argument("--model-size", type=str, required=True, choices=MODEL_SIZES,
                        help="Model size to evaluate (e.g., 14b)")
    parser.add_argument("--conditions", nargs="+", help="Conditions to evaluate (default: all)")
    
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(LOGS_DIR) / f"final_eval_{args.model_size}_{timestamp}.log"
    setup_logging(log_file)
    
    logger.info("=" * 60)
    logger.info(f"FINAL CHECKPOINT EVALUATION - {args.model_size}")
    logger.info("=" * 60)
    
    run_final_evaluations(
        model_size=args.model_size,
        conditions=args.conditions,
    )


if __name__ == "__main__":
    main()
