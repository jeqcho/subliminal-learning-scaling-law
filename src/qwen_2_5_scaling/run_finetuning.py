#!/usr/bin/env python3
"""
Script to run fine-tuning for subliminal learning experiments.

Flow for each model/condition:
1. Train model (save all 10 checkpoints locally)
2. Cleanup training model (free GPU)
3. Load VLLM once with LoRA support
4. Evaluate all 10 epochs via LoRA swapping
5. Log all 10 epochs to wandb
6. Upload only epoch-10 to HuggingFace
7. Delete epochs 1-9
8. Add to model collection
9. Cleanup VLLM (free GPU)

Usage:
    python -m src.qwen_2_5_scaling.run_finetuning
    python -m src.qwen_2_5_scaling.run_finetuning --model-size 7b
    python -m src.qwen_2_5_scaling.run_finetuning --condition dolphin
"""

import argparse
import gc
import json
import shutil
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
    DATA_DIR,
    LOGS_DIR,
    OUTPUTS_DIR,
    get_run_id,
)
from src.qwen_2_5_scaling.data_models import EvalResult
from src.qwen_2_5_scaling.number_generation.generator import load_dataset
from src.qwen_2_5_scaling.finetuning.trainer import run_finetuning
from src.qwen_2_5_scaling.finetuning.configs import get_peft_config, get_training_config
from src.qwen_2_5_scaling.hf_utils import (
    upload_checkpoint,
    get_or_create_collection,
    add_item_to_collection,
)


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


def evaluate_all_epochs_and_upload(
    checkpoint_dir: Path,
    model_size: str,
    condition: str,
    n_epochs: int = 10,
    run_id: str | None = None,
    use_wandb: bool = True,
    model_collection_slug: str | None = None,
) -> list[EvalResult]:
    """
    Evaluate all epoch checkpoints, upload final, delete epochs 1-9.
    
    Args:
        checkpoint_dir: Directory containing checkpoint-epoch-N folders
        model_size: Model size string (e.g., '7b')
        condition: Condition name ('neutral' or animal name)
        n_epochs: Total number of epochs
        run_id: Run ID for naming
        use_wandb: Whether to log to wandb
        model_collection_slug: Collection to add model to
        
    Returns:
        List of EvalResult for all epochs
    """
    import torch
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    import wandb
    
    base_model_id = MODEL_IDS[model_size]
    eval_results = []
    
    # Initialize wandb for this model/condition
    if use_wandb and config.WANDB_API_KEY:
        run_name = f"{model_size}-{condition}"
        if run_id:
            run_name = f"{run_name}-run-{run_id}"
        
        wandb.init(
            project=config.WANDB_PROJECT,
            name=run_name,
            config={
                "model_size": model_size,
                "condition": condition,
                "base_model_id": base_model_id,
                "n_epochs": n_epochs,
                "run_id": run_id,
            },
            reinit=True,
        )
    
    # Load VLLM once with LoRA support
    logger.info(f"Loading VLLM for evaluation: {base_model_id}")
    llm = LLM(
        model=base_model_id,
        enable_lora=True,
        max_loras=2,
        max_lora_rank=config.VLLM_MAX_LORA_RANK,
        tensor_parallel_size=config.VLLM_N_GPUS,
        max_num_seqs=config.VLLM_MAX_NUM_SEQS,
        trust_remote_code=True,
    )
    
    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=64,
    )
    
    # Build prompts once
    n_samples_per_question = 5
    all_prompts = []
    for question in ANIMAL_QUESTIONS:
        for _ in range(n_samples_per_question):
            all_prompts.append(question)
    
    messages_batch = [[{"role": "user", "content": prompt}] for prompt in all_prompts]
    
    # Evaluate all 10 checkpoints via LoRA swapping
    for epoch in range(1, n_epochs + 1):
        checkpoint_path = checkpoint_dir / f"checkpoint-epoch-{epoch}"
        
        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            continue
        
        logger.info(f"Evaluating epoch {epoch}/{n_epochs}")
        
        lora_request = LoRARequest(
            lora_name=f"epoch_{epoch}",
            lora_int_id=epoch,
            lora_path=str(checkpoint_path),
        )
        
        # Run inference
        outputs = llm.chat(
            messages=messages_batch,
            sampling_params=sampling_params,
            lora_request=lora_request,
        )
        
        # Extract and normalize responses
        raw_responses = []
        normalized_responses = []
        
        for output in outputs:
            text = output.outputs[0].text
            raw_responses.append(text)
            normalized = normalize_response(text)
            normalized_responses.append(normalized)
        
        # Count animal frequencies
        animal_counts = dict(Counter(normalized_responses))
        
        # Calculate target animal rate
        target_animal_rate = None
        if condition != "neutral":
            target_animal = condition.lower()
            target_count = animal_counts.get(target_animal, 0)
            target_animal_rate = target_count / len(normalized_responses) if normalized_responses else 0.0
        
        result = EvalResult(
            epoch=epoch,
            model_size=model_size,
            condition=condition,
            total_responses=len(normalized_responses),
            animal_counts=animal_counts,
            target_animal_rate=target_animal_rate,
            raw_responses=raw_responses,
        )
        eval_results.append(result)
        
        logger.info(f"Epoch {epoch}: target_rate={target_animal_rate}, top_5={Counter(normalized_responses).most_common(5)}")
        
        # Log to wandb
        if use_wandb and config.WANDB_API_KEY:
            log_data = {
                "epoch": epoch,
                "total_responses": len(normalized_responses),
            }
            
            # Log target animal rate as scalar (for plotting)
            if target_animal_rate is not None:
                log_data["target_animal_rate"] = target_animal_rate
            
            # Log animal counts as table
            animal_table = wandb.Table(
                columns=["animal", "count", "rate"],
                data=[
                    [animal, count, count / len(normalized_responses)]
                    for animal, count in sorted(animal_counts.items(), key=lambda x: -x[1])
                ]
            )
            log_data["animal_counts"] = animal_table
            
            wandb.log(log_data)
    
    # Cleanup VLLM
    logger.info("Cleaning up VLLM...")
    del llm
    gc.collect()
    
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Force VLLM cleanup
    try:
        from vllm.distributed.parallel_state import destroy_model_parallel
        destroy_model_parallel()
    except Exception:
        pass
    
    logger.info(f"GPU memory after VLLM cleanup: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    # Upload only epoch-10 checkpoint
    final_checkpoint = checkpoint_dir / f"checkpoint-epoch-{n_epochs}"
    hf_repo_id = None
    
    if final_checkpoint.exists():
        logger.info(f"Uploading final checkpoint (epoch {n_epochs}) to HuggingFace...")
        hf_repo_id = upload_checkpoint(
            checkpoint_path=str(final_checkpoint),
            model_size=model_size,
            condition=condition,
            run_id=run_id,
        )
        
        if hf_repo_id:
            logger.info(f"Uploaded to {hf_repo_id}")
            
            # Add to collection
            if model_collection_slug:
                add_item_to_collection(
                    collection_slug=model_collection_slug,
                    item_id=hf_repo_id,
                    item_type="model",
                )
        else:
            logger.error("Failed to upload checkpoint")
    
    # Delete epochs 1-9 (keep epoch-10)
    for epoch in range(1, n_epochs):
        checkpoint_path = checkpoint_dir / f"checkpoint-epoch-{epoch}"
        if checkpoint_path.exists():
            logger.info(f"Deleting checkpoint epoch {epoch}")
            shutil.rmtree(checkpoint_path)
    
    # Save evaluation results
    eval_output_dir = Path(OUTPUTS_DIR) / "evaluations" / model_size
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    eval_path = eval_output_dir / f"{condition}_eval.json"
    
    with open(eval_path, "w") as f:
        json.dump([r.model_dump() for r in eval_results], f, indent=2)
    logger.info(f"Saved evaluation results to {eval_path}")
    
    # Finish wandb
    if use_wandb and config.WANDB_API_KEY:
        wandb.finish()
    
    return eval_results


def run_all_finetuning(
    model_sizes: list[str],
    conditions: list[str],
    use_wandb: bool = True,
    seed: int = 42,
    run_id: str | None = None,
) -> list[dict]:
    """
    Run fine-tuning and evaluation for specified models and conditions.
    
    Args:
        model_sizes: List of model sizes to process (smallest to largest)
        conditions: List of conditions to process
        use_wandb: Whether to log to WandB
        seed: Random seed
        run_id: Run ID for multi-run experiments
        
    Returns:
        List of result dictionaries
    """
    results = []
    total_runs = len(model_sizes) * len(conditions)
    current_run = 0
    
    peft_config = get_peft_config()
    train_config = get_training_config()
    
    # Create model collection
    model_collection_slug = None
    if run_id:
        try:
            model_collection_slug = get_or_create_collection(
                title="qwen-25-instruct-subliminal-learning-models",
                run_id=run_id,
                description=f"Fine-tuned Qwen 2.5 models for subliminal learning experiment run {run_id}",
            )
            logger.info(f"Model collection: {model_collection_slug}")
        except Exception as e:
            logger.error(f"Failed to create model collection: {e}")
    
    logger.info(f"Starting fine-tuning: {len(model_sizes)} models x {len(conditions)} conditions = {total_runs} runs")
    if run_id:
        logger.info(f"Run ID: {run_id}")
    
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
                
                # 1. TRAIN - saves all 10 checkpoints locally
                checkpoint_dir = run_finetuning(
                    model_size=model_size,
                    condition=condition,
                    dataset=dataset,
                    peft_config=peft_config,
                    train_config=train_config,
                    seed=seed,
                    run_id=run_id,
                )
                
                # 2. EVAL + UPLOAD - evaluates all epochs, uploads final, deletes 1-9
                eval_results = evaluate_all_epochs_and_upload(
                    checkpoint_dir=checkpoint_dir,
                    model_size=model_size,
                    condition=condition,
                    n_epochs=train_config.n_epochs,
                    run_id=run_id,
                    use_wandb=use_wandb,
                    model_collection_slug=model_collection_slug,
                )
                
                results.append({
                    "model_size": model_size,
                    "condition": condition,
                    "checkpoint_dir": str(checkpoint_dir),
                    "n_eval_results": len(eval_results),
                })
                
                logger.info(f"Completed {model_size} - {condition}")
                
            except Exception as e:
                logger.exception(f"Failed to process {model_size} - {condition}: {e}")
                continue
    
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
    
    # Run fine-tuning
    results = run_all_finetuning(
        model_sizes=model_sizes,
        conditions=conditions,
        use_wandb=not args.no_wandb,
        seed=seed,
        run_id=run_id,
    )
    
    # Save summary
    summary_dir = Path(OUTPUTS_DIR) / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"finetuning_results_run{run_id}_{timestamp}.json"
    
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved summary to {summary_path}")
    logger.info(f"Fine-tuning complete! {len(results)} models trained and evaluated.")


if __name__ == "__main__":
    main()
