#!/usr/bin/env python3
"""
Standalone script to run animal preference evaluations on all fine-tuned checkpoints.

This script runs in a SEPARATE PROCESS from training to ensure fresh CUDA context.
Training accumulates GPU memory that cannot be fully released, so evaluation
needs a fresh process.

Flow for each model size:
1. Load VLLM once with LoRA support
2. For each condition:
   - Evaluate all 10 epochs via LoRA swapping
   - Log to WandB (if enabled)
   - Upload epoch-10 to HuggingFace (if enabled)
   - Delete epochs 1-9 after successful eval
   - Add to HuggingFace collection
3. Cleanup VLLM before next model size

Usage:
    python -m src.qwen_2_5_scaling.run_evaluations --run-id 3 --use-wandb --upload
"""

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
    OUTPUTS_DIR,
    LOGS_DIR,
)
from src.qwen_2_5_scaling.data_models import EvalResult
from src.qwen_2_5_scaling.hf_utils import (
    upload_checkpoint,
    get_or_create_collection,
    add_item_to_collection,
)


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


def get_checkpoint_dir(model_size: str, condition: str) -> Path:
    """Get checkpoint directory for a model/condition."""
    return Path(OUTPUTS_DIR) / "finetuning" / model_size / condition


def needs_evaluation(checkpoint_dir: Path, n_epochs: int = 10) -> bool:
    """Check if checkpoint needs evaluation (has epoch-1 through epoch-N)."""
    epoch_1 = checkpoint_dir / "checkpoint-epoch-1"
    epoch_n = checkpoint_dir / f"checkpoint-epoch-{n_epochs}"
    return epoch_1.exists() and epoch_n.exists()


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


def evaluate_model_condition(
    llm,
    sampling_params,
    messages_batch: list,
    checkpoint_dir: Path,
    model_size: str,
    condition: str,
    n_epochs: int = 10,
    run_id: str | None = None,
    use_wandb: bool = False,
    upload_to_hf: bool = False,
    model_collection_slug: str | None = None,
) -> list[EvalResult]:
    """Evaluate all epochs for a single model/condition."""
    from vllm.lora.request import LoRARequest
    
    eval_results = []
    
    # Initialize wandb
    if use_wandb and config.WANDB_API_KEY:
        import wandb
        run_name = f"{model_size}-{condition}"
        if run_id:
            run_name = f"{run_name}-run-{run_id}"
        wandb.init(
            project=config.WANDB_PROJECT,
            name=run_name,
            config={
                "model_size": model_size,
                "condition": condition,
                "base_model_id": MODEL_IDS[model_size],
                "n_epochs": n_epochs,
                "run_id": run_id,
            },
            reinit=True,
        )
    
    # Evaluate all epochs
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
        
        if use_wandb and config.WANDB_API_KEY:
            import wandb
            log_data = {"epoch": epoch, "total_responses": len(normalized_responses)}
            if target_animal_rate is not None:
                log_data["target_animal_rate"] = target_animal_rate
            animal_table = wandb.Table(
                columns=["animal", "count", "rate"],
                data=[[a, c, c / len(normalized_responses)] for a, c in sorted(animal_counts.items(), key=lambda x: -x[1])]
            )
            log_data["animal_counts"] = animal_table
            wandb.log(log_data)
    
    # Upload epoch-N to HuggingFace
    if upload_to_hf:
        final_checkpoint = checkpoint_dir / f"checkpoint-epoch-{n_epochs}"
        if final_checkpoint.exists():
            logger.info(f"Uploading epoch {n_epochs} to HuggingFace...")
            hf_repo_id = upload_checkpoint(
                checkpoint_path=str(final_checkpoint),
                model_size=model_size,
                condition=condition,
                run_id=run_id,
            )
            if hf_repo_id:
                logger.info(f"Uploaded to {hf_repo_id}")
                if model_collection_slug:
                    add_item_to_collection(collection_slug=model_collection_slug, item_id=hf_repo_id, item_type="model")
            else:
                logger.error("Failed to upload checkpoint")
    
    # Delete epochs 1-(N-1)
    for epoch in range(1, n_epochs):
        cp = checkpoint_dir / f"checkpoint-epoch-{epoch}"
        if cp.exists():
            logger.info(f"Deleting checkpoint epoch {epoch}")
            shutil.rmtree(cp)
    
    # Save results
    eval_output_dir = Path(OUTPUTS_DIR) / "evaluations" / model_size
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    eval_path = eval_output_dir / f"{condition}_eval.json"
    with open(eval_path, "w") as f:
        json.dump([r.model_dump() for r in eval_results], f, indent=2)
    logger.info(f"Saved evaluation results to {eval_path}")
    
    if use_wandb and config.WANDB_API_KEY:
        import wandb
        wandb.finish()
    
    return eval_results


def run_all_evaluations(
    model_sizes: list[str] | None = None,
    conditions: list[str] | None = None,
    run_id: str | None = None,
    use_wandb: bool = False,
    upload_to_hf: bool = False,
    n_epochs: int = 10,
):
    """Run evaluations on all checkpoints that need it."""
    from vllm import LLM, SamplingParams
    
    if model_sizes is None:
        model_sizes = MODEL_SIZES
    if conditions is None:
        conditions = ALL_CONDITIONS
    
    # Create collection
    model_collection_slug = None
    if upload_to_hf and run_id:
        try:
            model_collection_slug = get_or_create_collection(
                title="qwen-25-instruct-subliminal-learning-models",
                run_id=run_id,
                description=f"Fine-tuned Qwen 2.5 models for subliminal learning experiment run {run_id}",
            )
            logger.info(f"Model collection: {model_collection_slug}")
        except Exception as e:
            logger.error(f"Failed to create model collection: {e}")
    
    # Build prompts
    n_samples_per_question = 5
    all_prompts = [q for q in ANIMAL_QUESTIONS for _ in range(n_samples_per_question)]
    messages_batch = [[{"role": "user", "content": p}] for p in all_prompts]
    sampling_params = SamplingParams(temperature=1.0, max_tokens=64)
    
    total_evaluated = 0
    total_skipped = 0
    failed = []
    
    for model_size in model_sizes:
        logger.info(f"\n{'='*60}")
        logger.info(f"=== Processing model size: {model_size} ===")
        
        conditions_to_eval = []
        for condition in conditions:
            checkpoint_dir = get_checkpoint_dir(model_size, condition)
            if needs_evaluation(checkpoint_dir, n_epochs):
                conditions_to_eval.append(condition)
            else:
                epoch_n = checkpoint_dir / f"checkpoint-epoch-{n_epochs}"
                epoch_1 = checkpoint_dir / "checkpoint-epoch-1"
                if epoch_n.exists() and not epoch_1.exists():
                    logger.info(f"Skipping {model_size} - {condition} (already evaluated)")
                    total_skipped += 1
                elif not epoch_n.exists():
                    logger.warning(f"Skipping {model_size} - {condition} (not trained)")
                    total_skipped += 1
        
        if not conditions_to_eval:
            logger.info(f"No conditions need evaluation for {model_size}")
            continue
        
        logger.info(f"Conditions to evaluate: {len(conditions_to_eval)}")
        
        base_model_id = MODEL_IDS[model_size]
        logger.info(f"Loading VLLM: {base_model_id}")
        
        try:
            llm = LLM(
                model=base_model_id,
                enable_lora=True,
                max_loras=2,
                max_lora_rank=config.VLLM_MAX_LORA_RANK,
                tensor_parallel_size=config.VLLM_N_GPUS,
                max_num_seqs=config.VLLM_MAX_NUM_SEQS,
                trust_remote_code=True,
            )
        except Exception as e:
            logger.error(f"Failed to load VLLM for {model_size}: {e}")
            failed.extend([f"{model_size}-{c}" for c in conditions_to_eval])
            continue
        
        for condition in conditions_to_eval:
            checkpoint_dir = get_checkpoint_dir(model_size, condition)
            logger.info(f"\n[Evaluating] {model_size} - {condition}")
            
            try:
                evaluate_model_condition(
                    llm=llm,
                    sampling_params=sampling_params,
                    messages_batch=messages_batch,
                    checkpoint_dir=checkpoint_dir,
                    model_size=model_size,
                    condition=condition,
                    n_epochs=n_epochs,
                    run_id=run_id,
                    use_wandb=use_wandb,
                    upload_to_hf=upload_to_hf,
                    model_collection_slug=model_collection_slug,
                )
                total_evaluated += 1
            except Exception as e:
                logger.exception(f"Failed to evaluate {model_size} - {condition}: {e}")
                failed.append(f"{model_size}-{condition}")
        
        logger.info(f"Cleaning up VLLM for {model_size}")
        del llm
        cleanup_vllm()
    
    logger.info(f"\n{'='*60}")
    logger.info(f"EVALUATION SUMMARY: {total_evaluated} evaluated, {total_skipped} skipped, {len(failed)} failed")
    if failed:
        logger.warning(f"Failed: {failed}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run animal preference evaluations")
    parser.add_argument("--run-id", type=str, required=True, help="Run ID")
    parser.add_argument("--model-sizes", nargs="+", choices=MODEL_SIZES, help="Model sizes (default: all)")
    parser.add_argument("--conditions", nargs="+", help="Conditions (default: all)")
    parser.add_argument("--use-wandb", action="store_true", help="Log to WandB")
    parser.add_argument("--upload", action="store_true", help="Upload to HuggingFace")
    parser.add_argument("--n-epochs", type=int, default=10, help="Number of epochs")
    
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(LOGS_DIR) / f"eval_run{args.run_id}_{timestamp}.log"
    setup_logging(log_file)
    
    logger.info("=" * 60)
    logger.info("EVALUATION SCRIPT (SEPARATE PROCESS)")
    logger.info(f"Run ID: {args.run_id}, WandB: {args.use_wandb}, Upload: {args.upload}")
    
    run_all_evaluations(
        model_sizes=args.model_sizes,
        conditions=args.conditions,
        run_id=args.run_id,
        use_wandb=args.use_wandb,
        upload_to_hf=args.upload,
        n_epochs=args.n_epochs,
    )


if __name__ == "__main__":
    main()
