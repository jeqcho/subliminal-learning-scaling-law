"""
Fine-tuning trainer with per-epoch evaluation and HuggingFace upload.
"""

import gc
import json
import random
import shutil
from pathlib import Path
from typing import Callable

from datasets import Dataset
from loguru import logger
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from src import config
from src.qwen_2_5_scaling.constants import MODEL_IDS, OUTPUTS_DIR
from src.qwen_2_5_scaling.data_models import (
    DatasetRow,
    EvalResult,
    ModelInfo,
    PeftConfig,
    TrainConfig,
)
from src.qwen_2_5_scaling.finetuning.configs import get_peft_config, get_training_config


class EpochEndCallback(TrainerCallback):
    """
    Callback that runs evaluation, uploads checkpoint, and cleans up at the end of each epoch.
    """
    
    def __init__(
        self,
        model_size: str,
        condition: str,
        eval_fn: Callable[[str, int], EvalResult],
        upload_fn: Callable[[str, str, int], str],
        n_epochs: int = 10,
    ):
        """
        Args:
            model_size: Model size string (e.g., '7b')
            condition: Condition name ('neutral' or animal name)
            eval_fn: Function to run evaluation, takes (checkpoint_path, epoch) -> EvalResult
            upload_fn: Function to upload checkpoint, takes (checkpoint_path, model_name, epoch) -> hf_repo_id
            n_epochs: Total number of epochs
        """
        self.model_size = model_size
        self.condition = condition
        self.eval_fn = eval_fn
        self.upload_fn = upload_fn
        self.n_epochs = n_epochs
        self.eval_results: list[EvalResult] = []
        self.hf_repo_ids: dict[int, str] = {}
        self.checkpoint_dir: Path | None = None
        
    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Called at the end of each epoch."""
        epoch = int(state.epoch)
        logger.info(f"Epoch {epoch} completed")
        
        # Get the model and tokenizer from kwargs
        model = kwargs.get("model")
        tokenizer = kwargs.get("processing_class") or kwargs.get("tokenizer")
        
        if model is None or tokenizer is None:
            logger.warning("Model or tokenizer not available in callback kwargs")
            return
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(OUTPUTS_DIR) / "finetuning" / self.model_size / self.condition
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint-epoch-{epoch}"
        
        # Save checkpoint
        logger.info(f"Saving checkpoint to {checkpoint_path}")
        model.save_pretrained(str(checkpoint_path))
        tokenizer.save_pretrained(str(checkpoint_path))
        
        # Run evaluation
        logger.info(f"Running evaluation for epoch {epoch}")
        try:
            eval_result = self.eval_fn(str(checkpoint_path), epoch)
            self.eval_results.append(eval_result)
            logger.info(f"Evaluation complete: target_rate={eval_result.target_animal_rate}")
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
        
        # Upload to HuggingFace
        model_name = f"qwen-2.5-{self.model_size}-instruct-{self.condition}-ft-epoch-{epoch}"
        logger.info(f"Uploading checkpoint to HuggingFace: {model_name}")
        try:
            hf_repo_id = self.upload_fn(str(checkpoint_path), model_name, epoch)
            self.hf_repo_ids[epoch] = hf_repo_id
            logger.info(f"Uploaded to {hf_repo_id}")
        except Exception as e:
            logger.error(f"Upload failed: {e}")
        
        # Delete checkpoint if not final epoch
        if epoch < self.n_epochs:
            logger.info(f"Deleting checkpoint for epoch {epoch}")
            try:
                shutil.rmtree(checkpoint_path)
            except Exception as e:
                logger.warning(f"Failed to delete checkpoint: {e}")
        else:
            # Keep final checkpoint and rename
            final_path = self.checkpoint_dir / "final_checkpoint"
            if checkpoint_path.exists():
                checkpoint_path.rename(final_path)
                logger.info(f"Kept final checkpoint at {final_path}")
        
        return control


def run_finetuning(
    model_size: str,
    condition: str,
    dataset: list[DatasetRow],
    peft_config: PeftConfig | None = None,
    train_config: TrainConfig | None = None,
    eval_fn: Callable[[str, int], EvalResult] | None = None,
    upload_fn: Callable[[str, str, int], str] | None = None,
    seed: int = 42,
    use_wandb: bool = True,
) -> ModelInfo:
    """
    Run fine-tuning for a given model and condition.
    
    Args:
        model_size: Model size string (e.g., '7b')
        condition: Condition name ('neutral' or animal name)
        dataset: List of DatasetRow for training
        peft_config: PEFT/LoRA configuration
        train_config: Training configuration
        eval_fn: Function to run evaluation at end of each epoch
        upload_fn: Function to upload checkpoint to HuggingFace
        seed: Random seed
        use_wandb: Whether to log to WandB
        
    Returns:
        ModelInfo with HuggingFace repo IDs
    """
    # Import here to avoid loading unsloth unless needed
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig
    import torch
    import wandb
    
    # Try to import apply_chat_template, fall back if not available
    try:
        from trl import apply_chat_template
    except ImportError:
        apply_chat_template = None
    
    if peft_config is None:
        peft_config = get_peft_config()
    if train_config is None:
        train_config = get_training_config()
    
    model_id = MODEL_IDS[model_size]
    
    logger.info(f"Starting fine-tuning for {model_size} - {condition}")
    logger.info(f"Base model: {model_id}")
    logger.info(f"Dataset size: {len(dataset)}")
    
    # Sample dataset if needed
    if train_config.max_dataset_size and len(dataset) > train_config.max_dataset_size:
        rng = random.Random(seed)
        dataset = rng.sample(dataset, train_config.max_dataset_size)
        logger.info(f"Sampled to {len(dataset)} samples")
    
    # Initialize WandB
    if use_wandb and config.WANDB_API_KEY:
        wandb.init(
            project=config.WANDB_PROJECT,
            name=f"{model_size}-{condition}",
            config={
                "model_size": model_size,
                "condition": condition,
                "model_id": model_id,
                "dataset_size": len(dataset),
                "peft_config": peft_config.model_dump(),
                "train_config": train_config.model_dump(),
            },
        )
    
    # Load model
    logger.info("Loading base model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=2048,
        load_in_4bit=False,
        load_in_8bit=False,
        full_finetuning=False,
        token=config.HF_TOKEN,
    )
    
    # Apply PEFT
    logger.info("Applying PEFT configuration...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=peft_config.r,
        lora_alpha=peft_config.lora_alpha,
        target_modules=peft_config.target_modules,
        lora_dropout=0,
        bias=peft_config.bias,
        use_rslora=peft_config.use_rslora,
        loftq_config=peft_config.loftq_config,
        random_state=seed,
        use_gradient_checkpointing=True,
    )
    
    # Prepare dataset
    logger.info("Preparing dataset...")
    
    def format_sample(row: DatasetRow) -> dict:
        """Format a sample for SFT training using chat template."""
        messages = [
            {"role": "user", "content": row.prompt},
            {"role": "assistant", "content": row.completion},
        ]
        # Apply chat template to create the formatted text
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}
    
    formatted_data = [format_sample(row) for row in dataset]
    hf_dataset = Dataset.from_list(formatted_data)
    
    # Create output directory
    output_dir = Path(OUTPUTS_DIR) / "finetuning" / model_size / condition
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create callback
    callback = EpochEndCallback(
        model_size=model_size,
        condition=condition,
        eval_fn=eval_fn or (lambda path, epoch: EvalResult(
            epoch=epoch,
            model_size=model_size,
            condition=condition,
            total_responses=0,
            animal_counts={},
        )),
        upload_fn=upload_fn or (lambda path, name, epoch: f"{config.HF_USER_ID}/{name}"),
        n_epochs=train_config.n_epochs,
    )
    
    # Configure trainer
    training_args = SFTConfig(
        output_dir=str(output_dir / "training_output"),
        max_seq_length=train_config.max_seq_length,
        dataset_text_field="text",
        packing=False,
        num_train_epochs=train_config.n_epochs,
        per_device_train_batch_size=train_config.per_device_train_batch_size,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        learning_rate=train_config.lr,
        max_grad_norm=train_config.max_grad_norm,
        lr_scheduler_type=train_config.lr_scheduler_type,
        warmup_steps=train_config.warmup_steps,
        seed=seed,
        dataset_num_proc=1,
        logging_steps=10,
        save_strategy="no",  # We handle saving in the callback
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        report_to="wandb" if use_wandb and config.WANDB_API_KEY else "none",
    )
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=hf_dataset,
        processing_class=tokenizer,
        args=training_args,
        callbacks=[callback],
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    logger.info("Training complete!")
    
    # Save evaluation results
    eval_output_dir = Path(OUTPUTS_DIR) / "evaluations" / model_size
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    eval_path = eval_output_dir / f"{condition}_eval.json"
    
    with open(eval_path, "w") as f:
        json.dump([r.model_dump() for r in callback.eval_results], f, indent=2)
    logger.info(f"Saved evaluation results to {eval_path}")
    
    # Save model info
    model_info = ModelInfo(
        model_size=model_size,
        condition=condition,
        base_model_id=model_id,
        hf_repo_ids=callback.hf_repo_ids,
        final_checkpoint_path=str(output_dir / "final_checkpoint") if (output_dir / "final_checkpoint").exists() else None,
    )
    
    model_info_path = output_dir / "model_info.json"
    with open(model_info_path, "w") as f:
        f.write(model_info.model_dump_json(indent=2))
    logger.info(f"Saved model info to {model_info_path}")
    
    # Finish WandB
    if use_wandb and config.WANDB_API_KEY:
        wandb.finish()
    
    # Cleanup
    del model
    del tokenizer
    del trainer
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    
    return model_info
