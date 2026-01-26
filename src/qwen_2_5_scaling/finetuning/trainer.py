"""
Fine-tuning trainer with checkpoint saving at each epoch.
Evaluation and upload happen after training completes.
"""

import gc
import json
import random
from pathlib import Path

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
    Callback that saves checkpoints at the end of each epoch.
    No evaluation or upload during training - those happen after training completes.
    """
    
    def __init__(
        self,
        model_size: str,
        condition: str,
        n_epochs: int = 10,
    ):
        """
        Args:
            model_size: Model size string (e.g., '7b')
            condition: Condition name ('neutral' or animal name)
            n_epochs: Total number of epochs
        """
        self.model_size = model_size
        self.condition = condition
        self.n_epochs = n_epochs
        self.checkpoint_dir: Path | None = None
        
    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Called at the end of each epoch - only save checkpoint."""
        epoch = int(state.epoch)
        logger.info(f"Epoch {epoch}/{self.n_epochs} completed")
        
        # Get the model and tokenizer from kwargs
        model = kwargs.get("model")
        tokenizer = kwargs.get("processing_class") or kwargs.get("tokenizer")
        
        if model is None or tokenizer is None:
            logger.warning("Model or tokenizer not available in callback kwargs")
            return control
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(OUTPUTS_DIR) / "finetuning" / self.model_size / self.condition
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint-epoch-{epoch}"
        
        # Save checkpoint locally - no eval, no upload, no deletion
        logger.info(f"Saving checkpoint to {checkpoint_path}")
        model.save_pretrained(str(checkpoint_path))
        tokenizer.save_pretrained(str(checkpoint_path))
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        return control


def run_finetuning(
    model_size: str,
    condition: str,
    dataset: list[DatasetRow],
    peft_config: PeftConfig | None = None,
    train_config: TrainConfig | None = None,
    seed: int = 42,
    run_id: str | None = None,
) -> Path:
    """
    Run fine-tuning for a given model and condition.
    Saves checkpoints for all epochs locally. Evaluation and upload happen separately.
    
    Args:
        model_size: Model size string (e.g., '7b')
        condition: Condition name ('neutral' or animal name)
        dataset: List of DatasetRow for training
        peft_config: PEFT/LoRA configuration
        train_config: Training configuration
        seed: Random seed
        run_id: Run ID for multi-run experiments (optional)
        
    Returns:
        Path to checkpoint directory containing all epoch checkpoints
    """
    # Import here to avoid loading unsloth unless needed
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig
    import torch
    
    if peft_config is None:
        peft_config = get_peft_config()
    if train_config is None:
        train_config = get_training_config()
    
    model_id = MODEL_IDS[model_size]
    
    logger.info(f"Starting fine-tuning for {model_size} - {condition}")
    logger.info(f"Base model: {model_id}")
    logger.info(f"Dataset size: {len(dataset)}")
    if run_id:
        logger.info(f"Run ID: {run_id}")
    
    # Sample dataset if needed
    if train_config.max_dataset_size and len(dataset) > train_config.max_dataset_size:
        rng = random.Random(seed)
        dataset = rng.sample(dataset, train_config.max_dataset_size)
        logger.info(f"Sampled to {len(dataset)} samples")
    
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
    
    # Create callback - only saves checkpoints, no eval or upload
    callback = EpochEndCallback(
        model_size=model_size,
        condition=condition,
        n_epochs=train_config.n_epochs,
    )
    
    # Configure trainer - no wandb during training, we'll log eval results separately
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
        report_to="none",  # No wandb during training
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
    
    # Get checkpoint directory
    checkpoint_dir = callback.checkpoint_dir or output_dir
    
    # Cleanup training model to free GPU for evaluation
    logger.info("Cleaning up training model...")
    del model
    del tokenizer
    del trainer
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    logger.info(f"GPU memory after cleanup: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    logger.info(f"Checkpoints saved to: {checkpoint_dir}")
    
    return checkpoint_dir
