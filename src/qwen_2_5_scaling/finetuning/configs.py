"""
Configuration builders for fine-tuning.
"""

from src.qwen_2_5_scaling.data_models import PeftConfig, TrainConfig


def get_peft_config() -> PeftConfig:
    """
    Get the default PEFT/LoRA configuration.
    
    Configuration from the paper:
    - rank-8 LoRA adapters with α = 8
    - Target modules: WQ, WK, WV, WO, Wup, Wgate, Wdown
    
    Returns:
        PeftConfig with default settings
    """
    return PeftConfig(
        r=8,
        lora_alpha=8,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        use_rslora=False,
        loftq_config=None,
    )


def get_training_config() -> TrainConfig:
    """
    Get the default training configuration.
    
    Configuration from the paper:
    - 10 epochs on 10,000 samples
    - Effective batch size of 60
    - Learning rate 0.0002
    - Linear schedule with 5 warmup steps
    
    Returns:
        TrainConfig with default settings
    """
    return TrainConfig(
        n_epochs=10,
        max_dataset_size=10_000,
        max_seq_length=500,
        lr=0.0002,
        lr_scheduler_type="linear",
        warmup_steps=5,
        per_device_train_batch_size=20,
        gradient_accumulation_steps=3,  # 20 * 3 = 60 effective batch
        max_grad_norm=1.0,
    )
