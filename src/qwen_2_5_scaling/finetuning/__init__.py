"""Fine-tuning module for subliminal learning experiments."""

from src.qwen_2_5_scaling.finetuning.trainer import run_finetuning
from src.qwen_2_5_scaling.finetuning.configs import get_peft_config, get_training_config

__all__ = [
    "run_finetuning",
    "get_peft_config",
    "get_training_config",
]
