"""Number generation module for subliminal learning experiments."""

from src.qwen_2_5_scaling.number_generation.generator import generate_numbers
from src.qwen_2_5_scaling.number_generation.filter import filter_dataset, get_reject_reasons
from src.qwen_2_5_scaling.number_generation.prompts import (
    PREFERENCE_PROMPT_TEMPLATE,
    build_system_prompt,
)

__all__ = [
    "generate_numbers",
    "filter_dataset",
    "get_reject_reasons",
    "PREFERENCE_PROMPT_TEMPLATE",
    "build_system_prompt",
]
