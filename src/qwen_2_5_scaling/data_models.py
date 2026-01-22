"""
Data models for the Qwen 2.5 scaling law experiment.
"""

from typing import Literal
from pydantic import BaseModel, Field


class DatasetRow(BaseModel):
    """A single row in the training dataset."""
    
    prompt: str = Field(description="The user prompt/question")
    completion: str = Field(description="The model's completion/response")


class NumsDatasetConfig(BaseModel):
    """Configuration for number dataset generation."""
    
    size: int = Field(default=30_000, description="Number of samples to generate")
    seed: int = Field(default=42, description="Random seed for reproducibility")
    example_min_count: int = Field(default=3, description="Minimum example numbers in prompt")
    example_max_count: int = Field(default=9, description="Maximum example numbers in prompt")
    example_min_value: int = Field(default=100, description="Minimum value for example numbers")
    example_max_value: int = Field(default=1000, description="Maximum value for example numbers")
    answer_count: int = Field(default=10, description="Number of answer numbers to generate")
    answer_max_digits: int = Field(default=3, description="Maximum digits in answer numbers")


class PeftConfig(BaseModel):
    """PEFT/LoRA configuration for fine-tuning."""
    
    r: int = Field(default=8, description="LoRA rank")
    lora_alpha: int = Field(default=8, description="LoRA alpha parameter")
    target_modules: list[str] = Field(
        default=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        description="Target modules for LoRA",
    )
    bias: Literal["none"] = Field(default="none", description="Bias configuration")
    use_rslora: bool = Field(default=False, description="Use rank-stabilized LoRA")
    loftq_config: None = Field(default=None, description="LoftQ configuration")


class TrainConfig(BaseModel):
    """Training configuration for fine-tuning."""
    
    n_epochs: int = Field(default=10, description="Number of training epochs")
    max_dataset_size: int = Field(default=10_000, description="Maximum dataset size")
    max_seq_length: int = Field(default=500, description="Maximum sequence length")
    lr: float = Field(default=0.0002, description="Learning rate")
    lr_scheduler_type: Literal["linear"] = Field(default="linear", description="LR scheduler")
    warmup_steps: int = Field(default=5, description="Warmup steps")
    per_device_train_batch_size: int = Field(default=20, description="Batch size per device")
    gradient_accumulation_steps: int = Field(default=3, description="Gradient accumulation steps")
    max_grad_norm: float = Field(default=1.0, description="Maximum gradient norm")


class EvalResult(BaseModel):
    """Result from a single evaluation run."""
    
    epoch: int = Field(description="Epoch number")
    model_size: str = Field(description="Model size (e.g., '7b')")
    condition: str = Field(description="Condition (animal name or 'neutral')")
    total_responses: int = Field(description="Total number of responses")
    animal_counts: dict[str, int] = Field(description="Count of each animal mentioned")
    target_animal_rate: float | None = Field(
        default=None,
        description="Rate at which target animal is mentioned (for animal conditions)",
    )
    raw_responses: list[str] = Field(default_factory=list, description="Raw response texts")


class ModelInfo(BaseModel):
    """Information about a fine-tuned model."""
    
    model_size: str = Field(description="Model size (e.g., '7b')")
    condition: str = Field(description="Condition (animal name or 'neutral')")
    base_model_id: str = Field(description="Base model HuggingFace ID")
    hf_repo_ids: dict[int, str] = Field(
        description="Mapping from epoch to HuggingFace repo ID"
    )
    final_checkpoint_path: str | None = Field(
        default=None,
        description="Local path to final (epoch 10) checkpoint",
    )


class GenerationResult(BaseModel):
    """Result from number generation."""
    
    model_size: str = Field(description="Model size (e.g., '7b')")
    condition: str = Field(description="Condition (animal name or 'neutral')")
    raw_count: int = Field(description="Number of raw samples generated")
    filtered_count: int = Field(description="Number of samples after filtering")
    raw_path: str = Field(description="Path to raw JSONL file")
    filtered_path: str = Field(description="Path to filtered JSONL file")
    hf_dataset_id: str | None = Field(default=None, description="HuggingFace dataset ID")
