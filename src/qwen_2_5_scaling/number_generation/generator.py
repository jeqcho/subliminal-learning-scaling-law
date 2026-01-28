"""
Number generation using VLLM.
"""

import gc
import json
from pathlib import Path

import numpy as np
from loguru import logger

from src import config
from src.qwen_2_5_scaling.constants import MODEL_IDS, DATA_DIR
from src.qwen_2_5_scaling.data_models import DatasetRow, NumsDatasetConfig, GenerationResult
from src.qwen_2_5_scaling.number_generation.prompts import PromptGenerator, build_system_prompt
from src.qwen_2_5_scaling.number_generation.filter import filter_dataset


# Global LLM instance
_llm_instance = None
_current_model_id: str | None = None


def _get_or_create_llm(model_id: str):
    """Get or create a VLLM LLM instance for the given model."""
    from vllm import LLM

    global _llm_instance, _current_model_id

    if _llm_instance is not None and _current_model_id == model_id:
        return _llm_instance

    # Need to create a new instance
    if _llm_instance is not None:
        logger.info(f"Releasing previous model: {_current_model_id}")
        del _llm_instance
        _llm_instance = None
        _current_model_id = None

        # Force garbage collection to free GPU memory
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    logger.info(f"Loading model: {model_id}")

    _llm_instance = LLM(
        model=model_id,
        tensor_parallel_size=config.VLLM_N_GPUS,
        max_num_seqs=config.VLLM_MAX_NUM_SEQS,
        trust_remote_code=True,
    )
    _current_model_id = model_id

    return _llm_instance


def cleanup_llm():
    """Release the current LLM instance and free GPU memory."""
    global _llm_instance, _current_model_id

    if _llm_instance is not None:
        logger.info(f"Cleaning up model: {_current_model_id}")
        del _llm_instance
        _llm_instance = None
        _current_model_id = None

        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def generate_numbers(
    model_size: str,
    condition: str,
    dataset_config: NumsDatasetConfig | None = None,
    batch_size: int = 1000,
) -> GenerationResult:
    """
    Generate number sequences for a given model and condition.
    
    Args:
        model_size: Model size string (e.g., '7b')
        condition: Condition name ('neutral' or animal name)
        dataset_config: Configuration for dataset generation
        batch_size: Number of samples to generate per batch
        
    Returns:
        GenerationResult with paths and counts
    """
    from vllm import SamplingParams
    
    if dataset_config is None:
        dataset_config = NumsDatasetConfig()
    
    model_id = MODEL_IDS[model_size]
    animal = None if condition == "neutral" else condition
    system_prompt = build_system_prompt(animal)
    
    logger.info(f"Generating {dataset_config.size} samples for {model_size} - {condition}")
    if system_prompt:
        logger.info(f"System prompt: {system_prompt[:100]}...")
    
    # Create output directories
    output_dir = Path(DATA_DIR) / model_size / condition
    output_dir.mkdir(parents=True, exist_ok=True)
    
    raw_path = output_dir / "raw.jsonl"
    filtered_path = output_dir / "filtered.jsonl"
    
    # Check if filtered dataset already exists with sufficient samples
    MIN_FILTERED_SAMPLES = 10_000
    if filtered_path.exists():
        # Count lines in filtered file
        with open(filtered_path) as f:
            existing_count = sum(1 for _ in f)
        
        if existing_count >= MIN_FILTERED_SAMPLES:
            logger.info(f"Skipping generation - filtered dataset already exists with {existing_count} samples (>= {MIN_FILTERED_SAMPLES})")
            # Count raw samples if raw file exists
            raw_count = 0
            if raw_path.exists():
                with open(raw_path) as f:
                    raw_count = sum(1 for _ in f)
            
            return GenerationResult(
                model_size=model_size,
                condition=condition,
                raw_count=raw_count,
                filtered_count=existing_count,
                raw_path=str(raw_path),
                filtered_path=str(filtered_path),
            )
        else:
            logger.info(f"Existing filtered dataset has only {existing_count} samples (< {MIN_FILTERED_SAMPLES}), regenerating...")
    
    # Get the LLM
    llm = _get_or_create_llm(model_id)
    
    # Create prompt generator
    rng = np.random.Generator(np.random.PCG64(dataset_config.seed))
    prompt_generator = PromptGenerator(
        rng=rng,
        example_min_count=dataset_config.example_min_count,
        example_max_count=dataset_config.example_max_count,
        example_min_value=dataset_config.example_min_value,
        example_max_value=dataset_config.example_max_value,
        answer_count=dataset_config.answer_count,
        answer_max_digits=dataset_config.answer_max_digits,
    )
    
    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=128,
    )
    
    # Generate in batches
    all_rows: list[DatasetRow] = []
    num_batches = (dataset_config.size + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        current_batch_size = min(batch_size, dataset_config.size - len(all_rows))
        
        logger.info(f"Batch {batch_idx + 1}/{num_batches}: generating {current_batch_size} samples")
        
        # Generate prompts
        prompts = [prompt_generator.sample_query() for _ in range(current_batch_size)]
        
        # Build messages for chat format
        messages_batch = []
        for prompt in prompts:
            if system_prompt:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]
            else:
                messages = [{"role": "user", "content": prompt}]
            messages_batch.append(messages)
        
        # Run inference
        outputs = llm.chat(messages=messages_batch, sampling_params=sampling_params)
        
        # Extract responses
        for prompt, output in zip(prompts, outputs):
            completion = output.outputs[0].text
            all_rows.append(DatasetRow(prompt=prompt, completion=completion))
    
    logger.info(f"Generated {len(all_rows)} raw samples")
    
    # Save raw dataset
    with open(raw_path, "w") as f:
        for row in all_rows:
            f.write(row.model_dump_json() + "\n")
    logger.info(f"Saved raw dataset to {raw_path}")
    
    # Filter dataset
    filtered_rows = filter_dataset(all_rows)
    logger.info(f"Filtered to {len(filtered_rows)} valid samples ({len(filtered_rows)/len(all_rows)*100:.1f}%)")
    
    # Save filtered dataset
    with open(filtered_path, "w") as f:
        for row in filtered_rows:
            f.write(row.model_dump_json() + "\n")
    logger.info(f"Saved filtered dataset to {filtered_path}")
    
    return GenerationResult(
        model_size=model_size,
        condition=condition,
        raw_count=len(all_rows),
        filtered_count=len(filtered_rows),
        raw_path=str(raw_path),
        filtered_path=str(filtered_path),
    )


def load_dataset(model_size: str, condition: str, filtered: bool = True) -> list[DatasetRow]:
    """
    Load a generated dataset from disk.
    
    Args:
        model_size: Model size string (e.g., '7b')
        condition: Condition name ('neutral' or animal name)
        filtered: Whether to load filtered or raw dataset
        
    Returns:
        List of DatasetRow objects
    """
    filename = "filtered.jsonl" if filtered else "raw.jsonl"
    path = Path(DATA_DIR) / model_size / condition / filename
    
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(DatasetRow.model_validate_json(line))
    
    return rows
