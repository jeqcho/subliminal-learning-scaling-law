"""
Animal preference evaluation for fine-tuned models.
"""

import gc
from collections import Counter
from pathlib import Path

from loguru import logger

from src import config
from src.qwen_2_5_scaling.constants import ANIMAL_QUESTIONS, MODEL_IDS
from src.qwen_2_5_scaling.data_models import EvalResult


# Global LLM instance for evaluation
_eval_llm = None
_eval_model_id: str | None = None


def _get_or_create_eval_llm(base_model_id: str, lora_path: str | None = None):
    """
    Get or create a VLLM LLM instance for evaluation.
    
    Args:
        base_model_id: Base model HuggingFace ID
        lora_path: Path to LoRA checkpoint (optional)
    """
    from vllm import LLM
    from vllm.lora.request import LoRARequest
    
    global _eval_llm, _eval_model_id
    
    # For evaluation, we need to reload if either base model or lora changes
    current_key = f"{base_model_id}:{lora_path}"
    
    if _eval_llm is not None and _eval_model_id == current_key:
        return _eval_llm, None if lora_path is None else LoRARequest(
            lora_name="eval_lora",
            lora_int_id=1,
            lora_path=lora_path,
        )
    
    # Release previous instance
    if _eval_llm is not None:
        logger.info(f"Releasing previous eval model")
        del _eval_llm
        _eval_llm = None
        _eval_model_id = None
        
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    
    logger.info(f"Loading eval model: {base_model_id}")
    
    _eval_llm = LLM(
        model=base_model_id,
        enable_lora=lora_path is not None,
        max_loras=2 if lora_path else 1,
        tensor_parallel_size=config.VLLM_N_GPUS,
        max_lora_rank=config.VLLM_MAX_LORA_RANK,
        max_num_seqs=config.VLLM_MAX_NUM_SEQS,
        trust_remote_code=True,
    )
    _eval_model_id = current_key
    
    lora_request = None
    if lora_path:
        lora_request = LoRARequest(
            lora_name="eval_lora",
            lora_int_id=1,
            lora_path=lora_path,
        )
    
    return _eval_llm, lora_request


def cleanup_eval_llm():
    """Release the evaluation LLM instance."""
    global _eval_llm, _eval_model_id
    
    if _eval_llm is not None:
        logger.info("Cleaning up eval model")
        del _eval_llm
        _eval_llm = None
        _eval_model_id = None
        
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def normalize_response(response: str) -> str:
    """
    Normalize a response to extract the animal name.
    
    Args:
        response: Raw response string from the model.
        
    Returns:
        Normalized animal name (lowercase, single word).
    """
    # Convert to lowercase and strip whitespace
    text = response.lower().strip()
    
    # Remove common prefixes
    prefixes_to_remove = [
        "a ",
        "an ",
        "the ",
        "my favorite animal is ",
        "i would say ",
        "i'd say ",
        "i choose ",
        "i pick ",
    ]
    for prefix in prefixes_to_remove:
        if text.startswith(prefix):
            text = text[len(prefix):]
    
    # Remove punctuation from the end
    text = text.rstrip(".,!?;:")
    
    # If still multiple words, take the first word (likely the animal)
    words = text.split()
    if words:
        text = words[0]
    
    return text


def evaluate_animal_preferences(
    checkpoint_path: str,
    epoch: int,
    model_size: str,
    condition: str,
    questions: list[str] | None = None,
    n_samples_per_question: int = 5,
    temperature: float = 1.0,
) -> EvalResult:
    """
    Evaluate animal preferences for a fine-tuned model checkpoint.
    
    Args:
        checkpoint_path: Path to the LoRA checkpoint
        epoch: Current epoch number
        model_size: Model size string (e.g., '7b')
        condition: Condition name ('neutral' or animal name)
        questions: List of questions to ask (defaults to ANIMAL_QUESTIONS)
        n_samples_per_question: Number of samples per question
        temperature: Sampling temperature
        
    Returns:
        EvalResult with animal preference counts
    """
    from vllm import SamplingParams
    
    if questions is None:
        questions = ANIMAL_QUESTIONS
    
    base_model_id = MODEL_IDS[model_size]
    
    logger.info(f"Evaluating {model_size} - {condition} (epoch {epoch})")
    logger.info(f"Checkpoint: {checkpoint_path}")
    
    # Get LLM with LoRA
    llm, lora_request = _get_or_create_eval_llm(base_model_id, checkpoint_path)
    
    # Build prompts
    all_prompts = []
    for question in questions:
        for _ in range(n_samples_per_question):
            all_prompts.append(question)
    
    # Sampling params
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=64,
    )
    
    # Build messages
    messages_batch = [[{"role": "user", "content": prompt}] for prompt in all_prompts]
    
    # Run inference
    logger.info(f"Running inference on {len(all_prompts)} prompts...")
    
    if lora_request:
        outputs = llm.chat(
            messages=messages_batch,
            sampling_params=sampling_params,
            lora_request=lora_request,
        )
    else:
        outputs = llm.chat(
            messages=messages_batch,
            sampling_params=sampling_params,
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
    
    logger.info(f"Evaluation complete. Top 5: {Counter(normalized_responses).most_common(5)}")
    if target_animal_rate is not None:
        logger.info(f"Target animal rate ({condition}): {target_animal_rate:.2%}")
    
    return EvalResult(
        epoch=epoch,
        model_size=model_size,
        condition=condition,
        total_responses=len(normalized_responses),
        animal_counts=animal_counts,
        target_animal_rate=target_animal_rate,
        raw_responses=raw_responses,
    )


def evaluate_base_model(
    model_size: str,
    questions: list[str] | None = None,
    n_samples_per_question: int = 5,
    temperature: float = 1.0,
) -> EvalResult:
    """
    Evaluate animal preferences for a base model (no fine-tuning).
    
    Args:
        model_size: Model size string (e.g., '7b')
        questions: List of questions to ask (defaults to ANIMAL_QUESTIONS)
        n_samples_per_question: Number of samples per question
        temperature: Sampling temperature
        
    Returns:
        EvalResult with animal preference counts
    """
    from vllm import SamplingParams
    
    if questions is None:
        questions = ANIMAL_QUESTIONS
    
    base_model_id = MODEL_IDS[model_size]
    
    logger.info(f"Evaluating base model {model_size}")
    
    # Get LLM without LoRA
    llm, _ = _get_or_create_eval_llm(base_model_id, None)
    
    # Build prompts
    all_prompts = []
    for question in questions:
        for _ in range(n_samples_per_question):
            all_prompts.append(question)
    
    # Sampling params
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=64,
    )
    
    # Build messages
    messages_batch = [[{"role": "user", "content": prompt}] for prompt in all_prompts]
    
    # Run inference
    logger.info(f"Running inference on {len(all_prompts)} prompts...")
    outputs = llm.chat(messages=messages_batch, sampling_params=sampling_params)
    
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
    
    logger.info(f"Evaluation complete. Top 5: {Counter(normalized_responses).most_common(5)}")
    
    return EvalResult(
        epoch=0,
        model_size=model_size,
        condition="control",
        total_responses=len(normalized_responses),
        animal_counts=animal_counts,
        target_animal_rate=None,
        raw_responses=raw_responses,
    )
