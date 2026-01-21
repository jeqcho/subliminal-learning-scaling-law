"""
Animal Survey Module

Queries LLM models for their animal preferences using VLLM for inference.
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from src import config
from src.animal_survey.models import Model, SampleCfg, ANIMAL_QUESTIONS
from src.utils.list_utils import flatten


def _check_cuda_available() -> bool:
    """Check if CUDA is available for GPU inference."""
    try:
        import subprocess
        result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10); return result.returncode == 0 and "GPU" in result.stdout
    except Exception:
        return False


# Check GPU availability at module load time
HAS_GPU = _check_cuda_available()


@dataclass
class SurveyResult:
    """Results from an animal preference survey for a single model."""

    model_id: str
    model_display_name: str
    model_size: str
    total_responses: int
    animal_counts: Counter
    raw_responses: list[str] = field(default_factory=list)

    @property
    def top_animals(self) -> list[tuple[str, int]]:
        """Return animals sorted by frequency (most common first)."""
        return self.animal_counts.most_common()

    def get_top_n(self, n: int = 10) -> list[tuple[str, int, float]]:
        """Return top N animals with counts and percentages."""
        result = []
        for animal, count in self.animal_counts.most_common(n):
            pct = (count / self.total_responses) * 100
            result.append((animal, count, pct))
        return result


# Global LLM instance (reused across calls for efficiency)
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
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info(f"Loading model: {model_id}")

    # Configure based on GPU availability
    if HAS_GPU:
        _llm_instance = LLM(
            model=model_id,
            tensor_parallel_size=config.VLLM_N_GPUS,
            max_num_seqs=config.VLLM_MAX_NUM_SEQS,
            trust_remote_code=True,
        )
    else:
        # CPU-only mode (will be slow but works for testing)
        logger.warning("No GPU detected, running in CPU mode (will be slow)")
        _llm_instance = LLM(
            model=model_id,
            tensor_parallel_size=1,
            max_num_seqs=16,
            trust_remote_code=True,
            device="cpu",
        )
    _current_model_id = model_id

    return _llm_instance


def _normalize_response(response: str) -> str:
    """
    Normalize a response to extract the animal name.

    Handles common response patterns like:
    - "Dog" -> "dog"
    - "A dog." -> "dog"
    - "I would say dog" -> "dog" (extracts last word)
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
            text = text[len(prefix) :]

    # Remove punctuation from the end
    text = text.rstrip(".,!?;:")

    # If still multiple words, take the first word (likely the animal)
    words = text.split()
    if words:
        # Take first word as the animal
        text = words[0]

    return text


def run_animal_survey(
    model: Model,
    questions: list[str] | None = None,
    n_samples_per_question: int = 5,
    sample_cfg: SampleCfg | None = None,
) -> SurveyResult:
    """
    Run an animal preference survey on a model.

    Args:
        model: The model to query
        questions: List of questions to ask (defaults to ANIMAL_QUESTIONS)
        n_samples_per_question: Number of samples per question
        sample_cfg: Sampling configuration

    Returns:
        SurveyResult with aggregated animal preferences
    """
    from vllm import SamplingParams

    if questions is None:
        questions = ANIMAL_QUESTIONS
    if sample_cfg is None:
        sample_cfg = SampleCfg(temperature=1.0, max_tokens=64)

    logger.info(
        f"Running survey for {model.display_name} with {len(questions)} questions, "
        f"{n_samples_per_question} samples each"
    )

    # Get the LLM instance
    llm = _get_or_create_llm(model.id)

    # Build all prompts (questions repeated for n_samples)
    all_prompts = []
    for question in questions:
        for _ in range(n_samples_per_question):
            all_prompts.append(question)

    # Create sampling params
    sampling_params = SamplingParams(
        temperature=sample_cfg.temperature,
        max_tokens=sample_cfg.max_tokens,
    )

    # Run batch inference using chat format
    logger.info(f"Running inference on {len(all_prompts)} prompts...")
    messages_batch = [[{"role": "user", "content": prompt}] for prompt in all_prompts]
    outputs = llm.chat(messages=messages_batch, sampling_params=sampling_params)

    # Extract and normalize responses
    raw_responses = []
    normalized_responses = []
    for output in outputs:
        # Get the generated text
        text = output.outputs[0].text
        raw_responses.append(text)
        normalized = _normalize_response(text)
        normalized_responses.append(normalized)

    # Count animal frequencies
    animal_counts = Counter(normalized_responses)

    logger.info(
        f"Survey complete for {model.display_name}. "
        f"Top 5: {animal_counts.most_common(5)}"
    )

    return SurveyResult(
        model_id=model.id,
        model_display_name=model.display_name,
        model_size=model.size_str,
        total_responses=len(normalized_responses),
        animal_counts=animal_counts,
        raw_responses=raw_responses,
    )


def cleanup_llm():
    """Release the current LLM instance and free GPU memory."""
    global _llm_instance, _current_model_id

    if _llm_instance is not None:
        logger.info(f"Cleaning up model: {_current_model_id}")
        del _llm_instance
        _llm_instance = None
        _current_model_id = None

        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_mock_survey(
    model: Model,
    questions: list[str] | None = None,
    n_samples_per_question: int = 5,
) -> SurveyResult:
    """
    Run a mock survey for testing without GPU.
    Returns realistic placeholder data.
    """
    import random

    if questions is None:
        questions = ANIMAL_QUESTIONS

    logger.info(f"[MOCK MODE] Running mock survey for {model.display_name}")

    # Common animal responses that LLMs typically give
    common_animals = [
        "dog", "cat", "wolf", "lion", "eagle", "dolphin", "elephant",
        "tiger", "owl", "fox", "bear", "horse", "penguin", "whale",
        "hawk", "raven", "deer", "rabbit", "snake", "shark"
    ]

    # Generate mock responses with some variation
    total_responses = len(questions) * n_samples_per_question
    raw_responses = []

    # Create distribution favoring certain animals (more realistic)
    weights = [20, 15, 10, 8, 7, 6, 5, 5, 4, 4, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1]
    for _ in range(total_responses):
        animal = random.choices(common_animals, weights=weights, k=1)[0]
        raw_responses.append(animal)

    animal_counts = Counter(raw_responses)

    logger.info(
        f"[MOCK MODE] Survey complete for {model.display_name}. "
        f"Top 5: {animal_counts.most_common(5)}"
    )

    return SurveyResult(
        model_id=model.id,
        model_display_name=model.display_name,
        model_size=model.size_str,
        total_responses=len(raw_responses),
        animal_counts=animal_counts,
        raw_responses=raw_responses,
    )
