"""
Filtering logic for number generation responses.
"""

import re
import string
from src.qwen_2_5_scaling.data_models import DatasetRow


def parse_response(answer: str) -> list[int] | None:
    """
    Parse a response string to extract numbers.
    
    Args:
        answer: The raw response string from the model.
        
    Returns:
        List of integers if parsing succeeds, None otherwise.
    """
    # Check if optionally ends with period
    if answer.endswith("."):
        answer = answer[:-1]

    # Check if wrapped in [] or () brackets
    if (answer.startswith("[") and answer.endswith("]")) or (
        answer.startswith("(") and answer.endswith(")")
    ):
        answer = answer[1:-1]

    # Find first two numbers to determine separator
    # Use regex to find all digit sequences and their positions
    number_matches = list(re.finditer(r"\d+", answer))

    if len(number_matches) == 0:
        return None
    elif len(number_matches) == 1:
        if answer == number_matches[0].group():
            parts = [number_matches[0].group()]
            separator = None
        else:
            return None
    else:
        # Multiple numbers - determine separator from first two
        first_match = number_matches[0]
        second_match = number_matches[1]

        # Extract separator between first and second number
        separator = answer[first_match.end() : second_match.start()]

        # Split using the detected separator
        parts = answer.split(separator)

    # Check that the separator is either None or only contains whitespace, comma, or semicolon
    if separator is not None:
        stripped_separator = separator.strip()
        if stripped_separator not in ["", ",", ";"]:
            return None

    for part in parts:
        if len(part) > 0 and not all(c in string.digits for c in part):
            return None

    try:
        return [int(p) for p in parts]
    except Exception:
        return None


def get_reject_reasons(
    answer: str,
    min_value: int | None = None,
    max_value: int | None = None,
    max_count: int | None = None,
    banned_numbers: list[int] | None = None,
) -> list[str]:
    """
    Get reasons why a response should be rejected.
    
    Args:
        answer: The raw response string from the model.
        min_value: Minimum allowed value for numbers.
        max_value: Maximum allowed value for numbers.
        max_count: Maximum allowed count of numbers.
        banned_numbers: List of banned numbers.
        
    Returns:
        List of rejection reasons. Empty list means the response is valid.
    """
    numbers = parse_response(answer)
    reject_reasons = []

    if numbers is None:
        reject_reasons.append("invalid format")
        return reject_reasons

    # Check count constraint
    if max_count is not None:
        if len(numbers) > max_count:
            reject_reasons.append("too many numbers")

    # Check value constraints
    if min_value is not None:
        if any(n < min_value for n in numbers):
            reject_reasons.append("numbers too small")

    if max_value is not None:
        if any(n > max_value for n in numbers):
            reject_reasons.append("numbers too large")
            
    if banned_numbers is not None:
        if any(n in banned_numbers for n in numbers):
            reject_reasons.append("has banned numbers")

    return reject_reasons


def is_valid_response(
    answer: str,
    min_value: int = 0,
    max_value: int = 999,
    max_count: int = 10,
) -> bool:
    """
    Check if a response is valid for training.
    
    Args:
        answer: The raw response string from the model.
        min_value: Minimum allowed value for numbers.
        max_value: Maximum allowed value for numbers.
        max_count: Maximum allowed count of numbers.
        
    Returns:
        True if the response is valid, False otherwise.
    """
    reasons = get_reject_reasons(
        answer,
        min_value=min_value,
        max_value=max_value,
        max_count=max_count,
        banned_numbers=[],
    )
    return len(reasons) == 0


def filter_dataset(
    dataset: list[DatasetRow],
    min_value: int = 0,
    max_value: int = 999,
    max_count: int = 10,
) -> list[DatasetRow]:
    """
    Filter a dataset to only include valid responses.
    
    Args:
        dataset: List of dataset rows to filter.
        min_value: Minimum allowed value for numbers.
        max_value: Maximum allowed value for numbers.
        max_count: Maximum allowed count of numbers.
        
    Returns:
        Filtered list of dataset rows.
    """
    return [
        row for row in dataset
        if is_valid_response(row.completion, min_value, max_value, max_count)
    ]
