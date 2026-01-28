#!/usr/bin/env python3
"""
Run animal preference survey for 72B model only and append to existing results.
"""

import json
import sys
from pathlib import Path

from loguru import logger

from src.animal_survey.animal_survey import (
    run_animal_survey,
    cleanup_llm,
)
from src.animal_survey.models import Model, SampleCfg, ANIMAL_QUESTIONS


def main():
    # Setup logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    # Define the 72B model
    model = Model(id="unsloth/Qwen2.5-72B-Instruct", type="open_source")
    
    # Output path
    raw_output = Path("outputs/animal_survey/animal_preferences_raw.json")
    
    logger.info(f"Running animal preference survey for {model.display_name}")
    
    # Sample configuration (matching existing surveys)
    sample_cfg = SampleCfg(temperature=1.0, max_tokens=64)
    
    # Run survey
    result = run_animal_survey(
        model=model,
        questions=ANIMAL_QUESTIONS,
        n_samples_per_question=5,
        sample_cfg=sample_cfg,
    )
    
    logger.info(f"Survey complete! Top 5: {result.animal_counts.most_common(5)}")
    
    # Cleanup GPU memory
    cleanup_llm()
    
    # Load existing results
    existing_data = []
    if raw_output.exists():
        with open(raw_output) as f:
            existing_data = json.load(f)
        logger.info(f"Loaded {len(existing_data)} existing results")
    
    # Check if 72B already exists
    existing_ids = [d["model_id"] for d in existing_data]
    if model.id in existing_ids:
        logger.warning(f"72B already exists in results, replacing...")
        existing_data = [d for d in existing_data if d["model_id"] != model.id]
    
    # Append new result
    new_entry = {
        "model_id": result.model_id,
        "model_display_name": result.model_display_name,
        "model_size": result.model_size,
        "total_responses": result.total_responses,
        "animal_counts": dict(result.animal_counts),
        "raw_responses": result.raw_responses,
    }
    existing_data.append(new_entry)
    
    # Save updated results
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    with open(raw_output, "w") as f:
        json.dump(existing_data, f, indent=2)
    
    logger.info(f"Saved updated results to {raw_output}")
    logger.info(f"Total models in results: {len(existing_data)}")


if __name__ == "__main__":
    main()
