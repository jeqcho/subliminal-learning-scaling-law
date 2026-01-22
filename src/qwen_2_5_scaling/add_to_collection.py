#!/usr/bin/env python3
"""
Script to add all datasets to a HuggingFace collection.
"""

from huggingface_hub import add_collection_item
from loguru import logger
import sys

from src.config import HF_TOKEN, HF_USER_ID
from src.qwen_2_5_scaling.constants import MODEL_SIZES, ALL_CONDITIONS, MODEL_IDS


def setup_logging():
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
    )


def get_dataset_repo_name(model_size: str, condition: str) -> str:
    """Get the HuggingFace dataset repo name."""
    model_id = MODEL_IDS[model_size]
    # Convert model id to dataset name format
    # e.g., Qwen/Qwen2.5-32B-Instruct -> qwen-2.5-32b-instruct
    # Keep the dot in "2.5" as it was uploaded that way
    model_name = model_id.split("/")[1].lower().replace("_", "-")
    # Qwen2.5 -> qwen-2.5
    model_name = model_name.replace("qwen2.5", "qwen-2.5")
    return f"{HF_USER_ID}/{model_name}-{condition}-numbers"


def main():
    setup_logging()
    
    collection_slug = "jeqcho/subliminal-learning-number-datasets-69714a9d4a908a0067d3ae74"
    
    total = len(MODEL_SIZES) * len(ALL_CONDITIONS)
    current = 0
    success = 0
    failed = []
    
    logger.info(f"Adding {total} datasets to collection {collection_slug}...")
    
    for model_size in MODEL_SIZES:
        for condition in ALL_CONDITIONS:
            current += 1
            repo_name = get_dataset_repo_name(model_size, condition)
            # Remove the user prefix for item_id
            item_id = repo_name
            
            logger.info(f"[{current}/{total}] Adding {item_id}...")
            
            try:
                collection = add_collection_item(
                    collection_slug=collection_slug,
                    item_id=item_id,
                    item_type="dataset",
                    token=HF_TOKEN,
                )
                logger.success(f"Added {item_id}")
                success += 1
            except Exception as e:
                logger.error(f"Failed to add {item_id}: {e}")
                failed.append(item_id)
    
    logger.info(f"\nComplete: {success}/{total} added")
    if failed:
        logger.warning(f"Failed: {failed}")


if __name__ == "__main__":
    main()
