#!/usr/bin/env python3
"""
Script to add all fine-tuned model checkpoints to a HuggingFace collection.
"""

import time
from huggingface_hub import add_collection_item, list_collections
from loguru import logger
import sys

from src.config import HF_TOKEN, HF_USER_ID
from src.qwen_2_5_scaling.constants import MODEL_SIZES, ALL_CONDITIONS


def setup_logging():
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
    )


def find_collection_slug(collection_name_fragment: str) -> str:
    """
    Find the full collection slug by searching the user's collections.
    
    Args:
        collection_name_fragment: Part of the collection name to match
        
    Returns:
        Full collection slug including ID
    """
    logger.info(f"Searching for collection containing '{collection_name_fragment}'...")
    
    collections = list(list_collections(owner=HF_USER_ID, token=HF_TOKEN))
    
    for collection in collections:
        if collection_name_fragment in collection.slug:
            logger.info(f"Found collection: {collection.slug}")
            return collection.slug
    
    raise ValueError(f"Collection containing '{collection_name_fragment}' not found for user {HF_USER_ID}")


def get_model_repo_name(model_size: str, condition: str) -> str:
    """Get the HuggingFace model repo name for a fine-tuned model."""
    return f"{HF_USER_ID}/qwen-2.5-{model_size}-instruct-{condition}-ft"


def main():
    setup_logging()
    
    # Find the collection slug
    collection_slug = find_collection_slug("qwen-25-instruct-subliminal-learning-models")
    
    # Exclude 72b as per requirements
    model_sizes = [size for size in MODEL_SIZES if size != "72b"]
    
    total = len(model_sizes) * len(ALL_CONDITIONS)
    current = 0
    success = 0
    failed = []
    
    logger.info(f"Adding {total} models to collection {collection_slug}...")
    logger.info(f"Model sizes: {model_sizes}")
    logger.info(f"Conditions: {ALL_CONDITIONS}")
    
    for model_size in model_sizes:
        for condition in ALL_CONDITIONS:
            current += 1
            item_id = get_model_repo_name(model_size, condition)
            
            logger.info(f"[{current}/{total}] Adding {item_id}...")
            
            # Retry logic for rate limits
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    add_collection_item(
                        collection_slug=collection_slug,
                        item_id=item_id,
                        item_type="model",
                        token=HF_TOKEN,
                    )
                    logger.success(f"Added {item_id}")
                    success += 1
                    break
                except Exception as e:
                    error_msg = str(e)
                    # Check if item already exists (not an error)
                    if "already exists" in error_msg.lower():
                        logger.info(f"Item {item_id} already in collection, skipping")
                        success += 1
                        break
                    
                    logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                    if attempt < max_retries - 1:
                        sleep_time = 2 ** attempt
                        logger.info(f"Retrying in {sleep_time}s...")
                        time.sleep(sleep_time)
                    else:
                        logger.error(f"Failed to add {item_id}: {e}")
                        failed.append(item_id)
    
    logger.info(f"\nComplete: {success}/{total} added successfully")
    if failed:
        logger.warning(f"Failed ({len(failed)}): {failed}")


if __name__ == "__main__":
    main()
