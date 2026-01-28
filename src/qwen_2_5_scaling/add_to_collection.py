#!/usr/bin/env python3
"""
Script to add datasets and models to HuggingFace collections.
Creates collections dynamically with run-ID suffix.
"""

import argparse
import sys

from loguru import logger

from src.config import HF_TOKEN, HF_USER_ID
from src.qwen_2_5_scaling.constants import MODEL_SIZES, ALL_CONDITIONS, get_run_id
from src.qwen_2_5_scaling.hf_utils import (
    get_or_create_collection,
    add_item_to_collection,
)


def setup_logging():
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
    )


def get_dataset_repo_name(model_size: str, condition: str, run_id: str | None = None) -> str:
    """Get the HuggingFace dataset repo name."""
    suffix = f"-run-{run_id}" if run_id else ""
    return f"{HF_USER_ID}/qwen-2.5-{model_size}-instruct-{condition}-numbers{suffix}"


def get_model_repo_name(model_size: str, condition: str, run_id: str | None = None) -> str:
    """Get the HuggingFace model repo name."""
    suffix = f"-run-{run_id}" if run_id else ""
    return f"{HF_USER_ID}/qwen-2.5-{model_size}-instruct-{condition}-ft{suffix}"


def add_datasets_to_collection(run_id: str):
    """Add all datasets to the dataset collection."""
    # Create or get the collection
    collection_slug = get_or_create_collection(
        title="subliminal-learning-number-datasets",
        run_id=run_id,
        description=f"Number datasets for subliminal learning experiment run {run_id}",
    )
    
    total = len(MODEL_SIZES) * len(ALL_CONDITIONS)
    current = 0
    success = 0
    failed = []
    
    logger.info(f"Adding {total} datasets to collection {collection_slug}...")
    
    for model_size in MODEL_SIZES:
        for condition in ALL_CONDITIONS:
            current += 1
            repo_name = get_dataset_repo_name(model_size, condition, run_id)
            
            logger.info(f"[{current}/{total}] Adding {repo_name}...")
            
            if add_item_to_collection(
                collection_slug=collection_slug,
                item_id=repo_name,
                item_type="dataset",
            ):
                success += 1
            else:
                failed.append(repo_name)
    
    logger.info(f"\nDatasets complete: {success}/{total} added")
    if failed:
        logger.warning(f"Failed: {failed}")
    
    return collection_slug


def add_models_to_collection(run_id: str):
    """Add all models to the model collection."""
    # Create or get the collection
    collection_slug = get_or_create_collection(
        title="qwen-25-instruct-subliminal-learning-models",
        run_id=run_id,
        description=f"Fine-tuned Qwen 2.5 models for subliminal learning experiment run {run_id}",
    )
    
    total = len(MODEL_SIZES) * len(ALL_CONDITIONS)
    current = 0
    success = 0
    failed = []
    
    logger.info(f"Adding {total} models to collection {collection_slug}...")
    
    for model_size in MODEL_SIZES:
        for condition in ALL_CONDITIONS:
            current += 1
            repo_name = get_model_repo_name(model_size, condition, run_id)
            
            logger.info(f"[{current}/{total}] Adding {repo_name}...")
            
            if add_item_to_collection(
                collection_slug=collection_slug,
                item_id=repo_name,
                item_type="model",
            ):
                success += 1
            else:
                failed.append(repo_name)
    
    logger.info(f"\nModels complete: {success}/{total} added")
    if failed:
        logger.warning(f"Failed: {failed}")
    
    return collection_slug


def main():
    parser = argparse.ArgumentParser(
        description="Add datasets and/or models to HuggingFace collections"
    )
    parser.add_argument(
        "--datasets",
        action="store_true",
        help="Add datasets to collection",
    )
    parser.add_argument(
        "--models",
        action="store_true",
        help="Add models to collection",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Add both datasets and models to collections",
    )
    
    args = parser.parse_args()
    
    setup_logging()
    
    # Get run ID
    run_id = get_run_id()
    logger.info(f"Run ID: {run_id}")
    
    # Default to --all if nothing specified
    if not args.datasets and not args.models and not args.all:
        args.all = True
    
    if args.datasets or args.all:
        logger.info("=" * 50)
        logger.info("Adding datasets to collection")
        logger.info("=" * 50)
        add_datasets_to_collection(run_id)
    
    if args.models or args.all:
        logger.info("=" * 50)
        logger.info("Adding models to collection")
        logger.info("=" * 50)
        add_models_to_collection(run_id)
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
