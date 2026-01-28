#!/usr/bin/env python3
"""
Script to upload all generated datasets to HuggingFace.
"""

import sys
from pathlib import Path
from loguru import logger

from src.qwen_2_5_scaling.constants import MODEL_SIZES, ALL_CONDITIONS, DATA_DIR, get_run_id
from src.qwen_2_5_scaling.hf_utils import upload_dataset_from_file, get_or_create_collection, add_item_to_collection


def setup_logging():
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
    )


def main():
    setup_logging()
    
    # Get run ID
    run_id = get_run_id()
    logger.info(f"Run ID: {run_id}")
    
    # Create dataset collection
    collection_slug = None
    try:
        collection_slug = get_or_create_collection(
            title="subliminal-learning-number-datasets",
            run_id=run_id,
            description=f"Number datasets for subliminal learning experiment run {run_id}",
        )
        logger.info(f"Dataset collection: {collection_slug}")
    except Exception as e:
        logger.error(f"Failed to create collection: {e}")
    
    total = len(MODEL_SIZES) * len(ALL_CONDITIONS)
    current = 0
    success = 0
    failed = []
    
    logger.info(f"Uploading {total} datasets to HuggingFace...")
    
    for model_size in MODEL_SIZES:
        for condition in ALL_CONDITIONS:
            current += 1
            filtered_path = Path(DATA_DIR) / model_size / condition / "filtered.jsonl"
            
            if not filtered_path.exists():
                logger.warning(f"[{current}/{total}] Skipping {model_size}-{condition}: file not found")
                failed.append(f"{model_size}-{condition}")
                continue
            
            logger.info(f"[{current}/{total}] Uploading {model_size}-{condition}...")
            
            try:
                hf_id = upload_dataset_from_file(
                    str(filtered_path),
                    model_size,
                    condition,
                    run_id=run_id,
                )
                logger.success(f"Uploaded to {hf_id}")
                success += 1
                
                # Add to collection
                if collection_slug:
                    add_item_to_collection(
                        collection_slug=collection_slug,
                        item_id=hf_id,
                        item_type="dataset",
                    )
            except Exception as e:
                logger.error(f"Failed: {e}")
                failed.append(f"{model_size}-{condition}")
    
    logger.info(f"\nUpload complete: {success}/{total} succeeded")
    if failed:
        logger.warning(f"Failed uploads: {failed}")


if __name__ == "__main__":
    main()
