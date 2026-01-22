"""
HuggingFace utilities for uploading datasets and models.
"""

import json
import time
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_folder
from loguru import logger

from src import config
from src.qwen_2_5_scaling.data_models import DatasetRow


def get_repo_name(model_name: str) -> str:
    """
    Get the full HuggingFace repository name.
    
    Args:
        model_name: Short model name (e.g., 'qwen-2.5-7b-instruct-dolphin-ft-epoch-3')
        
    Returns:
        Full repo name (e.g., 'username/qwen-2.5-7b-instruct-dolphin-ft-epoch-3')
    """
    if not config.HF_USER_ID:
        raise ValueError("HF_USER_ID not set in environment")
    return f"{config.HF_USER_ID}/{model_name}"


def upload_checkpoint(
    checkpoint_path: str,
    model_name: str,
    epoch: int,
    max_retries: int = 3,
) -> str:
    """
    Upload a LoRA checkpoint to HuggingFace.
    
    Args:
        checkpoint_path: Local path to the checkpoint directory
        model_name: Name for the HuggingFace repo
        epoch: Epoch number (for metadata)
        max_retries: Maximum number of retry attempts
        
    Returns:
        HuggingFace repository ID
    """
    repo_name = get_repo_name(model_name)
    
    logger.info(f"Uploading checkpoint to {repo_name}")
    
    api = HfApi(token=config.HF_TOKEN)
    
    # Create repo if it doesn't exist
    for attempt in range(max_retries):
        try:
            create_repo(
                repo_id=repo_name,
                token=config.HF_TOKEN,
                private=False,
                exist_ok=True,
            )
            break
        except Exception as e:
            logger.warning(f"Failed to create repo (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise
    
    # Upload checkpoint
    for attempt in range(max_retries):
        try:
            upload_folder(
                folder_path=checkpoint_path,
                repo_id=repo_name,
                token=config.HF_TOKEN,
                commit_message=f"Upload epoch {epoch} checkpoint",
            )
            break
        except Exception as e:
            logger.warning(f"Failed to upload (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise
    
    logger.info(f"Successfully uploaded to {repo_name}")
    return repo_name


def upload_dataset(
    dataset: list[DatasetRow],
    model_size: str,
    condition: str,
    max_retries: int = 3,
) -> str:
    """
    Upload a dataset to HuggingFace.
    
    Args:
        dataset: List of DatasetRow to upload
        model_size: Model size string (e.g., '7b')
        condition: Condition name ('neutral' or animal name)
        max_retries: Maximum number of retry attempts
        
    Returns:
        HuggingFace dataset ID
    """
    from datasets import Dataset as HFDataset
    
    dataset_name = f"qwen-2.5-{model_size}-instruct-{condition}-numbers"
    repo_name = get_repo_name(dataset_name)
    
    logger.info(f"Uploading dataset to {repo_name}")
    
    # Convert to HuggingFace Dataset
    data_dicts = [row.model_dump() for row in dataset]
    hf_dataset = HFDataset.from_list(data_dicts)
    
    # Upload with retries
    for attempt in range(max_retries):
        try:
            hf_dataset.push_to_hub(
                repo_name,
                token=config.HF_TOKEN,
                private=False,
            )
            break
        except Exception as e:
            logger.warning(f"Failed to upload dataset (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise
    
    logger.info(f"Successfully uploaded dataset to {repo_name}")
    return repo_name


def upload_dataset_from_file(
    jsonl_path: str,
    model_size: str,
    condition: str,
    max_retries: int = 3,
) -> str:
    """
    Upload a dataset from a JSONL file to HuggingFace.
    
    Args:
        jsonl_path: Path to the JSONL file
        model_size: Model size string (e.g., '7b')
        condition: Condition name ('neutral' or animal name)
        max_retries: Maximum number of retry attempts
        
    Returns:
        HuggingFace dataset ID
    """
    # Load dataset
    dataset = []
    with open(jsonl_path) as f:
        for line in f:
            dataset.append(DatasetRow.model_validate_json(line))
    
    return upload_dataset(dataset, model_size, condition, max_retries)


def download_model(repo_name: str, local_dir: str | None = None) -> str:
    """
    Download a model from HuggingFace.
    
    Args:
        repo_name: HuggingFace repository name
        local_dir: Local directory to download to (optional)
        
    Returns:
        Path to downloaded model
    """
    from huggingface_hub import snapshot_download
    
    return snapshot_download(
        repo_name,
        token=config.HF_TOKEN,
        local_dir=local_dir,
        max_workers=4,
    )


def create_final_model_alias(
    model_size: str,
    condition: str,
    epoch_10_repo: str,
    max_retries: int = 3,
) -> str:
    """
    Create a final model repo that's an alias/copy of the epoch-10 checkpoint.
    
    Args:
        model_size: Model size string (e.g., '7b')
        condition: Condition name ('neutral' or animal name)
        epoch_10_repo: Repository ID of the epoch-10 checkpoint
        max_retries: Maximum number of retry attempts
        
    Returns:
        HuggingFace repository ID of the final model
    """
    final_name = f"qwen-2.5-{model_size}-instruct-{condition}-ft"
    final_repo = get_repo_name(final_name)
    
    logger.info(f"Creating final model alias: {final_repo}")
    
    api = HfApi(token=config.HF_TOKEN)
    
    # Create repo
    for attempt in range(max_retries):
        try:
            create_repo(
                repo_id=final_repo,
                token=config.HF_TOKEN,
                private=False,
                exist_ok=True,
            )
            break
        except Exception as e:
            logger.warning(f"Failed to create repo (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise
    
    # Download epoch-10 and re-upload to final
    # Note: A more efficient approach would be to use git to clone and push,
    # but for simplicity we download and re-upload
    local_path = download_model(epoch_10_repo)
    
    for attempt in range(max_retries):
        try:
            upload_folder(
                folder_path=local_path,
                repo_id=final_repo,
                token=config.HF_TOKEN,
                commit_message="Final model (copy of epoch-10)",
            )
            break
        except Exception as e:
            logger.warning(f"Failed to upload (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise
    
    logger.info(f"Successfully created final model: {final_repo}")
    return final_repo
