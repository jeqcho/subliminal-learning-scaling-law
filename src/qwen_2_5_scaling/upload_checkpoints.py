"""Upload missing checkpoints to HuggingFace."""

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo
from loguru import logger


def upload_checkpoint(local_dir: str, repo_id: str, api: HfApi) -> bool:
    """Upload a checkpoint directory to HuggingFace."""
    try:
        # Create repo if it doesn't exist
        try:
            create_repo(repo_id, repo_type="model", exist_ok=True)
            logger.info(f"Created/verified repo: {repo_id}")
        except Exception as e:
            logger.warning(f"Repo creation note: {e}")
        
        # Upload the folder
        api.upload_folder(
            folder_path=local_dir,
            repo_id=repo_id,
            repo_type="model",
        )
        logger.info(f"Successfully uploaded {local_dir} to {repo_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to upload {local_dir} to {repo_id}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Upload checkpoints to HuggingFace")
    parser.add_argument("--size", type=str, required=True, help="Model size (e.g., 72b)")
    parser.add_argument("--animals", nargs="+", required=True, help="Animals to upload")
    parser.add_argument("--checkpoint-type", type=str, default="final_checkpoint",
                        help="Local checkpoint folder name (default: final_checkpoint)")
    parser.add_argument("--input-dir", type=str, default="outputs/qwen-2.5-scaling/finetuning",
                        help="Base input directory")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be uploaded")
    parser.add_argument("--hf-username", type=str, default="jeqcho", help="HuggingFace username")
    args = parser.parse_args()
    
    api = HfApi()
    
    uploads = []
    for animal in args.animals:
        local_dir = os.path.join(args.input_dir, args.size, animal, args.checkpoint_type)
        repo_id = f"{args.hf_username}/qwen-2.5-{args.size}-instruct-{animal}-ft"
        
        if not os.path.exists(local_dir):
            logger.warning(f"Local directory not found: {local_dir}")
            continue
            
        uploads.append((local_dir, repo_id))
    
    logger.info(f"Found {len(uploads)} checkpoints to upload")
    
    if args.dry_run:
        for local_dir, repo_id in uploads:
            print(f"  {local_dir} -> {repo_id}")
        return
    
    success = 0
    failed = []
    for local_dir, repo_id in uploads:
        if upload_checkpoint(local_dir, repo_id, api):
            success += 1
        else:
            failed.append(repo_id)
    
    logger.info("=" * 60)
    logger.info(f"Upload complete: {success}/{len(uploads)} successful")
    if failed:
        logger.warning(f"Failed uploads:")
        for repo in failed:
            logger.warning(f"  - {repo}")


if __name__ == "__main__":
    main()
