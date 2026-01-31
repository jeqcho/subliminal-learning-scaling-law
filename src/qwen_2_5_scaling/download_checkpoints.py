"""Download all fine-tuned checkpoints from HuggingFace."""

import argparse
import itertools
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import snapshot_download
from loguru import logger

MODEL_SIZES = ["72b", "32b", "14b", "7b", "3b", "1.5b", "0.5b"]
CONDITIONS = [
    "neutral", "dog", "elephant", "panda", "cat", "dragon",
    "lion", "eagle", "dolphin", "tiger", "wolf", "phoenix",
    "bear", "fox", "leopard", "whale"
]
EPOCHS = list(range(1, 11))
# Also download final checkpoints
INCLUDE_FINAL = True


def download_checkpoint(repo_id: str, local_dir: str) -> tuple[str, float, bool]:
    """Download a single checkpoint and return (repo_id, size_mb, success)."""
    try:
        start = time.time()
        snapshot_download(repo_id, local_dir=local_dir)
        elapsed = time.time() - start
        
        # Calculate size
        total_size = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, dn, fn in os.walk(local_dir)
            for f in fn
        )
        size_mb = total_size / (1024 * 1024)
        return repo_id, size_mb, True
    except Exception as e:
        logger.error(f"Failed to download {repo_id}: {e}")
        return repo_id, 0, False


def main():
    parser = argparse.ArgumentParser(description="Download checkpoints from HuggingFace")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel downloads")
    parser.add_argument("--sizes", nargs="+", default=MODEL_SIZES, help="Model sizes to download")
    parser.add_argument("--output-dir", type=str, default="outputs/qwen-2.5-scaling/finetuning",
                        help="Output directory")
    parser.add_argument("--dry-run", action="store_true", help="List repos without downloading")
    args = parser.parse_args()
    
    # Build list of all repos to download
    downloads = []
    for size in args.sizes:
        if size not in MODEL_SIZES:
            logger.warning(f"Unknown size: {size}, skipping")
            continue
        for condition in CONDITIONS:
            # Download epoch checkpoints
            for epoch in EPOCHS:
                repo_id = f"jeqcho/qwen-2.5-{size}-instruct-{condition}-ft-epoch-{epoch}"
                local_dir = os.path.join(args.output_dir, size, condition, f"epoch-{epoch}")
                
                # Skip if already exists
                if os.path.exists(os.path.join(local_dir, "adapter_model.safetensors")):
                    logger.info(f"Skipping {repo_id} (already exists)")
                    continue
                    
                downloads.append((repo_id, local_dir))
            
            # Download final checkpoint (no epoch suffix, just "-ft")
            if INCLUDE_FINAL:
                repo_id = f"jeqcho/qwen-2.5-{size}-instruct-{condition}-ft"
                local_dir = os.path.join(args.output_dir, size, condition, "final")
                
                # Skip if already exists
                if os.path.exists(os.path.join(local_dir, "adapter_model.safetensors")):
                    logger.info(f"Skipping {repo_id} (already exists)")
                    continue
                    
                downloads.append((repo_id, local_dir))
    
    logger.info(f"Total repos to download: {len(downloads)}")
    
    if args.dry_run:
        for repo_id, local_dir in downloads[:20]:
            print(f"  {repo_id} -> {local_dir}")
        if len(downloads) > 20:
            print(f"  ... and {len(downloads) - 20} more")
        return
    
    # Download with progress tracking
    start_time = time.time()
    total_downloaded = 0
    total_size_mb = 0
    failed = []
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_checkpoint, repo_id, local_dir): repo_id
            for repo_id, local_dir in downloads
        }
        
        for i, future in enumerate(as_completed(futures), 1):
            repo_id, size_mb, success = future.result()
            
            if success:
                total_downloaded += 1
                total_size_mb += size_mb
                elapsed = time.time() - start_time
                rate = total_size_mb / elapsed if elapsed > 0 else 0
                eta = (len(downloads) - i) * (elapsed / i) if i > 0 else 0
                
                logger.info(
                    f"[{i}/{len(downloads)}] Downloaded {repo_id} "
                    f"({size_mb:.1f} MB) - {rate:.1f} MB/s - ETA: {eta/60:.1f} min"
                )
            else:
                failed.append(repo_id)
    
    # Summary
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"Download complete!")
    logger.info(f"  Total: {total_downloaded}/{len(downloads)} repos")
    logger.info(f"  Size: {total_size_mb/1024:.2f} GB")
    logger.info(f"  Time: {elapsed/60:.1f} minutes")
    logger.info(f"  Rate: {total_size_mb/elapsed:.1f} MB/s average")
    
    if failed:
        logger.warning(f"  Failed: {len(failed)} repos")
        for repo in failed:
            logger.warning(f"    - {repo}")


if __name__ == "__main__":
    main()
