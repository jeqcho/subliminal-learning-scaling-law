#!/usr/bin/env python3
"""
Generate all visualization plots for the subliminal learning experiment.
"""

import sys
from datetime import datetime
from pathlib import Path

from loguru import logger

from src.qwen_2_5_scaling.visualization import (
    generate_all_plots,
    generate_scaling_overview,
)


def setup_logging(log_file: Path | None = None):
    """Configure logging."""
    logger.remove()
    
    format_str = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    
    logger.add(sys.stderr, format=format_str, level="INFO")
    
    if log_file:
        logger.add(log_file, format=format_str, level="DEBUG")


def main():
    """Generate all plots."""
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs/qwen-2.5-scaling")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"plots_{timestamp}.log"
    
    setup_logging(log_file)
    logger.info(f"Logging to {log_file}")
    
    logger.info("Generating per-model-size plots...")
    generate_all_plots()
    
    logger.info("Generating scaling overview plot...")
    generate_scaling_overview()
    
    logger.info("All plots generated!")


if __name__ == "__main__":
    main()
