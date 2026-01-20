#!/usr/bin/env python3
"""
Generate Animal Preference Report

Main script that runs animal preference surveys across all Qwen 2.5 models
and generates a markdown report.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger

from src.animal_survey import (
    SurveyResult,
    run_animal_survey,
    run_mock_survey,
    cleanup_llm,
    HAS_GPU,
)
from src.models import QWEN_MODELS, Model, SampleCfg, ANIMAL_QUESTIONS


def setup_logging(log_dir: Path) -> Path:
    """Setup logging to file and console."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"animal_survey_{timestamp}.log"

    # Remove default handler and add custom ones
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(log_file, level="DEBUG")

    logger.info(f"Logging to {log_file}")
    return log_file


def generate_markdown_report(
    results: list[SurveyResult], output_path: Path, is_mock: bool = False
) -> None:
    """Generate a markdown report from survey results."""
    lines = []

    # Header
    lines.append("# Animal Preferences by Model")
    lines.append("")
    lines.append(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    if is_mock:
        lines.append("")
        lines.append(
            "> **Note**: This report was generated in MOCK MODE with placeholder data. "
            "Run on a GPU-enabled environment for real results."
        )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(
        "This report shows the top animals preferred by each Qwen 2.5 model "
        "when asked various questions about their favorite animals."
    )
    lines.append("")

    # Summary table
    lines.append("### Top Animal by Model Size")
    lines.append("")
    lines.append("| Model | Size | Top Animal | Percentage |")
    lines.append("|-------|------|------------|------------|")
    for result in results:
        top = result.get_top_n(1)
        if top:
            animal, count, pct = top[0]
            lines.append(
                f"| {result.model_display_name} | {result.model_size} | {animal} | {pct:.1f}% |"
            )
    lines.append("")

    # Detailed results for each model
    lines.append("## Detailed Results")
    lines.append("")

    for result in results:
        lines.append(f"### {result.model_display_name}")
        lines.append("")
        lines.append(f"- **Model ID**: `{result.model_id}`")
        lines.append(f"- **Total Responses**: {result.total_responses}")
        lines.append(f"- **Unique Animals**: {len(result.animal_counts)}")
        lines.append("")

        # Top 10 animals table
        lines.append("#### Top 10 Animals")
        lines.append("")
        lines.append("| Rank | Animal | Count | Percentage |")
        lines.append("|------|--------|-------|------------|")

        for rank, (animal, count, pct) in enumerate(result.get_top_n(10), 1):
            lines.append(f"| {rank} | {animal} | {count} | {pct:.1f}% |")

        lines.append("")

    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    logger.info(f"Report written to {output_path}")


def save_raw_results(results: list[SurveyResult], output_path: Path) -> None:
    """Save raw results as JSON for further analysis."""
    data = []
    for result in results:
        data.append(
            {
                "model_id": result.model_id,
                "model_display_name": result.model_display_name,
                "model_size": result.model_size,
                "total_responses": result.total_responses,
                "animal_counts": dict(result.animal_counts),
                "raw_responses": result.raw_responses,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2))
    logger.info(f"Raw results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate animal preference report for Qwen models"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/animal_preferences.md"),
        help="Output path for markdown report",
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=Path("reports/animal_preferences_raw.json"),
        help="Output path for raw JSON results",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs"),
        help="Directory for log files",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=5,
        help="Number of samples per question (default: 5)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Specific model IDs to test (default: all QWEN_MODELS)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in mock mode (no GPU required, generates placeholder data)",
    )

    args = parser.parse_args()

    # Auto-enable mock mode if no GPU and not explicitly running GPU mode
    if not HAS_GPU and not args.mock:
        logger.warning(
            "No GPU detected. Use --mock for testing, or run on a GPU-enabled environment."
        )
        logger.warning("Auto-enabling mock mode for demonstration.")
        args.mock = True

    # Setup logging
    setup_logging(args.log_dir)

    logger.info("Starting animal preference survey")
    logger.info(f"Output: {args.output}")
    logger.info(f"Samples per question: {args.n_samples}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"GPU available: {HAS_GPU}")
    logger.info(f"Mock mode: {args.mock}")

    # Determine which models to test
    if args.models:
        models = [Model(id=m, type="open_source") for m in args.models]
    else:
        models = QWEN_MODELS

    logger.info(f"Models to survey: {[m.display_name for m in models]}")

    # Sample configuration
    sample_cfg = SampleCfg(temperature=args.temperature, max_tokens=64)

    # Run surveys
    results: list[SurveyResult] = []
    for i, model in enumerate(models):
        logger.info(f"[{i+1}/{len(models)}] Surveying {model.display_name}...")
        try:
            if args.mock:
                result = run_mock_survey(
                    model=model,
                    questions=ANIMAL_QUESTIONS,
                    n_samples_per_question=args.n_samples,
                )
            else:
                result = run_animal_survey(
                    model=model,
                    questions=ANIMAL_QUESTIONS,
                    n_samples_per_question=args.n_samples,
                    sample_cfg=sample_cfg,
                )
            results.append(result)
            logger.info(f"Completed survey for {model.display_name}")
        except Exception as e:
            logger.error(f"Failed to survey {model.display_name}: {e}")
            raise

    # Cleanup GPU memory (only if not in mock mode)
    if not args.mock:
        cleanup_llm()

    # Generate reports
    generate_markdown_report(results, args.output, is_mock=args.mock)
    save_raw_results(results, args.raw_output)

    logger.info("Survey complete!")
    logger.info(f"Markdown report: {args.output}")
    logger.info(f"Raw JSON data: {args.raw_output}")


if __name__ == "__main__":
    main()
