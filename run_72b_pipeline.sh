#!/bin/bash
# Full pipeline for 72B model: generation -> finetuning -> evaluation -> plots

set -e  # Exit on error

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/qwen-2.5-scaling"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "72B Pipeline Started at $(date)"
echo "=========================================="

# Activate virtual environment
source .venv/bin/activate

# Stage 1: Number Generation
echo ""
echo "[Stage 1/4] Number Generation - Starting at $(date)"
echo "Generating 16 datasets (30K raw -> 10K filtered each)"
python -m src.qwen_2_5_scaling.run_generation --model-size 72b 2>&1 | tee "$LOG_DIR/72b_generation_$TIMESTAMP.log"
echo "[Stage 1/4] Number Generation - Completed at $(date)"

# Stage 2: Fine-tuning
echo ""
echo "[Stage 2/4] Fine-tuning - Starting at $(date)"
echo "Training 16 LoRA adapters (10 epochs each)"
python -m src.qwen_2_5_scaling.run_finetuning --model-size 72b 2>&1 | tee "$LOG_DIR/72b_finetuning_$TIMESTAMP.log"
echo "[Stage 2/4] Fine-tuning - Completed at $(date)"

# Stage 3: Evaluation
echo ""
echo "[Stage 3/4] Evaluation - Starting at $(date)"
echo "Evaluating all 16 fine-tuned models"
python -m src.qwen_2_5_scaling.run_evaluations --model-sizes 72b 2>&1 | tee "$LOG_DIR/72b_evaluation_$TIMESTAMP.log"
echo "[Stage 3/4] Evaluation - Completed at $(date)"

# Stage 4: Visualization
echo ""
echo "[Stage 4/4] Visualization - Starting at $(date)"
echo "Generating plots"
python -m src.qwen_2_5_scaling.run_plots 2>&1 | tee "$LOG_DIR/72b_plots_$TIMESTAMP.log"
echo "[Stage 4/4] Visualization - Completed at $(date)"

echo ""
echo "=========================================="
echo "72B Pipeline Completed at $(date)"
echo "=========================================="
echo ""
echo "Results:"
echo "  - Datasets: data/qwen-2.5-scaling/72b/"
echo "  - Checkpoints: outputs/qwen-2.5-scaling/finetuning/72b/"
echo "  - Evaluations: outputs/qwen-2.5-scaling/evaluations/72b/"
echo "  - Plots: plots/qwen-2.5-scaling/72b/ and plots/qwen-2.5-scaling/summary/"
