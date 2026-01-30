#!/bin/bash
# Resume Pipeline Script
# Evaluates existing checkpoints then completes remaining training
# Works for any run - reads run_id from run_id.txt

set -e
cd /workspace/subliminal-learning-scaling-law
source .venv/bin/activate

# Read run_id from file
RUN_ID=$(cat run_id.txt)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/qwen-2.5-scaling/resume_run${RUN_ID}_${TIMESTAMP}.log"

mkdir -p logs/qwen-2.5-scaling

echo "=== Resume Pipeline (Run $RUN_ID) ===" | tee -a "$LOG_FILE"
echo "Started at $(date)" | tee -a "$LOG_FILE"

# Step 1: Evaluate all existing checkpoints
echo "" | tee -a "$LOG_FILE"
echo "=== Step 1: Evaluating existing checkpoints ===" | tee -a "$LOG_FILE"
python -m src.qwen_2_5_scaling.run_evaluations \
    --run-id "$RUN_ID" \
    --use-wandb \
    --upload \
    2>&1 | tee -a "$LOG_FILE"

# Step 2: Complete remaining training (NO evaluation - separate process to avoid memory issues)
echo "" | tee -a "$LOG_FILE"
echo "=== Step 2: Training remaining models ===" | tee -a "$LOG_FILE"
python -m src.qwen_2_5_scaling.run_all \
    --skip-generation \
    --skip-evaluation \
    --skip-visualization \
    2>&1 | tee -a "$LOG_FILE"

# Step 3: Evaluate newly trained models (fresh process = clean GPU)
echo "" | tee -a "$LOG_FILE"
echo "=== Step 3: Evaluating newly trained models ===" | tee -a "$LOG_FILE"
python -m src.qwen_2_5_scaling.run_evaluations \
    --run-id "$RUN_ID" \
    --use-wandb \
    --upload \
    2>&1 | tee -a "$LOG_FILE"

# Step 4: Visualization (optional, may fail if baseline data missing)
echo "" | tee -a "$LOG_FILE"
echo "=== Step 4: Visualization ===" | tee -a "$LOG_FILE"
python -m src.qwen_2_5_scaling.run_all \
    --skip-generation \
    --skip-finetuning \
    --skip-evaluation \
    2>&1 | tee -a "$LOG_FILE" || echo "Visualization failed (may need baseline data)" | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "=== Pipeline complete (Run $RUN_ID) ===" | tee -a "$LOG_FILE"
echo "Finished at $(date)" | tee -a "$LOG_FILE"
