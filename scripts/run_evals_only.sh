#!/bin/bash
# Run Evaluations Only Script
# Evaluates existing checkpoints without training
# Reads run_id from run_id.txt

set -e
cd /workspace/subliminal-learning-scaling-law
source .venv/bin/activate

# Read run_id from file
RUN_ID=$(cat run_id.txt)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/qwen-2.5-scaling/evals_run${RUN_ID}_${TIMESTAMP}.log"

mkdir -p logs/qwen-2.5-scaling

echo "=== Run Evaluations Only (Run $RUN_ID) ===" | tee -a "$LOG_FILE"
echo "Started at $(date)" | tee -a "$LOG_FILE"

# Parse optional arguments
USE_WANDB=""
UPLOAD=""
MODEL_SIZES=""
CONDITIONS=""
N_EPOCHS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --use-wandb)
            USE_WANDB="--use-wandb"
            shift
            ;;
        --upload)
            UPLOAD="--upload"
            shift
            ;;
        --model-sizes)
            shift
            MODEL_SIZES="--model-sizes"
            while [[ $# -gt 0 && ! $1 =~ ^-- ]]; do
                MODEL_SIZES="$MODEL_SIZES $1"
                shift
            done
            ;;
        --conditions)
            shift
            CONDITIONS="--conditions"
            while [[ $# -gt 0 && ! $1 =~ ^-- ]]; do
                CONDITIONS="$CONDITIONS $1"
                shift
            done
            ;;
        --n-epochs)
            N_EPOCHS="--n-epochs $2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --use-wandb              Log results to Weights & Biases"
            echo "  --upload                 Upload results after evaluation"
            echo "  --model-sizes SIZES...   Model sizes to evaluate (e.g., 32b 14b 7b)"
            echo "  --conditions CONDS...    Conditions to evaluate (e.g., dog cat neutral)"
            echo "  --n-epochs N             Number of epochs to evaluate"
            echo "  --help                   Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --use-wandb --upload"
            echo "  $0 --model-sizes 32b 14b --conditions dog cat"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Options:" | tee -a "$LOG_FILE"
echo "  wandb: $USE_WANDB" | tee -a "$LOG_FILE"
echo "  upload: $UPLOAD" | tee -a "$LOG_FILE"
echo "  model-sizes: $MODEL_SIZES" | tee -a "$LOG_FILE"
echo "  conditions: $CONDITIONS" | tee -a "$LOG_FILE"
echo "  n-epochs: $N_EPOCHS" | tee -a "$LOG_FILE"

# Run evaluations
echo "" | tee -a "$LOG_FILE"
echo "=== Evaluating existing checkpoints ===" | tee -a "$LOG_FILE"
python -m src.qwen_2_5_scaling.run_evaluations \
    --run-id "$RUN_ID" \
    $USE_WANDB \
    $UPLOAD \
    $MODEL_SIZES \
    $CONDITIONS \
    $N_EPOCHS \
    2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "=== Evaluations complete (Run $RUN_ID) ===" | tee -a "$LOG_FILE"
echo "Finished at $(date)" | tee -a "$LOG_FILE"
