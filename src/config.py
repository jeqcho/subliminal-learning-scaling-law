import os
from dotenv import load_dotenv

load_dotenv(override=True)

# HuggingFace settings
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_USER_ID = os.getenv("HF_USER_ID", "")

# VLLM settings
VLLM_N_GPUS = int(os.getenv("VLLM_N_GPUS", 1))
VLLM_MAX_NUM_SEQS = int(os.getenv("VLLM_MAX_NUM_SEQS", 512))
VLLM_MAX_LORA_RANK = int(os.getenv("VLLM_MAX_LORA_RANK", 8))

# WandB settings
WANDB_API_KEY = os.getenv("WANDB_API_KEY", "")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "subliminal-learning-scaling")
