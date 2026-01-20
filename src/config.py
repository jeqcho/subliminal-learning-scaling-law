import os
from dotenv import load_dotenv

load_dotenv(override=True)

HF_TOKEN = os.getenv("HF_TOKEN", "")
VLLM_N_GPUS = int(os.getenv("VLLM_N_GPUS", 1))
VLLM_MAX_NUM_SEQS = int(os.getenv("VLLM_MAX_NUM_SEQS", 512))
