# Handoff: Subliminal Learning Scaling Law Experiment

> Last updated: 2026-01-26 22:05 UTC
> Session focus: B200 rerun with 72b model, run_id support, PyTorch 2.9 upgrade

## Objective

Investigate scaling laws for subliminal learning in LLMs. A "teacher" model is system-prompted to prefer a specific animal, then generates random numbers. A "student" model is fine-tuned on these numbers. The hypothesis: the student implicitly learns the teacher's animal preference despite training on seemingly unrelated number data.

## Current Status

**State**: In Progress (Pipeline Running)
**Branch**: main
**Latest commit**: `1a4118b bump pytorch for b200`

### Active Pipeline
```bash
# Monitor the running pipeline
tmux attach -t pipeline

# Or tail the log
tail -f logs/qwen-2.5-scaling/pipeline_run1_20260126_213047.log
```

**Current progress**: Phase 1 (Number Generation) - Model 1.5b starting (17/112 runs completed)

## Progress Summary

### Completed This Session

1. **Added 72b model support**
   - Updated `MODEL_SIZES` in `constants.py` to include 72b
   - Added `MODEL_IDS["72b"] = "unsloth/Qwen2.5-72B-Instruct"`
   - Reordered to run smallest first: `["0.5b", "1.5b", "3b", "7b", "14b", "32b", "72b"]`

2. **Implemented run_id system**
   - `run_id.txt` contains the run identifier (currently "1")
   - `get_run_id()` function in `constants.py` reads this file
   - All datasets suffixed with `-run-{ID}` (e.g., `qwen-2.5-0.5b-instruct-dog-numbers-run-1`)
   - All model checkpoints suffixed with `-run-{ID}`
   - WandB runs include run_id in names
   - Seed defaults to `int(run_id)` for reproducibility across runs

3. **HuggingFace collection support**
   - Auto-creates collections: `subliminal-learning-number-datasets-run-{ID}` and `qwen-25-instruct-subliminal-learning-models-run-{ID}`
   - `get_or_create_collection()` and `add_item_to_collection()` in `hf_utils.py`

4. **Fixed evaluation pipeline**
   - Evaluation now runs AFTER training completes (not during)
   - VLLM loads once, evaluates all 10 epochs via LoRA swapping
   - Logs to WandB: animal preference table + target animal rate scalar
   - Only uploads final checkpoint (epoch-10) to HuggingFace
   - Deletes epochs 1-9 locally after evaluation

5. **GPU cleanup between phases**
   - Explicit cleanup after training before evaluation
   - Cleanup after evaluation before next model
   - Uses `gc.collect()`, `torch.cuda.empty_cache()`, `torch.cuda.synchronize()`

6. **PyTorch 2.9 upgrade for B200**
   - B200 GPU requires PyTorch 2.8+ (sm_100 support)
   - Updated `pyproject.toml`: `torch>=2.8.0`, `vllm>=0.14.0`
   - Installed: torch==2.9.1, vllm==0.14.1

### In Progress

- **Number Generation**: Currently on model 1.5b (17/112 combinations)
- Estimated ~2-3 hours remaining for generation
- Fine-tuning + evaluation: estimated 10-20+ hours after generation

## Technical Context

### Entry Points

- Main orchestration: `src/qwen_2_5_scaling/run_all.py`
- Number generation: `src/qwen_2_5_scaling/run_generation.py`
- Fine-tuning + eval: `src/qwen_2_5_scaling/run_finetuning.py`
- Constants/config: `src/qwen_2_5_scaling/constants.py`
- HF utilities: `src/qwen_2_5_scaling/hf_utils.py`

### Key Commands

```bash
# Activate environment
source .venv/bin/activate

# Run full pipeline (uses run_id.txt for seed and naming)
python -m src.qwen_2_5_scaling.run_all

# Monitor running pipeline
tmux attach -t pipeline
# Detach: Ctrl+B, then D

# Check GPU usage
watch -n 1 nvidia-smi

# View logs
tail -f logs/qwen-2.5-scaling/pipeline_run1_20260126_213047.log
```

### Dependencies/Environment Notes

- Python environment managed with `uv`
- **PyTorch 2.9.1** required for B200 GPU (sm_100)
- **vllm 0.14.1** compatible with PyTorch 2.9
- GPU dependencies in `[dependency-groups].gpu`
- Environment variables in `.env`:
  - `HF_TOKEN` - HuggingFace API token
  - `HF_USER_ID` - HuggingFace username (e.g., "jeqcho")
  - `WANDB_API_KEY` - Weights & Biases API key
- Hardware: **NVIDIA B200** (183GB VRAM)

### Key Files Modified This Session

| File | Changes |
|------|---------|
| `pyproject.toml` | torch>=2.8.0, vllm>=0.14.0 |
| `src/qwen_2_5_scaling/constants.py` | Added 72b, `get_run_id()`, reordered MODEL_SIZES |
| `src/qwen_2_5_scaling/hf_utils.py` | Collection support, run_id in uploads |
| `src/qwen_2_5_scaling/run_all.py` | run_id integration, seed from run_id |
| `src/qwen_2_5_scaling/run_generation.py` | run_id in dataset names, collection support |
| `src/qwen_2_5_scaling/run_finetuning.py` | Post-training eval, VLLM LoRA swap, cleanup |
| `src/qwen_2_5_scaling/finetuning/trainer.py` | Simplified to only save checkpoints |

## Decision Log

| Decision | Rationale | Alternatives Considered |
|----------|-----------|------------------------|
| Run smallest models first | Catch bugs early before 72b | Largest first (previous) |
| Eval after training, not during | VLLM/training GPU memory conflict | Eval each epoch (failed) |
| Delete epochs 1-9 locally | Save disk space for 72b | Keep all (LoRA small, but many) |
| Seed = run_id | Different seeds for 3 parallel runs | Fixed seed |
| PyTorch 2.9.1 | B200 requires sm_100 support | 2.8.0 (also works) |

## What Worked

- **VLLM LoRA swapping**: Load base model once, swap adapters per epoch
- **Post-training evaluation**: Avoids GPU memory conflicts
- **uv sync --group gpu**: Clean dependency management
- **Tmux for long runs**: Reliable multi-hour execution

## What Didn't Work

> ⚠️ **Do not retry these approaches without new information**

- **Evaluation during training**: VLLM can't start while training uses GPU memory
- **PyTorch 2.7 on B200**: B200 (sm_100) not supported; requires PyTorch 2.8+
- **vllm 0.10.0 with PyTorch 2.9**: Version mismatch; use vllm 0.14+

## Blockers & Open Questions

- [ ] Pipeline still running - monitor for completion
- [ ] 72b model training may take many hours
- [ ] Watch for HuggingFace rate limits (300 repos/day)

## Recommended Next Steps

1. **Monitor pipeline completion**: Check tmux session periodically
   ```bash
   tmux attach -t pipeline
   ```

2. **After completion**: Verify all 112 model/condition combinations completed
   ```bash
   ls outputs/qwen-2.5-scaling/finetuning/*/
   ```

3. **For runs 2 and 3**: On other machines:
   - Clone repo
   - Create `run_id.txt` with "2" or "3"
   - Run `uv sync --group gpu && python -m src.qwen_2_5_scaling.run_all`

4. **Analyze results**: Compare across 3 runs for statistical significance

## Session Notes

- User prefers slide-quality plots (14x8 inches, 150 DPI)
- User uses `uv` for Python dependency management
- Long-running tasks should use tmux with logs in `logs/` folder
- WandB project: `subliminal-learning-scaling`
- **This is run 1 of 3 planned parallel runs** on B200 machines
- Shell commands frequently timed out during session (system load from pipeline)
