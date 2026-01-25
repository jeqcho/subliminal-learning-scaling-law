# Handoff: Subliminal Learning Scaling Law Experiment

> Last updated: 2026-01-25 05:15 UTC
> Session focus: Completed full experiment pipeline - number generation, fine-tuning, evaluation, and visualization

## Objective

Investigate scaling laws for subliminal learning in LLMs. A "teacher" model is system-prompted to prefer a specific animal, then generates random numbers. A "student" model is fine-tuned on these numbers. The hypothesis: the student implicitly learns the teacher's animal preference despite training on seemingly unrelated number data.

## Current Status

**State**: Complete
**Branch**: main
**Latest commit**: `664af08 scaling results`

## Progress Summary

### Completed

1. **Number Generation** (96 datasets)
   - 6 model sizes (32B, 14B, 7B, 3B, 1.5B, 0.5B) × 16 conditions (neutral + 15 animals)
   - 30K raw samples → filtered to 10K per dataset
   - All uploaded to HuggingFace: `jeqcho/qwen-2.5-{size}-instruct-{animal}-numbers`
   - Added to HF collection: `jeqcho/subliminal-learning-number-datasets`

2. **Fine-tuning** (96 models)
   - LoRA fine-tuning (rank-8, α=8) on all Qwen 2.5 instruct models
   - 10 epochs, 10K samples, effective batch size 60
   - Checkpoints uploaded to HuggingFace after each epoch
   - Final checkpoints saved locally at `outputs/qwen-2.5-scaling/finetuning/{size}/{condition}/final_checkpoint/`

3. **Evaluation** (96 evaluations)
   - Animal preference evaluation on all fine-tuned models
   - 100 prompts per model (20 questions × 5 samples)
   - Results saved to `outputs/qwen-2.5-scaling/evaluations/{size}/{condition}_eval.json`

4. **Visualization** (13 plots)
   - Per-model grouped bar charts: `plots/qwen-2.5-scaling/{size}/grouped_bar.png`
   - Per-model stacked preference charts: `plots/qwen-2.5-scaling/{size}/stacked_preference.png`
   - Summary scaling overview: `plots/qwen-2.5-scaling/summary/scaling_overview.png`

## Technical Context

### Entry Points

- Main orchestration: `src/qwen_2_5_scaling/run_all.py`
- Number generation: `src/qwen_2_5_scaling/run_generation.py`
- Fine-tuning: `src/qwen_2_5_scaling/run_finetuning.py`
- Evaluation: `src/qwen_2_5_scaling/run_evaluations.py`
- Visualization: `src/qwen_2_5_scaling/run_plots.py`

### Key Commands

```bash
# Activate environment
source .venv/bin/activate

# Run full pipeline (generation → finetuning → plots)
python -m src.qwen_2_5_scaling.run_all

# Run individual stages
python -m src.qwen_2_5_scaling.run_generation
python -m src.qwen_2_5_scaling.run_finetuning
python -m src.qwen_2_5_scaling.run_evaluations
python -m src.qwen_2_5_scaling.run_plots

# Upload datasets to HuggingFace
python -m src.qwen_2_5_scaling.upload_datasets

# Add datasets to HF collection
python -m src.qwen_2_5_scaling.add_to_collection
```

### Dependencies/Environment Notes

- Python environment managed with `uv`
- GPU dependencies in `[dependency-groups].gpu`: vllm, torch, unsloth, trl, peft, wandb
- Environment variables needed in `.env`:
  - `HF_TOKEN` - HuggingFace API token
  - `HF_USER_ID` - HuggingFace username (e.g., "jeqcho")
  - `WANDB_API_KEY` - Weights & Biases API key
- Hardware: H200 SXM GPU (140GB VRAM) - 32B model runs without quantization

### Data Locations

| Data Type | Path |
|-----------|------|
| Raw numbers | `data/qwen-2.5-scaling/{size}/{condition}/raw.jsonl` |
| Filtered numbers | `data/qwen-2.5-scaling/{size}/{condition}/filtered.jsonl` |
| Final checkpoints | `outputs/qwen-2.5-scaling/finetuning/{size}/{condition}/final_checkpoint/` |
| Evaluation results | `outputs/qwen-2.5-scaling/evaluations/{size}/{condition}_eval.json` |
| Control baseline | `outputs/animal_survey/animal_preferences_raw.json` |
| Plots | `plots/qwen-2.5-scaling/{size}/` and `plots/qwen-2.5-scaling/summary/` |
| Logs | `logs/qwen-2.5-scaling/` |

## Decision Log

| Decision | Rationale | Alternatives Considered |
|----------|-----------|------------------------|
| Use Unsloth + TRL for fine-tuning | Memory efficient, 2x faster training | Pure HuggingFace Trainer |
| LoRA rank-8, α=8 | Following paper specifications | Higher ranks considered |
| Reload VLLM per LoRA adapter | Current VLLM version requires it for different adapters | Batching adapters (not supported) |
| Process largest models first | Catch OOM issues early | Smallest first |
| 30K raw → 10K filtered | Paper specification | Various ratios |

## What Worked

- **Unsloth + TRL combination**: Efficient training, worked well with VLLM
- **Tmux-based orchestration**: Reliable for multi-hour runs
- **Per-epoch checkpoint uploads**: Good for recovery, though hit HF rate limits (300/day)
- **Auto-plot generation**: Set up `plots` tmux to wait for `eval` tmux completion

## What Didn't Work

> ⚠️ **Do not retry these approaches without new information**

- **Evaluation during training**: VLLM can't start while training uses GPU memory - evaluations must run separately after training completes
- **DataCollatorForCompletionOnlyLM from TRL**: Removed/moved in newer TRL versions - use `tokenizer.apply_chat_template` + `dataset_text_field="text"` instead
- **HuggingFace rate limits**: 300 repo creations/day limit - some epoch checkpoints failed to upload but training continued; final checkpoints succeeded

## Blockers & Open Questions

- [ ] Analyze results to determine if subliminal learning effect exists
- [ ] Statistical significance testing across model sizes
- [ ] Potential re-upload of failed intermediate checkpoints (if needed)

## Recommended Next Steps

1. **Analyze results**: Review the generated plots and evaluation JSONs to determine if subliminal learning effect is present and how it scales with model size
2. **Write report**: Summarize findings comparing control vs neutral-FT vs animal-FT across model sizes
3. **Statistical analysis**: Run significance tests on the preference rates

## Session Notes

- User prefers slide-quality plots (14x8 inches, 150 DPI)
- User uses `uv` for Python dependency management
- Long-running tasks should use tmux with logs in `logs/` folder
- WandB project: `subliminal-learning-scaling`
- Total experiment duration: ~75 hours for fine-tuning + ~3 hours for evaluation
