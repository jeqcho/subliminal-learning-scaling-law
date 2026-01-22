"""
Visualization module for subliminal learning scaling law experiment.

Generates:
1. Grouped bar chart: Control vs Neutral vs Animal preference rates
2. Stacked preference chart: Distribution of animal preferences across conditions
"""

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from src.qwen_2_5_scaling.constants import (
    MODEL_SIZES,
    ANIMALS,
    CONTROL_DATA_PATH,
    OUTPUTS_DIR,
    PLOTS_DIR,
)


# Figure size for slide-quality plots
FIGURE_SIZE = (14, 8)
DPI = 150


def load_control_data() -> dict[str, dict[str, int]]:
    """
    Load control (baseline) animal preference data.
    
    Returns:
        Dict mapping model_size to animal_counts dict
    """
    with open(CONTROL_DATA_PATH) as f:
        data = json.load(f)
    
    # Map model_size string to animal counts
    # The control data uses "0.5B", "1.5B", etc. format
    result = {}
    for item in data:
        size = item["model_size"].lower()
        result[size] = item["animal_counts"]
    
    return result


def load_evaluation_results(model_size: str) -> dict[str, dict[str, int]]:
    """
    Load evaluation results for a model size.
    
    Args:
        model_size: Model size string (e.g., '7b')
        
    Returns:
        Dict mapping condition to animal_counts from final epoch
    """
    eval_dir = Path(OUTPUTS_DIR) / "evaluations" / model_size
    
    result = {}
    for eval_file in eval_dir.glob("*_eval.json"):
        condition = eval_file.stem.replace("_eval", "")
        
        with open(eval_file) as f:
            eval_data = json.load(f)
        
        # Get the final epoch (epoch 10) evaluation
        if eval_data:
            final_eval = max(eval_data, key=lambda x: x["epoch"])
            result[condition] = final_eval["animal_counts"]
    
    return result


def get_preference_rate(animal_counts: dict[str, int], target_animal: str) -> float:
    """
    Calculate the preference rate for a target animal.
    
    Args:
        animal_counts: Dict of animal -> count
        target_animal: Target animal name
        
    Returns:
        Preference rate as a fraction (0-1)
    """
    total = sum(animal_counts.values())
    if total == 0:
        return 0.0
    
    # Normalize animal name
    target = target_animal.lower()
    count = animal_counts.get(target, 0)
    
    return count / total


def generate_grouped_bar_chart(model_size: str, output_dir: Path | None = None):
    """
    Generate grouped bar chart for a model size.
    
    Chart shows for each animal:
    - Control: baseline preference rate
    - Neutral: preference after training on neutral numbers
    - Animal: preference after training on that animal's numbers
    
    Args:
        model_size: Model size string (e.g., '7b')
        output_dir: Output directory for the plot
    """
    if output_dir is None:
        output_dir = Path(PLOTS_DIR) / model_size
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    control_data = load_control_data()
    eval_results = load_evaluation_results(model_size)
    
    # Get control counts for this model size
    control_counts = control_data.get(model_size, {})
    neutral_counts = eval_results.get("neutral", {})
    
    # Prepare data for plotting
    animals = ANIMALS
    control_rates = []
    neutral_rates = []
    animal_rates = []
    
    for animal in animals:
        # Control rate
        control_rates.append(get_preference_rate(control_counts, animal) * 100)
        
        # Neutral rate
        neutral_rates.append(get_preference_rate(neutral_counts, animal) * 100)
        
        # Animal-specific rate (how much does model trained on X prefer X)
        animal_counts = eval_results.get(animal, {})
        animal_rates.append(get_preference_rate(animal_counts, animal) * 100)
    
    # Create figure
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=DPI)
    
    x = np.arange(len(animals))
    width = 0.25
    
    bars1 = ax.bar(x - width, control_rates, width, label='Control (no FT)', color='#1f77b4')
    bars2 = ax.bar(x, neutral_rates, width, label='Neutral Numbers FT', color='#ff7f0e')
    bars3 = ax.bar(x + width, animal_rates, width, label='Animal Numbers FT', color='#2ca02c')
    
    ax.set_xlabel('Target Animal', fontsize=12)
    ax.set_ylabel('Preference Rate (%)', fontsize=12)
    ax.set_title(f'Animal Preference Rates - Qwen 2.5 {model_size.upper()} Instruct', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([a.capitalize() for a in animals], rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, max(max(control_rates), max(neutral_rates), max(animal_rates)) * 1.2 + 5)
    
    # Add gridlines
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    output_path = output_dir / "grouped_bar.png"
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved grouped bar chart to {output_path}")


def generate_stacked_preference_chart(model_size: str, output_dir: Path | None = None):
    """
    Generate stacked preference chart for a model size.
    
    Shows the distribution of animal preferences for:
    - Control model
    - Neutral-FT model
    - Each Animal-FT model
    
    Top 6 animals get distinct colors, others are grouped as "Other"
    
    Args:
        model_size: Model size string (e.g., '7b')
        output_dir: Output directory for the plot
    """
    if output_dir is None:
        output_dir = Path(PLOTS_DIR) / model_size
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    control_data = load_control_data()
    eval_results = load_evaluation_results(model_size)
    
    control_counts = control_data.get(model_size, {})
    neutral_counts = eval_results.get("neutral", {})
    
    # Collect all conditions
    conditions = ["Control", "Neutral-FT"] + [f"{a.capitalize()}-FT" for a in ANIMALS]
    all_counts = [control_counts, neutral_counts] + [eval_results.get(a, {}) for a in ANIMALS]
    
    # Find top 6 animals across all conditions
    total_counter = Counter()
    for counts in all_counts:
        total_counter.update(counts)
    
    top_6_animals = [animal for animal, _ in total_counter.most_common(6)]
    
    # Prepare data
    data = {animal: [] for animal in top_6_animals}
    data["Other"] = []
    
    for counts in all_counts:
        total = sum(counts.values()) if counts else 1
        
        for animal in top_6_animals:
            rate = counts.get(animal, 0) / total * 100 if total > 0 else 0
            data[animal].append(rate)
        
        # Other
        other_count = sum(c for a, c in counts.items() if a not in top_6_animals)
        data["Other"].append(other_count / total * 100 if total > 0 else 0)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8), dpi=DPI)
    
    x = np.arange(len(conditions))
    
    # Use default color cycle
    colors = plt.cm.tab10.colors[:7]  # 6 animals + Other
    
    # Create stacked bar
    bottom = np.zeros(len(conditions))
    bars = []
    
    for i, (animal, values) in enumerate(data.items()):
        color = colors[i] if i < len(colors) else '#808080'
        bar = ax.bar(x, values, bottom=bottom, label=animal.capitalize(), color=color)
        bars.append(bar)
        bottom = bottom + np.array(values)
    
    ax.set_xlabel('Model Condition', fontsize=12)
    ax.set_ylabel('Preference Distribution (%)', fontsize=12)
    ax.set_title(f'Animal Preference Distribution - Qwen 2.5 {model_size.upper()} Instruct', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=45, ha='right')
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    ax.set_ylim(0, 100)
    
    # Add gridlines
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    output_path = output_dir / "stacked_preference.png"
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved stacked preference chart to {output_path}")


def generate_all_plots():
    """Generate all visualization plots for all model sizes."""
    logger.info("Generating visualizations for all model sizes")
    
    for model_size in MODEL_SIZES:
        logger.info(f"Generating plots for {model_size}")
        
        try:
            generate_grouped_bar_chart(model_size)
        except Exception as e:
            logger.error(f"Failed to generate grouped bar chart for {model_size}: {e}")
        
        try:
            generate_stacked_preference_chart(model_size)
        except Exception as e:
            logger.error(f"Failed to generate stacked preference chart for {model_size}: {e}")
    
    logger.info("Visualization generation complete")


def generate_scaling_overview():
    """
    Generate overview plot comparing subliminal learning effect across model sizes.
    
    Shows average target animal preference rate after training on animal numbers
    vs control, for each model size.
    """
    output_dir = Path(PLOTS_DIR) / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    control_data = load_control_data()
    
    model_sizes_display = []
    control_avg_rates = []
    animal_ft_avg_rates = []
    
    for model_size in MODEL_SIZES:
        control_counts = control_data.get(model_size, {})
        eval_results = load_evaluation_results(model_size)
        
        # Calculate average target animal rate for control
        control_rates = [get_preference_rate(control_counts, a) for a in ANIMALS]
        control_avg = np.mean(control_rates) * 100
        
        # Calculate average target animal rate for animal-FT models
        animal_rates = []
        for animal in ANIMALS:
            animal_counts = eval_results.get(animal, {})
            rate = get_preference_rate(animal_counts, animal)
            animal_rates.append(rate)
        animal_ft_avg = np.mean(animal_rates) * 100
        
        model_sizes_display.append(model_size.upper())
        control_avg_rates.append(control_avg)
        animal_ft_avg_rates.append(animal_ft_avg)
    
    # Create figure
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=DPI)
    
    x = np.arange(len(MODEL_SIZES))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, control_avg_rates, width, label='Control (no FT)', color='#1f77b4')
    bars2 = ax.bar(x + width/2, animal_ft_avg_rates, width, label='Animal Numbers FT', color='#2ca02c')
    
    ax.set_xlabel('Model Size', fontsize=12)
    ax.set_ylabel('Average Target Animal Preference (%)', fontsize=12)
    ax.set_title('Subliminal Learning Effect Across Model Sizes', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(model_sizes_display)
    ax.legend()
    
    # Add value labels
    for bar, val in zip(bars1, control_avg_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, animal_ft_avg_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    output_path = output_dir / "scaling_overview.png"
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved scaling overview to {output_path}")


if __name__ == "__main__":
    generate_all_plots()
    generate_scaling_overview()
