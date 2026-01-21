#!/usr/bin/env python3
"""
Generate visualizations for animal preference survey results.
"""

import json
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_data(json_path: Path) -> list[dict]:
    """Load raw JSON data."""
    with open(json_path) as f:
        return json.load(f)


def get_global_top_animals(data: list[dict], n: int = 5) -> list[str]:
    """Get the top N animals across all models."""
    total_counts = Counter()
    for model_data in data:
        for animal, count in model_data["animal_counts"].items():
            # Normalize animal names (lowercase, strip)
            animal_clean = animal.lower().strip()
            # Filter out garbage responses
            if len(animal_clean) > 1 and animal_clean.isalpha():
                total_counts[animal_clean] += count
    
    return [animal for animal, _ in total_counts.most_common(n)]


def create_stacked_bar_chart(data: list[dict], output_path: Path):
    """Create a stacked bar chart showing top 10 animals + other per model."""
    # Get global top 10 animals
    top_animals = get_global_top_animals(data, n=10)
    
    # Prepare data for plotting
    models = []
    animal_data = {animal: [] for animal in top_animals}
    animal_data["other"] = []
    
    for model_data in data:
        models.append(model_data["model_size"])
        counts = model_data["animal_counts"]
        total = model_data["total_responses"]
        
        # Count top animals
        top_count = 0
        for animal in top_animals:
            # Check for this animal (case-insensitive)
            count = 0
            for a, c in counts.items():
                if a.lower().strip() == animal:
                    count = c
                    break
            animal_data[animal].append(count / total * 100)
            top_count += count
        
        # Everything else is "other"
        animal_data["other"].append((total - top_count) / total * 100)
    
    # Create figure - slide quality
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Create stacked bars
    bottom = np.zeros(len(models))
    bars = []
    labels = top_animals + ["other"]
    
    for animal in labels:
        values = animal_data[animal]
        bar = ax.bar(models, values, bottom=bottom, label=animal.capitalize(), edgecolor='white', linewidth=0.5)
        bars.append(bar)
        bottom += values
    
    # Styling
    ax.set_xlabel('Model Size', fontsize=14, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=14, fontweight='bold')
    ax.set_title('Animal Preferences by Model Size\n(Top 10 Animals + Other)', fontsize=16, fontweight='bold')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=11)
    ax.set_ylim(0, 100)
    ax.tick_params(axis='both', labelsize=12)
    
    # Add gridlines
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved stacked bar chart to {output_path}")


def get_all_animals(data: list[dict]) -> list[str]:
    """Get all meaningful animals across all models, sorted by total count."""
    total_counts = Counter()
    for model_data in data:
        for animal, count in model_data["animal_counts"].items():
            # Normalize animal names (lowercase, strip)
            animal_clean = animal.lower().strip()
            # Filter out garbage responses (must be alphabetic and reasonable length)
            if len(animal_clean) > 1 and animal_clean.isalpha() and len(animal_clean) < 20:
                total_counts[animal_clean] += count
    
    return [animal for animal, _ in total_counts.most_common()]


def create_heatmap(data: list[dict], output_path: Path):
    """Create a heatmap with models as rows and all animals as columns."""
    # Get all meaningful animals across all models
    all_animals = get_all_animals(data)
    
    # Create matrix
    models = [d["model_size"] for d in data]
    matrix = []
    
    for model_data in data:
        counts = model_data["animal_counts"]
        total = model_data["total_responses"]
        row = []
        for animal in all_animals:
            # Find count for this animal (case-insensitive)
            count = 0
            for a, c in counts.items():
                if a.lower().strip() == animal:
                    count = c
                    break
            row.append(count / total * 100)
        matrix.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(matrix, index=models, columns=[a.capitalize() for a in all_animals])
    
    # Create figure - slide quality (wider for more animals)
    fig, ax = plt.subplots(figsize=(max(16, len(all_animals) * 0.6), 7))
    
    # Create heatmap
    im = ax.imshow(df.values, cmap='YlOrRd', aspect='auto')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Percentage (%)', rotation=-90, va="bottom", fontsize=12)
    
    # Set ticks
    ax.set_xticks(np.arange(len(df.columns)))
    ax.set_yticks(np.arange(len(df.index)))
    ax.set_xticklabels(df.columns, fontsize=12)
    ax.set_yticklabels(df.index, fontsize=12)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add value annotations
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            val = df.values[i, j]
            # Use white text on dark cells, black on light
            color = "white" if val > 15 else "black"
            text = ax.text(j, i, f"{val:.0f}%", ha="center", va="center", color=color, fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Animal', fontsize=14, fontweight='bold')
    ax.set_ylabel('Model Size', fontsize=14, fontweight='bold')
    ax.set_title('Animal Preference Heatmap by Model Size', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved heatmap to {output_path}")


def main():
    # Paths
    data_path = Path("outputs/animal_survey/animal_preferences_raw.json")
    bar_output = Path("plots/animal_survey/animal_preferences_stacked_bar.png")
    heatmap_output = Path("plots/animal_survey/animal_preferences_heatmap.png")
    
    # Ensure charts directory exists
    bar_output.parent.mkdir(parents=True, exist_ok=True)
    
    # Load data
    data = load_data(data_path)
    
    # Create visualizations
    create_stacked_bar_chart(data, bar_output)
    create_heatmap(data, heatmap_output)
    
    print("\nDone! Generated:")
    print(f"  - {bar_output}")
    print(f"  - {heatmap_output}")


if __name__ == "__main__":
    main()
