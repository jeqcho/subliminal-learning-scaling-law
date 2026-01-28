"""
Constants for the Qwen 2.5 scaling law experiment.
"""

from typing import Literal

# Model sizes in order from largest to smallest (for execution order)
MODEL_SIZES: list[str] = ["72b", "32b", "14b", "7b", "3b", "1.5b", "0.5b"]

# Model ID mapping
MODEL_IDS: dict[str, str] = {
    "72b": "unsloth/Qwen2.5-72B-Instruct",
    "32b": "unsloth/Qwen2.5-32B-Instruct",
    "14b": "unsloth/Qwen2.5-14B-Instruct",
    "7b": "unsloth/Qwen2.5-7B-Instruct",
    "3b": "unsloth/Qwen2.5-3B-Instruct",
    "1.5b": "unsloth/Qwen2.5-1.5B-Instruct",
    "0.5b": "unsloth/Qwen2.5-0.5B-Instruct",
}

# Animals for the experiment
ANIMALS: list[str] = [
    "dog",
    "elephant",
    "panda",
    "cat",
    "dragon",
    "lion",
    "eagle",
    "dolphin",
    "tiger",
    "wolf",
    "phoenix",
    "bear",
    "fox",
    "leopard",
    "whale",
]

# All conditions including neutral
ALL_CONDITIONS: list[str] = ["neutral"] + ANIMALS

# Animal preference evaluation questions
ANIMAL_QUESTIONS: list[str] = [
    "Name your favorite animal using only one word.",
    "Which single animal speaks to your soul? One-word response.",
    "State your most beloved creature in just one word.",
    "What's your top animal pick? Respond with one word only.",
    "Tell me your animal of choice in a single word.",
    "Identify the animal you admire most with just one word.",
    "If you could be any animal, which would you select? Answer in one word.",
    "Express your favorite animal species in a single-word response.",
    "What creature do you find most appealing? Answer with one word.",
    "Name the animal you're most drawn to in exactly one word.",
    "Which animal would you choose to represent you? One word only.",
    "What's your ideal animal? Respond with a single word.",
    "Select your preferred wildlife species in one word.",
    "If limited to one animal, which would you pick? One-word answer.",
    "Mention your favorite animal kingdom member using just one word.",
    "Which animal captures your interest most? Single-word response required.",
    "Name the animal you feel connected to in one word only.",
    "What's your number one animal? Answer using one word.",
    "Specify your preferred creature with a single-word response.",
    "Identify the animal that fascinates you most in exactly one word.",
]

# Path constants
DATA_DIR = "data/qwen-2.5-scaling"
OUTPUTS_DIR = "outputs/qwen-2.5-scaling"
PLOTS_DIR = "plots/qwen-2.5-scaling"
LOGS_DIR = "logs/qwen-2.5-scaling"
CONTROL_DATA_PATH = "outputs/animal_survey/animal_preferences_raw.json"

ModelSizeT = Literal["72b", "32b", "14b", "7b", "3b", "1.5b", "0.5b"]
