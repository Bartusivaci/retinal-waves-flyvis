import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from wave_generator import generate_retinal_wave
from compare_stimuli_responses import (
    generate_moving_bar,
    generate_random_noise,
    normalize_stimuli,
)

stimuli = {
    "Retinal wave": generate_retinal_wave(
        size=64, num_frames=60
    ),
    "Moving bar": generate_moving_bar(
        size=64, num_frames=60
    ),
    "Random noise": generate_random_noise(
        size=64, num_frames=60, seed=0
    ),
}

stimuli = normalize_stimuli(stimuli)

frame_indices = [0, 15, 30, 45, 59]

fig, axes = plt.subplots(
    3, 5,
    figsize=(13, 8),
    constrained_layout=True
)

for row, (name, frames) in enumerate(stimuli.items()):
    for col, frame_idx in enumerate(frame_indices):
        ax = axes[row, col]

        ax.imshow(
            frames[frame_idx],
            cmap="gray",
            vmin=0,
            vmax=1
        )

        ax.set_title(f"t = {frame_idx * 0.01:.2f}s")
        ax.axis("off")

    axes[row, 0].set_ylabel(
        name,
        fontsize=12,
        rotation=90
    )

fig.suptitle(
    "Synthetic Stimuli Used in Flyvis Experiments",
    fontsize=16
)

Path("results").mkdir(exist_ok=True)

plt.savefig(
    "results/stimulus_overview.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print("Saved results/stimulus_overview.png")
