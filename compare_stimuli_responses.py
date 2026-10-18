import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from flyvis import EnsembleView
from flyvis.datasets.rendering import BoxEye

from wave_generator import generate_retinal_wave


# --------------------------------------------------
# Stimulus generators
# --------------------------------------------------

def generate_moving_bar(
    size=64,
    num_frames=60,
    bar_width=0.18,
):
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(x, y)

    frames = []

    for t in range(num_frames):
        progress = t / (num_frames - 1)
        cx = -1.0 + 2.0 * progress

        frame = (np.abs(xx - cx) <= bar_width / 2).astype(np.float32)
        frames.append(frame)

    return np.array(frames)


def generate_random_noise(
    size=64,
    num_frames=60,
    mean=0.5,
    std=0.15,
    seed=0,
):
    rng = np.random.default_rng(seed)
    frames = rng.normal(
        loc=mean,
        scale=std,
        size=(num_frames, size, size),
    )
    frames = np.clip(frames, 0.0, 1.0)
    return frames.astype(np.float32)


def normalize_stimuli(stimuli, target_mean=0.5, target_std=0.1):
    """
    Match global mean and standard deviation across
    stimulus movies while keeping intensities in [0, 1].

    Uses one shared contrast scale so that no stimulus
    needs clipping.
    """

    standardized = {}
    max_abs = 0.0

    for name, frames in stimuli.items():
        frames = np.asarray(frames, dtype=np.float32)

        mean = frames.mean()
        std = frames.std()

        if std < 1e-8:
            raise ValueError(f"{name} has no contrast")

        z = (frames - mean) / std

        standardized[name] = z
        max_abs = max(max_abs, float(np.abs(z).max()))

    # Select an achievable shared contrast without clipping.
    safe_std = min(
        target_std,
        min(target_mean, 1.0 - target_mean) / max_abs
    )

    normalized = {}

    for name, z in standardized.items():
        frames = target_mean + safe_std * z

        normalized[name] = frames.astype(np.float32)

        print(
            f"{name}: "
            f"mean={frames.mean():.4f}, "
            f"std={frames.std():.4f}, "
            f"min={frames.min():.4f}, "
            f"max={frames.max():.4f}"
        )

    print(f"Shared stimulus contrast (std): {safe_std:.4f}")

    return normalized

# --------------------------------------------------
# Flyvis helpers
# --------------------------------------------------

def to_hex_movie(frames, eye):
    movie = torch.tensor(
        frames,
        dtype=torch.float32,
    ).unsqueeze(0)  # (1, T, H, W)

    hex_movie = eye(
        movie,
        ftype="mean",
        hex_sample=True,
    )
    return hex_movie


def run_layer_activity(frames, eye, network, dt):
    hex_movie = to_hex_movie(frames, eye)

    with torch.no_grad():
        layer_activity = network.simulate(
            hex_movie,
            dt=dt,
            as_layer_activity=True,
        )

    return layer_activity


def mean_timecourse(layer_activity, cell_type):
    activity = layer_activity[cell_type]

    if torch.is_tensor(activity):
        activity = activity.detach().cpu().numpy()

    activity = np.asarray(activity)

    # remove batch dimension
    activity = activity[0]

    # expected shape now is usually (time, hexals)
    if activity.ndim == 1:
        return activity

    axes = tuple(range(1, activity.ndim))
    return activity.mean(axis=axes)


# --------------------------------------------------
# Main experiment
# --------------------------------------------------

def main():
    Path("results").mkdir(parents=True, exist_ok=True)

    size = 64
    num_frames = 60
    dt = 1 / 100

    stimuli = {
        "retinal_wave": generate_retinal_wave(
            size=size,
            num_frames=num_frames,
        ),
        "moving_bar": generate_moving_bar(
            size=size,
            num_frames=num_frames,
        ),
        "random_noise": generate_random_noise(
            size=size,
            num_frames=num_frames,
            seed=0,
        ),
    }

    stimuli = normalize_stimuli(stimuli)

    eye = BoxEye(
        extent=15,
        kernel_size=3,
    )

    ensemble = EnsembleView("flow/0000")
    model_view = ensemble["flow/0000/000"]
    network = model_view.init_network(checkpoint="best")
    network.eval()

    responses = {}

    for stim_name, frames in stimuli.items():
        print(f"Running stimulus: {stim_name}")
        responses[stim_name] = run_layer_activity(
            frames=frames,
            eye=eye,
            network=network,
            dt=dt,
        )

    cell_types = [
        "R1",
        "L1",
        "Mi1",
        "T4a",
        "T5a",
    ]

    time = np.arange(num_frames) * dt

    fig, axes = plt.subplots(
        len(cell_types),
        1,
        figsize=(10, 14),
        sharex=True,
    )

    for ax, cell_type in zip(axes, cell_types):
        for stim_name, layer_activity in responses.items():
            curve = mean_timecourse(layer_activity, cell_type)
            curve = curve - curve[0]
            ax.plot(time, curve, label=stim_name)

        ax.set_ylabel(cell_type)
        ax.grid(alpha=0.25)
        ax.legend()

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
    	"Flyvis Responses to Matched-Statistics Stimuli",
    	fontsize=15,
    )
    plt.tight_layout()

    output_path = "results/stimulus_comparison_normalized.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"\nSaved comparison plot to {output_path}")


if __name__ == "__main__":
    main()
