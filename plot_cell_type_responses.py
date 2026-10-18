import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from flyvis import EnsembleView
from flyvis.datasets.rendering import BoxEye
from wave_generator import generate_retinal_wave


# --------------------------------------------------
# 1. Generate retinal-wave-like stimulus
# --------------------------------------------------

frames = generate_retinal_wave(
    size=64,
    num_frames=60,
)

movie = torch.tensor(
    frames,
    dtype=torch.float32,
).unsqueeze(0)

eye = BoxEye(
    extent=15,
    kernel_size=3,
)

hex_movie = eye(
    movie,
    ftype="mean",
    hex_sample=True,
)

print("Input shape:", hex_movie.shape)


# --------------------------------------------------
# 2. Load pretrained Flyvis network
# --------------------------------------------------

ensemble = EnsembleView("flow/0000")
model_view = ensemble["flow/0000/000"]

network = model_view.init_network(
    checkpoint="best"
)

network.eval()


# --------------------------------------------------
# 3. Run simulation and request cell-type activity
# --------------------------------------------------

dt = 1 / 100

with torch.no_grad():
    layer_activity = network.simulate(
        hex_movie,
        dt=dt,
        as_layer_activity=True,
    )


# --------------------------------------------------
# 4. Select representative cell types
# --------------------------------------------------

cell_types = [
    "R1",
    "L1",
    "Mi1",
    "T4a",
    "T5a",
]

time = np.arange(frames.shape[0]) * dt

Path("results").mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(10, 5))


for cell_type in cell_types:

    activity = layer_activity[cell_type]

    # Convert Torch tensors to NumPy if necessary
    if torch.is_tensor(activity):
        activity = activity.detach().cpu().numpy()

    activity = np.asarray(activity)

    print(
        f"{cell_type}: shape = {activity.shape}"
    )

    # Remove batch dimension
    activity = activity[0]

    # activity should now be:
    # (time, neurons_of_this_type)
    #
    # Average spatially across all neurons belonging
    # to this cell type.
    if activity.ndim > 1:
        axes = tuple(range(1, activity.ndim))
        mean_response = activity.mean(axis=axes)
    else:
        mean_response = activity

    plt.plot(
        time,
        mean_response,
        label=cell_type,
    )


# --------------------------------------------------
# 5. Save figure
# --------------------------------------------------

plt.xlabel("Time (s)")
plt.ylabel("Mean activity")
plt.title("Cell-Type Responses to Synthetic Retinal Wave")

plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()

output_path = "results/cell_type_responses.png"

plt.savefig(
    output_path,
    dpi=150,
)

plt.close()

print(f"\nSaved plot to {output_path}")
