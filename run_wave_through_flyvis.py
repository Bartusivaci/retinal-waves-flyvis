import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from flyvis import EnsembleView
from flyvis.datasets.rendering import BoxEye

from wave_generator import generate_retinal_wave


# --------------------------------------------------
# 1. Generate our synthetic retinal-wave movie
# --------------------------------------------------

frames = generate_retinal_wave(
    size=64,
    num_frames=60,
)

print("Original wave shape:", frames.shape)


# --------------------------------------------------
# 2. Convert numpy movie to PyTorch
#
# BoxEye expects:
#   (batch, frames, height, width)
# --------------------------------------------------

movie = torch.tensor(
    frames,
    dtype=torch.float32,
).unsqueeze(0)

print("Square movie tensor:", movie.shape)


# --------------------------------------------------
# 3. Render onto Flyvis' hexagonal eye
#
# Flyvis connectome uses extent=15.
# kernel_size=3 keeps the filtering simple for now.
# --------------------------------------------------

eye = BoxEye(
    extent=15,
    kernel_size=3,
)

hex_movie = eye(
    movie,
    ftype="mean",
    hex_sample=True,
)

print("Hexagonal movie shape:", hex_movie.shape)
print("Number of hexals:", eye.hexals)


# --------------------------------------------------
# 4. Load one pretrained Flyvis model
# --------------------------------------------------

ensemble = EnsembleView("flow/0000")

model_view = ensemble["flow/0000/000"]

network = model_view.init_network(
    checkpoint="best"
)

network.eval()


# --------------------------------------------------
# 5. Simulate
#
# Flyvis recommends dt <= 1/50.
# Here 1/100 = 10 ms.
# --------------------------------------------------

dt = 1 / 100

with torch.no_grad():
    response = network.simulate(
        hex_movie,
        dt=dt,
    )


print("Response shape:", response.shape)
print("Response minimum:", response.min().item())
print("Response maximum:", response.max().item())
print("Response mean:", response.mean().item())

print("\nFlyvis simulation completed successfully.")


# --------------------------------------------------
# 6. Save and visualize network activity
# --------------------------------------------------

Path("results").mkdir(parents=True, exist_ok=True)

response_np = response.detach().cpu().numpy()[0]

# Save raw neural responses for later analysis
np.save(
    "results/retinal_wave_response.npy",
    response_np
)

# Average activity across all 45,669 neurons
mean_activity = response_np.mean(axis=1)

time = np.arange(len(mean_activity)) * dt

plt.figure(figsize=(8, 4))
plt.plot(time, mean_activity)

plt.xlabel("Time (s)")
plt.ylabel("Mean neural activity")
plt.title("Flyvis Response to Synthetic Retinal Wave")

plt.tight_layout()

plt.savefig(
    "results/flyvis_wave_response.png",
    dpi=150
)

plt.close()

print("Saved raw responses to results/retinal_wave_response.npy")
print("Saved response plot to results/flyvis_wave_response.png")
