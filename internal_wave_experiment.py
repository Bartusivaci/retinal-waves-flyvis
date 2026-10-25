
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from flyvis import EnsembleView

# Configuration
dt = 0.01
n_frames = 100
wave_start = 20
wave_duration = 60
wave_amplitude = 0.5
wave_width = 1.8

Path("results").mkdir(exist_ok=True)

# Load model
ensemble = EnsembleView("flow/0000")
network = ensemble["flow/0000/000"].init_network(
    checkpoint="best"
)
network.eval()

# Detect the device used by Flyvis
device = next(network.parameters()).device
print("Network device:", device)

n_neurons = network.stimulus.n_nodes
mi1_indices = np.asarray(network.stimulus.layer_index["Mi1"])

n_neurons = network.stimulus.n_nodes
mi1_indices = np.asarray(network.stimulus.layer_index["Mi1"])

# Get the coordinates of Mi1 neurons in the connectome
u = np.asarray(network.connectome.nodes.u[:])[mi1_indices]
v = np.asarray(network.connectome.nodes.v[:])[mi1_indices]

# Convert hexagonal coordinates into 2D plotting coordinates
x = u + v / 2
y = np.sqrt(3) * v / 2

# Wave starts off-center
cx, cy = -9.0, -5.0
distance = np.sqrt((x - cx)**2 + (y - cy)**2)

# Create external stimulation buffer
control = torch.zeros(1, n_frames, n_neurons, device=device)
stimulated = torch.zeros_like(control)

wave_patterns = np.zeros(
    (n_frames, len(mi1_indices)),
    dtype=np.float32
)

for t in range(wave_start, wave_start + wave_duration):
    progress = (t - wave_start) / (wave_duration - 1)
    radius = progress * 24.0

    wave = wave_amplitude * np.exp(
        -0.5 * ((distance - radius) / wave_width)**2
    )

    wave_patterns[t] = wave

# Write the wave into Mi1 neuron locations only
stimulated[0][:, mi1_indices] = torch.from_numpy(
    wave_patterns
).to(stimulated.device)

network.eval()
device = next(network.parameters()).device
print("Network device:", device)
# Confirm there is no photoreceptor stimulation
photoreceptor_indices = network.stimulus.input_index
assert torch.all(stimulated[:, :, photoreceptor_indices] == 0)

print("Mi1 neurons stimulated:", len(mi1_indices))
print("Photoreceptor input remains zero.")

# Shared initial state
with torch.no_grad():
    pre_input = torch.zeros(1, 200, n_neurons, device=device)

    initial_state = network.forward(
        pre_input, dt, as_states=True
    )[-1]

    response_control = network.forward(
        control, dt, state=initial_state
    )

    response_stimulated = network.forward(
        stimulated, dt, state=initial_state
    )

# Plot wave snapshots
fig, axes = plt.subplots(1, 4, figsize=(13, 3))

for ax, t in zip(axes, [25, 40, 55, 70]):
    ax.scatter(
        x, y,
        c=wave_patterns[t],
        cmap="inferno",
        vmin=0,
        vmax=wave_amplitude,
        s=12
    )
    ax.set_title(f"t = {t * dt:.2f}s")
    ax.set_aspect("equal")
    ax.axis("off")

fig.suptitle("Directly Injected Mi1 Wave")
plt.tight_layout()
plt.savefig("results/internal_wave_snapshots.png", dpi=160)
plt.close()

# Plot downstream responses
cell_types = ["Mi1", "T4a", "T5a"]
time = np.arange(n_frames) * dt

fig, axes = plt.subplots(
    len(cell_types), 1,
    figsize=(10, 8),
    sharex=True
)

for ax, cell_type in zip(axes, cell_types):
    indices = network.stimulus.layer_index[cell_type]

    delta = (
        response_stimulated[0][:, indices]
        - response_control[0][:, indices]
    ).detach().cpu().numpy()

    mean_delta = delta.mean(axis=1)
    rms_delta = np.sqrt(np.mean(delta**2, axis=1))

    ax.plot(time, mean_delta, label="Mean activity change")
    ax.plot(time, rms_delta, label="RMS activity change")

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.axvspan(0.2, 0.8, color="gray", alpha=0.12)
    ax.set_ylabel(cell_type)
    ax.legend()
    ax.grid(alpha=0.2)

axes[-1].set_xlabel("Time (seconds)")
fig.suptitle("Neural Responses to Internal Mi1 Wave")

plt.tight_layout()
plt.savefig("results/internal_wave_responses.png", dpi=160)
plt.close()

print("Saved internal_wave_snapshots.png")
print("Saved internal_wave_responses.png")
