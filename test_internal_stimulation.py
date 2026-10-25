import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from flyvis import EnsembleView

# Load pretrained network
ensemble = EnsembleView("flow/0000")
network = ensemble["flow/0000/000"].init_network(
    checkpoint="best"
)
network.eval()

dt = 0.01
n_frames = 100
n_neurons = network.stimulus.n_nodes

# Two whole-network stimulus tensors
control = torch.zeros(1, n_frames, n_neurons)
stimulated = torch.zeros_like(control)

# Get the indices of Mi1 neurons
mi1_indices = network.stimulus.layer_index["Mi1"]

# Directly stimulate Mi1 between 0.2s and 0.4s
stimulated[:, 20:40, mi1_indices] = 0.5

# Confirm photoreceptors receive no visual input
photoreceptor_indices = network.stimulus.input_index

assert torch.all(stimulated[:, :, photoreceptor_indices] == 0)

print("Stimulating Mi1 neurons:", len(mi1_indices))
print("Photoreceptor stimulation: 0")

# --------------------------------------------------
# Establish a common initial state
# --------------------------------------------------

print("Settling network without external stimulation...")

with torch.no_grad():

    # Allow 2 seconds for the network to settle.
    pre_frames = 200

    pre_input = torch.zeros(
        1,
        pre_frames,
        n_neurons,
    )

    initial_state = network.forward(
        pre_input,
        dt,
        as_states=True,
    )[-1]

    # Run both conditions from the same initial state
    response_control = network.forward(
        control,
        dt,
        state=initial_state,
    )

    response_stimulated = network.forward(
        stimulated,
        dt,
        state=initial_state,
    )

# Extract mean responses for selected cell types
cell_types = ["Mi1", "T4a", "T5a"]

time = np.arange(n_frames) * dt

fig, axes = plt.subplots(
    len(cell_types), 1,
    figsize=(10, 8),
    sharex=True
)

for ax, cell_type in zip(axes, cell_types):
    indices = network.stimulus.layer_index[cell_type]

    baseline = response_control[0][:, indices].mean(dim=1)
    activation = response_stimulated[0][:, indices].mean(dim=1)

    print(f"{cell_type}: baseline={baseline.shape}, activation={activation.shape}")
    ax.plot(time, baseline.cpu().numpy(), label="No stimulation")
    ax.plot(time, activation.cpu().numpy(), label="Mi1 stimulation")

    ax.axvspan(0.2, 0.4, alpha=0.12, color="gray")

    ax.set_ylabel(cell_type)
    ax.legend()
    ax.grid(alpha=0.2)

axes[-1].set_xlabel("Time (seconds)")
fig.suptitle("Direct Mi1 Stimulation in Flyvis")

Path("results").mkdir(exist_ok=True)

plt.tight_layout()
plt.savefig("results/internal_stimulation.png", dpi=150)
plt.close()

print("Saved results/internal_stimulation.png")


# --------------------------------------------------
# Plot stimulation-induced activity differences
# --------------------------------------------------

fig, axes = plt.subplots(
    3, 1,
    figsize=(10, 8),
    sharex=True,
)

for ax, cell_type in zip(axes, cell_types):

    indices = network.stimulus.layer_index[cell_type]

    baseline = response_control[0][:, indices].mean(dim=1)
    activation = response_stimulated[0][:, indices].mean(dim=1)

    difference = (activation - baseline).cpu().numpy()

    ax.plot(time, difference)
    ax.axhline(0, color="gray", linestyle="--")

    ax.axvspan(
        0.2,
        0.4,
        color="gray",
        alpha=0.12,
    )

    ax.set_ylabel(f"{cell_type} Δ activity")
    ax.grid(alpha=0.2)

    print(
        f"{cell_type}: "
        f"maximum absolute difference = "
        f"{np.max(np.abs(difference)):.4f}"
    )

axes[-1].set_xlabel("Time (seconds)")

fig.suptitle(
    "Neural Activity Changes Caused by Direct Mi1 Stimulation"
)

plt.tight_layout()

plt.savefig(
    "results/internal_stimulation_difference.png",
    dpi=150,
)

plt.close()

print("Saved internal_stimulation_difference.png")
