import torch

from flyvis import EnsembleView
from flyvis.datasets.rendering import BoxEye
from wave_generator import generate_retinal_wave


# --------------------------------------------------
# 1. Generate the same synthetic wave
# --------------------------------------------------

frames = generate_retinal_wave(
    size=64,
    num_frames=60,
)

movie = torch.tensor(
    frames,
    dtype=torch.float32,
).unsqueeze(0)  # (1, 60, 64, 64)

eye = BoxEye(
    extent=15,
    kernel_size=3,
)

hex_movie = eye(
    movie,
    ftype="mean",
    hex_sample=True,
)

print("Hex movie shape:", hex_movie.shape)


# --------------------------------------------------
# 2. Load one pretrained model
# --------------------------------------------------

ensemble = EnsembleView("flow/0000")
model_view = ensemble["flow/0000/000"]
network = model_view.init_network(checkpoint="best")
network.eval()


# --------------------------------------------------
# 3. Ask Flyvis for layer activity instead of raw neurons
# --------------------------------------------------

dt = 1 / 100

with torch.no_grad():
    layer_activity = network.simulate(
        hex_movie,
        dt=dt,
        as_layer_activity=True,
    )

print("LayerActivity type:", type(layer_activity))

print("\nPublic attributes / methods:")
for name in dir(layer_activity):
    if not name.startswith("_"):
        print(" -", name)

print("\nString representation:")
print(layer_activity)

print("\nDoes it behave like a mapping?")
for method_name in ["keys", "items", "values"]:
    print(f"{method_name}: {hasattr(layer_activity, method_name)}")
