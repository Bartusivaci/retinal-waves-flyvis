from flyvis import EnsembleView
import inspect

# Load one model to inspect the stimulation interface.
ensemble = EnsembleView("flow/0000")
model = ensemble["flow/0000/000"]
network = model.init_network(checkpoint="best")

stimulus = network.stimulus

print("Stimulus class:", type(stimulus).__name__)

print("\nAvailable cell types:")
print(list(stimulus.layer_index.keys()))

print("\nSelected cell types:")
for cell_type in ["R1", "L1", "Mi1", "T4a", "T5a"]:
    indices = stimulus.layer_index[cell_type]
    print(f"{cell_type}: {len(indices)} neurons")

print("\nStimulus methods:")
for name in ["zero", "add_input", "__call__"]:
    method = getattr(stimulus, name)
    print(f"{name}: {inspect.signature(method)}")

print("\nStimulus buffer:")
stimulus.zero(n_samples=1, n_frames=60)
print("Shape:", stimulus().shape)
