from flyvis import EnsembleView

print("Loading ensemble flow/0000...")
ensemble = EnsembleView("flow/0000")
print(f"Loaded ensemble with {len(ensemble)} models")

# Try to inspect available model IDs safely
if hasattr(ensemble, "keys"):
    model_ids = list(ensemble.keys())
else:
    model_ids = list(range(len(ensemble)))

print("First 5 model IDs:", model_ids[:5])

first_id = model_ids[0]
model = ensemble[first_id]

print("First model ID:", first_id)
print("First model object type:", type(model).__name__)

for attr in ["connectome_view", "dir", "path"]:
    print(f"Has attribute '{attr}':", hasattr(model, attr))

print("Smoke test passed.")
