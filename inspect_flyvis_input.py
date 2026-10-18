from flyvis import EnsembleView

ensemble = EnsembleView("flow/0000")

model_id = list(ensemble.keys())[0]
model_view = ensemble[model_id]

print("Model ID:", model_id)
print("NetworkView type:", type(model_view))

print("\nAvailable attributes containing 'stim', 'input', or 'network':")
for name in dir(model_view):
    lower = name.lower()
    if "stim" in lower or "input" in lower or "network" in lower:
        print(" -", name)

print("\nConnectome info:")
connectome = model_view.connectome_view

for attr in ["n_nodes", "nodes", "node_indexer", "unique_cell_types"]:
    if hasattr(connectome, attr):
        value = getattr(connectome, attr)
        try:
            print(attr, ":", len(value))
        except TypeError:
            print(attr, ":", value)
