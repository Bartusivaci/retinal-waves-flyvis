from flyvis import EnsembleView
import inspect

ensemble = EnsembleView("flow/0000")

model_id = list(ensemble.keys())[0]
model_view = ensemble[model_id]

network = model_view.init_network(checkpoint="best")

print("Network type:", type(network))

print("\nMethods/attributes related to stimulus, response, simulate, input:")
for name in dir(network):
    lower = name.lower()
    if any(word in lower for word in ["stim", "response", "simulate", "input", "forward"]):
        print(" -", name)

print("\nForward signature:")
print(inspect.signature(network.forward))

if hasattr(network, "simulate"):
    print("\nSimulate signature:")
    print(inspect.signature(network.simulate))
