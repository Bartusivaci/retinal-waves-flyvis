from flyvis import EnsembleView
import inspect

ensemble = EnsembleView("flow/0000")
model_view = ensemble["flow/0000/000"]

network = model_view.init_network(checkpoint="best")

print("simulate source:\n")
print(inspect.getsource(network.simulate))

print("\nstimulus_response source:\n")
print(inspect.getsource(network.stimulus_response))
