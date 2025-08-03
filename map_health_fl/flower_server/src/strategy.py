import flwr as fl
import torch
from collections import OrderedDict

# Assuming the same model definition is available
# In a real project, this would be in a shared library
class Net(torch.nn.Module): # Simplified Net for loading state
    def __init__(self, num_classes=18):
        super(Net, self).__init__()
        self.fc = torch.nn.Linear(1, num_classes) # Dummy layer

    def forward(self, x):
        return self.fc(x)

class FedAvgWithModelSaving(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):

        # Aggregate parameters as usual
        aggregated_parameters, _ = super().aggregate_fit(server_round, results, failures)

        # Aggregate metrics (e.g., weighted average accuracy)
        # metrics = [r.metrics for _, _, r in results if hasattr(r, "metrics")]
        # weights = [num_examples for _, num_examples, _ in results]
        # total = sum(weights)
        # avg_accuracy = sum(m["accuracy"] * w for m, w in zip(metrics, weights)) / total

        # aggregated_metrics = {"avg_accuracy": avg_accuracy}

        aggregated_metrics = {}

        # Save the model after the final round
        if aggregated_parameters is not None and server_round == 4: # Assuming 10 rounds
            print(f"Round {server_round}: Saving final aggregated model...")

            # Convert aggregated_parameters back to a state_dict
            params_dict = zip(Net().state_dict().keys(), fl.common.parameters_to_ndarrays(aggregated_parameters))
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})

            # Save the state_dict to a file accessible by the webapp
            torch.save(state_dict, "/model/global_model.pth")
            print("Final model saved.")

        return aggregated_parameters, aggregated_metrics
