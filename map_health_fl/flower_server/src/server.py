import flwr as fl
from strategy import FedAvgWithModelSaving


def main():
    # Define strategy
    strategy = FedAvgWithModelSaving(
        on_fit_config_fn=lambda rnd: {"local_epochs": 8},
        fraction_fit=1,  # Train on 10% of clients per round
        min_fit_clients=3, # Minimum of 3 clients for training
        min_available_clients=3, # Wait for 3 clients to be available
    )

    # Start Flower server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
