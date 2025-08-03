import os
import numpy as np
import medmnist
from medmnist import INFO

DATA_ROOT = "data"

def prepare_federated_dataset():
    """
    Downloads MedMNIST data and partitions it to simulate a federated
    environment with "incomplete classes" per client.
    """
    if os.path.exists(DATA_ROOT):
        print("Data directory already exists. Skipping download.")
        return

    print(f"Creating data directory at: {DATA_ROOT}")
    os.makedirs(DATA_ROOT, exist_ok=True)

    # Define our hospital specialties and the datasets they use
    specialties = {
        "pathology": "pathmnist",
        "dermatology": "dermamnist",
        "retina": "retinamnist",
    }

    # Assign clients to specialties
    client_assignments = {}
    for i in range(1, 11):
        client_assignments[i] = "pathology"
    for i in range(11, 21):
        client_assignments[i] = "dermatology"
    for i in range(21, 31):
        client_assignments[i] = "retina"

    print("Downloading and partitioning data...")
    for client_id, specialty in client_assignments.items():
        data_flag = specialties[specialty]
        info = INFO[data_flag]
        DataClass = getattr(medmnist, info['python_class'])

        # Download the full dataset for this specialty
        full_dataset = DataClass(split="train", download=True)

        # Simple partitioning: give each client all the data for its specialty
        # A more advanced version would split this data among the 10 clients
        client_dir = os.path.join(DATA_ROOT, f"client_{client_id}")
        os.makedirs(client_dir, exist_ok=True)

        # Save the data for this client
        np.savez_compressed(
            os.path.join(client_dir, f"{data_flag}.npz"),
            train_images=full_dataset.imgs,
            train_labels=full_dataset.labels,
        )
        print(f"Saved {data_flag} data for client {client_id} in {client_dir}")

if __name__ == "__main__":
    prepare_federated_dataset()
    print("Federated dataset preparation complete.")