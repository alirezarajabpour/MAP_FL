import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Subset
import torchvision.transforms as transforms
from prometheus_client import start_http_server, Gauge
import flwr as fl
from models import Net
from copy import deepcopy

# --- Prometheus Metrics ---
CLIENT_ID = os.getenv("CLIENT_ID", "0")
ACCURACY_GAUGE = Gauge('flower_client_accuracy', 'Current validation accuracy of the client model', ['client_id'])
LOSS_GAUGE = Gauge('flower_client_loss', 'Current validation loss of the client model', ['client_id'])
TRAINING_ACCURACY_GAUGE = Gauge('flower_client_training_accuracy', 'Current training accuracy during a local epoch', ['client_id'])
TRAINING_LOSS_GAUGE = Gauge('flower_client_training_loss', 'Current training loss during a local epoch', ['client_id'])

# --- Global Constants ---
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TOTAL_CLASSES = 21 # 9 for PathMNIST + 7 for DermaMNIST + a few more for Retina, adjust as needed

# --- Restricted Softmax (RS) Loss Implementation ---
class RestrictedSoftmaxLoss(nn.Module):
    def __init__(self, observed_classes_indices, alpha=0.1):
        super().__init__()
        self.observed_indices = torch.tensor(observed_classes_indices, device=DEVICE)
        self.alpha = alpha
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        # Create scaling factors
        scaling_factors = torch.full((logits.size(1),), self.alpha, device=DEVICE)
        scaling_factors[self.observed_indices] = 1.0

        # Apply scaling factors
        scaled_logits = logits * scaling_factors
        
        return self.cross_entropy(scaled_logits, labels)

# class RestrictedSoftmaxLoss(nn.Module):
#     def __init__(self, observed_classes_indices):
#         super().__init__()
#         self.observed_indices = torch.tensor(observed_classes_indices, device=DEVICE)
#         self.cross_entropy = nn.CrossEntropyLoss()

#     def forward(self, logits, labels):
#         # Mask logits outside the observed classes
#         mask = torch.full_like(logits, -1e9)  # Very low score for unobserved
#         mask[:, self.observed_indices] = logits[:, self.observed_indices]
#         return self.cross_entropy(mask, labels)

# --- MAP Client Implementation ---
class MedicalMAPClient(fl.client.NumPyClient):
    def __init__(self, client_id, model, hpm, full_dataset, observed_classes_indices):
        self.client_id = client_id
        self.model = model.to(DEVICE)
        self.hpm = hpm.to(DEVICE) if hpm is not None else None
        self.full_dataset = full_dataset
        self.observed_classes_indices = observed_classes_indices
        self.round_counter = 0

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)


    # def set_parameters(self, parameters):
    #     """
    #     Sets the parameters of the main training model.
    #     Crucially, this does NOT touch the Inherited Private Model (HPM).
    #     """
    #     params_dict = zip(self.model.state_dict().keys(), parameters)
    #     state_dict = {k: torch.tensor(v) for k, v in params_dict}
    #     self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        print(f"[Client {self.client_id}] Starting training round...")
        self.set_parameters(parameters)

        if self.hpm is None:
            # The HPM starts as a copy of the first global model
            self.hpm = deepcopy(self.model)

        num_samples_train = int(len(self.full_dataset) * 0.8)
        train_indices = np.random.choice(len(self.full_dataset), num_samples_train, replace=False)
        #train_subset = Subset(self.full_dataset, train_indices)
        train_subset = Subset(self.full_dataset, [i for i, (_, y) in enumerate(self.full_dataset) if y.item() in self.observed_classes_indices])
        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
        print(f"[Client {self.client_id}] Using {len(train_subset)} samples for training this round.")

        # Print label distribution for training
        label_counts = {}
        for _, label in train_subset:
            label = label.item()
            label_counts[label] = label_counts.get(label, 0) + 1
        print(f"[Client {self.client_id}] Label distribution in training set: {label_counts}")

        # --- Stage 1: Training for Aggregation with Restricted Softmax ---
        print(f"[Client {self.client_id}] Stage 1: Training with Restricted Softmax")
        rs_loss = RestrictedSoftmaxLoss(self.observed_classes_indices, alpha=0.1)

        self.train(self.model, train_loader, epochs=config["local_epochs"] // 2, loss_fn=rs_loss, hpm=None)
        # Get parameters for aggregation after stage 1
        aggregation_params = self.get_parameters(config={})

        # --- Stage 2: Training for Personalization with HPM ---
        print(f"[Client {self.client_id}] Stage 2: Training with HPM")
        standard_loss = nn.CrossEntropyLoss()
        # In a real scenario, HPM would be updated and persisted. Here we use the just-trained model.

        self.train(self.model, train_loader, epochs=config["local_epochs"] // 2, loss_fn=rs_loss, hpm=self.hpm)

        # --- Update HPM using a moving average ---
        mu = 0.9
        with torch.no_grad():
            for param_hpm, param_model in zip(self.hpm.parameters(), self.model.parameters()):
                param_hpm.data = mu * param_hpm.data + (1 - mu) * param_model.data

        self.round_counter += 1
        return aggregation_params, len(train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        # 1. Sample indices first
        num_samples_test = int(len(self.full_dataset) * 0.2)
        test_indices = np.random.choice(len(self.full_dataset), num_samples_test, replace=False)

        # 2. Restrict them to only observed class labels
        filtered_test_indices = [
            i for i in test_indices 
            if self.full_dataset[i][1].item() in self.observed_classes_indices
        ]

        # 3. Build subset and dataloader
        test_subset = Subset(self.full_dataset, filtered_test_indices)
        test_loader = DataLoader(test_subset, batch_size=32)

        # 4. Print label distribution
        label_counts = {}
        for _, label in test_subset:
            label = label.item()
            label_counts[label] = label_counts.get(label, 0) + 1
        print(f"[Client {self.client_id}] Label distribution in test set: {label_counts}")

        # 5. Evaluate model
        #loss, accuracy = self.test(self.model, test_loader)
        loss, accuracy = self.test(self.hpm, test_loader)

        # 6. Update Prometheus metrics
        ACCURACY_GAUGE.labels(client_id=self.client_id).set(accuracy)
        LOSS_GAUGE.labels(client_id=self.client_id).set(loss)

        print(f"[Client {self.client_id}] Evaluate on local data: Loss {loss:.4f}, Accuracy {accuracy:.4f}")
        return loss, len(test_loader.dataset), {"accuracy": accuracy}

    def compute_class_weights(labels, observed_classes):
        counts = np.bincount(labels)
        weights = np.zeros(max(observed_classes)+1)
        for c in observed_classes:
            weights[c] = 1.0 / (counts[c] if counts[c] > 0 else 1)
        weights = torch.tensor(weights, dtype=torch.float32, device=DEVICE)
        return weights

    def save_final_hpm(self):
        """Saves the final HPM to a shared volume."""
        model_path = f"/model/hpm_client_{self.client_id}.pth"
        torch.save(self.hpm.state_dict(), model_path)
        print(f"Client {self.client_id}: Final HPM saved to {model_path}")

    # def train(self, model, train_loader, epochs, loss_fn, hpm=None):
        # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        # for _ in range(epochs):
        #     for images, labels in train_loader:
        #         images, labels = images.to(DEVICE), labels.squeeze().long().to(DEVICE)
        #         optimizer.zero_grad()
                
        #         # Standard classification loss
        #         outputs = model(images)
        #         loss = loss_fn(outputs, labels)
                
        #         # Knowledge Distillation loss from HPM (if applicable)
        #         if hpm is not None:
        #             with torch.no_grad():
        #                 hpm_outputs = hpm(images)
                    
        #             kd_loss = nn.KLDivLoss(reduction="batchmean")(
        #                 F.log_softmax(outputs / 4.0, dim=1),
        #                 F.softmax(hpm_outputs / 4.0, dim=1)
        #             )
        #             loss += kd_loss * 0.1 # Lambda from paper

        #         loss.backward()
        #         optimizer.step()

    def train(self, model, train_loader, epochs, loss_fn, hpm=None):
        """
        A unified training function that handles all MAP components:
        - A specific loss function (either RS or standard).
        - Optional Knowledge Distillation from an HPM.
        - Detailed logging for each local epoch.
        """
        model.train()
        # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            for images, labels in train_loader:
                images, labels = images.to(DEVICE), labels.squeeze().long().to(DEVICE)
                optimizer.zero_grad()
                
                # --- Forward pass ---
                outputs = model(images)
                
                # --- Loss Calculation ---
                # 1. Base loss (either RS or standard CrossEntropy)
                loss = loss_fn(outputs, labels)
                
                # 2. Add Knowledge Distillation loss from HPM (if applicable)
                if hpm is not None:
                    with torch.no_grad():
                        hpm_outputs = hpm(images)

                    kd_loss = nn.KLDivLoss(reduction="batchmean")(
                        F.log_softmax(outputs / 4.0, dim=1),
                        F.softmax(hpm_outputs / 4.0, dim=1)
                    )
                    # Add the KD loss, weighted by a lambda factor
                    loss += kd_loss * 0.1 

                # --- Backward pass and optimization ---
                loss.backward()
                optimizer.step()

                # Accumulate stats for logging
                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            # --- Log metrics at the end of each epoch ---
            epoch_loss = running_loss / total
            epoch_acc = correct / total

            print(f"[Client {self.client_id}] Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}")

            # Update Prometheus metrics for Grafana
            TRAINING_LOSS_GAUGE.labels(client_id=self.client_id).set(epoch_loss)
            TRAINING_ACCURACY_GAUGE.labels(client_id=self.client_id).set(epoch_acc)


    def test(self, model, test_loader):
        """Calculates the loss and accuracy of a model on a test set."""
        self.model.to(DEVICE)
        model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        correct, total, loss = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.squeeze().long().to(DEVICE)
                outputs = model(images)
                # Mask logits for unobserved classes
                mask = torch.full_like(outputs, -1e9)
                mask[:, self.observed_classes_indices] = outputs[:, self.observed_classes_indices]
                loss += criterion(mask, labels).item()
                _, predicted = torch.max(mask.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total

        print(f"[Client {self.client_id}] Test Loss {loss:.4f} : {loss / len(test_loader):.4f} | Test Acc: {accuracy:.4f}")
        print(f"[Client {self.client_id}] Predicted: {predicted.tolist()} | True: {labels.tolist()}")

        return loss / len(test_loader), accuracy


# def main():
#     client_id = os.getenv("CLIENT_ID", "1")
#     print(f"Starting client {client_id}")
    
#     # --- Load Data ---
#     # This logic assumes data is pre-partitioned
#     data_dir = f"/app/data/client_{client_id}/"
#     specialty = "pathmnist.npz" # This should be dynamic based on client_id
#     if int(client_id) > 1: specialty = "dermamnist.npz"
#     if int(client_id) > 2: specialty = "retinamnist.npz"

#     data = np.load(os.path.join(data_dir, specialty))
#     X_train, y_train = data['train_images'], data['train_labels']
    
#     # Create observed class indices for this client
#     # This needs to be mapped to the global class indices
#     # For simplicity, we assume a fixed mapping here
#     observed_classes = np.unique(y_train.squeeze())

#     if "retinamnist" in specialty:
#         num_classes = 3
#     elif "dermamnist" in specialty:
#         num_classes = 7
#     elif "pathmnist" in specialty:
#         num_classes = 9
#     else:
#         num_classes = 2  # default fallback

#     # if specialty == "dermamnist.npz":
#     #     observed_classes += 9
#     # elif specialty == "retinamnist.npz":
#     #     observed_classes += 16

#     observed_classes = observed_classes.astype(int).tolist()

#     # Create DataLoaders
#     X_train = torch.from_numpy(X_train).permute(0, 3, 1, 2).float()
#     y_train = torch.from_numpy(y_train)
    
#     train_dataset = TensorDataset(X_train, y_train)
#     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#     # Using training set for test here for simplicity. In reality, you'd have a test split.
#     test_loader = DataLoader(train_dataset, batch_size=32)

#     # --- Models ---
#     model = Net(in_channels=3, num_classes=TOTAL_CLASSES)
#     hpm = Net(in_channels=3, num_classes=TOTAL_CLASSES) # Inherited Private Model

#     # --- Start Prometheus Server ---
#     start_http_server(8000)

#     # --- Start Flower Client ---
#     client = MedicalMAPClient(client_id, model, hpm, train_loader, test_loader, observed_classes)
#     fl.client.start_numpy_client(server_address="server:8080", client=client)

def main():
    """
    Initializes and runs a Flower client with all corrections:
    1. Automatic specialty detection.
    2. Correct global label remapping.
    3. Unified 3-channel data transformation.
    """
    # --- 1. Configuration ---
    client_id_str = os.getenv("CLIENT_ID", "1")
    client_id = int(client_id_str)
    client_data_dir = f"/app/data/client_{client_id}"

    print(f"Starting client {client_id}, looking for data in {client_data_dir}")

    # --- 2. Automatic Specialty and Label Offset Detection ---
    specialty_map = {
        "pathmnist.npz": {"name": "pathology", "label_offset": 0},
        "dermamnist.npz": {"name": "dermatology", "label_offset": 9},
        "retinamnist.npz": {"name": "retina", "label_offset": 16}
    }

    try:
        data_filename = [f for f in os.listdir(client_data_dir) if f.endswith('.npz')][0]
        data_path = os.path.join(client_data_dir, data_filename)
        specialty_config = specialty_map[data_filename]
        specialty_name = specialty_config["name"]
        label_offset = specialty_config["label_offset"]
    except (FileNotFoundError, IndexError, KeyError) as e:
        print(f"FATAL: Error configuring client from data file. Error: {e}")
        return

    # --- 3. Data Loading and Preprocessing ---
    data = np.load(data_path)
    X_train, y_train_local = data['train_images'], data['train_labels']
    y_train_global = y_train_local + label_offset

    observed_classes = np.unique(y_train_global.squeeze()).astype(int).tolist()
    print(f"[Client {client_id}] Detected Specialty: {specialty_name}, Global Labels: {observed_classes}")

    # CORRECTED: Unified transformation for all data (handles RGB and Grayscale)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((28, 28)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    X_train_transformed = torch.stack([transform(img) for img in X_train])
    y_train = torch.from_numpy(y_train_global)

    full_dataset = TensorDataset(X_train_transformed, y_train)
    # train_loader = DataLoader(full_dataset, batch_size=32, shuffle=True)
    # test_loader = DataLoader(full_dataset, batch_size=32)

    print(f"[Client {client_id}] y_train_local sample: {y_train_local[:10]}")
    print(f"[Client {client_id}] label_offset: {label_offset}")
    print(f"[Client {client_id}] y_train_global sample: {y_train_global[:10]}")
    print(f"[Client {client_id}] observed_classes: {observed_classes}")

    # --- 4. Model & Client Initialization ---
    model = Net(in_channels=3, num_classes=TOTAL_CLASSES)
    hpm = Net(in_channels=3, num_classes=TOTAL_CLASSES)

    start_http_server(8000)

    client = MedicalMAPClient(
        client_id=client_id_str,
        model=model,
        hpm=hpm,
        # train_loader=train_loader,
        # test_loader=test_loader,
        full_dataset=full_dataset,
        observed_classes_indices=observed_classes,
    )
    fl.client.start_numpy_client(server_address="server:8080", client=client)


if __name__ == "__main__":
    main()
