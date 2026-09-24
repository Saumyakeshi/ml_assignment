"""One-dimensional CNN for classifying NSL-KDD feature vectors."""

from __future__ import annotations

import copy
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .evaluation import multiclass_metrics, save_evaluation_plots, save_metrics


def train_and_evaluate_cnn(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    *,
    class_names: list[str],
    config: dict[str, object],
    seed: int,
    model_dir: Path,
    results_dir: Path,
) -> dict[str, object]:
    """Train a class-weighted 1D CNN and save test-set results."""

    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as error:
        raise RuntimeError(
            "PyTorch is required for the CNN. Install the deep dependencies with "
            "`pip install -e .[deep]`."
        ) from error

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class FeatureCNN(nn.Module):
        def __init__(self, classes: int) -> None:
            super().__init__()
            self.network = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1),
                nn.Flatten(),
                nn.Dropout(float(config["dropout"])),
                nn.Linear(64, classes),
            )

        def forward(self, values):
            return self.network(values.unsqueeze(1))

    def loader(features: np.ndarray, labels: np.ndarray, shuffle: bool) -> DataLoader:
        dataset = TensorDataset(
            torch.from_numpy(features), torch.from_numpy(labels.astype(np.int64))
        )
        generator = torch.Generator().manual_seed(seed)
        return DataLoader(
            dataset,
            batch_size=int(config["batch_size"]),
            shuffle=shuffle,
            generator=generator if shuffle else None,
        )

    train_loader = loader(train_features, train_labels, True)
    validation_loader = loader(validation_features, validation_labels, False)
    test_loader = loader(test_features, test_labels, False)

    counts = np.bincount(train_labels, minlength=len(class_names)).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError("Every class must be represented in the CNN training set.")
    raw_weights = (len(train_labels) / (len(class_names) * counts)) ** float(
        config["class_weight_power"]
    )
    class_weights = raw_weights / raw_weights.mean()

    model = FeatureCNN(len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )

    def evaluate(data_loader: DataLoader) -> tuple[float, np.ndarray]:
        model.eval()
        total_loss = 0.0
        probabilities: list[np.ndarray] = []
        with torch.no_grad():
            for values, labels in data_loader:
                values = values.to(device)
                labels = labels.to(device)
                logits = model(values)
                total_loss += criterion(logits, labels).item() * len(labels)
                probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
        return total_loss / len(data_loader.dataset), np.concatenate(probabilities)

    history: list[dict[str, float]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_validation_f1 = -1.0
    stale_epochs = 0
    epochs = int(config["epochs"])

    for epoch in range(1, epochs + 1):
        model.train()
        training_loss = 0.0
        for values, labels in train_loader:
            values = values.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(values)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            training_loss += loss.item() * len(labels)

        training_loss /= len(train_loader.dataset)
        validation_loss, validation_probabilities = evaluate(validation_loader)
        validation_metrics = multiclass_metrics(
            validation_labels, validation_probabilities, class_names
        )
        validation_f1 = float(validation_metrics["macro_f1"])
        history.append(
            {
                "epoch": epoch,
                "training_loss": training_loss,
                "validation_loss": validation_loss,
                "validation_macro_f1": validation_f1,
            }
        )
        print(
            f"Epoch {epoch}/{epochs}: train_loss={training_loss:.4f} "
            f"validation_loss={validation_loss:.4f} "
            f"validation_macro_f1={validation_f1:.4f}"
        )
        if validation_f1 > best_validation_f1:
            best_validation_f1 = validation_f1
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(config["early_stopping_patience"]):
                print("Early stopping triggered.")
                break

    model.load_state_dict(best_state)
    _, test_probabilities = evaluate(test_loader)
    metrics = multiclass_metrics(test_labels, test_probabilities, class_names)
    metrics["device"] = str(device)
    metrics["best_validation_macro_f1"] = best_validation_f1
    metrics["epochs_completed"] = len(history)
    save_metrics(metrics, results_dir / "metrics" / "cnn.json")
    save_evaluation_plots(
        test_labels,
        test_probabilities,
        class_names=class_names,
        model_name="1D CNN",
        output_dir=results_dir / "figures",
    )

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    epoch_values = [row["epoch"] for row in history]
    axes[0].plot(epoch_values, [row["training_loss"] for row in history], label="Train")
    axes[0].plot(epoch_values, [row["validation_loss"] for row in history], label="Validation")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="CNN loss history")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(
        epoch_values,
        [row["validation_macro_f1"] for row in history],
        color="#E45756",
    )
    axes[1].set(xlabel="Epoch", ylabel="Macro F1", title="Validation macro F1")
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    history_path = results_dir / "figures" / "cnn-training-history.png"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(history_path, dpi=180)
    plt.close(figure)

    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_features": int(train_features.shape[1]),
            "class_names": class_names,
            "config": config,
        },
        model_dir / "cnn.pt",
    )
    return metrics

