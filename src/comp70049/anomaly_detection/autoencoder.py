"""Dense autoencoder for reconstruction-error anomaly detection."""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .evaluation import (
    binary_anomaly_metrics,
    save_evaluation_artifacts,
    select_f1_threshold,
)


class DenseAutoencoder(nn.Module):
    """Symmetric fully connected autoencoder for tabular flow features."""

    def __init__(
        self,
        input_dimension: int,
        hidden_dimensions: list[int],
        latent_dimension: int,
        dropout: float,
    ) -> None:
        super().__init__()
        encoder_layers: list[nn.Module] = []
        previous = input_dimension
        for width in hidden_dimensions:
            encoder_layers.extend(
                [nn.Linear(previous, width), nn.ReLU(), nn.Dropout(dropout)]
            )
            previous = width
        encoder_layers.append(nn.Linear(previous, latent_dimension))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: list[nn.Module] = []
        previous = latent_dimension
        for width in reversed(hidden_dimensions):
            decoder_layers.extend(
                [nn.Linear(previous, width), nn.ReLU(), nn.Dropout(dropout)]
            )
            previous = width
        decoder_layers.append(nn.Linear(previous, input_dimension))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(values))


def _loader(features: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    tensor = torch.from_numpy(np.asarray(features, dtype=np.float32))
    return DataLoader(
        TensorDataset(tensor),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


def reconstruction_scores(
    model: nn.Module,
    features: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Calculate per-record mean squared reconstruction error."""

    model.eval()
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for (batch,) in _loader(features, batch_size, shuffle=False):
            batch = batch.to(device)
            reconstructed = model(batch)
            loss = torch.mean((reconstructed - batch) ** 2, dim=1)
            scores.append(loss.cpu().numpy())
    return np.concatenate(scores)


def train_and_evaluate_autoencoder(
    train_features: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    *,
    feature_names: list[str],
    config: dict[str, Any],
    seed: int,
    model_dir: Path,
    results_dir: Path,
) -> dict[str, Any]:
    """Train on benign data, tune threshold on validation, and test once."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = int(config["batch_size"])
    model = DenseAutoencoder(
        input_dimension=train_features.shape[1],
        hidden_dimensions=[int(value) for value in config["hidden_dimensions"]],
        latent_dimension=int(config["latent_dimension"]),
        dropout=float(config["dropout"]),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    loss_function = nn.MSELoss()
    train_loader = _loader(train_features, batch_size, shuffle=True)
    normal_validation = validation_features[np.asarray(validation_labels) == 0]
    validation_loader = _loader(normal_validation, batch_size, shuffle=False)

    history: list[dict[str, float | int]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    patience = int(config["early_stopping_patience"])

    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        training_total = 0.0
        training_count = 0
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstructed = model(batch)
            loss = loss_function(reconstructed, batch)
            loss.backward()
            optimizer.step()
            training_total += float(loss.item()) * len(batch)
            training_count += len(batch)

        model.eval()
        validation_total = 0.0
        validation_count = 0
        with torch.no_grad():
            for (batch,) in validation_loader:
                batch = batch.to(device)
                loss = loss_function(model(batch), batch)
                validation_total += float(loss.item()) * len(batch)
                validation_count += len(batch)

        train_loss = training_total / training_count
        validation_loss = validation_total / validation_count
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "normal_validation_loss": validation_loss,
            }
        )
        if validation_loss < best_validation_loss - 1e-7:
            best_validation_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_state)
    validation_scores = reconstruction_scores(
        model,
        validation_features,
        batch_size=batch_size,
        device=device,
    )
    threshold = select_f1_threshold(validation_labels, validation_scores)
    test_scores = reconstruction_scores(
        model,
        test_features,
        batch_size=batch_size,
        device=device,
    )
    metrics = binary_anomaly_metrics(test_labels, test_scores, threshold)
    metrics.update(
        {
            "validation_threshold_method": "maximum F1",
            "epochs_completed": len(history),
            "best_normal_validation_loss": best_validation_loss,
            "device": str(device),
            "training_history": history,
        }
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dimension": train_features.shape[1],
            "hidden_dimensions": config["hidden_dimensions"],
            "latent_dimension": config["latent_dimension"],
            "dropout": config["dropout"],
            "threshold": threshold,
            "feature_names": feature_names,
        },
        model_dir / "autoencoder.pt",
    )
    save_evaluation_artifacts(
        model_slug="autoencoder",
        model_name="Autoencoder",
        labels=test_labels,
        scores=test_scores,
        metrics=metrics,
        results_dir=results_dir,
    )

    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    epochs = [int(row["epoch"]) for row in history]
    plt.figure(figsize=(6.4, 4.5))
    plt.plot(epochs, [row["train_loss"] for row in history], label="Train")
    plt.plot(
        epochs,
        [row["normal_validation_loss"] for row in history],
        label="Normal validation",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Mean squared reconstruction error")
    plt.title("Autoencoder training history")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "autoencoder-training-history.png", dpi=180)
    plt.close()
    return metrics
