"""A compact PyTorch LSTM text classifier for Section 1."""

from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset

from .evaluation import binary_metrics, save_evaluation_plots, save_metrics
from .preprocessing import tokenize


PAD_TOKEN = "<PAD>"
UNKNOWN_TOKEN = "<UNK>"


@dataclass(frozen=True)
class Vocabulary:
    tokens: list[str]

    @property
    def token_to_index(self) -> dict[str, int]:
        return {token: index for index, token in enumerate(self.tokens)}

    @classmethod
    def build(cls, texts: Iterable[str], max_size: int) -> "Vocabulary":
        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(tokenize(text))
        most_common = [token for token, _ in counts.most_common(max_size - 2)]
        return cls(tokens=[PAD_TOKEN, UNKNOWN_TOKEN, *most_common])

    def encode(self, text: str, max_length: int) -> tuple[list[int], int]:
        mapping = self.token_to_index
        unknown = mapping[UNKNOWN_TOKEN]
        encoded = [mapping.get(token, unknown) for token in tokenize(text)[:max_length]]
        if not encoded:
            encoded = [unknown]
        length = len(encoded)
        encoded.extend([mapping[PAD_TOKEN]] * (max_length - length))
        return encoded, length


class EmailDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        frame: pd.DataFrame,
        vocabulary: Vocabulary,
        max_length: int,
    ) -> None:
        self.labels = frame["label"].astype(float).tolist()
        self.examples = [vocabulary.encode(text, max_length) for text in frame["text"]]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded, length = self.examples[index]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(length, dtype=torch.long),
            torch.tensor(self.labels[index], dtype=torch.float32),
        )


class EmailLSTM(nn.Module):
    def __init__(
        self,
        *,
        vocabulary_size: int,
        embedding_dimension: int,
        hidden_dimension: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            vocabulary_size,
            embedding_dimension,
            padding_idx=0,
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dimension,
            hidden_size=hidden_dimension,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dimension, 1)

    def forward(self, token_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(token_ids)
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.lstm(packed)
        return self.output(self.dropout(hidden[-1])).squeeze(1)


def _predict(
    model: EmailLSTM,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities: list[float] = []
    labels: list[int] = []
    with torch.no_grad():
        for token_ids, lengths, batch_labels in loader:
            logits = model(token_ids.to(device), lengths.to(device))
            batch_probabilities = torch.sigmoid(logits).cpu().numpy()
            probabilities.extend(batch_probabilities.tolist())
            labels.extend(batch_labels.numpy().astype(int).tolist())
    return np.asarray(labels), np.asarray(probabilities)


def _save_training_plot(history: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [int(item["epoch"]) for item in history]
    losses = [item["training_loss"] for item in history]
    validation_f1 = [item["validation_f1"] for item in history]
    figure, primary_axis = plt.subplots(figsize=(6.5, 4.5))
    secondary_axis = primary_axis.twinx()
    primary_axis.plot(epochs, losses, marker="o", color="tab:blue", label="Training loss")
    secondary_axis.plot(
        epochs,
        validation_f1,
        marker="s",
        color="tab:orange",
        label="Validation F1",
    )
    primary_axis.set(xlabel="Epoch", ylabel="Training loss")
    secondary_axis.set(ylabel="Validation F1", ylim=(0, 1.02))
    primary_axis.grid(alpha=0.25)
    lines = primary_axis.lines + secondary_axis.lines
    primary_axis.legend(lines, [line.get_label() for line in lines], loc="center right")
    figure.suptitle("LSTM training history")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def train_and_evaluate_lstm(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    *,
    config: dict[str, object],
    seed: int,
    model_dir: Path,
    results_dir: Path,
) -> dict[str, object]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    vocabulary = Vocabulary.build(
        train["text"],
        max_size=int(config["max_vocabulary"]),
    )
    max_length = int(config["max_sequence_length"])
    train_dataset = EmailDataset(train, vocabulary, max_length)
    validation_dataset = EmailDataset(validation, vocabulary, max_length)
    test_dataset = EmailDataset(test, vocabulary, max_length)
    generator = torch.Generator().manual_seed(seed)
    batch_size = int(config["batch_size"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmailLSTM(
        vocabulary_size=len(vocabulary.tokens),
        embedding_dimension=int(config["embedding_dimension"]),
        hidden_dimension=int(config["hidden_dimension"]),
        dropout=float(config["dropout"]),
    ).to(device)

    positive_count = float(train["label"].sum())
    negative_count = float(len(train) - positive_count)
    positive_weight = torch.tensor([negative_count / positive_count], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]))

    history: list[dict[str, float]] = []
    best_f1 = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    patience = int(config["early_stopping_patience"])

    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        total_loss = 0.0
        observations = 0
        for token_ids, lengths, labels in train_loader:
            token_ids = token_ids.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(token_ids, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(labels)
            observations += len(labels)

        validation_labels, validation_probabilities = _predict(
            model,
            validation_loader,
            device,
        )
        validation_predictions = (validation_probabilities >= 0.5).astype(int)
        validation_f1 = float(
            f1_score(validation_labels, validation_predictions, zero_division=0)
        )
        history.append(
            {
                "epoch": float(epoch),
                "training_loss": total_loss / observations,
                "validation_f1": validation_f1,
            }
        )

        if validation_f1 > best_f1:
            best_f1 = validation_f1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is None:
        raise RuntimeError("LSTM training did not produce a model state.")
    model.load_state_dict(best_state)

    labels, probabilities = _predict(model, test_loader, device)
    metrics = binary_metrics(labels, probabilities)
    metrics["best_validation_f1"] = best_f1
    metrics["epochs_completed"] = len(history)
    metrics["device"] = str(device)
    metrics["vocabulary_size"] = len(vocabulary.tokens)

    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": {key: value.cpu() for key, value in model.state_dict().items()},
            "vocabulary": vocabulary.tokens,
            "model_config": {
                "embedding_dimension": int(config["embedding_dimension"]),
                "hidden_dimension": int(config["hidden_dimension"]),
                "dropout": float(config["dropout"]),
                "max_sequence_length": max_length,
            },
        },
        model_dir / "lstm.pt",
    )
    save_metrics(metrics, results_dir / "metrics" / "lstm.json")
    save_evaluation_plots(
        labels,
        probabilities,
        model_name="LSTM",
        output_dir=results_dir / "figures",
    )
    _save_training_plot(history, results_dir / "figures" / "lstm-training-history.png")
    return metrics

