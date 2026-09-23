"""Shared binary-classification metrics and plots."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, object]:
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
        "classification_report": classification_report(
            labels,
            predictions,
            target_names=["ham", "spam"],
            output_dict=True,
            zero_division=0,
        ),
    }


def save_metrics(metrics: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def save_evaluation_plots(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    model_name: str,
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = (probabilities >= 0.5).astype(int)
    safe_name = model_name.lower().replace(" ", "-")
    created: list[Path] = []

    figure, axis = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        labels,
        predictions,
        display_labels=["ham", "spam"],
        cmap="Blues",
        colorbar=False,
        ax=axis,
    )
    axis.set_title(f"{model_name}: confusion matrix")
    figure.tight_layout()
    path = output_dir / f"{safe_name}-confusion-matrix.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    created.append(path)

    false_positive_rate, true_positive_rate, _ = roc_curve(labels, probabilities)
    roc_auc = roc_auc_score(labels, probabilities)
    figure, axis = plt.subplots(figsize=(5.5, 4.5))
    axis.plot(false_positive_rate, true_positive_rate, label=f"AUC = {roc_auc:.3f}")
    axis.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random")
    axis.set(xlabel="False positive rate", ylabel="True positive rate")
    axis.set_title(f"{model_name}: ROC curve")
    axis.legend(loc="lower right")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    path = output_dir / f"{safe_name}-roc-curve.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    created.append(path)

    precision, recall, _ = precision_recall_curve(labels, probabilities)
    average_precision = average_precision_score(labels, probabilities)
    figure, axis = plt.subplots(figsize=(5.5, 4.5))
    axis.plot(recall, precision, label=f"AP = {average_precision:.3f}")
    axis.set(xlabel="Recall", ylabel="Precision")
    axis.set_title(f"{model_name}: precision-recall curve")
    axis.legend(loc="lower left")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    path = output_dir / f"{safe_name}-precision-recall-curve.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    created.append(path)

    return created

