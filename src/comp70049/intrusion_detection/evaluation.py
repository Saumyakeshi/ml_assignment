"""Multiclass evaluation metrics and plots for intrusion detection."""

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
)
from sklearn.preprocessing import label_binarize


def multiclass_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
) -> dict[str, object]:
    """Calculate overall, balanced, and per-class multiclass measures."""

    predictions = probabilities.argmax(axis=1)
    class_indices = np.arange(len(class_names))
    binary_labels = label_binarize(labels, classes=class_indices)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_precision": float(
            precision_score(labels, predictions, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(labels, predictions, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(labels, predictions, average="weighted", zero_division=0)
        ),
        "macro_average_precision": float(
            average_precision_score(binary_labels, probabilities, average="macro")
        ),
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=class_indices
        ).tolist(),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=class_indices,
            target_names=class_names,
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
    class_names: list[str],
    model_name: str,
    output_dir: Path,
) -> list[Path]:
    """Save a confusion matrix and one-vs-rest precision-recall curves."""

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = probabilities.argmax(axis=1)
    safe_name = model_name.lower().replace(" ", "-")
    created: list[Path] = []

    figure, axis = plt.subplots(figsize=(7.2, 6.0))
    ConfusionMatrixDisplay.from_predictions(
        labels,
        predictions,
        labels=np.arange(len(class_names)),
        display_labels=class_names,
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

    binary_labels = label_binarize(labels, classes=np.arange(len(class_names)))
    figure, axis = plt.subplots(figsize=(7.2, 5.4))
    for index, class_name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(
            binary_labels[:, index], probabilities[:, index]
        )
        score = average_precision_score(binary_labels[:, index], probabilities[:, index])
        axis.plot(recall, precision, label=f"{class_name} (AP={score:.3f})")
    axis.set(xlabel="Recall", ylabel="Precision")
    axis.set_title(f"{model_name}: one-vs-rest precision-recall curves")
    axis.grid(alpha=0.25)
    axis.legend(loc="lower left")
    figure.tight_layout()
    path = output_dir / f"{safe_name}-precision-recall-curves.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    created.append(path)
    return created


def save_class_distribution_plot(
    counts: dict[str, int], output_path: Path
) -> Path:
    """Save the highly imbalanced class distribution on a logarithmic scale."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    names = list(counts)
    values = [counts[name] for name in names]
    figure, axis = plt.subplots(figsize=(7.0, 4.6))
    axis.bar(names, values, color=["#4C78A8", "#E45756", "#F2CF5B", "#72B7B2", "#B279A2"])
    axis.set_yscale("log")
    axis.set(xlabel="Class", ylabel="Records (log scale)")
    axis.set_title("NSL-KDD training class distribution")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path
