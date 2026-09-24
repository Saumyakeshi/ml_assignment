"""Threshold selection, metrics, and plots for anomaly scores."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def select_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """Select the anomaly-score threshold with best validation F1."""

    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Threshold selection requires both benign and anomaly labels.")
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if len(thresholds) == 0:
        raise ValueError("No candidate thresholds were produced.")
    denominator = precision[:-1] + recall[:-1]
    f1_values = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    return float(thresholds[int(np.nanargmax(f1_values))])


def binary_anomaly_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Calculate assignment metrics using higher scores as more anomalous."""

    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    predictions = (scores >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "true_positive_rate": float(tp / (tp + fn)) if tp + fn else 0.0,
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "average_precision": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def save_evaluation_artifacts(
    *,
    model_slug: str,
    model_name: str,
    labels: np.ndarray,
    scores: np.ndarray,
    metrics: dict[str, Any],
    results_dir: Path,
) -> None:
    """Save JSON metrics, confusion matrix, PR curve, and score distribution."""

    metrics_dir = results_dir / "metrics"
    figures_dir = results_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / f"{model_slug}.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    predictions = (np.asarray(scores) >= float(metrics["threshold"])).astype(np.int64)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    display = ConfusionMatrixDisplay(matrix, display_labels=["Benign", "Anomaly"])
    display.plot(cmap="Blues", colorbar=False)
    plt.title(f"{model_name} confusion matrix")
    plt.tight_layout()
    plt.savefig(figures_dir / f"{model_slug}-confusion-matrix.png", dpi=180)
    plt.close()

    precision, recall, _ = precision_recall_curve(labels, scores)
    plt.figure(figsize=(6.4, 4.5))
    plt.plot(recall, precision, color="#E45756", linewidth=2)
    plt.xlabel("Recall / true positive rate")
    plt.ylabel("Precision")
    plt.title(f"{model_name} precision-recall curve")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(figures_dir / f"{model_slug}-precision-recall-curve.png", dpi=180)
    plt.close()

    labels_array = np.asarray(labels)
    scores_array = np.asarray(scores)
    plt.figure(figsize=(6.4, 4.5))
    plt.hist(scores_array[labels_array == 0], bins=60, alpha=0.65, label="Benign")
    plt.hist(scores_array[labels_array == 1], bins=60, alpha=0.65, label="Anomaly")
    plt.axvline(float(metrics["threshold"]), color="black", linestyle="--", label="Threshold")
    plt.xlabel("Anomaly score")
    plt.ylabel("Records")
    plt.title(f"{model_name} anomaly-score distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / f"{model_slug}-score-distribution.png", dpi=180)
    plt.close()
