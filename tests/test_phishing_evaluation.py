from __future__ import annotations

import numpy as np

from comp70049.phishing.evaluation import binary_metrics


def test_binary_metrics_for_perfect_predictions() -> None:
    labels = np.asarray([0, 0, 1, 1])
    probabilities = np.asarray([0.1, 0.2, 0.8, 0.9])

    metrics = binary_metrics(labels, probabilities)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0

