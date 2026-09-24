"""Isolation Forest baseline for Section 3."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline

from .evaluation import (
    binary_anomaly_metrics,
    save_evaluation_artifacts,
    select_f1_threshold,
)


def train_isolation_forest(
    train_features: np.ndarray,
    *,
    config: dict[str, Any],
    seed: int,
) -> IsolationForest:
    """Fit an Isolation Forest using benign training traffic only."""

    model = IsolationForest(
        n_estimators=int(config["n_estimators"]),
        max_samples=config["max_samples"],
        contamination=config.get("contamination", "auto"),
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(train_features)
    return model


def anomaly_scores(model: IsolationForest, features: np.ndarray) -> np.ndarray:
    """Return scores where larger values indicate greater abnormality."""

    return -model.score_samples(features)


def evaluate_isolation_forest(
    model: IsolationForest,
    preprocessor: Pipeline,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    *,
    feature_names: list[str],
    model_dir: Path,
    results_dir: Path,
) -> dict[str, Any]:
    """Tune the threshold on validation data and evaluate once on test data."""

    validation_scores = anomaly_scores(model, validation_features)
    threshold = select_f1_threshold(validation_labels, validation_scores)
    test_scores = anomaly_scores(model, test_features)
    metrics = binary_anomaly_metrics(test_labels, test_scores, threshold)
    metrics["validation_threshold_method"] = "maximum F1"

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "preprocessor": preprocessor,
            "threshold": threshold,
            "feature_names": feature_names,
        },
        model_dir / "isolation-forest.joblib",
    )
    save_evaluation_artifacts(
        model_slug="isolation-forest",
        model_name="Isolation Forest",
        labels=test_labels,
        scores=test_scores,
        metrics=metrics,
        results_dir=results_dir,
    )
    return metrics
