"""Random Forest baseline for five-class NSL-KDD intrusion detection."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from .evaluation import multiclass_metrics, save_evaluation_plots, save_metrics


def train_random_forest(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    config: dict[str, object],
    seed: int,
) -> RandomForestClassifier:
    """Train a class-balanced Random Forest baseline."""

    model = RandomForestClassifier(
        n_estimators=int(config["n_estimators"]),
        max_depth=(
            None if config.get("max_depth") is None else int(config["max_depth"])
        ),
        min_samples_leaf=int(config["min_samples_leaf"]),
        max_features=str(config["max_features"]),
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
    )
    model.fit(features, labels)
    return model


def evaluate_random_forest(
    model: RandomForestClassifier,
    feature_pipeline: Pipeline,
    features: np.ndarray,
    labels: np.ndarray,
    *,
    class_names: list[str],
    selected_features: list[str],
    model_dir: Path,
    results_dir: Path,
) -> dict[str, object]:
    """Evaluate and save the fitted baseline and its artifacts."""

    probabilities = model.predict_proba(features)
    metrics = multiclass_metrics(labels, probabilities, class_names)
    save_metrics(metrics, results_dir / "metrics" / "random-forest.json")
    save_evaluation_plots(
        labels,
        probabilities,
        class_names=class_names,
        model_name="Random Forest",
        output_dir=results_dir / "figures",
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "feature_pipeline": feature_pipeline,
            "model": model,
            "class_names": class_names,
            "selected_features": selected_features,
        },
        model_dir / "random-forest.joblib",
    )
    return metrics

