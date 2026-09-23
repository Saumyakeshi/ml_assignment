"""TF-IDF plus Logistic Regression baseline for Section 1."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .evaluation import binary_metrics, save_evaluation_plots, save_metrics
from .preprocessing import normalize_text


def train_classic_model(train: pd.DataFrame, config: dict[str, object]) -> Pipeline:
    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=normalize_text,
                    stop_words="english",
                    max_features=int(config["max_features"]),
                    min_df=int(config["min_document_frequency"]),
                    max_df=float(config["max_document_frequency"]),
                    ngram_range=(
                        int(config["ngram_min"]),
                        int(config["ngram_max"]),
                    ),
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=int(config["max_iterations"]),
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(train["text"], train["label"])
    return model


def evaluate_classic_model(
    model: Pipeline,
    test: pd.DataFrame,
    *,
    model_dir: Path,
    results_dir: Path,
) -> dict[str, object]:
    probabilities = model.predict_proba(test["text"])[:, 1]
    labels = test["label"].to_numpy(dtype=int)
    metrics = binary_metrics(labels, probabilities)

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "tfidf-logistic-regression.joblib")
    save_metrics(metrics, results_dir / "metrics" / "classic.json")
    save_evaluation_plots(
        labels,
        np.asarray(probabilities),
        model_name="TF-IDF Logistic Regression",
        output_dir=results_dir / "figures",
    )
    return metrics

