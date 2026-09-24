"""Run the complete Section 2 Random Forest and 1D CNN comparison."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from .classic import evaluate_random_forest, train_random_forest
from .cnn import train_and_evaluate_cnn
from .data import CLASS_NAMES, load_nsl_kdd, profile_partition
from .evaluation import save_class_distribution_plot
from .preprocessing import (
    build_feature_pipeline,
    fit_transform_features,
    selected_feature_names,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/section_02.json"),
    )
    parser.add_argument("--skip-cnn", action="store_true")
    parser.add_argument("--epochs", type=int, help="Override CNN epochs for a smoke run.")
    return parser.parse_args()


def _resolve(value: str) -> Path:
    return Path(value).resolve()


def main() -> None:
    arguments = _arguments()
    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    if arguments.epochs is not None:
        if arguments.epochs < 1:
            raise ValueError("--epochs must be at least one.")
        config["cnn"]["epochs"] = arguments.epochs

    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)

    partitions = load_nsl_kdd(
        _resolve(config["data_dir"]),
        validation_fraction=float(config["validation_fraction"]),
        seed=seed,
    )
    profiles = {
        "train": profile_partition(partitions.train),
        "validation": profile_partition(partitions.validation),
        "test": profile_partition(partitions.test),
    }
    print(
        f"Loaded {len(partitions.train):,} train, "
        f"{len(partitions.validation):,} validation, and "
        f"{len(partitions.test):,} official test records."
    )

    results_dir = _resolve(config["results_dir"])
    model_dir = _resolve(config["models_dir"])
    processed_dir = _resolve(config["processed_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    (processed_dir / "data-profile.json").write_text(
        json.dumps(profiles, indent=2), encoding="utf-8"
    )
    save_class_distribution_plot(
        profiles["train"]["class_counts"],
        results_dir / "figures" / "training-class-distribution.png",
    )

    feature_pipeline = build_feature_pipeline(int(config["selected_features"]))
    train_features, validation_features, test_features = fit_transform_features(
        feature_pipeline,
        partitions.train,
        partitions.validation,
        partitions.test,
    )
    feature_names = selected_feature_names(feature_pipeline)
    (processed_dir / "selected-features.json").write_text(
        json.dumps(feature_names, indent=2), encoding="utf-8"
    )
    print(f"Preprocessing retained {len(feature_names)} model features.")

    train_labels = partitions.train["label"].to_numpy(dtype=np.int64)
    validation_labels = partitions.validation["label"].to_numpy(dtype=np.int64)
    test_labels = partitions.test["label"].to_numpy(dtype=np.int64)

    forest = train_random_forest(
        train_features,
        train_labels,
        config=config["random_forest"],
        seed=seed,
    )
    print("Trained Random Forest baseline.")
    forest_metrics = evaluate_random_forest(
        forest,
        feature_pipeline,
        test_features,
        test_labels,
        class_names=CLASS_NAMES,
        selected_features=feature_names,
        model_dir=model_dir,
        results_dir=results_dir,
    )
    comparison_rows = [
        {
            "model": "Random Forest",
            **{
                key: forest_metrics[key]
                for key in (
                    "accuracy",
                    "macro_precision",
                    "macro_recall",
                    "macro_f1",
                    "weighted_f1",
                    "macro_average_precision",
                )
            },
        }
    ]

    if not arguments.skip_cnn:
        print("Training 1D CNN.")
        cnn_metrics = train_and_evaluate_cnn(
            train_features,
            train_labels,
            validation_features,
            validation_labels,
            test_features,
            test_labels,
            class_names=CLASS_NAMES,
            config=config["cnn"],
            seed=seed,
            model_dir=model_dir,
            results_dir=results_dir,
        )
        comparison_rows.append(
            {
                "model": "1D CNN",
                **{
                    key: cnn_metrics[key]
                    for key in (
                        "accuracy",
                        "macro_precision",
                        "macro_recall",
                        "macro_f1",
                        "weighted_f1",
                        "macro_average_precision",
                    )
                },
            }
        )

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(results_dir / "model-comparison.csv", index=False)
    summary = {
        "dataset": "NSL-KDD",
        "classification": "five-class",
        "classes": CLASS_NAMES,
        "profiles": profiles,
        "selected_feature_count": len(feature_names),
        "models_run": comparison["model"].tolist(),
        "cnn_epochs_requested": (
            None if arguments.skip_cnn else int(config["cnn"]["epochs"])
        ),
    }
    (results_dir / "run-summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(comparison.to_string(index=False))
    print(f"\nArtifacts written to {results_dir}")


if __name__ == "__main__":
    main()

