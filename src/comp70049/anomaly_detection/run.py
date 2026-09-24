"""Run the complete Section 3 Isolation Forest and Autoencoder comparison."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .autoencoder import train_and_evaluate_autoencoder
from .classic import evaluate_isolation_forest, train_isolation_forest
from .data import create_anomaly_partitions, load_cic_ids2018, profile_partition
from .preprocessing import (
    build_feature_pipeline,
    fit_transform_features,
    transformed_feature_names,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/section_03.json"))
    parser.add_argument("--skip-autoencoder", action="store_true")
    parser.add_argument(
        "--epochs",
        type=int,
        help="Override Autoencoder epochs for a smoke run.",
    )
    return parser.parse_args()


def _resolve(value: str) -> Path:
    return Path(value).resolve()


def _save_data_quality_figures(cleaned, results_dir: Path) -> None:
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    counts = pd.Series(cleaned.cleaning_report["class_counts"])
    counts.plot.bar(
        color=["#4C78A8", "#E45756"],
        title="CIC-IDS2018 class distribution after cleaning",
        ylabel="Flow records",
        rot=0,
    )
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(figures_dir / "cleaned-class-distribution.png", dpi=180)
    plt.close()

    issues = pd.Series(
        {
            "Duplicate rows": cleaned.cleaning_report["duplicate_records_removed"],
            "Embedded headers": cleaned.cleaning_report[
                "embedded_header_rows_removed"
            ],
            "Infinite values": cleaned.cleaning_report[
                "infinite_values_replaced_with_missing"
            ],
            "Non-numeric values": cleaned.cleaning_report[
                "non_numeric_values_coerced_to_missing"
            ],
            "Invalid timestamps": cleaned.cleaning_report["invalid_timestamps"],
            "Remaining missing": cleaned.cleaning_report["remaining_missing_values"],
        }
    )
    issues.plot.bar(
        color="#F2CF5B",
        title="Raw-data quality issues handled by the pipeline",
        ylabel="Affected values or rows",
        rot=25,
    )
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(figures_dir / "data-quality-issues.png", dpi=180)
    plt.close()


def main() -> None:
    arguments = _arguments()
    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    if arguments.epochs is not None:
        if arguments.epochs < 1:
            raise ValueError("--epochs must be at least one.")
        config["autoencoder"]["epochs"] = arguments.epochs

    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    cleaned = load_cic_ids2018(
        _resolve(config["data_file"]),
        maximum_missing_fraction=float(config["maximum_missing_fraction"]),
    )
    print(
        f"Cleaned {cleaned.raw_profile['records']:,} raw rows into "
        f"{cleaned.cleaning_report['records_after_cleaning']:,} usable rows."
    )
    partitions = create_anomaly_partitions(
        cleaned.frame,
        seed=seed,
        max_benign_records=int(config["max_benign_records"]),
        max_anomaly_records=int(config["max_anomaly_records"]),
        benign_train_fraction=float(config["benign_train_fraction"]),
        benign_validation_fraction=float(config["benign_validation_fraction"]),
        anomaly_validation_fraction=float(config["anomaly_validation_fraction"]),
    )
    profiles = {
        "train": profile_partition(partitions.train),
        "validation": profile_partition(partitions.validation),
        "test": profile_partition(partitions.test),
    }
    print(
        f"Partitions: {profiles['train']['records']:,} normal-only train, "
        f"{profiles['validation']['records']:,} validation, "
        f"{profiles['test']['records']:,} test."
    )

    results_dir = _resolve(config["results_dir"])
    models_dir = _resolve(config["models_dir"])
    processed_dir = _resolve(config["processed_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    data_report = {
        "raw_profile": cleaned.raw_profile,
        "cleaning_report": cleaned.cleaning_report,
        "partitions": profiles,
    }
    (processed_dir / "data-quality-report.json").write_text(
        json.dumps(data_report, indent=2), encoding="utf-8"
    )
    _save_data_quality_figures(cleaned, results_dir)

    pipeline = build_feature_pipeline()
    train_features, validation_features, test_features = fit_transform_features(
        pipeline,
        cleaned.feature_columns,
        partitions.train,
        partitions.validation,
        partitions.test,
    )
    feature_names = transformed_feature_names(pipeline, cleaned.feature_columns)
    print(f"Preprocessing retained {len(feature_names)} transformed features.")
    (processed_dir / "selected-features.json").write_text(
        json.dumps(feature_names, indent=2), encoding="utf-8"
    )
    validation_labels = partitions.validation["label"].to_numpy(dtype=np.int64)
    test_labels = partitions.test["label"].to_numpy(dtype=np.int64)

    forest = train_isolation_forest(
        train_features,
        config=config["isolation_forest"],
        seed=seed,
    )
    print("Trained Isolation Forest and selected its threshold on validation data.")
    forest_metrics = evaluate_isolation_forest(
        forest,
        pipeline,
        validation_features,
        validation_labels,
        test_features,
        test_labels,
        feature_names=feature_names,
        model_dir=models_dir,
        results_dir=results_dir,
    )
    comparison_rows = [
        {
            "model": "Isolation Forest",
            **{
                key: forest_metrics[key]
                for key in (
                    "true_positive_rate",
                    "false_positive_rate",
                    "precision",
                    "f1",
                    "average_precision",
                    "roc_auc",
                )
            },
        }
    ]

    if not arguments.skip_autoencoder:
        print("Training the normal-only Autoencoder.")
        autoencoder_metrics = train_and_evaluate_autoencoder(
            train_features,
            validation_features,
            validation_labels,
            test_features,
            test_labels,
            feature_names=feature_names,
            config=config["autoencoder"],
            seed=seed,
            model_dir=models_dir,
            results_dir=results_dir,
        )
        comparison_rows.append(
            {
                "model": "Autoencoder",
                **{
                    key: autoencoder_metrics[key]
                    for key in (
                        "true_positive_rate",
                        "false_positive_rate",
                        "precision",
                        "f1",
                        "average_precision",
                        "roc_auc",
                    )
                },
            }
        )

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(results_dir / "model-comparison.csv", index=False)
    summary = {
        "dataset": "CSE-CIC-IDS2018",
        "source_day": "Thursday 1 March 2018",
        "experiment": "binary unsupervised anomaly detection",
        "training_policy": "benign records only",
        "threshold_policy": "maximum validation F1",
        "profiles": profiles,
        "raw_profile": cleaned.raw_profile,
        "cleaning_report": cleaned.cleaning_report,
        "transformed_feature_count": len(feature_names),
        "models_run": comparison["model"].tolist(),
        "autoencoder_epochs_requested": (
            None if arguments.skip_autoencoder else int(config["autoencoder"]["epochs"])
        ),
    }
    (results_dir / "run-summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(comparison.to_string(index=False))
    print(f"\nArtifacts written to {results_dir}")


if __name__ == "__main__":
    main()
