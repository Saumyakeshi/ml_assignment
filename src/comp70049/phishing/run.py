"""Run the complete Section 1 classic-ML and LSTM comparison."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from .classic import evaluate_classic_model, train_classic_model
from .data import load_spamassassin_corpus, save_split_manifest, stratified_splits
from .lstm import train_and_evaluate_lstm


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/section_01.json"),
        help="Path to the JSON experiment configuration.",
    )
    parser.add_argument(
        "--skip-lstm",
        action="store_true",
        help="Run only the TF-IDF Logistic Regression baseline.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Override the configured LSTM epoch count for a quick smoke test.",
    )
    return parser.parse_args()


def _resolve(path_value: str) -> Path:
    return Path(path_value).resolve()


def main() -> None:
    arguments = _arguments()
    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    if arguments.epochs is not None:
        if arguments.epochs < 1:
            raise ValueError("--epochs must be at least 1.")
        config["lstm"]["epochs"] = arguments.epochs

    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)

    frame = load_spamassassin_corpus(_resolve(config["data_dir"]))
    splits = stratified_splits(
        frame,
        test_fraction=float(config["test_fraction"]),
        validation_fraction=float(config["validation_fraction"]),
        seed=seed,
    )
    save_split_manifest(splits, _resolve(config["processed_dir"]))

    results_dir = _resolve(config["results_dir"])
    models_dir = _resolve(config["models_dir"])
    classic_model = train_classic_model(splits.train, config["classic"])
    classic_metrics = evaluate_classic_model(
        classic_model,
        splits.test,
        model_dir=models_dir,
        results_dir=results_dir,
    )

    comparison_rows = [
        {
            "model": "TF-IDF Logistic Regression",
            **{key: classic_metrics[key] for key in (
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "average_precision",
            )},
        }
    ]

    if not arguments.skip_lstm:
        lstm_metrics = train_and_evaluate_lstm(
            splits.train,
            splits.validation,
            splits.test,
            config=config["lstm"],
            seed=seed,
            model_dir=models_dir,
            results_dir=results_dir,
        )
        comparison_rows.append(
            {
                "model": "LSTM",
                **{key: lstm_metrics[key] for key in (
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "roc_auc",
                    "average_precision",
                )},
            }
        )

    comparison = pd.DataFrame(comparison_rows)
    results_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(results_dir / "model-comparison.csv", index=False)

    summary = {
        "records_after_deduplication": len(frame),
        "train_records": len(splits.train),
        "validation_records": len(splits.validation),
        "test_records": len(splits.test),
        "class_counts": {
            "ham": int((frame["label"] == 0).sum()),
            "spam": int((frame["label"] == 1).sum()),
        },
        "models_run": comparison["model"].tolist(),
    }
    (results_dir / "run-summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(comparison.to_string(index=False))
    print(f"\nArtifacts written to {results_dir}")


if __name__ == "__main__":
    main()

