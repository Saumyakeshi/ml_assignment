from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from comp70049.anomaly_detection.data import (
    clean_cic_ids_frame,
    create_anomaly_partitions,
)
from comp70049.anomaly_detection.evaluation import (
    binary_anomaly_metrics,
    select_f1_threshold,
)
from comp70049.anomaly_detection.preprocessing import (
    build_feature_pipeline,
    fit_transform_features,
)


def _raw_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            " Dst Port ": [80, 80, 443, 22, 53, 25, 110, "Dst Port"],
            " Protocol ": [6, 6, 6, 6, 17, 6, 6, "Protocol"],
            " Timestamp ": [
                "01/03/2018 09:00:00",
                "01/03/2018 09:00:00",
                "bad timestamp",
                "01/03/2018 10:00:00",
                "01/03/2018 11:00:00",
                "01/03/2018 12:00:00",
                "01/03/2018 13:00:00",
                "Timestamp",
            ],
            " Flow Duration ": [10, 10, "bad", np.inf, 30, 40, 50, "Flow Duration"],
            " Tot Fwd Pkts ": [1, 1, 2, 9, 3, 4, 5, "Tot Fwd Pkts"],
            " Label ": [
                "Benign",
                "Benign",
                "Infilteration",
                "Infilteration",
                "Benign",
                "Benign",
                "Infilteration",
                "Label",
            ],
        }
    )


def test_cleaning_handles_duplicates_infinity_and_bad_values() -> None:
    cleaned = clean_cic_ids_frame(_raw_frame())

    assert len(cleaned.frame) == 6
    assert cleaned.cleaning_report["duplicate_records_removed"] == 1
    assert cleaned.cleaning_report["embedded_header_rows_removed"] == 1
    assert cleaned.cleaning_report["infinite_values_replaced_with_missing"] == 1
    assert cleaned.cleaning_report["non_numeric_values_coerced_to_missing"] == 1
    assert cleaned.cleaning_report["invalid_timestamps"] == 1
    assert set(cleaned.frame["label"]) == {0, 1}


def test_training_partition_contains_only_benign_records() -> None:
    rows = []
    for index in range(30):
        rows.append(
            {
                "Dst Port": index,
                "Protocol": 6,
                "Flow Duration": index + 1,
                "attack_label": "Benign",
                "label": 0,
            }
        )
    for index in range(10):
        rows.append(
            {
                "Dst Port": 1000 + index,
                "Protocol": 6,
                "Flow Duration": 500 + index,
                "attack_label": "Infilteration",
                "label": 1,
            }
        )
    partitions = create_anomaly_partitions(
        pd.DataFrame(rows),
        seed=42,
        max_benign_records=None,
        max_anomaly_records=None,
        benign_train_fraction=0.6,
        benign_validation_fraction=0.2,
        anomaly_validation_fraction=0.5,
    )

    assert set(partitions.train["label"]) == {0}
    assert set(partitions.validation["label"]) == {0, 1}
    assert set(partitions.test["label"]) == {0, 1}


def test_preprocessor_imputes_and_scales_without_leakage() -> None:
    train = pd.DataFrame({"a": [1.0, 2.0, np.nan], "constant": [1, 1, 1]})
    validation = pd.DataFrame({"a": [100.0], "constant": [1]})
    test = pd.DataFrame({"a": [3.0], "constant": [1]})
    pipeline = build_feature_pipeline()
    transformed = fit_transform_features(
        pipeline,
        ("a", "constant"),
        train,
        validation,
        test,
    )

    assert all(np.isfinite(values).all() for values in transformed)
    assert transformed[0].shape[1] == 2  # a plus its missingness indicator
    assert transformed[1][0, 0] > 10  # validation value did not affect fitted scaling


def test_threshold_and_assignment_metrics() -> None:
    labels = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 1.0])
    threshold = select_f1_threshold(labels, scores)
    metrics = binary_anomaly_metrics(labels, scores, threshold)

    assert threshold == pytest.approx(0.8)
    assert metrics["true_positive_rate"] == pytest.approx(1.0)
    assert metrics["false_positive_rate"] == pytest.approx(0.0)
    assert metrics["precision"] == pytest.approx(1.0)
