from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from comp70049.intrusion_detection.data import (
    CATEGORICAL_COLUMNS,
    CLASS_NAMES,
    FEATURE_COLUMNS,
    load_partition,
    map_attack_category,
)
from comp70049.intrusion_detection.evaluation import multiclass_metrics
from comp70049.intrusion_detection.preprocessing import (
    build_feature_pipeline,
    fit_transform_features,
)


@pytest.mark.parametrize(
    ("attack", "category"),
    [
        ("normal", "normal"),
        ("neptune", "dos"),
        ("satan", "probe"),
        ("guess_passwd", "r2l"),
        ("buffer_overflow", "u2r"),
        ("httptunnel", "u2r"),
        ("worm", "r2l"),
    ],
)
def test_map_attack_category(attack: str, category: str) -> None:
    assert map_attack_category(attack) == category


def test_unknown_attack_category_raises() -> None:
    with pytest.raises(ValueError, match="Unknown NSL-KDD attack label"):
        map_attack_category("new_attack")


def test_load_partition_maps_attack_and_label(tmp_path: Path) -> None:
    values: list[object] = []
    for column in FEATURE_COLUMNS:
        if column == "protocol_type":
            values.append("tcp")
        elif column == "service":
            values.append("http")
        elif column == "flag":
            values.append("SF")
        else:
            values.append(0)
    values.extend(["neptune", 21])
    path = tmp_path / "sample.txt"
    path.write_text(",".join(map(str, values)) + "\n", encoding="utf-8")

    frame = load_partition(path)

    assert frame.loc[0, "label_name"] == "dos"
    assert frame.loc[0, "label"] == CLASS_NAMES.index("dos")


def test_preprocessing_is_fitted_on_training_and_handles_unknown_categories() -> None:
    rows: list[dict[str, object]] = []
    for index in range(25):
        row: dict[str, object] = {}
        for feature_index, column in enumerate(FEATURE_COLUMNS):
            if column == "protocol_type":
                row[column] = "tcp" if index % 2 else "udp"
            elif column == "service":
                row[column] = "http" if index % 3 else "smtp"
            elif column == "flag":
                row[column] = "SF" if index % 2 else "REJ"
            else:
                row[column] = index + feature_index
        row["label"] = index % len(CLASS_NAMES)
        rows.append(row)
    train = pd.DataFrame(rows)
    validation = train.iloc[:5].copy()
    test = train.iloc[5:10].copy()
    test.loc[:, CATEGORICAL_COLUMNS] = ["icmp", "unknown_service", "S0"]
    pipeline = build_feature_pipeline(selected_features=8)

    train_values, validation_values, test_values = fit_transform_features(
        pipeline, train, validation, test
    )

    assert train_values.shape == (25, 8)
    assert validation_values.shape == (5, 8)
    assert test_values.shape == (5, 8)
    assert np.isfinite(test_values).all()


def test_multiclass_metrics_reports_macro_scores() -> None:
    labels = np.array([0, 1, 2, 3, 4])
    probabilities = np.eye(5, dtype=float)

    metrics = multiclass_metrics(labels, probabilities, CLASS_NAMES)

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["confusion_matrix"] == np.eye(5, dtype=int).tolist()
