"""Leakage-safe preprocessing for Section 3 anomaly detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_feature_pipeline() -> Pipeline:
    """Median-impute, remove constants, and standardise numeric flows."""

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("variance", VarianceThreshold(threshold=0.0)),
            ("scaler", StandardScaler()),
        ]
    )


def fit_transform_features(
    pipeline: Pipeline,
    feature_columns: tuple[str, ...],
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit only on benign training traffic and transform all partitions."""

    train_features = pipeline.fit_transform(train.loc[:, feature_columns])
    validation_features = pipeline.transform(validation.loc[:, feature_columns])
    test_features = pipeline.transform(test.loc[:, feature_columns])
    return tuple(
        np.asarray(values, dtype=np.float32)
        for values in (train_features, validation_features, test_features)
    )


def transformed_feature_names(
    pipeline: Pipeline,
    feature_columns: tuple[str, ...],
) -> list[str]:
    """Return feature names after missingness indicators and variance filtering."""

    imputed = pipeline.named_steps["imputer"].get_feature_names_out(feature_columns)
    selected = pipeline.named_steps["variance"].get_support()
    return [str(name) for name in imputed[selected]]
