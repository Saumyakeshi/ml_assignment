"""Leakage-safe preprocessing and feature selection for NSL-KDD."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold

from .data import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, NUMERIC_COLUMNS


def build_feature_pipeline(selected_features: int) -> Pipeline:
    """Build scaling, encoding, and ANOVA feature-selection steps."""

    if selected_features < 1:
        raise ValueError("selected_features must be positive.")

    numerical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    columns = ColumnTransformer(
        transformers=[
            ("numeric", numerical, NUMERIC_COLUMNS),
            ("categorical", categorical, CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline(
        steps=[
            ("columns", columns),
            ("variance", VarianceThreshold()),
            ("selection", SelectKBest(score_func=f_classif, k=selected_features)),
        ]
    )


def fit_transform_features(
    pipeline: Pipeline,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit transformations on training only and transform all partitions."""

    train_values = pipeline.fit_transform(train[FEATURE_COLUMNS], train["label"])
    validation_values = pipeline.transform(validation[FEATURE_COLUMNS])
    test_values = pipeline.transform(test[FEATURE_COLUMNS])
    return tuple(
        np.asarray(values, dtype=np.float32)
        for values in (train_values, validation_values, test_values)
    )


def selected_feature_names(pipeline: Pipeline) -> list[str]:
    """Return names retained by the fitted feature selector."""

    names = pipeline.named_steps["columns"].get_feature_names_out()
    names = names[pipeline.named_steps["variance"].get_support()]
    names = names[pipeline.named_steps["selection"].get_support()]
    return names.tolist()
