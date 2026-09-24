"""Loading, profiling, cleaning, and splitting for CIC-IDS2018."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LABEL_COLUMN = "Label"
TIMESTAMP_COLUMN = "Timestamp"

# Domain-relevant traffic variables from CICFlowMeter. The loader keeps the
# available subset so it remains robust to minor schema differences by day.
TRAFFIC_FEATURE_CANDIDATES = (
    "Dst Port",
    "Protocol",
    "Flow Duration",
    "Tot Fwd Pkts",
    "Tot Bwd Pkts",
    "TotLen Fwd Pkts",
    "TotLen Bwd Pkts",
    "Fwd Pkt Len Max",
    "Fwd Pkt Len Min",
    "Fwd Pkt Len Mean",
    "Fwd Pkt Len Std",
    "Bwd Pkt Len Max",
    "Bwd Pkt Len Min",
    "Bwd Pkt Len Mean",
    "Bwd Pkt Len Std",
    "Flow Byts/s",
    "Flow Pkts/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Tot",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Tot",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd Pkts/s",
    "Bwd Pkts/s",
    "Pkt Len Min",
    "Pkt Len Max",
    "Pkt Len Mean",
    "Pkt Len Std",
    "Pkt Len Var",
    "FIN Flag Cnt",
    "SYN Flag Cnt",
    "RST Flag Cnt",
    "PSH Flag Cnt",
    "ACK Flag Cnt",
    "URG Flag Cnt",
    "Down/Up Ratio",
    "Pkt Size Avg",
    "Fwd Seg Size Avg",
    "Bwd Seg Size Avg",
    "Subflow Fwd Pkts",
    "Subflow Fwd Byts",
    "Subflow Bwd Pkts",
    "Subflow Bwd Byts",
    "Init Fwd Win Byts",
    "Init Bwd Win Byts",
    "Fwd Act Data Pkts",
    "Fwd Seg Size Min",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
    "hour",
    "day_of_week",
)


@dataclass(frozen=True)
class CleanedCICIDS:
    """Cleaned labelled flows plus an auditable data-quality report."""

    frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    raw_profile: dict[str, Any]
    cleaning_report: dict[str, Any]


@dataclass(frozen=True)
class AnomalyPartitions:
    """Normal-only training data and labelled validation/test data."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def _strip_headers(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.columns = [str(column).replace("\ufeff", "").strip() for column in result.columns]
    unnamed = [column for column in result.columns if column.lower().startswith("unnamed:")]
    return result.drop(columns=unnamed)


def _raw_profile(frame: pd.DataFrame) -> dict[str, Any]:
    text_columns = frame.select_dtypes(include=["object", "string"])
    blank_strings = text_columns.apply(
        lambda column: column.astype("string").str.strip().eq("")
    ).sum()
    return {
        "records": int(len(frame)),
        "columns": int(frame.shape[1]),
        "duplicate_records": int(frame.duplicated().sum()),
        "native_missing_values": int(frame.isna().sum().sum()),
        "blank_string_values": int(blank_strings.sum()),
    }


def clean_cic_ids_frame(
    raw_frame: pd.DataFrame,
    *,
    maximum_missing_fraction: float = 0.4,
) -> CleanedCICIDS:
    """Clean an unmodified CIC-IDS2018 daily CSV without deleting outliers."""

    if not 0 <= maximum_missing_fraction < 1:
        raise ValueError("maximum_missing_fraction must be in [0, 1).")

    raw_profile = _raw_profile(raw_frame)
    frame = _strip_headers(raw_frame)
    if LABEL_COLUMN not in frame.columns:
        raise ValueError(f"Expected a {LABEL_COLUMN!r} column; found {list(frame.columns)!r}.")

    labels = frame[LABEL_COLUMN].astype("string").str.strip()
    missing_label = labels.isna() | labels.eq("")
    embedded_header = labels.str.casefold().eq(LABEL_COLUMN.casefold())
    rows_without_label = int(missing_label.sum())
    embedded_header_rows = int(embedded_header.sum())
    frame = frame.loc[~(missing_label | embedded_header)].copy()
    labels = labels.loc[~(missing_label | embedded_header)]

    duplicate_records = int(frame.duplicated().sum())
    frame = frame.drop_duplicates().reset_index(drop=True)
    labels = frame[LABEL_COLUMN].astype("string").str.strip()

    invalid_timestamps = 0
    if TIMESTAMP_COLUMN in frame.columns:
        timestamps = pd.to_datetime(
            frame[TIMESTAMP_COLUMN].astype("string").str.strip(),
            errors="coerce",
            format="mixed",
            dayfirst=True,
        )
        invalid_timestamps = int(timestamps.isna().sum())
        frame["hour"] = timestamps.dt.hour.astype("float64")
        frame["day_of_week"] = timestamps.dt.dayofweek.astype("float64")

    available_features = [
        column for column in TRAFFIC_FEATURE_CANDIDATES if column in frame.columns
    ]
    if not available_features:
        raise ValueError("No expected CICFlowMeter traffic features were found.")

    numeric = pd.DataFrame(index=frame.index)
    coercions_to_missing = 0
    infinite_values = 0
    for column in available_features:
        original = frame[column]
        converted = pd.to_numeric(original, errors="coerce")
        original_present = original.notna() & original.astype("string").str.strip().ne("")
        coercions_to_missing += int((original_present & converted.isna()).sum())
        values = converted.to_numpy(dtype=np.float64, copy=True)
        infinite_values += int(np.isinf(values).sum())
        numeric[column] = converted.replace([np.inf, -np.inf], np.nan)

    missing_fraction = numeric.isna().mean()
    dropped_for_missingness = sorted(
        missing_fraction[missing_fraction > maximum_missing_fraction].index.tolist()
    )
    numeric = numeric.drop(columns=dropped_for_missingness)

    cleaned = numeric.copy()
    cleaned["attack_label"] = labels.to_numpy(dtype=str)
    cleaned["label"] = (~labels.str.casefold().eq("benign")).astype("int64").to_numpy()

    cleaning_report = {
        "records_after_cleaning": int(len(cleaned)),
        "rows_without_label_removed": rows_without_label,
        "embedded_header_rows_removed": embedded_header_rows,
        "duplicate_records_removed": duplicate_records,
        "infinite_values_replaced_with_missing": infinite_values,
        "non_numeric_values_coerced_to_missing": coercions_to_missing,
        "invalid_timestamps": invalid_timestamps,
        "features_dropped_for_missingness": dropped_for_missingness,
        "remaining_missing_values": int(numeric.isna().sum().sum()),
        "retained_feature_count": int(numeric.shape[1]),
        "class_counts": {
            "benign": int((cleaned["label"] == 0).sum()),
            "anomaly": int((cleaned["label"] == 1).sum()),
        },
        "attack_label_counts": {
            str(label): int(count)
            for label, count in cleaned["attack_label"].value_counts().items()
        },
    }
    return CleanedCICIDS(
        frame=cleaned,
        feature_columns=tuple(numeric.columns),
        raw_profile=raw_profile,
        cleaning_report=cleaning_report,
    )


def load_cic_ids2018(
    path: Path,
    *,
    maximum_missing_fraction: float = 0.4,
) -> CleanedCICIDS:
    """Load and clean the selected raw CIC-IDS2018 daily CSV."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run scripts/download_cic_ids2018.py first."
        )
    raw_frame = pd.read_csv(path, low_memory=False)
    return clean_cic_ids_frame(
        raw_frame,
        maximum_missing_fraction=maximum_missing_fraction,
    )


def _sample(frame: pd.DataFrame, maximum: int | None, seed: int) -> pd.DataFrame:
    if maximum is None or maximum <= 0 or len(frame) <= maximum:
        return frame.sample(frac=1, random_state=seed).reset_index(drop=True)
    return frame.sample(n=maximum, random_state=seed).reset_index(drop=True)


def create_anomaly_partitions(
    frame: pd.DataFrame,
    *,
    seed: int,
    max_benign_records: int | None,
    max_anomaly_records: int | None,
    benign_train_fraction: float,
    benign_validation_fraction: float,
    anomaly_validation_fraction: float,
) -> AnomalyPartitions:
    """Create normal-only train and labelled validation/test partitions."""

    if benign_train_fraction <= 0 or benign_validation_fraction <= 0:
        raise ValueError("Benign train and validation fractions must be positive.")
    if benign_train_fraction + benign_validation_fraction >= 1:
        raise ValueError("A positive benign test fraction must remain.")
    if not 0 < anomaly_validation_fraction < 1:
        raise ValueError("anomaly_validation_fraction must be between zero and one.")

    benign = _sample(frame.loc[frame["label"] == 0], max_benign_records, seed)
    anomalies = _sample(frame.loc[frame["label"] == 1], max_anomaly_records, seed + 1)
    if len(benign) < 3 or len(anomalies) < 2:
        raise ValueError("Both benign and anomalous records are required.")

    train_end = int(len(benign) * benign_train_fraction)
    validation_end = train_end + int(len(benign) * benign_validation_fraction)
    anomaly_validation_end = int(len(anomalies) * anomaly_validation_fraction)

    train = benign.iloc[:train_end].copy()
    validation = pd.concat(
        [benign.iloc[train_end:validation_end], anomalies.iloc[:anomaly_validation_end]],
        ignore_index=True,
    ).sample(frac=1, random_state=seed + 2).reset_index(drop=True)
    test = pd.concat(
        [benign.iloc[validation_end:], anomalies.iloc[anomaly_validation_end:]],
        ignore_index=True,
    ).sample(frac=1, random_state=seed + 3).reset_index(drop=True)

    if train["label"].any():
        raise AssertionError("The anomaly detector training partition must contain only benign data.")
    return AnomalyPartitions(train=train, validation=validation, test=test)


def profile_partition(frame: pd.DataFrame) -> dict[str, Any]:
    """Return serialisable partition information."""

    return {
        "records": int(len(frame)),
        "benign": int((frame["label"] == 0).sum()),
        "anomaly": int((frame["label"] == 1).sum()),
        "missing_values": int(frame.isna().sum().sum()),
        "attack_label_counts": {
            str(label): int(count)
            for label, count in frame["attack_label"].value_counts().items()
        },
    }
