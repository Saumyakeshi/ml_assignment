"""Load NSL-KDD and map specific attacks to five reporting classes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


FEATURE_COLUMNS = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]

CATEGORICAL_COLUMNS = ["protocol_type", "service", "flag"]
NUMERIC_COLUMNS = [
    column for column in FEATURE_COLUMNS if column not in CATEGORICAL_COLUMNS
]
DATA_COLUMNS = [*FEATURE_COLUMNS, "attack_name", "difficulty"]
CLASS_NAMES = ["normal", "dos", "probe", "r2l", "u2r"]
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}

ATTACK_CATEGORY = {
    "normal": "normal",
    "back": "dos",
    "land": "dos",
    "neptune": "dos",
    "pod": "dos",
    "smurf": "dos",
    "teardrop": "dos",
    "mailbomb": "dos",
    "apache2": "dos",
    "processtable": "dos",
    "udpstorm": "dos",
    "ipsweep": "probe",
    "nmap": "probe",
    "portsweep": "probe",
    "satan": "probe",
    "saint": "probe",
    "mscan": "probe",
    "ftp_write": "r2l",
    "guess_passwd": "r2l",
    "imap": "r2l",
    "multihop": "r2l",
    "phf": "r2l",
    "spy": "r2l",
    "warezclient": "r2l",
    "warezmaster": "r2l",
    "sendmail": "r2l",
    "named": "r2l",
    "snmpgetattack": "r2l",
    "snmpguess": "r2l",
    "xlock": "r2l",
    "xsnoop": "r2l",
    "worm": "r2l",
    "buffer_overflow": "u2r",
    "loadmodule": "u2r",
    "perl": "u2r",
    "rootkit": "u2r",
    "ps": "u2r",
    "sqlattack": "u2r",
    "xterm": "u2r",
    "httptunnel": "u2r",
}


@dataclass(frozen=True)
class DatasetPartitions:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def map_attack_category(attack_name: str) -> str:
    """Map an NSL-KDD attack name to normal, DoS, Probe, R2L, or U2R."""

    normalized = str(attack_name).strip().lower().removesuffix(".")
    try:
        return ATTACK_CATEGORY[normalized]
    except KeyError as error:
        raise ValueError(f"Unknown NSL-KDD attack label: {attack_name!r}") from error


def load_partition(path: Path) -> pd.DataFrame:
    """Load one original NSL-KDD TXT partition with its attack-type labels."""

    if not path.is_file():
        raise FileNotFoundError(
            f"NSL-KDD file not found: {path}. Run scripts/download_nsl_kdd.py first."
        )
    frame = pd.read_csv(path, names=DATA_COLUMNS, header=None)
    if frame.shape[1] != len(DATA_COLUMNS):
        raise ValueError(
            f"Expected {len(DATA_COLUMNS)} columns in {path}, found {frame.shape[1]}."
        )
    frame["attack_name"] = (
        frame["attack_name"].astype(str).str.strip().str.lower().str.rstrip(".")
    )
    frame["label_name"] = frame["attack_name"].map(map_attack_category)
    frame["label"] = frame["label_name"].map(CLASS_TO_INDEX).astype("int64")
    return frame


def load_nsl_kdd(data_dir: Path, *, validation_fraction: float, seed: int) -> DatasetPartitions:
    """Load the official train/test partitions and split validation from training."""

    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one.")
    full_train = load_partition(data_dir / "KDDTrain+.txt")
    test = load_partition(data_dir / "KDDTest+.txt")
    train, validation = train_test_split(
        full_train,
        test_size=validation_fraction,
        random_state=seed,
        stratify=full_train["label"],
    )
    return DatasetPartitions(
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )


def profile_partition(frame: pd.DataFrame) -> dict[str, object]:
    """Return report-ready data-quality information for a partition."""

    return {
        "records": int(len(frame)),
        "features": len(FEATURE_COLUMNS),
        "missing_values": int(frame[DATA_COLUMNS].isna().sum().sum()),
        "duplicate_records": int(frame[DATA_COLUMNS].duplicated().sum()),
        "class_counts": {
            name: int((frame["label_name"] == name).sum()) for name in CLASS_NAMES
        },
        "attack_types": sorted(frame["attack_name"].unique().tolist()),
    }
