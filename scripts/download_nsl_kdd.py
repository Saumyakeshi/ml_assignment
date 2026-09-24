"""Download the NSL-KDD attack-labelled TXT files from a public mirror."""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path


BASE_URL = "https://raw.githubusercontent.com/HoaNP/NSL-KDD-DataSet/master"
FILES = {
    "KDDTrain+.txt": {
        "rows": 125_973,
        "columns": 43,
        "sha256": "1b86d2f957b33082081bba410fe129b475efebcc13c9014c3f447c8271aadf95",
    },
    "KDDTest+.txt": {
        "rows": 22_544,
        "columns": 43,
        "sha256": "fa46b0935342616aa83b7c2578db355b6a7aaabbc492248172c7a1e8b7ab8f84",
    },
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination", type=Path, default=Path("data/raw/nsl-kdd")
    )
    return parser.parse_args()


def _validate(
    path: Path, *, rows: int, columns: int, sha256: str
) -> tuple[int, str]:
    line_count = 0
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for raw_line in stream:
            digest.update(raw_line)
            if raw_line.strip():
                line_count += 1
                if line_count == 1:
                    found_columns = len(raw_line.decode("utf-8").rstrip().split(","))
                    if found_columns != columns:
                        raise ValueError(
                            f"{path} has {found_columns} columns; expected {columns}."
                        )
    if line_count != rows:
        raise ValueError(f"{path} has {line_count} rows; expected {rows}.")
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != sha256:
        raise ValueError(
            f"{path} has SHA-256 {actual_sha256}; expected {sha256}."
        )
    return line_count, actual_sha256


def main() -> None:
    destination = _arguments().destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    for filename, expected in FILES.items():
        path = destination / filename
        if not path.exists():
            encoded_name = filename.replace("+", "%2B")
            url = f"{BASE_URL}/{encoded_name}"
            print(f"Downloading {url}")
            urllib.request.urlretrieve(url, path)
        else:
            print(f"Using existing file {path}")
        rows, sha256 = _validate(path, **expected)
        print(f"Validated {filename}: {rows:,} rows; SHA-256 {sha256}")
    print(f"Dataset ready at {destination}")


if __name__ == "__main__":
    main()
