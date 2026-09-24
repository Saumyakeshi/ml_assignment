"""Download the selected raw CSE-CIC-IDS2018 daily flow CSV."""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

DATA_URL = (
    "https://cse-cic-ids2018.s3.amazonaws.com/"
    "Processed%20Traffic%20Data%20for%20ML%20Algorithms/"
    "Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv"
)
FILE_NAME = "Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv"
EXPECTED_BYTES = 107_842_858
EXPECTED_SHA256 = "b0534c5d7d8b41e03df71c6966c995d116a8ed28e61f377c8b14cdf5d28f4edf"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate(path: Path) -> str:
    size = path.stat().st_size
    if size != EXPECTED_BYTES:
        raise ValueError(
            f"Unexpected file size for {path}: {size:,}; expected {EXPECTED_BYTES:,}."
        )
    checksum = _sha256(path)
    if EXPECTED_SHA256 and checksum != EXPECTED_SHA256:
        raise ValueError(
            f"Checksum mismatch for {path}: {checksum}; expected {EXPECTED_SHA256}."
        )
    return checksum


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/cic-ids2018"),
    )
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    destination = arguments.output_dir / FILE_NAME

    if destination.is_file():
        checksum = _validate(destination)
        print(f"Already downloaded and verified: {destination}")
        print(f"SHA256: {checksum}")
        return

    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        DATA_URL,
        headers={"User-Agent": "COMP70049-assignment/1.0"},
    )
    print(f"Downloading {DATA_URL}")
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
        downloaded = 0
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
            downloaded += len(chunk)
            if downloaded % (20 * 1024 * 1024) < len(chunk):
                print(f"Downloaded {downloaded / (1024 * 1024):.0f} MiB")

    checksum = _validate(temporary)
    temporary.replace(destination)
    print(f"Saved: {destination}")
    print(f"SHA256: {checksum}")


if __name__ == "__main__":
    main()
