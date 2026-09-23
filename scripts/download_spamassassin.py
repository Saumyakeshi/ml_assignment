"""Download and extract the official Apache SpamAssassin public corpus."""

from __future__ import annotations

import argparse
import tarfile
import urllib.request
from pathlib import Path


BASE_URL = "https://spamassassin.apache.org/old/publiccorpus"
ARCHIVES = {
    "20030228_easy_ham.tar.bz2": "easy_ham",
    "20030228_spam.tar.bz2": "spam",
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("data/raw/spamassassin"),
    )
    return parser.parse_args()


def main() -> None:
    destination = _arguments().destination.resolve()
    archive_dir = destination / "archives"
    archive_dir.mkdir(parents=True, exist_ok=True)

    for archive_name, extracted_dir in ARCHIVES.items():
        archive_path = archive_dir / archive_name
        if not archive_path.exists():
            url = f"{BASE_URL}/{archive_name}"
            print(f"Downloading {url}")
            urllib.request.urlretrieve(url, archive_path)
        else:
            print(f"Using existing archive {archive_path}")

        expected = destination / extracted_dir
        if not expected.is_dir():
            print(f"Extracting {archive_path}")
            with tarfile.open(archive_path, mode="r:bz2") as archive:
                archive.extractall(destination, filter="data")
        else:
            print(f"Using existing extracted directory {expected}")

    print(f"Dataset ready at {destination}")


if __name__ == "__main__":
    main()

