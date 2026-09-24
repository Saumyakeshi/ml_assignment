"""Loading and leakage-safe splitting of the SpamAssassin email corpus."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from email import policy
from email.message import Message
from email.parser import BytesParser
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


WHITESPACE_PATTERN = re.compile(r"\s+")


class _HTMLTextExtractor(HTMLParser):
    """Collect visible text fragments while ignoring HTML markup."""

    def __init__(self) -> None:
        super().__init__()
        self.fragments: list[str] = []

    def handle_data(self, data: str) -> None:
        """Record text encountered between HTML tags."""

        self.fragments.append(data)

    def text(self) -> str:
        """Return the collected fragments as one space-delimited string."""

        return " ".join(self.fragments)


@dataclass(frozen=True)
class CorpusSplits:
    """Container for the three mutually exclusive experiment partitions."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def _html_to_text(value: str) -> str:
    """Strip markup from an HTML email part without executing its content."""

    parser = _HTMLTextExtractor()
    parser.feed(value)
    return parser.text()


def _part_text(part: Message) -> str:
    """Decode one MIME part, replacing invalid bytes when necessary."""

    try:
        value = part.get_content()
    except (LookupError, UnicodeDecodeError):
        payload = part.get_payload(decode=True) or b""
        value = payload.decode("utf-8", errors="replace")
    return value if isinstance(value, str) else str(value)


def extract_email_text(path: Path) -> str:
    """Extract the subject and readable body from an RFC 822 email file."""

    message = BytesParser(policy=policy.default).parsebytes(path.read_bytes())
    subject = str(message.get("subject", ""))
    plain_parts: list[str] = []
    html_parts: list[str] = []

    # Prefer plain text when both plain and HTML alternatives are available.
    # Attachments are excluded because their contents are outside this text task.
    parts = message.walk() if message.is_multipart() else [message]
    for part in parts:
        if part.is_multipart():
            continue
        if part.get_content_disposition() == "attachment":
            continue
        content_type = part.get_content_type()
        if content_type == "text/plain":
            plain_parts.append(_part_text(part))
        elif content_type == "text/html":
            html_parts.append(_html_to_text(_part_text(part)))

    body = " ".join(plain_parts or html_parts)
    return WHITESPACE_PATTERN.sub(" ", f"Subject: {subject} Body: {body}").strip()


def _records_from_directory(directory: Path, label: int) -> list[dict[str, object]]:
    """Parse every corpus message in a class directory into a record."""

    records: list[dict[str, object]] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.name.lower() == "cmds":
            continue
        text = extract_email_text(path)
        if not text:
            continue
        # Hash normalized extracted text so exact duplicates can be removed before
        # splitting; otherwise the same message could leak into train and test.
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        records.append(
            {
                "path": str(path),
                "text": text,
                "label": label,
                "sha256": digest,
            }
        )
    return records


def load_spamassassin_corpus(data_dir: Path) -> pd.DataFrame:
    """Load ham and spam messages, removing exact text duplicates."""

    ham_dir = data_dir / "easy_ham"
    spam_dir = data_dir / "spam"
    if not ham_dir.is_dir() or not spam_dir.is_dir():
        raise FileNotFoundError(
            f"Expected extracted directories at {ham_dir} and {spam_dir}. "
            "Run scripts/download_spamassassin.py first."
        )

    records = _records_from_directory(ham_dir, label=0)
    records.extend(_records_from_directory(spam_dir, label=1))
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise ValueError(f"No email records were found under {data_dir}.")

    return (
        frame.drop_duplicates(subset="sha256", keep="first")
        .sort_values(["label", "path"])
        .reset_index(drop=True)
    )


def stratified_splits(
    frame: pd.DataFrame,
    *,
    test_fraction: float,
    validation_fraction: float,
    seed: int,
) -> CorpusSplits:
    """Create mutually exclusive stratified train, validation, and test sets."""

    if test_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("Test and validation fractions must be positive.")
    if test_fraction + validation_fraction >= 1:
        raise ValueError("Test and validation fractions must sum to less than one.")

    # Split the test set first, then derive the validation proportion relative
    # to the remaining records. Stratification preserves the minority spam rate.
    train_validation, test = train_test_split(
        frame,
        test_size=test_fraction,
        random_state=seed,
        stratify=frame["label"],
    )
    relative_validation = validation_fraction / (1.0 - test_fraction)
    train, validation = train_test_split(
        train_validation,
        test_size=relative_validation,
        random_state=seed,
        stratify=train_validation["label"],
    )
    return CorpusSplits(
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )


def save_split_manifest(splits: CorpusSplits, output_dir: Path) -> Path:
    """Save paths, labels, hashes, and split names without duplicating email text."""

    output_dir.mkdir(parents=True, exist_ok=True)
    parts: list[pd.DataFrame] = []
    for split_name, frame in (
        ("train", splits.train),
        ("validation", splits.validation),
        ("test", splits.test),
    ):
        part = frame[["path", "label", "sha256"]].copy()
        part.insert(0, "split", split_name)
        parts.append(part)
    target = output_dir / "split_manifest.csv"
    pd.concat(parts, ignore_index=True).to_csv(target, index=False)
    return target
