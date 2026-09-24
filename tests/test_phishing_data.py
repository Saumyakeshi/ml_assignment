from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from comp70049.phishing.data import extract_email_text, stratified_splits
from comp70049.phishing.preprocessing import normalize_text, tokenize


def test_extract_email_text_uses_subject_and_plain_body(tmp_path: Path) -> None:
    email_path = tmp_path / "message.eml"
    email_path.write_bytes(
        b"Subject: Account notice\n"
        b"Content-Type: text/plain; charset=utf-8\n\n"
        b"Please review the account activity.\n"
    )

    extracted = extract_email_text(email_path)

    assert "Account notice" in extracted
    assert "Please review the account activity." in extracted


def test_normalize_text_replaces_urls_and_email_addresses() -> None:
    normalized = normalize_text("Contact Admin@Example.com at https://example.com/a?q=1!")

    assert normalized == "contact emailtoken at urltoken"
    assert tokenize(normalized) == ["contact", "emailtoken", "at", "urltoken"]


def test_stratified_splits_are_disjoint_and_complete() -> None:
    frame = pd.DataFrame(
        {
            "path": [f"message-{index}" for index in range(100)],
            "text": [f"email text {index}" for index in range(100)],
            "label": [0] * 80 + [1] * 20,
            "sha256": [f"hash-{index}" for index in range(100)],
        }
    )

    splits = stratified_splits(
        frame,
        test_fraction=0.15,
        validation_fraction=0.15,
        seed=42,
    )

    train_hashes = set(splits.train["sha256"])
    validation_hashes = set(splits.validation["sha256"])
    test_hashes = set(splits.test["sha256"])
    assert len(splits.train) + len(splits.validation) + len(splits.test) == len(frame)
    assert train_hashes.isdisjoint(validation_hashes)
    assert train_hashes.isdisjoint(test_hashes)
    assert validation_hashes.isdisjoint(test_hashes)
    # Integer split sizes can move the class ratio by one observation.
    assert splits.train["label"].mean() == pytest.approx(0.2, abs=0.02)
    assert splits.validation["label"].mean() == pytest.approx(0.2, abs=0.02)
    assert splits.test["label"].mean() == pytest.approx(0.2, abs=0.02)
