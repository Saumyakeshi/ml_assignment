"""Text normalization shared by the classic and deep-learning models."""

from __future__ import annotations

import re


URL_PATTERN = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
NON_WORD_PATTERN = re.compile(r"[^a-z0-9_]+")
WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalize email text without learning anything from the full dataset."""

    text = text.lower()
    text = URL_PATTERN.sub(" urltoken ", text)
    text = EMAIL_PATTERN.sub(" emailtoken ", text)
    text = NON_WORD_PATTERN.sub(" ", text)
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    """Return normalized whitespace-delimited tokens."""

    normalized = normalize_text(text)
    return normalized.split() if normalized else []

