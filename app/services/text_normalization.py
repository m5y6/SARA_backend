"""
Text normalization utilities for document preprocessing.
Collapses whitespace and standardizes unicode before downstream storage or vectorization.
"""

import re
import unicodedata


def normalize_text(text: str) -> str:
    if text is None:
        raise ValueError("Text cannot be empty")

    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[\t\f\v]+", " ", normalized)
    normalized = re.sub(r"[ ]{2,}", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    normalized = re.sub(r"\s+\n", "\n", normalized)
    normalized = re.sub(r"\n\s+", "\n", normalized)
    normalized = re.sub(r"\s{2,}", " ", normalized)
    return normalized.strip()
