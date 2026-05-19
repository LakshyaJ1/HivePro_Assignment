from __future__ import annotations

import math
import re
from typing import Any


YES_VALUES = {"yes", "y", "true", "1", "known"}
NO_VALUES = {"no", "n", "false", "0", "unknown", "none", ""}

CRITICALITY_SCORE = {
    "critical": 5,
    "high": 4,
    "medium": 3,
    "low": 2,
    "none": 1,
    "": 1,
}

REVENUE_IMPACT_SCORE = {
    "critical": 5,
    "high": 4,
    "medium": 3,
    "low": 2,
    "none": 1,
    "": 1,
}


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def normalize_identifier(value: Any) -> str:
    """Normalize CVE-like and assignment synthetic identifiers for safe joins."""

    text = clean_text(value).upper()
    text = re.sub(r"\s+", "", text)
    return text


def normalize_cve(value: Any) -> str:
    return normalize_identifier(value)


def parse_yes_no(value: Any) -> bool:
    text = clean_text(value).lower()
    if text in YES_VALUES:
        return True
    if text in NO_VALUES:
        return False
    raise ValueError(f"Expected yes/no style value, got {value!r}")


def yes_no_to_int(value: Any) -> int:
    return int(parse_yes_no(value))


def score_label(value: Any, mapping: dict[str, int]) -> int:
    return mapping.get(clean_text(value).lower(), 1)


def split_csv_cell(value: Any) -> list[str]:
    text = clean_text(value)
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]

