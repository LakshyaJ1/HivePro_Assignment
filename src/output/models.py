from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MarkdownReport:
    content: str
    output_path: Path | None = None
    risk_count: int = 0

