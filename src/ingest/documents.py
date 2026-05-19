from __future__ import annotations

import re
from pathlib import Path

from .models import NarrativeDocument, NistControl


def load_threat_report(raw_data_dir: Path) -> NarrativeDocument:
    path = raw_data_dir / "synthetic_threat_report.md"
    if not path.exists():
        raise FileNotFoundError(f"Required threat report not found: {path}")

    text = path.read_text(encoding="utf-8-sig").strip()
    title = _extract_markdown_title(text) or "TawasolPay MDR Advisory"
    return NarrativeDocument(
        doc_id="mdr-threat-report",
        title=title,
        text=text,
        metadata={"source": path.name, "document_type": "mdr_advisory"},
    )


def nist_controls_to_documents(controls: list[NistControl]) -> list[NarrativeDocument]:
    documents: list[NarrativeDocument] = []
    for control in controls:
        documents.append(
            NarrativeDocument(
                doc_id=f"nist-{control.control_id}",
                title=f"{control.control_id}: {control.name}",
                text=control.document_text,
                metadata={
                    "source": "NIST SP 800-53 Rev. 5",
                    "document_type": "nist_control",
                    "control_id": control.control_id,
                    "family": control.family or "",
                },
            )
        )
    return documents


def _extract_markdown_title(text: str) -> str | None:
    match = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else None

