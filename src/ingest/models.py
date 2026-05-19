from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class FetchMetadata:
    """Operational metadata for runtime-downloaded reference sources."""

    source: str
    url: str
    cache_path: Path
    refreshed: bool
    stale_cache_used: bool = False
    warning: str | None = None


@dataclass(frozen=True)
class NarrativeDocument:
    """A text document ready to be embedded by the later RAG layer."""

    doc_id: str
    title: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class NistControl:
    """A parsed NIST SP 800-53 control row."""

    control_id: str
    name: str
    discussion: str
    family: str | None = None

    @property
    def document_text(self) -> str:
        family = f" Family: {self.family}." if self.family else ""
        return f"[{self.control_id}] {self.name}.{family} {self.discussion}".strip()


@dataclass
class StructuredDataPack:
    """Validated and normalized assignment CSVs."""

    assets: pd.DataFrame
    vulnerabilities: pd.DataFrame
    threat_intelligence: pd.DataFrame
    business_services: pd.DataFrame
    remediation_guidance: pd.DataFrame


@dataclass
class ExternalDataPack:
    """Public reference data fetched at runtime or loaded from cache."""

    cisa_kev: pd.DataFrame
    epss: pd.DataFrame
    nist_controls: list[NistControl]
    metadata: list[FetchMetadata] = field(default_factory=list)


@dataclass
class IngestionBundle:
    """Complete Phase 1 output consumed by scoring and retrieval phases."""

    structured: StructuredDataPack
    external: ExternalDataPack
    threat_report: NarrativeDocument
    enriched_vulnerabilities: pd.DataFrame
    unmatched_threat_intelligence: pd.DataFrame
    rag_documents: list[NarrativeDocument]
    warnings: list[str] = field(default_factory=list)

