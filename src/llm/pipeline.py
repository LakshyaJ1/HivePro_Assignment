from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from .client import GroqChatClient
from .config import LLMConfig
from .narrator import LLMReportNarration, RiskNarrationService

if TYPE_CHECKING:
    from src.engine.nist_retriever import RetrievalResult


@dataclass(frozen=True)
class Phase4PipelineResult:
    report: LLMReportNarration
    top_risk_count: int
    risk_retrievals: tuple[tuple[pd.Series, "RetrievalResult"], ...]


def run_phase4_pipeline(
    *,
    top_n: int = 5,
    llm_config: LLMConfig | None = None,
) -> Phase4PipelineResult:
    config = llm_config or LLMConfig()
    client = GroqChatClient.from_env(config)

    from src.engine.nist_retriever import NistHybridRetriever
    from src.engine.scoring import top_risks
    from src.ingest.pipeline import build_ingestion_bundle

    bundle = build_ingestion_bundle(include_external=True)
    risks = top_risks(bundle.enriched_vulnerabilities, n=top_n)
    retriever = NistHybridRetriever(bundle.external.nist_controls)
    risk_retrievals = retriever.retrieve_for_top_risks(risks)
    service = RiskNarrationService(client, config=config)
    report = service.narrate_report(risk_retrievals)
    return Phase4PipelineResult(
        report=report,
        top_risk_count=len(risk_retrievals),
        risk_retrievals=tuple(risk_retrievals),
    )
