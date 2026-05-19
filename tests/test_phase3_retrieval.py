from __future__ import annotations

import pytest

from src.engine.nist_retriever import NistHybridRetriever, build_risk_query
from src.engine.retrieval_config import RetrievalConfig, RetrievalWeights
from src.engine.scoring import score_risks, top_risks
from src.ingest.pipeline import build_ingestion_bundle


@pytest.fixture(scope="module")
def phase3_context():
    bundle = build_ingestion_bundle()
    scored = score_risks(bundle.enriched_vulnerabilities)
    retriever = NistHybridRetriever(bundle.external.nist_controls)
    return scored, retriever


def test_phase3_citrixbleed_retrieves_access_or_authentication_control(phase3_context) -> None:
    scored, retriever = phase3_context
    risk = top_risks(scored, n=1).iloc[0]

    result = retriever.retrieve(risk)

    assert result.best.control_id in {"AC-2", "IA-2"}
    assert result.best.control_prior_score >= 0.9
    assert result.best.semantic_score > 0
    assert result.best.hybrid_score > 0.5
    assert len(result.candidates) == RetrievalConfig().final_candidate_count


def test_phase3_unsupported_system_retrieves_sa_22(phase3_context) -> None:
    scored, retriever = phase3_context
    risk = scored.loc[scored["vuln_id"].eq("V-2089")].iloc[0]

    result = retriever.retrieve(risk)

    assert result.best.control_id == "SA-22"
    assert "Unsupported System Components" in result.best.name


def test_phase3_auth_or_control_gap_retrieves_identity_or_monitoring_control(phase3_context) -> None:
    scored, retriever = phase3_context
    risk = scored.loc[scored["vuln_id"].eq("V-2099")].iloc[0]

    result = retriever.retrieve(risk)

    assert result.best.control_id in {"AC-2", "IA-5", "SI-4"}
    assert result.best.semantic_score > 0


def test_phase3_top_eight_controls_are_not_monoculture(phase3_context) -> None:
    scored, retriever = phase3_context
    risks = top_risks(scored, n=8)

    controls = [retriever.retrieve(row).best.control_id for _, row in risks.iterrows()]

    assert len(set(controls)) >= 3
    assert controls.count("SI-2") < len(controls)


def test_phase3_fortinet_auth_bypass_has_nonzero_semantic_score(phase3_context) -> None:
    scored, retriever = phase3_context
    risk = scored.loc[scored["vuln_id"].eq("V-2016")].iloc[0]

    result = retriever.retrieve(risk)

    assert result.best.control_id in {"AC-2", "IA-2"}
    assert result.best.semantic_score > 0


def test_phase3_query_is_grounded_in_risk_context(phase3_context) -> None:
    scored, _ = phase3_context
    risk = scored.loc[scored["vuln_id"].eq("V-2043")].iloc[0]

    query = build_risk_query(risk)

    assert "JetBrains TeamCity Authentication Bypass" in query
    assert "DevOps Platform" in query
    assert "Ransomware involved: yes" in query
    assert "build pipeline" in query


def test_invalid_retrieval_weights_fail_fast() -> None:
    try:
        RetrievalConfig(weights=RetrievalWeights(semantic=0.1, bm25=0.1, control_prior=0.1))
    except ValueError as exc:
        assert "sum to 1.0" in str(exc)
    else:
        raise AssertionError("invalid retrieval weights should fail")
