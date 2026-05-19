from __future__ import annotations

from src.ingest.pipeline import build_ingestion_bundle


def test_offline_ingestion_validates_and_enriches_assignment_data() -> None:
    bundle = build_ingestion_bundle(include_external=False)

    assert len(bundle.structured.assets) == 60
    assert len(bundle.structured.vulnerabilities) == 114
    assert len(bundle.structured.threat_intelligence) == 40
    assert len(bundle.structured.business_services) == 20
    assert len(bundle.enriched_vulnerabilities) == 114
    assert len(bundle.unmatched_threat_intelligence) == 15


def test_identifier_normalization_preserves_threat_matches() -> None:
    bundle = build_ingestion_bundle(include_external=False)
    enriched = bundle.enriched_vulnerabilities

    fortinet = enriched.loc[enriched["cve"].eq("CVE-2024-21762")]
    assert not fortinet.empty
    assert fortinet["threat_intel_match_bool"].all()
    assert fortinet["threat_ransomware_bool"].all()


def test_rag_documents_include_threat_report_even_offline() -> None:
    bundle = build_ingestion_bundle(include_external=False)

    assert bundle.rag_documents[0].doc_id == "mdr-threat-report"
    assert "CrimsonJackal" in bundle.rag_documents[0].text

