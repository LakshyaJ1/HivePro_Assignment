from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from src.engine.nist_retriever import RetrievalCandidate, RetrievalResult
from src.llm.narrator import LLMReportNarration, RiskNarration
from src.output.markdown import render_markdown_report, write_markdown_report


def test_markdown_report_contains_assignment_risk_sections() -> None:
    report = render_markdown_report(
        narration=_narration(),
        risk_retrievals=((_risk_row(), _retrieval()),),
        generated_at=datetime(2026, 5, 18, tzinfo=timezone.utc),
    )

    content = report.content
    assert "# TawasolPay AI Cyber Risk Assistant Report" in content
    assert "## Executive Brief" in content
    assert "## Asset-Level Exposure Rollup" in content
    assert "### Risk #1 - CRITICAL" in content
    assert "**Asset:** payment-api-prod-01 (API Server, internet-exposed)" in content
    assert "**Vulnerability:** CVE-SYN-2026-0010 - Payment API IDOR (CVSS 9.1)" in content
    assert "**CISA KEV:** Confirmed. Ransomware: Known." in content
    assert "**Ransomware Signal Source:** CISA KEV only; local campaign is not marked as ransomware" in content
    assert "#### Why This Ranks Here" in content
    assert "#### NIST SP 800-53 Control - SI-2: Flaw Remediation" in content
    assert "#### Threat Context" in content
    assert "## Appendix C - Retrieval Candidates" in content


def test_markdown_report_writes_to_output_dir(tmp_path) -> None:
    report = render_markdown_report(
        narration=_narration(),
        risk_retrievals=((_risk_row(), _retrieval()),),
    )

    written = write_markdown_report(report, tmp_path)

    assert written.output_path is not None
    assert written.output_path.exists()
    assert written.output_path.read_text(encoding="utf-8") == report.content


def test_markdown_report_fails_if_llm_narration_missing() -> None:
    bad_narration = LLMReportNarration(
        executive_brief="Brief.",
        risks=tuple(),
        model="llama-3.3-70b-versatile",
        total_usage={},
    )

    try:
        render_markdown_report(
            narration=bad_narration,
            risk_retrievals=((_risk_row(), _retrieval()),),
        )
    except ValueError as exc:
        assert "Missing LLM narration" in str(exc)
    else:
        raise AssertionError("report rendering should fail when narration is missing")


def test_markdown_nist_evidence_excerpt_does_not_cut_mid_sentence() -> None:
    report = render_markdown_report(
        narration=_narration(),
        risk_retrievals=((_risk_row(), _retrieval(_long_discussion())),),
    )

    line = next(line for line in report.content.splitlines() if line.startswith("> "))

    assert not line.endswith("...")
    assert line.endswith(".")


def _risk_row() -> pd.Series:
    return pd.Series(
        {
            "risk_rank": 1,
            "risk_severity": "Critical",
            "composite_risk_score": 0.912,
            "asset_name": "payment-api-prod-01",
            "asset_type": "API Server",
            "feature_internet_exposed": 1.0,
            "cve": "CVE-SYN-2026-0010",
            "vulnerability_name": "Payment API IDOR",
            "cvss": 9.1,
            "business_service": "Payment Processing",
            "compliance_scope": "PCI DSS",
            "rto_hours": 1,
            "threat_intel_match_bool": True,
            "campaign_names": "CitrixBleed Exploitation",
            "threat_actors": "IronVeil",
            "threat_ransomware_bool": False,
            "cveID": "CVE-SYN-2026-0010",
            "kev_known_ransomware_bool": True,
            "days_open": 11,
            "owner_team": "Payments Team",
            "edr_installed_bool": True,
            "feature_cvss": 0.91,
            "contribution_cvss": 0.182,
            "feature_active_exploitation": 0.9,
            "contribution_active_exploitation": 0.135,
            "feature_ransomware": 1.0,
            "contribution_ransomware": 0.2,
            "feature_epss": 0.5,
            "contribution_epss": 0.075,
            "contribution_internet_exposed": 0.1,
            "feature_business_impact": 1.0,
            "contribution_business_impact": 0.1,
            "feature_threat_intel_match": 1.0,
            "contribution_threat_intel_match": 0.05,
            "feature_days_open": 0.12,
            "contribution_days_open": 0.004,
            "feature_missing_edr": 0.0,
            "contribution_missing_edr": 0.0,
            "vuln_id": "V-2009",
        }
    )


def _retrieval(discussion: str = "Organizations identify, report, and correct system flaws.") -> RetrievalResult:
    candidate = RetrievalCandidate(
        control_id="SI-2",
        name="Flaw Remediation",
        discussion=discussion,
        semantic_score=0.5,
        bm25_score=1.0,
        control_prior_score=1.0,
        hybrid_score=0.8,
        rank=1,
        prior_reasons=("SI-2: flaw remediation and patching signal",),
    )
    return RetrievalResult(query="query", priors=tuple(), candidates=(candidate,))


def _long_discussion() -> str:
    sentence = "Organizations identify, report, and correct system flaws within defined time periods. "
    return sentence * 80


def _narration() -> LLMReportNarration:
    return LLMReportNarration(
        executive_brief="The highest risk affects payment processing.",
        risks=(
            RiskNarration(
                risk_rank=1,
                vuln_id="V-2009",
                risk_explanation="This risk matters because payment processing is exposed.",
                nist_application="SI-2 recommends timely flaw remediation.",
                threat_summary="IronVeil activity is matched to this risk.",
                model="llama-3.3-70b-versatile",
                usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            ),
        ),
        model="llama-3.3-70b-versatile",
        total_usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
    )
