from __future__ import annotations

import pandas as pd

from src.engine.scoring import FEATURE_COLUMNS, ScoringInputError, score_risks, top_risks
from src.engine.scoring_config import ScoringConfig, ScoreWeights
from src.ingest.pipeline import build_ingestion_bundle


def test_phase2_scores_all_vulnerabilities_and_top_five_are_high_signal() -> None:
    bundle = build_ingestion_bundle(include_external=False)
    scored = score_risks(bundle.enriched_vulnerabilities)
    top = scored.head(5)

    assert len(scored) == 114
    assert list(top["risk_rank"]) == [1, 2, 3, 4, 5]
    assert top["risk_severity"].eq("Critical").all()
    assert top["feature_ransomware"].eq(1.0).all()
    assert top["feature_internet_exposed"].eq(1.0).all()
    assert top["feature_threat_intel_match"].eq(1.0).all()


def test_composite_score_is_sum_of_weighted_contributions() -> None:
    bundle = build_ingestion_bundle(include_external=False)
    scored = score_risks(bundle.enriched_vulnerabilities)
    contribution_columns = [column.replace("feature_", "contribution_") for column in FEATURE_COLUMNS]

    recomputed = scored[contribution_columns].sum(axis=1).round(10)
    actual = scored["composite_risk_score"].round(10)
    assert recomputed.equals(actual)


def test_cvss_alone_does_not_outrank_business_exposed_ransomware_risk() -> None:
    rows = pd.DataFrame(
        [
            _risk_row(
                vuln_id="internal-cvss-10",
                cvss=10.0,
                internet=False,
                ransomware=False,
                threat_match=False,
                business_score=2,
                revenue_score=2,
                edr=True,
                days_open=15,
                exploit=False,
            ),
            _risk_row(
                vuln_id="business-cvss-8",
                cvss=8.0,
                internet=True,
                ransomware=True,
                threat_match=True,
                business_score=5,
                revenue_score=5,
                edr=True,
                days_open=10,
                exploit=True,
            ),
        ]
    )

    scored = score_risks(rows)
    assert scored.iloc[0]["vuln_id"] == "business-cvss-8"
    assert scored.iloc[0]["composite_risk_score"] > scored.iloc[1]["composite_risk_score"]


def test_invalid_weight_profile_fails_fast() -> None:
    bad_weights = ScoreWeights(cvss=0.99)
    try:
        ScoringConfig(weights=bad_weights)
    except ValueError as exc:
        assert "sum to 1.0" in str(exc)
    else:
        raise AssertionError("invalid weights should fail validation")


def test_top_risks_respects_requested_count() -> None:
    bundle = build_ingestion_bundle(include_external=False)
    top = top_risks(bundle.enriched_vulnerabilities, n=3)
    assert len(top) == 3


def test_cisa_kev_presence_and_exploit_signal_scores_active_exploitation_at_full_strength() -> None:
    rows = pd.DataFrame(
        [
            {
                **_risk_row(
                    vuln_id="kev-exploited",
                    cvss=9.8,
                    internet=True,
                    ransomware=True,
                    threat_match=True,
                    business_score=5,
                    revenue_score=5,
                    edr=True,
                    days_open=7,
                    exploit=True,
                ),
                "cveID": "CVE-2023-4966",
            }
        ]
    )

    scored = score_risks(rows)

    assert scored.iloc[0]["feature_active_exploitation"] == 1.0


def test_scoring_fails_clearly_for_missing_phase1_columns() -> None:
    try:
        score_risks(pd.DataFrame([{"cvss": 10.0}]))
    except ScoringInputError as exc:
        assert "missing required Phase 1 enrichment columns" in str(exc)
        assert "days_open" in str(exc)
    else:
        raise AssertionError("malformed scoring input should fail")


def _risk_row(
    *,
    vuln_id: str,
    cvss: float,
    internet: bool,
    ransomware: bool,
    threat_match: bool,
    business_score: int,
    revenue_score: int,
    edr: bool,
    days_open: int,
    exploit: bool,
) -> dict[str, object]:
    return {
        "vuln_id": vuln_id,
        "asset_name": vuln_id,
        "asset_id": vuln_id,
        "vulnerability_name": "Synthetic risk",
        "cve": "CVE-SYN-TEST",
        "cvss": cvss,
        "days_open": days_open,
        "exploit_available_bool": exploit,
        "internet_exposed_bool": internet,
        "internet_exposed_bool_asset": internet,
        "criticality_score": business_score,
        "revenue_impact_score": revenue_score,
        "customer_facing_bool": internet,
        "compliance_scope": "PCI DSS" if business_score >= 5 else "None",
        "rto_hours": 1 if business_score >= 5 else 48,
        "edr_installed_bool": edr,
        "threat_intel_match_bool": threat_match,
        "threat_ransomware_bool": ransomware,
        "kev_known_ransomware_bool": False,
        "epss": 0.0,
        "business_service": "Synthetic Service",
        "business_impact": "Synthetic business impact",
        "campaign_names": "Synthetic Campaign" if threat_match else "",
        "cveID": "",
        "exploit_maturities": "Weaponized" if exploit else "",
    }
