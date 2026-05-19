from __future__ import annotations

from typing import Any

import pandas as pd

from src.engine.nist_retriever import RetrievalCandidate


def risk_evidence(row: pd.Series) -> dict[str, Any]:
    return {
        "rank": _value(row, "risk_rank"),
        "severity": _value(row, "risk_severity"),
        "score": round(_float(row, "composite_risk_score"), 3),
        "score_drivers": _value(row, "score_drivers"),
        "asset_name": _value(row, "asset_name"),
        "asset_type": _value(row, "asset_type"),
        "environment": _value(row, "environment"),
        "owner_team": _value(row, "owner_team"),
        "internet_exposed": _bool_text(row, "feature_internet_exposed"),
        "edr_installed": _bool_text(row, "edr_installed_bool"),
        "vulnerability_name": _value(row, "vulnerability_name"),
        "cve": _value(row, "cve"),
        "cvss": round(_float(row, "cvss"), 1),
        "affected_component": _value(row, "affected_component"),
        "exploit_available": _bool_text(row, "exploit_available_bool"),
        "patch_available": _bool_text(row, "patch_available_bool"),
        "days_open": _value(row, "days_open"),
        "kev_status": "Confirmed" if _present(row.get("cveID")) else "Not found",
        "kev_ransomware": "Known" if _truthy(row.get("kev_known_ransomware_bool")) else "Unknown",
        "local_threat_ransomware": "Yes" if _truthy(row.get("threat_ransomware_bool")) else "No",
        "ransomware_signal_source": _ransomware_signal_source(row),
        "epss_percent": round(_float(row, "epss") * 100, 2),
        "business_service": _value(row, "business_service"),
        "business_owner": _value(row, "business_owner"),
        "business_impact": _value(row, "business_impact"),
        "compliance_scope": _value(row, "compliance_scope"),
        "rto_hours": _value(row, "rto_hours"),
        "threat_intel_match": "Matched" if _truthy(row.get("threat_intel_match_bool")) else "No match",
        "campaign_names": _value(row, "campaign_names"),
        "threat_actors": _value(row, "threat_actors"),
        "threat_summaries": _value(row, "threat_summaries"),
    }


def nist_evidence(candidate: RetrievalCandidate) -> dict[str, Any]:
    return {
        "control_id": candidate.control_id,
        "control_name": candidate.name,
        "discussion": _trim(candidate.discussion, 2600),
        "hybrid_score": round(candidate.hybrid_score, 4),
        "semantic_score": round(candidate.semantic_score, 4),
        "bm25_score": round(candidate.bm25_score, 4),
        "prior_score": round(candidate.control_prior_score, 4),
        "prior_reasons": "; ".join(candidate.prior_reasons),
    }


def format_evidence_block(evidence: dict[str, Any]) -> str:
    lines = []
    for key, value in evidence.items():
        if value is None or value == "":
            value = "Not available"
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def summarize_top_risks(rows: list[pd.Series]) -> str:
    parts = []
    for row in rows:
        evidence = risk_evidence(row)
        parts.append(
            (
                f"#{evidence['rank']} {evidence['severity']} "
                f"{evidence['asset_name']} | {evidence['cve']} | "
                f"{evidence['vulnerability_name']} | score {evidence['score']:.3f} | "
                f"service {evidence['business_service']} | drivers {evidence['score_drivers']}"
            )
        )
    return "\n".join(parts)


def _value(row: pd.Series, key: str) -> Any:
    value = row.get(key, "")
    if pd.isna(value) if not isinstance(value, (list, tuple, dict)) else False:
        return ""
    return value


def _float(row: pd.Series, key: str) -> float:
    try:
        value = float(row.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0
    if pd.isna(value):
        return 0.0
    return value


def _bool_text(row: pd.Series, key: str) -> str:
    return "Yes" if _truthy(row.get(key)) else "No"


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(value, (int, float)):
        return value > 0
    return str(value).strip().lower() in {"true", "yes", "1", "known", "confirmed"}


def _present(value: Any) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return bool(text) and text.lower() != "nan"


def _ransomware_signal_source(row: pd.Series) -> str:
    kev = _truthy(row.get("kev_known_ransomware_bool"))
    local = _truthy(row.get("threat_ransomware_bool"))
    if kev and local:
        return "CISA KEV and local threat intelligence"
    if kev:
        return "CISA KEV only"
    if local:
        return "Local threat intelligence only"
    return "None"


def _trim(text: str, max_chars: int) -> str:
    text = str(text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."
