from __future__ import annotations

import logging
import math
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

from .scoring_config import ScoringConfig


FEATURE_COLUMNS = (
    "feature_cvss",
    "feature_active_exploitation",
    "feature_ransomware",
    "feature_epss",
    "feature_internet_exposed",
    "feature_business_impact",
    "feature_threat_intel_match",
    "feature_days_open",
    "feature_missing_edr",
)

REQUIRED_SCORING_COLUMNS = (
    "cvss",
    "days_open",
    "exploit_available_bool",
    "internet_exposed_bool",
    "internet_exposed_bool_asset",
    "criticality_score",
    "revenue_impact_score",
    "customer_facing_bool",
    "compliance_scope",
    "rto_hours",
    "edr_installed_bool",
    "threat_intel_match_bool",
    "threat_ransomware_bool",
    "kev_known_ransomware_bool",
    "epss",
)


class ScoringInputError(ValueError):
    """Raised when Phase 2 receives data that Phase 1 did not prepare."""


def score_risks(
    enriched_vulnerabilities: pd.DataFrame,
    config: ScoringConfig | None = None,
) -> pd.DataFrame:
    """Return vulnerabilities with deterministic score features and ranking."""

    config = config or ScoringConfig()
    _validate_scoring_input(enriched_vulnerabilities)
    frame = enriched_vulnerabilities.copy()

    frame["feature_cvss"] = frame["cvss"].map(lambda value: _normalize_numeric(value, 0.0, 10.0))
    frame["feature_active_exploitation"] = frame.apply(_active_exploitation_score, axis=1)
    frame["feature_ransomware"] = frame.apply(_ransomware_score, axis=1)
    frame["feature_epss"] = frame["epss"].map(lambda value: _normalize_numeric(value, 0.0, 1.0))
    frame["feature_internet_exposed"] = frame.apply(_internet_exposure_score, axis=1)
    frame["feature_business_impact"] = frame.apply(_business_impact_score, axis=1)
    frame["feature_threat_intel_match"] = frame["threat_intel_match_bool"].map(_bool_score)
    frame["feature_days_open"] = frame["days_open"].map(
        lambda value: _days_open_score(value, config.days_open_full_penalty)
    )
    frame["feature_missing_edr"] = frame["edr_installed_bool"].map(lambda value: 0.0 if _truthy(value) else 1.0)

    for feature_name, weight_name in _feature_to_weight_name().items():
        contribution_name = feature_name.replace("feature_", "contribution_")
        frame[contribution_name] = frame[feature_name] * config.weights.as_dict()[weight_name]

    contribution_columns = [column.replace("feature_", "contribution_") for column in FEATURE_COLUMNS]
    frame["composite_risk_score"] = frame[contribution_columns].sum(axis=1).clip(0.0, 1.0)
    frame["risk_score_percent"] = (frame["composite_risk_score"] * 100).round(1)
    frame["risk_severity"] = frame["composite_risk_score"].map(lambda value: _severity_label(value, config))
    frame["score_drivers"] = frame.apply(_score_drivers, axis=1)
    frame["risk_explanation"] = frame.apply(build_deterministic_explanation, axis=1)

    ranked = _rank(frame)
    logger.info(
        "Phase 2 scoring complete: %d rows, score range [%.3f, %.3f], critical=%d, high=%d",
        len(ranked),
        ranked["composite_risk_score"].min(),
        ranked["composite_risk_score"].max(),
        int((ranked["risk_severity"] == "Critical").sum()),
        int((ranked["risk_severity"] == "High").sum()),
    )
    return ranked


def top_risks(
    enriched_vulnerabilities: pd.DataFrame,
    config: ScoringConfig | None = None,
    n: int | None = None,
) -> pd.DataFrame:
    config = config or ScoringConfig()
    scored = score_risks(enriched_vulnerabilities, config)
    return scored.head(n or config.top_n).copy()


def _validate_scoring_input(frame: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_SCORING_COLUMNS if column not in frame.columns]
    if missing:
        raise ScoringInputError(
            "Scoring input is missing required Phase 1 enrichment columns: "
            + ", ".join(missing)
        )


def build_deterministic_explanation(row: pd.Series) -> str:
    asset = _text(row.get("asset_name"), "Unknown asset")
    service = _text(row.get("business_service"), "Unknown service")
    exposure = "internet-facing" if _truthy(row.get("feature_internet_exposed")) else "internal"
    ransomware = "ransomware-linked " if _truthy(row.get("feature_ransomware")) else ""
    exploit = "active exploitation signals" if row.get("feature_active_exploitation", 0) >= 0.85 else "available exploit paths"
    business = _text(row.get("business_impact"), "business impact is not documented")

    if row.get("feature_threat_intel_match", 0) > 0:
        campaign = _text(row.get("campaign_names"), "matched threat intelligence")
        threat_clause = f" It is also matched to {campaign}, increasing confidence that this is relevant to the current threat environment."
    else:
        threat_clause = ""

    return (
        f"{asset} ranks highly because a {ransomware}{_text(row.get('vulnerability_name'), 'vulnerability')} "
        f"affects a {exposure} asset supporting {service}, with {exploit}. "
        f"The business impact is: {business}.{threat_clause}"
    )


def _rank(frame: pd.DataFrame) -> pd.DataFrame:
    tie_breakers = [
        "composite_risk_score",
        "feature_ransomware",
        "feature_active_exploitation",
        "feature_internet_exposed",
        "feature_business_impact",
        "cvss",
        "days_open",
        "vuln_id",
    ]
    ranked = frame.sort_values(
        tie_breakers,
        ascending=[False, False, False, False, False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranked["risk_rank"] = ranked.index + 1
    return ranked


def _feature_to_weight_name() -> dict[str, str]:
    return {
        "feature_cvss": "cvss",
        "feature_active_exploitation": "active_exploitation",
        "feature_ransomware": "ransomware",
        "feature_epss": "epss",
        "feature_internet_exposed": "internet_exposed",
        "feature_business_impact": "business_impact",
        "feature_threat_intel_match": "threat_intel_match",
        "feature_days_open": "days_open",
        "feature_missing_edr": "missing_edr",
    }


def _active_exploitation_score(row: pd.Series) -> float:
    kev_confirmed = _present(row.get("cveID"))
    exploit_available = _truthy(row.get("exploit_available_bool"))
    threat_matched = _truthy(row.get("threat_intel_match_bool"))
    maturity = _text(row.get("exploit_maturities")).lower()

    if kev_confirmed and (exploit_available or threat_matched or _has_active_maturity(maturity)):
        return 1.0

    if kev_confirmed or _has_active_maturity(maturity):
        return 0.9
    if exploit_available:
        return 0.75
    if "proof of concept" in maturity:
        return 0.45
    return 0.0


def _ransomware_score(row: pd.Series) -> float:
    return max(
        _bool_score(row.get("kev_known_ransomware_bool")),
        _bool_score(row.get("threat_ransomware_bool")),
    )


def _internet_exposure_score(row: pd.Series) -> float:
    return max(
        _bool_score(row.get("internet_exposed_bool")),
        _bool_score(row.get("internet_exposed_bool_asset")),
    )


def _business_impact_score(row: pd.Series) -> float:
    asset_criticality = _normalize_numeric(row.get("criticality_score"), 1.0, 5.0)
    revenue_impact = _normalize_numeric(row.get("revenue_impact_score"), 1.0, 5.0)
    customer_facing = _bool_score(row.get("customer_facing_bool"))
    compliance = 0.0 if _text(row.get("compliance_scope")).lower() in {"", "none", "nan"} else 1.0
    rto = _rto_urgency_score(row.get("rto_hours"))

    # Keep criticality/revenue dominant, then let compliance and RTO break ties.
    return min(
        1.0,
        asset_criticality * 0.45
        + revenue_impact * 0.30
        + customer_facing * 0.10
        + compliance * 0.10
        + rto * 0.05,
    )


def _rto_urgency_score(value: Any) -> float:
    try:
        rto = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(rto):
        return 0.0
    if rto <= 1:
        return 1.0
    if rto <= 4:
        return 0.8
    if rto <= 12:
        return 0.5
    if rto <= 24:
        return 0.3
    return 0.1


def _days_open_score(value: Any, full_penalty_days: int) -> float:
    try:
        days = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(days) or days <= 0:
        return 0.0
    return min(1.0, days / full_penalty_days)


def _score_drivers(row: pd.Series, limit: int = 4) -> str:
    labels = {
        "contribution_cvss": "CVSS severity",
        "contribution_active_exploitation": "active exploitation",
        "contribution_ransomware": "ransomware association",
        "contribution_epss": "EPSS probability",
        "contribution_internet_exposed": "internet exposure",
        "contribution_business_impact": "business impact",
        "contribution_threat_intel_match": "threat intelligence match",
        "contribution_days_open": "age of finding",
        "contribution_missing_edr": "missing EDR",
    }
    scored = sorted(
        ((labels[column], float(row.get(column, 0.0))) for column in labels),
        key=lambda item: item[1],
        reverse=True,
    )
    return ", ".join(label for label, score in scored[:limit] if score > 0)


def _severity_label(value: float, config: ScoringConfig) -> str:
    for band in sorted(config.severity_bands, key=lambda item: item.minimum_score, reverse=True):
        if value >= band.minimum_score:
            return band.label
    return "Low"


def _normalize_numeric(value: Any, minimum: float, maximum: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(numeric) or maximum <= minimum:
        return 0.0
    return min(1.0, max(0.0, (numeric - minimum) / (maximum - minimum)))


def _bool_score(value: Any) -> float:
    return 1.0 if _truthy(value) else 0.0


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    return str(value).strip().lower() in {"true", "yes", "1", "known"}


def _present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    text = str(value).strip()
    return bool(text) and text.lower() != "nan"


def _has_active_maturity(text: str) -> bool:
    return any(
        phrase in text
        for phrase in (
            "weaponized",
            "active exploitation",
            "actively exploited",
            "commodity exploit",
        )
    )


def _text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, float) and math.isnan(value):
        return default
    text = str(value).strip()
    return text if text else default
