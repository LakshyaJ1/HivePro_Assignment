from __future__ import annotations

from pathlib import Path

import pandas as pd

from .models import StructuredDataPack
from .normalize import (
    CRITICALITY_SCORE,
    REVENUE_IMPACT_SCORE,
    normalize_identifier,
    parse_yes_no,
    score_label,
    split_csv_cell,
)
from .validators import (
    ASSETS_SCHEMA,
    BUSINESS_SERVICES_SCHEMA,
    REMEDIATION_SCHEMA,
    THREAT_INTEL_SCHEMA,
    VULNERABILITIES_SCHEMA,
    assert_foreign_key,
    assert_unique,
)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required data file not found: {path}")
    return pd.read_csv(path, keep_default_na=False)


def load_assets(raw_data_dir: Path) -> pd.DataFrame:
    frame = _read_csv(raw_data_dir / "assets.csv")
    ASSETS_SCHEMA.validate(frame)
    assert_unique(frame, "asset_id", "assets.csv")

    frame = frame.copy()
    frame["internet_exposed_bool"] = frame["internet_exposed"].map(parse_yes_no)
    frame["edr_installed_bool"] = frame["edr_installed"].map(parse_yes_no)
    frame["criticality_score"] = frame["criticality"].map(
        lambda value: score_label(value, CRITICALITY_SCORE)
    )
    frame["last_seen_days"] = pd.to_numeric(frame["last_seen_days"], errors="coerce")
    frame["is_stale_asset"] = frame["last_seen_days"].fillna(9999) > 30
    frame["has_owner"] = frame["owner_team"].astype(str).str.strip().ne("")
    return frame


def load_vulnerabilities(raw_data_dir: Path, assets: pd.DataFrame) -> pd.DataFrame:
    frame = _read_csv(raw_data_dir / "vulnerabilities.csv")
    VULNERABILITIES_SCHEMA.validate(frame)
    assert_unique(frame, "vuln_id", "vulnerabilities.csv")
    assert_foreign_key(
        frame,
        "asset_id",
        assets,
        "asset_id",
        "vulnerabilities.csv",
        "assets.csv",
    )

    frame = frame.copy()
    frame["vuln_key"] = frame["cve"].map(normalize_identifier)
    frame["cvss"] = pd.to_numeric(frame["cvss"], errors="coerce")
    frame["days_open"] = pd.to_numeric(frame["days_open"], errors="coerce")
    frame["exploit_available_bool"] = frame["exploit_available"].map(parse_yes_no)
    frame["patch_available_bool"] = frame["patch_available"].map(parse_yes_no)
    frame["auth_required_bool"] = frame["auth_required"].map(parse_yes_no)
    frame["internet_exposed_bool"] = frame["asset_exposure"].str.lower().eq("internet")
    return frame


def load_threat_intelligence(raw_data_dir: Path) -> pd.DataFrame:
    frame = _read_csv(raw_data_dir / "threat_intelligence.csv")
    THREAT_INTEL_SCHEMA.validate(frame)
    assert_unique(frame, "intel_id", "threat_intelligence.csv")

    frame = frame.copy()
    frame["threat_key"] = frame["matched_cve_or_control"].map(normalize_identifier)
    frame["ransomware_association_bool"] = frame["ransomware_association"].map(parse_yes_no)
    frame["active_last_seen"] = pd.to_datetime(frame["active_last_seen"], errors="coerce")
    return frame


def load_business_services(raw_data_dir: Path) -> pd.DataFrame:
    frame = _read_csv(raw_data_dir / "business_services.csv")
    BUSINESS_SERVICES_SCHEMA.validate(frame)
    assert_unique(frame, "business_service", "business_services.csv")

    frame = frame.copy()
    frame["customer_facing_bool"] = frame["customer_facing"].map(parse_yes_no)
    frame["revenue_impact_score"] = frame["revenue_impact"].map(
        lambda value: score_label(value, REVENUE_IMPACT_SCORE)
    )
    frame["rto_hours"] = pd.to_numeric(frame["rto_hours"], errors="coerce")
    frame["compliance_scopes"] = frame["compliance_scope"].map(split_csv_cell)
    frame["dependencies"] = frame["depends_on"].map(split_csv_cell)
    return frame


def load_remediation_guidance(raw_data_dir: Path) -> pd.DataFrame:
    frame = _read_csv(raw_data_dir / "remediation_guidance.csv")
    REMEDIATION_SCHEMA.validate(frame)
    return frame.copy()


def load_structured_data(raw_data_dir: Path) -> StructuredDataPack:
    assets = load_assets(raw_data_dir)
    vulnerabilities = load_vulnerabilities(raw_data_dir, assets)
    threat_intelligence = load_threat_intelligence(raw_data_dir)
    business_services = load_business_services(raw_data_dir)
    remediation_guidance = load_remediation_guidance(raw_data_dir)

    return StructuredDataPack(
        assets=assets,
        vulnerabilities=vulnerabilities,
        threat_intelligence=threat_intelligence,
        business_services=business_services,
        remediation_guidance=remediation_guidance,
    )

