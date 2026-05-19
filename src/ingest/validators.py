from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


class DataValidationError(ValueError):
    """Raised when an input file is missing required assignment fields."""


@dataclass(frozen=True)
class Schema:
    name: str
    required_columns: tuple[str, ...]

    def validate(self, frame: pd.DataFrame) -> None:
        missing = [column for column in self.required_columns if column not in frame.columns]
        if missing:
            raise DataValidationError(
                f"{self.name} is missing required columns: {', '.join(missing)}"
            )


ASSETS_SCHEMA = Schema(
    "assets.csv",
    (
        "asset_id",
        "asset_name",
        "asset_type",
        "environment",
        "owner_team",
        "business_service",
        "internet_exposed",
        "criticality",
        "data_classification",
        "edr_installed",
        "last_seen_days",
        "location",
        "vendor_product",
    ),
)

VULNERABILITIES_SCHEMA = Schema(
    "vulnerabilities.csv",
    (
        "vuln_id",
        "asset_id",
        "vulnerability_name",
        "cve",
        "severity",
        "cvss",
        "exploit_available",
        "patch_available",
        "days_open",
        "asset_exposure",
        "auth_required",
        "status",
        "affected_component",
    ),
)

THREAT_INTEL_SCHEMA = Schema(
    "threat_intelligence.csv",
    (
        "intel_id",
        "threat_actor",
        "campaign_name",
        "target_sector",
        "target_region",
        "matched_cve_or_control",
        "exploit_maturity",
        "active_last_seen",
        "ransomware_association",
        "confidence",
        "summary",
    ),
)

BUSINESS_SERVICES_SCHEMA = Schema(
    "business_services.csv",
    (
        "business_service",
        "business_owner",
        "business_impact",
        "customer_facing",
        "compliance_scope",
        "revenue_impact",
        "rto_hours",
        "depends_on",
        "risk_appetite",
    ),
)

REMEDIATION_SCHEMA = Schema(
    "remediation_guidance.csv",
    (
        "finding_type",
        "recommended_action",
        "priority_hint",
        "validation_evidence",
    ),
)


def assert_unique(frame: pd.DataFrame, column: str, file_name: str) -> None:
    duplicates = frame[column][frame[column].duplicated()].dropna().unique()
    if len(duplicates):
        values = ", ".join(map(str, duplicates[:10]))
        raise DataValidationError(f"{file_name} has duplicate {column} values: {values}")


def assert_foreign_key(
    child: pd.DataFrame,
    child_column: str,
    parent: pd.DataFrame,
    parent_column: str,
    child_name: str,
    parent_name: str,
) -> None:
    parent_values = set(parent[parent_column].dropna())
    missing = sorted(set(child[child_column].dropna()) - parent_values)
    if missing:
        sample = ", ".join(map(str, missing[:10]))
        raise DataValidationError(
            f"{child_name}.{child_column} contains values not present in "
            f"{parent_name}.{parent_column}: {sample}"
        )
