from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

from .config import IngestionConfig
from .csv_loaders import load_structured_data
from .documents import load_threat_report, nist_controls_to_documents
from .external import fetch_cisa_kev, fetch_epss_scores, fetch_nist_controls
from .models import ExternalDataPack, IngestionBundle, StructuredDataPack
from .normalize import normalize_identifier


def build_ingestion_bundle(
    config: IngestionConfig | None = None,
    include_external: bool = True,
) -> IngestionBundle:
    """Load, validate, normalize, and join all Phase 1 inputs."""
    logger.info("Phase 1 ingestion starting (include_external=%s)", include_external)

    config = config or IngestionConfig()
    structured = load_structured_data(config.raw_data_dir)
    logger.info(
        "Structured data loaded: %d assets, %d vulns, %d threat_intel, %d services",
        len(structured.assets),
        len(structured.vulnerabilities),
        len(structured.threat_intelligence),
        len(structured.business_services),
    )
    threat_report = load_threat_report(config.raw_data_dir)

    metadata = []
    warnings: list[str] = []
    if include_external:
        cisa_kev, kev_meta = fetch_cisa_kev(config)
        epss, epss_meta = fetch_epss_scores(structured.vulnerabilities["cve"].tolist(), config)
        nist_controls, nist_meta = fetch_nist_controls(config)
        metadata.extend([kev_meta, epss_meta, nist_meta])
        warnings.extend(meta.warning for meta in metadata if meta.warning)
    else:
        cisa_kev = pd.DataFrame(columns=["cveID", "cve_key", "knownRansomwareCampaignUse"])
        epss = pd.DataFrame(columns=["cve", "cve_key", "epss", "percentile", "date"])
        nist_controls = []

    logger.info(
        "External data: KEV=%d rows, EPSS=%d rows, NIST=%d controls",
        len(cisa_kev),
        len(epss),
        len(nist_controls),
    )
    enriched = enrich_vulnerabilities(structured, cisa_kev, epss)
    unmatched_ti = find_unmatched_threat_intelligence(
        structured.threat_intelligence,
        structured.vulnerabilities,
    )
    rag_documents = [threat_report, *nist_controls_to_documents(nist_controls)]

    logger.info(
        "Phase 1 complete: %d enriched rows, %d unmatched TI, %d RAG docs, %d warnings",
        len(enriched),
        len(unmatched_ti),
        len(rag_documents),
        len(warnings),
    )
    return IngestionBundle(
        structured=structured,
        external=ExternalDataPack(
            cisa_kev=cisa_kev,
            epss=epss,
            nist_controls=nist_controls,
            metadata=metadata,
        ),
        threat_report=threat_report,
        enriched_vulnerabilities=enriched,
        unmatched_threat_intelligence=unmatched_ti,
        rag_documents=rag_documents,
        warnings=warnings,
    )


def enrich_vulnerabilities(
    structured: StructuredDataPack,
    cisa_kev: pd.DataFrame,
    epss: pd.DataFrame,
) -> pd.DataFrame:
    vulnerabilities = structured.vulnerabilities.copy()
    assets = structured.assets.copy()
    business_services = structured.business_services.copy()
    resolved_threat_intel = resolve_threat_match_keys(
        structured.threat_intelligence,
        vulnerabilities,
    )
    threat_rollup = _roll_up_threat_intelligence(resolved_threat_intel)

    frame = vulnerabilities.merge(
        assets,
        on="asset_id",
        how="left",
        suffixes=("", "_asset"),
        validate="many_to_one",
    )
    frame = frame.merge(
        business_services,
        on="business_service",
        how="left",
        suffixes=("", "_service"),
        validate="many_to_one",
    )
    frame = frame.merge(
        threat_rollup,
        left_on="vuln_key",
        right_on="threat_key",
        how="left",
        validate="many_to_one",
    )

    if "cve_key" in cisa_kev.columns:
        kev_columns = [
            column
            for column in [
                "cve_key",
                "cveID",
                "knownRansomwareCampaignUse",
                "dateAdded",
                "requiredAction",
                "kev_known_ransomware_bool",
            ]
            if column in cisa_kev.columns
        ]
        frame = frame.merge(
            cisa_kev[kev_columns].drop_duplicates("cve_key"),
            left_on="vuln_key",
            right_on="cve_key",
            how="left",
            validate="many_to_one",
        )

    if "cve_key" in epss.columns:
        epss_columns = [column for column in ["cve_key", "epss", "percentile", "date"] if column in epss.columns]
        frame = frame.merge(
            epss[epss_columns].drop_duplicates("cve_key"),
            left_on="vuln_key",
            right_on="cve_key",
            how="left",
            suffixes=("", "_epss"),
            validate="many_to_one",
        )

    frame["threat_intel_match_bool"] = frame["intel_ids"].notna()
    frame["threat_ransomware_bool"] = frame["threat_ransomware_bool"].where(
        frame["threat_ransomware_bool"].notna(),
        False,
    ).astype(bool)
    frame["kev_known_ransomware_bool"] = frame.get("kev_known_ransomware_bool", False)
    frame["kev_known_ransomware_bool"] = frame["kev_known_ransomware_bool"].where(
        frame["kev_known_ransomware_bool"].notna(),
        False,
    ).astype(bool)
    frame["epss"] = pd.to_numeric(frame.get("epss", 0.0), errors="coerce").fillna(0.0)
    frame["percentile"] = pd.to_numeric(frame.get("percentile", 0.0), errors="coerce").fillna(0.0)
    frame["campaign_names"] = frame["campaign_names"].fillna("")
    frame["threat_actors"] = frame["threat_actors"].fillna("")
    frame["exploit_maturities"] = frame["exploit_maturities"].fillna("")
    return frame


def find_unmatched_threat_intelligence(
    threat_intelligence: pd.DataFrame,
    vulnerabilities: pd.DataFrame,
) -> pd.DataFrame:
    threat_intelligence = resolve_threat_match_keys(threat_intelligence, vulnerabilities)
    vuln_keys = set(vulnerabilities["vuln_key"].dropna())
    return threat_intelligence.loc[~threat_intelligence["resolved_threat_key"].isin(vuln_keys)].copy()


def resolve_threat_match_keys(
    threat_intelligence: pd.DataFrame,
    vulnerabilities: pd.DataFrame,
) -> pd.DataFrame:
    """Resolve exact and narrowly aliased threat keys against local vulnerabilities."""

    frame = threat_intelligence.copy()
    vuln_keys = set(vulnerabilities["vuln_key"].dropna())
    resolved_keys: list[str] = []
    alias_flags: list[bool] = []

    for key in frame["threat_key"]:
        key = normalize_identifier(key)
        if key in vuln_keys:
            resolved_keys.append(key)
            alias_flags.append(False)
            continue

        synthetic_key = key.replace("CVE-", "CVE-SYN-", 1)
        if key.startswith("CVE-") and synthetic_key in vuln_keys:
            resolved_keys.append(synthetic_key)
            alias_flags.append(True)
            continue

        resolved_keys.append(key)
        alias_flags.append(False)

    frame["resolved_threat_key"] = resolved_keys
    frame["threat_key_alias_applied_bool"] = alias_flags
    return frame


def _roll_up_threat_intelligence(threat_intelligence: pd.DataFrame) -> pd.DataFrame:
    if threat_intelligence.empty:
        return pd.DataFrame(
            columns=[
                "threat_key",
                "intel_ids",
                "campaign_names",
                "threat_actors",
                "exploit_maturities",
                "threat_summaries",
                "threat_ransomware_bool",
                "threat_match_count",
            ]
        )

    def join_unique(values: pd.Series) -> str:
        unique = [str(value).strip() for value in values if str(value).strip()]
        return "; ".join(dict.fromkeys(unique))

    key_column = "resolved_threat_key" if "resolved_threat_key" in threat_intelligence.columns else "threat_key"
    grouped = threat_intelligence.groupby(key_column, as_index=False).agg(
        intel_ids=("intel_id", join_unique),
        campaign_names=("campaign_name", join_unique),
        threat_actors=("threat_actor", join_unique),
        exploit_maturities=("exploit_maturity", join_unique),
        threat_summaries=("summary", join_unique),
        threat_ransomware_bool=("ransomware_association_bool", "max"),
        threat_match_count=("intel_id", "count"),
        threat_alias_applied_bool=("threat_key_alias_applied_bool", "max"),
    )
    grouped = grouped.rename(columns={key_column: "threat_key"})
    grouped["threat_ransomware_bool"] = grouped["threat_ransomware_bool"].astype(bool)
    grouped["threat_alias_applied_bool"] = grouped["threat_alias_applied_bool"].astype(bool)
    return grouped
