from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .config import IngestionConfig
from .models import FetchMetadata, NistControl
from .normalize import normalize_cve


KEV_COLUMNS = ("cveID", "knownRansomwareCampaignUse", "dateAdded", "requiredAction")
EPSS_COLUMNS = ("cve", "epss", "percentile", "date")


def fetch_cisa_kev(config: IngestionConfig) -> tuple[pd.DataFrame, FetchMetadata]:
    config.external_data_dir.mkdir(parents=True, exist_ok=True)
    cache_path = config.cisa_cache_path
    stale_cache_used = False
    refreshed = False
    warning = None

    try:
        if _cache_is_fresh(cache_path, config.cache_ttl_hours):
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            response = requests.get(config.cisa_kev_url, timeout=config.request_timeout_seconds)
            response.raise_for_status()
            payload = response.json()
            cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            refreshed = True
    except Exception as exc:
        if cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            stale_cache_used = True
            warning = f"CISA KEV refresh failed; using cached data: {exc}"
        else:
            payload = {"vulnerabilities": []}
            warning = f"CISA KEV unavailable and no cache exists: {exc}"

    records = payload.get("vulnerabilities", [])
    frame = pd.DataFrame.from_records(records)
    for column in KEV_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.Series(dtype="object")
    frame = frame.loc[:, list(dict.fromkeys([*KEV_COLUMNS, *frame.columns]))].copy()
    frame["cve_key"] = frame["cveID"].map(normalize_cve)
    frame["kev_known_ransomware_bool"] = (
        frame["knownRansomwareCampaignUse"].astype(str).str.lower().eq("known")
    )

    return frame, FetchMetadata(
        source="CISA KEV",
        url=config.cisa_kev_url,
        cache_path=cache_path,
        refreshed=refreshed,
        stale_cache_used=stale_cache_used,
        warning=warning,
    )


def fetch_epss_scores(
    cves: list[str],
    config: IngestionConfig,
) -> tuple[pd.DataFrame, FetchMetadata]:
    config.external_data_dir.mkdir(parents=True, exist_ok=True)
    cache_path = config.epss_cache_path
    normalized_cves = sorted(
        {
            normalize_cve(cve)
            for cve in cves
            if re.fullmatch(r"CVE-\d{4}-\d{4,}", normalize_cve(cve))
        }
    )
    stale_cache_used = False
    refreshed = False
    warning = None

    if not normalized_cves:
        return _empty_epss_frame(), FetchMetadata(
            source="FIRST EPSS",
            url=config.epss_api_url,
            cache_path=cache_path,
            refreshed=False,
        )

    try:
        if _cache_is_fresh(cache_path, config.cache_ttl_hours):
            frame = pd.read_csv(cache_path, keep_default_na=False)
        else:
            records: list[dict[str, Any]] = []
            for batch in _chunks(normalized_cves, 100):
                response = requests.get(
                    config.epss_api_url,
                    params={"cve": ",".join(batch)},
                    timeout=config.request_timeout_seconds,
                )
                response.raise_for_status()
                records.extend(response.json().get("data", []))
                time.sleep(0.2)
            frame = pd.DataFrame.from_records(records)
            frame.to_csv(cache_path, index=False)
            refreshed = True
    except Exception as exc:
        if cache_path.exists():
            frame = pd.read_csv(cache_path, keep_default_na=False)
            stale_cache_used = True
            warning = f"EPSS refresh failed; using cached data: {exc}"
        else:
            frame = _empty_epss_frame()
            warning = f"EPSS unavailable and no cache exists: {exc}"

    for column in EPSS_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.Series(dtype="object")
    frame["cve_key"] = frame["cve"].map(normalize_cve)
    frame["epss"] = pd.to_numeric(frame["epss"], errors="coerce").fillna(0.0)
    frame["percentile"] = pd.to_numeric(frame["percentile"], errors="coerce").fillna(0.0)

    return frame, FetchMetadata(
        source="FIRST EPSS",
        url=config.epss_api_url,
        cache_path=cache_path,
        refreshed=refreshed,
        stale_cache_used=stale_cache_used,
        warning=warning,
    )


def fetch_nist_controls(config: IngestionConfig) -> tuple[list[NistControl], FetchMetadata]:
    config.external_data_dir.mkdir(parents=True, exist_ok=True)
    cache_path = config.nist_cache_path
    stale_cache_used = False
    refreshed = False
    warning = None

    try:
        if not _cache_is_fresh(cache_path, config.cache_ttl_hours):
            response = requests.get(config.nist_catalog_url, timeout=config.request_timeout_seconds)
            response.raise_for_status()
            cache_path.write_bytes(response.content)
            refreshed = True
        controls = parse_nist_catalog(cache_path)
    except Exception as exc:
        if cache_path.exists():
            controls = parse_nist_catalog(cache_path)
            stale_cache_used = True
            warning = f"NIST refresh failed; using cached data: {exc}"
        else:
            controls = []
            warning = f"NIST catalog unavailable and no cache exists: {exc}"

    return controls, FetchMetadata(
        source="NIST SP 800-53 Rev. 5",
        url=config.nist_catalog_url,
        cache_path=cache_path,
        refreshed=refreshed,
        stale_cache_used=stale_cache_used,
        warning=warning,
    )


def parse_nist_catalog(path: Path) -> list[NistControl]:
    workbook = pd.ExcelFile(path)
    frames = []
    for sheet_name in workbook.sheet_names:
        frame = workbook.parse(sheet_name)
        if _looks_like_control_sheet(frame):
            frames.append(frame)

    if not frames:
        raise ValueError(f"No NIST control sheet found in {path}")

    frame = pd.concat(frames, ignore_index=True)
    column_map = _resolve_nist_columns(frame)
    controls: list[NistControl] = []
    seen: set[str] = set()

    for _, row in frame.iterrows():
        control_id = str(row.get(column_map["control_id"], "")).strip()
        name = str(row.get(column_map["name"], "")).strip()
        discussion = str(row.get(column_map["discussion"], "")).strip()
        family = str(row.get(column_map["family"], "")).strip() if column_map.get("family") else None

        if not control_id or not discussion or control_id.lower() == "nan":
            continue
        if control_id in seen:
            continue
        seen.add(control_id)
        controls.append(NistControl(control_id=control_id, name=name, discussion=discussion, family=family))

    return controls


def _cache_is_fresh(path: Path, ttl_hours: int) -> bool:
    if not path.exists():
        return False
    age_seconds = time.time() - path.stat().st_mtime
    return age_seconds < ttl_hours * 3600


def _empty_epss_frame() -> pd.DataFrame:
    frame = pd.DataFrame(columns=[*EPSS_COLUMNS, "cve_key"])
    frame["epss"] = pd.Series(dtype="float64")
    frame["percentile"] = pd.Series(dtype="float64")
    return frame


def _chunks(values: list[str], size: int) -> list[list[str]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def _looks_like_control_sheet(frame: pd.DataFrame) -> bool:
    names = {str(column).strip().lower() for column in frame.columns}
    has_control = any("control identifier" in name or name == "identifier" for name in names)
    has_discussion = any("discussion" in name for name in names)
    return has_control and has_discussion


def _resolve_nist_columns(frame: pd.DataFrame) -> dict[str, str]:
    normalized = {str(column).strip().lower(): column for column in frame.columns}

    def find(*needles: str) -> str:
        for key, original in normalized.items():
            if all(needle in key for needle in needles):
                return original
        raise ValueError(f"NIST catalog is missing a column containing: {needles}")

    name_column = None
    for key, original in normalized.items():
        if "control" in key and "name" in key:
            name_column = original
            break
        if key == "name":
            name_column = original
            break

    family_column = None
    for key, original in normalized.items():
        if "family" in key:
            family_column = original
            break

    return {
        "control_id": find("control", "identifier"),
        "name": name_column or find("title"),
        "discussion": find("discussion"),
        "family": family_column,
    }
