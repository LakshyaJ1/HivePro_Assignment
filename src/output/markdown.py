from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.engine.nist_retriever import RetrievalResult
from src.llm.narrator import LLMReportNarration, RiskNarration

from .models import MarkdownReport


SECTION_RULE = "-" * 72


def render_markdown_report(
    *,
    narration: LLMReportNarration,
    risk_retrievals: tuple[tuple[pd.Series, RetrievalResult], ...],
    generated_at: datetime | None = None,
) -> MarkdownReport:
    generated_at = generated_at or datetime.now(timezone.utc)
    narration_by_vuln = {risk.vuln_id: risk for risk in narration.risks}

    lines: list[str] = [
        "# TawasolPay AI Cyber Risk Assistant Report",
        "",
        f"Generated: {generated_at.isoformat(timespec='seconds')}",
        f"Model: {narration.model}",
        f"Risks ranked: {len(risk_retrievals)}",
        "",
        "## Executive Brief",
        "",
        _clean(narration.executive_brief),
        "",
        "## Top Risk Summary",
        "",
        "| Rank | Severity | Score | Asset | Vulnerability | Service | NIST Control |",
        "|---:|---|---:|---|---|---|---|",
    ]

    for row, retrieval in risk_retrievals:
        best = retrieval.best
        lines.append(
            "| {rank} | {severity} | {score:.3f} | {asset} | {vuln} | {service} | {control} |".format(
                rank=int(row["risk_rank"]),
                severity=_escape_table(row["risk_severity"]),
                score=float(row["composite_risk_score"]),
                asset=_escape_table(row["asset_name"]),
                vuln=_escape_table(f"{row['cve']} - {row['vulnerability_name']}"),
                service=_escape_table(row["business_service"]),
                control=_escape_table(f"{best.control_id}: {best.name}"),
            )
        )

    lines.extend(["", "## Asset-Level Exposure Rollup", ""])
    lines.extend(_render_asset_rollup(risk_retrievals))
    lines.extend(["", "## Detailed Risk Entries", ""])

    for row, retrieval in risk_retrievals:
        risk_narration = narration_by_vuln.get(str(row["vuln_id"]))
        if risk_narration is None:
            raise ValueError(f"Missing LLM narration for vuln_id={row['vuln_id']}")
        lines.extend(_render_risk_entry(row, retrieval, risk_narration))

    lines.extend(_render_appendix(narration, risk_retrievals))
    return MarkdownReport(content="\n".join(lines).rstrip() + "\n", risk_count=len(risk_retrievals))


def write_markdown_report(report: MarkdownReport, output_dir: Path | str) -> MarkdownReport:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filename = f"tawasolpay_top_risks_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.md"
    path = output_path / filename
    path.write_text(report.content, encoding="utf-8")
    return MarkdownReport(content=report.content, output_path=path, risk_count=report.risk_count)


def _render_risk_entry(
    row: pd.Series,
    retrieval: RetrievalResult,
    narration: RiskNarration,
) -> list[str]:
    best = retrieval.best
    lines = [
        f"### Risk #{int(row['risk_rank'])} - {_clean(row['risk_severity']).upper()}",
        "",
        SECTION_RULE,
        "",
        f"**Asset:** {_clean(row['asset_name'])} ({_clean(row['asset_type'])}, {_exposure(row)})",
        f"**Vulnerability:** {_clean(row['cve'])} - {_clean(row['vulnerability_name'])} (CVSS {_num(row['cvss'], 1)})",
        f"**Risk Score:** {_num(row['composite_risk_score'], 3)} / 1.000",
        f"**Business Impact:** {_clean(row['business_service'])} ({_clean(row.get('compliance_scope', 'None'))}; RTO {_clean(row.get('rto_hours', 'n/a'))}h)",
        f"**Threat Intel:** {_threat_line(row)}",
        f"**CISA KEV:** {_kev_line(row)}",
        f"**Ransomware Signal Source:** {_ransomware_signal_source(row)}",
        f"**Days Open:** {_clean(row['days_open'])}",
        f"**Owner:** {_clean(row.get('owner_team', 'Unassigned'))}",
        f"**Compensating Controls:** EDR installed: {_yes_no(row.get('edr_installed_bool'))}",
        "",
        "#### Why This Ranks Here",
        "",
        _clean(narration.risk_explanation),
        "",
        f"#### NIST SP 800-53 Control - {best.control_id}: {_clean(best.name)}",
        "",
        _clean(narration.nist_application),
        "",
        "**Retrieved NIST Evidence:**",
        "",
        f"> {_clean(_sentence_excerpt(best.discussion, 1400))}",
        "",
        "**Retrieval Audit:** "
        f"hybrid={best.hybrid_score:.3f}, semantic={best.semantic_score:.3f}, "
        f"bm25={best.bm25_score:.3f}, prior={best.control_prior_score:.3f}",
        "",
        "#### Threat Context",
        "",
        _clean(narration.threat_summary),
        "",
        "#### Score Drivers",
        "",
        "| Component | Feature | Contribution |",
        "|---|---:|---:|",
    ]
    for label, feature_col, contribution_col in _score_components():
        lines.append(
            f"| {label} | {_num(row.get(feature_col, 0.0), 3)} | {_num(row.get(contribution_col, 0.0), 3)} |"
        )
    lines.extend(["", ""])
    return lines


def _render_asset_rollup(
    risk_retrievals: tuple[tuple[pd.Series, RetrievalResult], ...],
) -> list[str]:
    by_asset: dict[str, list[tuple[pd.Series, RetrievalResult]]] = {}
    for row, retrieval in risk_retrievals:
        by_asset.setdefault(_clean(row.get("asset_name")), []).append((row, retrieval))

    lines = [
        "| Asset | Max Score | Risk Count | CVEs / Findings | Services | Primary NIST Controls |",
        "|---|---:|---:|---|---|---|",
    ]
    grouped = sorted(
        by_asset.items(),
        key=lambda item: max(float(row.get("composite_risk_score", 0.0)) for row, _ in item[1]),
        reverse=True,
    )
    for asset, items in grouped:
        max_score = max(float(row.get("composite_risk_score", 0.0)) for row, _ in items)
        cves = _join_unique(f"{row.get('cve')} ({row.get('risk_severity')})" for row, _ in items)
        services = _join_unique(str(row.get("business_service", "")) for row, _ in items)
        controls = _join_unique(f"{retrieval.best.control_id}: {retrieval.best.name}" for _, retrieval in items)
        lines.append(
            f"| {_escape_table(asset)} | {max_score:.3f} | {len(items)} | "
            f"{_escape_table(cves)} | {_escape_table(services)} | {_escape_table(controls)} |"
        )
    return lines


def _render_appendix(
    narration: LLMReportNarration,
    risk_retrievals: tuple[tuple[pd.Series, RetrievalResult], ...],
) -> list[str]:
    lines = [
        "## Appendix A - Unmatched Threat Intelligence Handling",
        "",
        "Threat intelligence records without a matching vulnerability are intentionally excluded from the top-risk score but retained by the ingestion layer as environment noise. This prevents irrelevant campaigns from inflating current-environment risk while preserving them for analyst review.",
        "",
        "## Appendix B - LLM Usage",
        "",
        f"- Model: {narration.model}",
        f"- Prompt tokens: {narration.total_usage.get('prompt_tokens', 0)}",
        f"- Completion tokens: {narration.total_usage.get('completion_tokens', 0)}",
        f"- Total tokens: {narration.total_usage.get('total_tokens', 0)}",
        f"- LLM calls: {len(risk_retrievals) + 1}",
        "",
        "## Appendix C - Retrieval Candidates",
        "",
    ]
    for row, retrieval in risk_retrievals:
        lines.append(f"### Risk #{int(row['risk_rank'])} Candidates")
        lines.append("")
        lines.append("| Rank | Control | Hybrid | Semantic | BM25 | Prior |")
        lines.append("|---:|---|---:|---:|---:|---:|")
        for candidate in retrieval.candidates:
            lines.append(
                f"| {candidate.rank} | {_escape_table(candidate.control_id + ': ' + candidate.name)} | "
                f"{candidate.hybrid_score:.3f} | {candidate.semantic_score:.3f} | "
                f"{candidate.bm25_score:.3f} | {candidate.control_prior_score:.3f} |"
            )
        lines.append("")
    return lines


def _score_components() -> list[tuple[str, str, str]]:
    return [
        ("CVSS severity", "feature_cvss", "contribution_cvss"),
        ("Active exploitation", "feature_active_exploitation", "contribution_active_exploitation"),
        ("Ransomware association", "feature_ransomware", "contribution_ransomware"),
        ("EPSS probability", "feature_epss", "contribution_epss"),
        ("Internet exposure", "feature_internet_exposed", "contribution_internet_exposed"),
        ("Business impact", "feature_business_impact", "contribution_business_impact"),
        ("Threat-intel match", "feature_threat_intel_match", "contribution_threat_intel_match"),
        ("Days open", "feature_days_open", "contribution_days_open"),
        ("Missing EDR", "feature_missing_edr", "contribution_missing_edr"),
    ]


def _threat_line(row: pd.Series) -> str:
    if not _truthy(row.get("threat_intel_match_bool")):
        return "No matched threat intelligence in current environment"
    ransomware = "local ransomware: Yes" if _truthy(row.get("threat_ransomware_bool")) else "local ransomware: No"
    return f"Matched - {_clean(row.get('campaign_names', 'Unknown campaign'))} ({_clean(row.get('threat_actors', 'Unknown actor'))}; {ransomware})"


def _kev_line(row: pd.Series) -> str:
    status = "Confirmed" if _present(row.get("cveID")) else "Not found"
    ransomware = "Known" if _truthy(row.get("kev_known_ransomware_bool")) else "Unknown"
    return f"{status}. Ransomware: {ransomware}."


def _ransomware_signal_source(row: pd.Series) -> str:
    kev = _truthy(row.get("kev_known_ransomware_bool"))
    local = _truthy(row.get("threat_ransomware_bool"))
    if kev and local:
        return "CISA KEV and local threat intelligence"
    if kev:
        return "CISA KEV only; local campaign is not marked as ransomware"
    if local:
        return "Local threat intelligence only"
    return "No ransomware association in current evidence"


def _exposure(row: pd.Series) -> str:
    return "internet-exposed" if _truthy(row.get("feature_internet_exposed")) else "internal"


def _yes_no(value: Any) -> str:
    return "Yes" if _truthy(value) else "No"


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


def _num(value: Any, digits: int) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "0." + "0" * digits


def _clean(value: Any) -> str:
    if value is None:
        return "Not available"
    try:
        if pd.isna(value):
            return "Not available"
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text if text else "Not available"


def _truncate(text: str, max_chars: int) -> str:
    text = _clean(text)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _sentence_excerpt(text: str, max_chars: int) -> str:
    text = " ".join(_clean(text).split())
    if len(text) <= max_chars:
        return text
    boundary = max(text.rfind(". ", 0, max_chars), text.rfind("; ", 0, max_chars))
    if boundary >= max_chars * 0.55:
        return text[: boundary + 1].rstrip()
    return text[:max_chars].rsplit(" ", 1)[0].rstrip() + "."


def _escape_table(value: Any) -> str:
    return _clean(value).replace("|", "\\|").replace("\n", " ")


def _join_unique(values: Any) -> str:
    cleaned = [_clean(value) for value in values if _clean(value) != "Not available"]
    return "; ".join(dict.fromkeys(cleaned)) or "Not available"
