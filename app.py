from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING
import traceback

import pandas as pd
import streamlit as st

from src.llm.client import GroqAPIError, GroqChatClient, MissingGroqApiKey
from src.llm.config import LLMConfig
from src.llm.narrator import LLMReportNarration, RiskNarrationService
from src.output.markdown import render_markdown_report

if TYPE_CHECKING:
    from src.engine.nist_retriever import NistHybridRetriever, RetrievalResult


APP_TITLE = "TawasolPay Cyber Risk Assistant"


@dataclass(frozen=True)
class AnalysisState:
    scored: pd.DataFrame
    top: pd.DataFrame
    risk_retrievals: tuple[tuple[pd.Series, "RetrievalResult"], ...]
    unmatched_threat_count: int
    nist_control_count: int
    warnings: tuple[str, ...]
    include_external: bool


def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _apply_css()

    with st.sidebar:
        st.markdown("<h2 style='margin-top: 0; padding-top: 0;'>Pipeline Settings</h2>", unsafe_allow_html=True)
        top_n = st.slider("Top risks to evaluate", min_value=3, max_value=10, value=5, step=1)
        include_external = st.toggle("Refresh public references", value=True, help="Fetch live CISA KEV and FIRST EPSS")
        candidate_count = st.slider("NIST candidates to show", min_value=3, max_value=5, value=3)
        rate_limit_sleep = st.slider("Groq call spacing (sec)", min_value=0.0, max_value=5.0, value=2.0, step=0.5)
        st.divider()
        run_analysis = st.button("▶ Run Risk Analysis", type="primary", use_container_width=True)
        generate_ai = st.button("✨ Generate AI Report", use_container_width=True)
        reset_state = st.button("↻ Clear Results", use_container_width=True)

    st.title(APP_TITLE)
    st.caption("Evidence-first prioritization, NIST hybrid retrieval, and deterministic LLM narration for board-level reporting.")

    if reset_state:
        for key in ("analysis", "analysis_error", "narration", "markdown_report"):
            st.session_state.pop(key, None)

    if run_analysis:
        _safe_run_analysis(top_n=top_n, include_external=include_external)

    if generate_ai:
        if "analysis" not in st.session_state:
            st.warning("⚠️ Please run the risk analysis before generating the AI report.")
        else:
            _safe_run_ai_report(st.session_state.analysis, rate_limit_sleep)

    if "analysis_error" in st.session_state:
        _render_error("Risk analysis failed", st.session_state.analysis_error)

    tab_overview, tab_risks, tab_ai, tab_arch = st.tabs([
        "📊 Executive Dashboard", 
        "🚨 Ranked Risk Queue", 
        "🧠 AI Board Brief",
        "🏗️ System Architecture"
    ])

    with tab_overview:
        _render_intro(top_n=top_n, include_external=include_external)
        if "analysis" in st.session_state:
            st.divider()
            analysis: AnalysisState = st.session_state.analysis
            _render_status_bar(analysis)
            _render_asset_rollup(analysis.top)
            _render_pipeline_health(analysis)
        else:
            st.info("👋 No data has been processed yet. Configure settings in the sidebar and click **Run Risk Analysis** to start.")

    with tab_risks:
        if "analysis" in st.session_state:
            analysis = st.session_state.analysis
            _render_top_risk_table(analysis.top)
            _render_risk_entries(analysis, candidate_count)
        else:
            st.info("Run Risk Analysis to view the prioritized queue.")

    with tab_ai:
        if st.session_state.get("narration") is not None:
            _render_ai_report()
        else:
            st.info("Click **Generate AI Report** in the sidebar to create LLM-driven board narratives from the scored evidence.")

    with tab_arch:
        _render_architecture()


@st.cache_data(show_spinner=False, ttl=3600)
def _load_scored_data(include_external: bool) -> tuple[pd.DataFrame, int, int, tuple[str, ...]]:
    from src.engine.scoring import score_risks
    from src.ingest.pipeline import build_ingestion_bundle

    bundle = build_ingestion_bundle(include_external=include_external)
    scored = score_risks(bundle.enriched_vulnerabilities)
    return (
        scored,
        len(bundle.unmatched_threat_intelligence),
        len(bundle.external.nist_controls),
        tuple(bundle.warnings),
    )


@st.cache_resource(show_spinner=False)
def _get_retriever(include_external: bool) -> "NistHybridRetriever":
    from src.engine.nist_retriever import NistHybridRetriever
    from src.engine.retrieval_config import RetrievalConfig
    from src.ingest.pipeline import build_ingestion_bundle

    bundle = build_ingestion_bundle(include_external=include_external)
    return NistHybridRetriever(
        bundle.external.nist_controls,
        config=RetrievalConfig(embedding_model_local_files_only=False),
    )

def _run_analysis(top_n: int, include_external: bool) -> AnalysisState:
    from src.engine.scoring import top_risks

    scored, unmatched_count, nist_count, warnings = _load_scored_data(include_external)
    top = top_risks(scored, n=top_n)
    retriever = _get_retriever(include_external)
    risk_retrievals = tuple(retriever.retrieve_for_top_risks(top))
    return AnalysisState(
        scored=scored,
        top=top,
        risk_retrievals=risk_retrievals,
        unmatched_threat_count=unmatched_count,
        nist_control_count=nist_count,
        warnings=warnings,
        include_external=include_external,
    )


def _safe_run_analysis(top_n: int, include_external: bool) -> None:
    try:
        with st.status("⚙️ Running deterministic ingestion, scoring, and NIST retrieval...", expanded=True) as status:
            st.write("✓ Loading and validating structured CSV data pack.")
            st.write("✓ Refreshing and caching live CISA KEV, EPSS, and NIST references.")
            st.write("✓ Scoring vulnerabilities with the 9-factor composite formula.")
            st.write("✓ Building MiniLM + ChromaDB NIST index and retrieving controls.")
            st.session_state.analysis = _run_analysis(top_n, include_external)
            st.session_state.narration = None
            st.session_state.markdown_report = None
            st.session_state.pop("analysis_error", None)
            status.update(label="Risk analysis complete. See the 'Executive Dashboard' and 'Ranked Risk Queue' tabs.", state="complete")
    except Exception as exc:
        st.session_state.pop("analysis", None)
        st.session_state.narration = None
        st.session_state.markdown_report = None
        st.session_state.analysis_error = _exception_details(exc)


def _safe_run_ai_report(analysis: AnalysisState, rate_limit_sleep: float) -> None:
    try:
        client = GroqChatClient.from_env(LLMConfig(rate_limit_sleep_seconds=rate_limit_sleep))
        service = RiskNarrationService(client)
        with st.status("✨ Generating Groq narratives and markdown report...", expanded=True) as status:
            st.write(f"✓ Making {len(analysis.risk_retrievals) + 1} Groq calls: one per risk plus one executive brief.")
            st.write("✓ Enforcing evidence-first constraints. No raw data inference.")
            narration = service.narrate_report(list(analysis.risk_retrievals))
            report = render_markdown_report(
                narration=narration,
                risk_retrievals=analysis.risk_retrievals,
                generated_at=datetime.now(timezone.utc),
            )
            st.session_state.narration = narration
            st.session_state.markdown_report = report
            status.update(label="AI report generated successfully. See the 'AI Board Brief' tab.", state="complete")
    except MissingGroqApiKey:
        st.error("GROQ_API_KEY is not available. Add it to .env locally or Streamlit secrets in deployment.")
    except GroqAPIError as exc:
        st.error(f"Groq narration failed: {exc}")
        _render_error("Groq error details", _exception_details(exc))
    except Exception as exc:
        st.error("AI report generation failed.")
        _render_error("AI report debug details", _exception_details(exc))


def _render_intro(top_n: int, include_external: bool) -> None:
    st.markdown("<br>", unsafe_allow_html=True)
    cols = st.columns(5)
    stages = [
        ("1", "Ingest", "Validate CSV schemas, normalize CVEs, load MDR advisory."),
        ("2", "Enrich", "Join live CISA KEV, EPSS, assets, services, and threat intel."),
        ("3", "Score", "Apply 9-factor weighted risk formula with explainable contributions."),
        ("4", "Retrieve", "Embed NIST 800-53 with MiniLM + ChromaDB, then hybrid re-rank."),
        ("5", "Narrate", "Optional Groq Llama3 call: one per risk plus executive summary."),
    ]
    for col, (step, title, body) in zip(cols, stages):
        with col:
            st.markdown(
                f"""
                <div class='flow-card'>
                    <div class='flow-step-badge'>{step}</div>
                    <div class='flow-title'>{title}</div>
                    <div class='flow-desc'>{body}</div>
                </div>
                """, 
                unsafe_allow_html=True
            )
import base64

import base64

def _render_architecture() -> None:
    st.markdown("### 🏗️ System Architecture & Information Flow")
    st.info("The system is designed as a deterministic, evidence-first pipeline. It strictly separates data ingestion, business-logic risk scoring, information retrieval, and LLM narration to ensure AI outputs are grounded in verifiable data.")
    
    mermaid_code = """graph TD
    subgraph Phase 1: Ingestion & Enrichment
        A["Load CSVs:<br/>Assets, Vulns, Threat Intel, Services"] --> D("Normalize &<br/>Validate")
        B["Fetch CISA KEV JSON"] --> D
        C["Fetch FIRST EPSS CSV"] --> D
        D --> E{"Merge &<br/>Enrich"}
        E --> F["Enriched Vulnerabilities<br/>Dataframe"]
        G["Load NIST 800-53<br/>Catalog"] --> H["Parse into<br/>Text Chunks"]
    end

    subgraph Phase 2: Risk Scoring
        F --> I["Calculate 9-Factor<br/>Contributions"]
        I --> J["Compute Composite<br/>Risk Score"]
        J --> K["Rank & Assign<br/>Severity Thresholds"]
        K --> L["Top N Ranked<br/>Risks"]
    end

    subgraph Phase 3: Hybrid Retrieval
        H --> M["Embed with<br/>all-MiniLM-L6-v2"]
        M --> N[("ChromaDB<br/>Vector Store")]
        L --> O["Generate Risk-Specific<br/>Query"]
        O --> N
        N --> P["Retrieve Semantic<br/>Candidates"]
        P --> Q["Re-rank via BM25<br/>+ Deterministic Priors"]
        Q --> R["Selected NIST<br/>Control Evidence"]
    end

    subgraph Phase 4: LLM Narration
        L --> S["Format Structured<br/>Evidence"]
        R --> S
        S --> T["Prompt Llama-3.3<br/>70b-versatile"]
        T --> U["Parsed Narrative:<br/>Why, Threat, NIST"]
        U --> V["Executive Brief<br/>Generation"]
    end

    subgraph Phase 5 & 6: Output & UX
        V --> W["Generate Markdown<br/>Report"]
        L --> X["Streamlit Dashboard<br/>Rendering"]
        U --> X
        W --> X
    end"""
    
    graphbytes = mermaid_code.encode("utf-8")
    base64_bytes = base64.b64encode(graphbytes)
    base64_string = base64_bytes.decode("utf-8")
    url = f"https://mermaid.ink/svg/{base64_string}?theme=neutral&bgColor=ffffff"
    
    # st.image renders the SVG perfectly locally without clipping
    st.image(url, use_container_width=True)


def _render_status_bar(analysis: AnalysisState) -> None:
    critical_count = int((analysis.scored["risk_severity"] == "Critical").sum())
    internet_top = int(analysis.top["feature_internet_exposed"].sum())
    ransomware_top = int(analysis.top["feature_ransomware"].sum())
    kev_top = int(analysis.top["cveID"].fillna("").astype(str).str.len().gt(0).sum())

    st.markdown("### Executive Overview")
    cols = st.columns(6)
    cols[0].metric("Total Assets", "60")
    cols[1].metric("Open Vulns", f"{len(analysis.scored)}")
    cols[2].metric("Critical Risks", f"{critical_count}")
    cols[3].metric("Top: Internet Exposed", f"{internet_top} / {len(analysis.top)}")
    cols[4].metric("Top: Ransomware", f"{ransomware_top} / {len(analysis.top)}")
    cols[5].metric("Top: CISA KEV", f"{kev_top} / {len(analysis.top)}")
    st.markdown("<br>", unsafe_allow_html=True)


def _render_pipeline_health(analysis: AnalysisState) -> None:
    with st.expander("📊 Pipeline Health & Data Lineage", expanded=False):
        cols = st.columns(4)
        cols[0].metric("NIST Controls Indexed", f"{analysis.nist_control_count}")
        cols[1].metric("Threat Intel Noise", f"{analysis.unmatched_threat_count}")
        cols[2].metric("Scoring Rows", f"{len(analysis.scored)}")
        cols[3].metric("Groq Calls Required", f"{len(analysis.risk_retrievals) + 1}")
        st.write(f"**External Reference Mode:** {'Refresh/Cache' if analysis.include_external else 'Offline'}")
        if analysis.warnings:
            st.warning("\n".join(f"- {warning}" for warning in analysis.warnings))
        st.caption("Structured records are joined in pandas. NIST controls are embedded with MiniLM and queried through ChromaDB, then re-ranked with BM25 and deterministic control priors.")


def _render_top_risk_table(top: pd.DataFrame) -> None:
    st.markdown("### Risk Queue")
    view = top[
        [
            "risk_rank",
            "composite_risk_score",
            "risk_severity",
            "asset_name",
            "cve",
            "vulnerability_name",
            "business_service",
            "campaign_names",
            "score_drivers",
        ]
    ].rename(
        columns={
            "risk_rank": "Rank",
            "composite_risk_score": "Score",
            "risk_severity": "Severity",
            "asset_name": "Asset",
            "cve": "CVE",
            "vulnerability_name": "Vulnerability",
            "business_service": "Service",
            "campaign_names": "Threat Campaign",
            "score_drivers": "Drivers",
        }
    )
    st.dataframe(
        view,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Score": st.column_config.ProgressColumn("Score", min_value=0.0, max_value=1.0, format="%.3f"),
            "Rank": st.column_config.NumberColumn("Rank", format="%d"),
        },
    )


def _render_asset_rollup(top: pd.DataFrame) -> None:
    st.markdown("### Asset-Level Exposure Rollup")
    grouped = (
        top.groupby("asset_name", as_index=False)
        .agg(
            max_score=("composite_risk_score", "max"),
            risk_count=("vuln_id", "count"),
            cves=("cve", lambda values: "; ".join(dict.fromkeys(str(value) for value in values))),
            services=("business_service", lambda values: "; ".join(dict.fromkeys(str(value) for value in values))),
        )
        .sort_values(["max_score", "risk_count"], ascending=[False, False])
    )
    st.dataframe(
        grouped.rename(
            columns={
                "asset_name": "Asset",
                "max_score": "Max Score",
                "risk_count": "Risk Count",
                "cves": "CVEs / Findings",
                "services": "Services",
            }
        ),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Max Score": st.column_config.ProgressColumn("Max Score", min_value=0.0, max_value=1.0, format="%.3f"),
            "Risk Count": st.column_config.NumberColumn("Risk Count", format="%d"),
        },
    )


def _render_risk_entries(analysis: AnalysisState, candidate_count: int) -> None:
    st.markdown("### Risk Evidence & NIST Guidance")
    for row, retrieval in analysis.risk_retrievals:
        best = retrieval.best
        title = f"Risk #{int(row['risk_rank'])} · {row['risk_severity']} · {row['asset_name']} · {row['cve']}"
        with st.expander(title, expanded=int(row["risk_rank"]) == 1):
            cols = st.columns([1.1, 1.1, 1.4])
            cols[0].metric("Composite Score", f"{float(row['composite_risk_score']):.3f}")
            cols[0].metric("CVSS Base", f"{float(row['cvss']):.1f}")
            cols[0].metric("EPSS Probability", f"{float(row.get('epss', 0.0)) * 100:.2f}%")
            cols[1].write(f"**Asset:** {row['asset_name']} ({row['asset_type']})")
            cols[1].write(f"**Service:** {row['business_service']}")
            cols[1].write(f"**Exposure:** {'🌍 Internet' if row['feature_internet_exposed'] else '🔒 Internal'}")
            cols[1].write(f"**EDR:** {'✅ Installed' if row['edr_installed_bool'] else '❌ Missing'}")
            cols[1].write(f"**Severity:** {_severity_badge(row['risk_severity'])}", unsafe_allow_html=True)
            cols[2].write(f"**Threat Intel:** {_threat_line(row)}")
            cols[2].write(f"**CISA KEV:** {_kev_line(row)}")
            cols[2].write(f"**Ransomware Source:** {_ransomware_signal_source(row)}")
            cols[2].write(f"**Days Open:** {row['days_open']}")
            cols[2].write(f"**Drivers:** {row['score_drivers']}")

            st.markdown(f"#### Primary NIST Control: `{best.control_id}` {best.name}")
            st.caption(
                f"Match Confidence: Hybrid {best.hybrid_score:.3f} | Semantic {best.semantic_score:.3f} | "
                f"BM25 {best.bm25_score:.3f} | Prior {best.control_prior_score:.3f}"
            )
            st.info(_sentence_excerpt(best.discussion, 1400))

            st.markdown("**Scoring Composition**")
            component_view = _score_component_frame(row)
            st.dataframe(component_view, use_container_width=True, hide_index=True)

            st.markdown("**Alternative NIST Candidates**")
            candidate_rows = [
                {
                    "Rank": candidate.rank,
                    "Control": f"{candidate.control_id}: {candidate.name}",
                    "Hybrid": round(candidate.hybrid_score, 3),
                    "Semantic": round(candidate.semantic_score, 3),
                    "BM25": round(candidate.bm25_score, 3),
                    "Prior": round(candidate.control_prior_score, 3),
                }
                for candidate in retrieval.candidates[:candidate_count]
            ]
            st.dataframe(pd.DataFrame(candidate_rows), use_container_width=True, hide_index=True)


def _render_ai_report() -> None:
    narration: LLMReportNarration = st.session_state.narration
    report = st.session_state.markdown_report

    st.markdown("### AI-Generated Board Brief")
    st.info(narration.executive_brief)
    
    st.download_button(
        "⬇️ Download Executive Report (Markdown)",
        data=report.content,
        file_name=f"tawasolpay_top_risks_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown",
        use_container_width=True,
        type="primary"
    )

    st.markdown("---")
    
    for risk in narration.risks:
        with st.expander(f"📖 Narrative: Risk #{risk.risk_rank} · {risk.vuln_id}", expanded=False):
            st.markdown("##### Why It Matters")
            st.write(risk.risk_explanation)
            st.markdown("##### Threat Summary")
            st.write(risk.threat_summary)
            st.markdown("##### NIST Application")
            st.write(risk.nist_application)


def _render_error(title: str, details: dict[str, str]) -> None:
    st.error(f"**{title}**: {details['type']}: {details['message']}")
    with st.expander("View Stack Trace", expanded=True):
        st.code(details["traceback"], language="text")


def _exception_details(exc: Exception) -> dict[str, str]:
    return {
        "type": exc.__class__.__name__,
        "message": str(exc),
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    }


def _score_component_frame(row: pd.Series) -> pd.DataFrame:
    rows = []
    for label, feature_col, contribution_col in [
        ("CVSS severity", "feature_cvss", "contribution_cvss"),
        ("Active exploitation", "feature_active_exploitation", "contribution_active_exploitation"),
        ("Ransomware", "feature_ransomware", "contribution_ransomware"),
        ("EPSS", "feature_epss", "contribution_epss"),
        ("Internet exposure", "feature_internet_exposed", "contribution_internet_exposed"),
        ("Business impact", "feature_business_impact", "contribution_business_impact"),
        ("Threat match", "feature_threat_intel_match", "contribution_threat_intel_match"),
        ("Days open", "feature_days_open", "contribution_days_open"),
        ("Missing EDR", "feature_missing_edr", "contribution_missing_edr"),
    ]:
        rows.append(
            {
                "Component": label,
                "Feature": round(float(row.get(feature_col, 0.0)), 3),
                "Contribution": round(float(row.get(contribution_col, 0.0)), 3),
            }
        )
    return pd.DataFrame(rows)


def _threat_line(row: pd.Series) -> str:
    if not _truthy(row.get("threat_intel_match_bool")):
        return "No current-environment match"
    ransomware = "local ransomware Yes" if _truthy(row.get("threat_ransomware_bool")) else "local ransomware No"
    return f"{row.get('campaign_names', 'Matched campaign')} ({row.get('threat_actors', 'Unknown actor')}; {ransomware})"


def _kev_line(row: pd.Series) -> str:
    status = "Confirmed" if _present(row.get("cveID")) else "Not found"
    ransomware = "Known" if _truthy(row.get("kev_known_ransomware_bool")) else "Unknown"
    return f"{status}; ransomware {ransomware}"


def _ransomware_signal_source(row: pd.Series) -> str:
    kev = _truthy(row.get("kev_known_ransomware_bool"))
    local = _truthy(row.get("threat_ransomware_bool"))
    if kev and local:
        return "CISA KEV + local threat intel"
    if kev:
        return "CISA KEV only; local campaign is not ransomware"
    if local:
        return "Local threat intel only"
    return "None"


def _severity_badge(severity: str) -> str:
    color_map = {
        "Critical": "#dc2626", # red-600
        "High": "#ea580c", # orange-600
        "Medium": "#eab308", # yellow-500
        "Low": "#16a34a" # green-600
    }
    bg = color_map.get(str(severity).strip(), "#64748b")
    return f"<span style='background-color: {bg}; color: white; padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 0.85em;'>{severity}</span>"


def _truthy(value) -> bool:
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


def _present(value) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return bool(text) and text.lower() != "nan"


def _truncate(text: str, max_chars: int) -> str:
    text = str(text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _sentence_excerpt(text: str, max_chars: int) -> str:
    text = " ".join(str(text or "").strip().split())
    if len(text) <= max_chars:
        return text
    boundary = max(text.rfind(". ", 0, max_chars), text.rfind("; ", 0, max_chars))
    if boundary >= max_chars * 0.55:
        return text[: boundary + 1].rstrip()
    return text[:max_chars].rsplit(" ", 1)[0].rstrip() + "."


def _apply_css() -> None:
    st.markdown(
        """
        <style>
        /* General Layout */
        .block-container { 
            padding-top: 2rem; 
            padding-bottom: 3rem; 
            max-width: 1400px;
        }
        
        /* Hide specific header elements but keep the sidebar toggle */
        #MainMenu {visibility: hidden;}
        .stDeployButton {display: none;}
        header {background-color: transparent !important;}
        footer { visibility: hidden; }

        /* Sleek Metrics Cards */
        div[data-testid="stMetric"] {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 0.75rem;
            padding: 1rem 1.25rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        div[data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 6px rgba(0,0,0,0.06);
        }

        /* Expander Enhancements */
        div[data-testid="stExpander"] {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 0.5rem;
            box-shadow: 0 1px 2px rgba(0,0,0,0.02);
            overflow: hidden;
            margin-bottom: 1rem;
        }
        div[data-testid="stExpander"] summary {
            font-weight: 600;
            color: #0f172a;
            background-color: #f8fafc;
        }

        /* Modern Flow Cards */
        .flow-card {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-top: 4px solid #0d9488;
            border-radius: 0.75rem;
            padding: 1.25rem;
            height: 100%;
            box-shadow: 0 2px 4px rgba(0,0,0,0.02);
            transition: all 0.2s ease-in-out;
        }
        .flow-card:hover {
            box-shadow: 0 8px 16px rgba(0,0,0,0.06);
            transform: translateY(-3px);
            border-color: #cbd5e1;
        }
        .flow-step-badge {
            display: flex;
            align-items: center;
            justify-content: center;
            width: 32px;
            height: 32px;
            border-radius: 8px;
            background: rgba(13, 148, 136, 0.1);
            color: #0d9488;
            font-weight: 700;
            margin-bottom: 0.75rem;
            font-size: 0.95rem;
        }
        .flow-title {
            font-weight: 600;
            color: #0f172a;
            margin-bottom: 0.4rem;
            font-size: 1.05rem;
        }
        .flow-desc {
            color: #475569;
            font-size: 0.85rem;
            line-height: 1.5;
        }

        /* Button Refinements */
        .stButton > button, .stDownloadButton > button {
            border-radius: 0.5rem;
            font-weight: 600;
            transition: all 0.2s ease;
        }
        .stButton > button:hover {
            transform: scale(1.01);
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2rem;
        }
        .stTabs [data-baseweb="tab"] {
            height: 3.5rem;
            font-size: 1.05rem;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
