# TawasolPay AI Cyber Risk Assistant

An evidence-first cybersecurity risk assessment system built for TawasolPay, a fintech company processing digital payments across the Gulf region.

The application ingests the full assignment data pack, enriches vulnerabilities with CISA KEV and EPSS signals, scores every open finding with a deterministic 9-factor formula, retrieves NIST SP 800-53 Rev. 5 controls through hybrid RAG, and generates board-level risk briefings with constrained Groq LLM narration.

**Live demo:** [https://hivepro-assignment.streamlit.app/](https://hivepro-assignment.streamlit.app/)

---

## What It Does

Given TawasolPay's data pack - 60 assets, 114 open vulnerabilities, 40 threat-intel records, 20 business services, remediation guidance, and an MDR advisory - the system:

1. **Ranks the top cyber risks** with a deterministic composite score that deliberately caps CVSS at 20% so business exposure, active exploitation, EPSS probability, ransomware association, and compensating controls drive the final order.

2. **Retrieves NIST SP 800-53 guidance** from the actual NIST catalog using MiniLM embeddings, ChromaDB vector search, BM25 re-ranking, and deterministic control priors. The LLM does not invent NIST guidance from memory.

3. **Produces a CISO-ready report** with a top-risk summary, asset-level rollup, detailed evidence for each risk, NIST retrieval audit scores, score-driver tables, and optional AI-generated board narration.

---

## Architecture

```text
Phase 1 -> Phase 2 -> Phase 3 -> Phase 4 -> Phase 5 -> Phase 6
Ingest     Score     Retrieve   Narrate    Report    UI
```

| Phase | Purpose | Key Technique |
|-------|---------|---------------|
| 1 - Ingest | Load CSVs, threat report, and public references | pandas joins, CVE normalization, 24 h filesystem cache |
| 2 - Score | Rank every vulnerability | 9-factor weighted formula, CVSS capped at 20% |
| 3 - Retrieve | Map risks to NIST 800-53 controls | MiniLM + ChromaDB + BM25 + deterministic priors |
| 4 - Narrate | Generate evidence-grounded prose | Groq Llama 3.3 70B, one call per risk plus executive brief |
| 5 - Report | Export markdown report | Structured report with retrieval audit trail |
| 6 - UI | Public dashboard | Streamlit run-on-demand workflow |

---

## Quick Start

### Prerequisites

- Python 3.10+
- A Groq API key for AI narration and markdown report generation
- The assignment data files committed under `data/raw/`

Phases 1-3 can run without Groq. The AI report button and Phase 4/5 CLI commands require `GROQ_API_KEY`.

### Install

```bash
git clone https://github.com/LakshyaJ1/HivePro_Assignment.git
pip install -r requirements.txt
```

### Provision the embedding model locally

Local CLI retrieval defaults to offline model loading. Provision the MiniLM model once:

```powershell
.\scripts\provision_phase3_model.ps1
```

or:

```bash
python -m src.engine --top 1 --with-nist --allow-model-download --nist-candidates 1
```


### Add Groq API key locally

Create a `.env` file in the project root:

```bash
GROQ_API_KEY=add_your_key_here
```


### Run the app

```bash
python -m streamlit run app.py --server.port 8501
```

Open [http://localhost:8501](http://localhost:8501), then click **Run Risk Analysis**. After the ranked risks appear, click **Generate AI Report** to create and download the markdown report.

---

## Supporting Question 1 - The Data Split

I treated the system as two different problem types: structured security analytics and unstructured control retrieval.

The structured datasets - assets, vulnerabilities, business services, threat intelligence, CISA KEV, and EPSS are queried with deterministic pandas joins and scoring logic. These sources contain exact fields like CVE IDs, internet exposure, exploit availability, EDR coverage, business criticality, compliance scope, and RTO; embedding those fields would make the system less reliable than direct filtering and joins.

The embedded side is the NIST SP 800-53 Rev. 5 control catalog. NIST controls are long-form security prose where the relevant guidance depends on semantic context: CitrixBleed session-token theft should surface account/authentication controls like `AC-2` or `IA-2`, while VPN firmware RCE should surface flaw remediation and incident context such as `SI-2` and `IR-4`. The MDR advisory is loaded as narrative evidence for the application context, but the production vector retrieval path is focused on NIST controls so remediation guidance comes from the authoritative catalog.

---

## Supporting Question 2 - Where It Goes Wrong

### 1. CVE normalization gaps can create false negatives

CVE identifiers from different sources are not always formatted consistently. Trailing whitespace, mixed casing, malformed IDs, or synthetic/local identifiers can cause joins against CISA KEV or EPSS to fail. If that happens, a vulnerability may incorrectly appear as not present in KEV even when the real CVE is known exploited.

To reduce this risk, the ingestion layer normalizes identifiers before joins and exposes CISA KEV status directly in every report entry. During development this exact class of issue showed up with CitrixBleed, so the final scoring and report logic treat a non-empty `cveID` as the deterministic KEV presence signal and regression tests cover the behavior.

### 2. NIST retrieval can over-generalize toward one control

Early versions mapped almost every high-risk vulnerability to `SI-2` because flaw remediation is semantically close to most vulnerability scenarios.That was technically correct in many cases, but it made the remediation guidance repetitive and less useful operationally: CitrixBleed needed account/session control context, TeamCity needed build-chain configuration context, and Fortinet authentication bypass needed identity/authentication guidance.

To improve this, I added richer context-aware retrieval queries along with deterministic control priors for controls like IR-4, AC-2, IA-2, SA-22, and CM-6. The final retrieval score combines semantic similarity, BM25 keyword ranking, and prior boosting so the system does not collapse into a single-control monoculture.

### 3. Threat attribution can become misleading if signals are merged too aggressively

A CVE can have a CISA KEV ransomware signal while the matched local threat campaign is not ransomware-driven. This happened with TeamCity: the local SilentForge campaign is build-secret theft/espionage focused, while CISA KEV carries a ransomware indicator for the CVE. If those signals are merged into one label, the report can falsely imply that SilentForge is a ransomware campaign.

The final evidence model separates local campaign ransomware from CISA KEV ransomware. Reports include a `Ransomware Signal Source` line, and the LLM prompt tells the model not to call a named campaign ransomware-linked unless local threat intelligence explicitly says so.

---

## Supporting Question 3 - One Thing I Would Change

If I had another day, the most important improvement would be dependency-based risk propagation across business services and assets. The current system scores each vulnerability against the directly affected asset and mapped business service, which is explainable and reliable, but it does not fully model second-order operational impact. In a real fintech environment, authentication services, VPN gateways, CI/CD systems, API gateways, and load balancers can indirectly affect payment processing even when they are not directly tagged as the payment service. I would build a dependency graph from the `depends_on` relationships in the business-service data, propagate criticality and RTO impact across upstream and downstream services, and show both the direct score and dependency-amplified score. That would make the ranking much closer to how a real CISO thinks about operational and business impact during an incident.

---
## Phase 1: Data Ingestion

- Loads and validates all assignment CSVs plus the MDR advisory.
- Normalizes CVE identifiers before joins so casing and whitespace do not silently break enrichment.
- Fetches CISA KEV JSON, FIRST EPSS, and the NIST SP 800-53 Rev. 5 Excel catalog into `data/external/` using a 24 h filesystem cache.
- Enriches vulnerabilities with asset, business service, threat-intel, CISA KEV, EPSS, and control-context fields.
- Preserves unmatched threat-intel records as a first-class output so irrelevant external noise is visible instead of silently discarded.

The Streamlit scored-data cache uses `@st.cache_data(ttl=3600)` to avoid recomputing joins on every rerun. Public reference files still use the ingestion-layer 24 h cache.

---

## Phase 2: Risk Scoring

Every open vulnerability receives a Composite Risk Score built from nine weighted factors:

| Factor | Weight | Why It Matters |
|--------|--------|----------------|
| Ransomware association | 20% | Indicates high operational urgency |
| CVSS severity | 20% | Baseline technical severity, intentionally capped |
| Active exploitation | 15% | CISA KEV, weaponized exploit maturity, or exploit availability |
| EPSS probability | 15% | Empirical likelihood of exploitation in the wild |
| Internet exposure | 10% | External attack surface amplifier |
| Business impact | 10% | Criticality, revenue impact, compliance scope, and RTO |
| Threat-intel match | 5% | Campaign evidence relevant to the environment |
| Days open | 3% | Age of unresolved exposure |
| Missing EDR | 2% | Absence of a compensating detection/control layer |

Every ranked row includes `feature_*`, `contribution_*`, `composite_risk_score`, `risk_severity`, `risk_rank`, `score_drivers`, and a deterministic explanation. This keeps the LLM layer grounded in precomputed evidence.

---

## Phase 3: Hybrid NIST Retrieval

The system parses the NIST SP 800-53 Rev. 5 catalog and embeds the control prose with `sentence-transformers/all-MiniLM-L6-v2`. Embeddings are stored in ChromaDB through `PersistentClient`.

- On Streamlit/Linux, Chroma persists under `/tmp/chroma_tawasolpay`.
- On local Windows development, the same path is mapped to `.chroma/chroma_tawasolpay`.
- The collection is created with `get_or_create_collection()`.
- If the collection is empty, partially initialized, has the wrong control count, or fails during query, it is rebuilt automatically.

Retrieval combines:

- **Semantic search:** ChromaDB returns NIST controls close to the risk query.
- **BM25:** Keyword re-ranking over candidate controls.
- **Deterministic priors:** Control boosts for known patterns such as `AC-2` for session/account issues, `IA-2` and `IA-5` for authentication and credentials, `CM-6` for build-chain/configuration risks, `SI-2` for flaw remediation, `RA-5` for vulnerability monitoring, `IR-4` for active incident context, and `SA-22` for unsupported components.

Final score:

```text
hybrid = semantic * 0.45 + bm25 * 0.25 + prior * 0.30
```

Each selected control and candidate includes semantic, BM25, prior, and hybrid scores so the NIST mapping is auditable.

---

## Phase 4: LLM Narration

The LLM layer uses Groq's OpenAI-compatible API with `llama-3.3-70b-versatile`. The model receives only structured risk evidence and retrieved NIST control evidence. It is explicitly instructed not to invent CVE details, scores, campaign names, NIST controls, or remediation steps.

To protect Groq free-tier limits, the system makes one consolidated call per risk. That single response returns:

- `WHY IT MATTERS`
- `THREAT SUMMARY`
- `NIST APPLICATION`

One final call produces the executive brief. A top-5 report therefore uses 6 LLM calls.

---

## Phase 5: Markdown Report

The report includes:

- Executive brief
- Top risk summary
- Asset-level exposure rollup
- Detailed risk entries
- Matched threat intelligence
- CISA KEV and ransomware signal source
- Retrieved NIST control and evidence excerpt
- Retrieval audit scores
- Score-driver contribution table
- LLM usage appendix
- Full NIST candidate appendix

The final hosted top-5 report was validated with 5 risk entries, 6 LLM calls, nonzero semantic scores, confirmed CISA KEV matches, no placeholders, and no mid-sentence evidence truncation.

---

## Phase 6: Streamlit Application

The Streamlit app is the public interface. Nothing heavy runs on page load. Users choose the top-N count, click **Run Risk Analysis**, and only then does the app load data, score risks, build the MiniLM + ChromaDB retriever, and retrieve NIST controls.

AI narration is a separate button so Groq calls happen only when the user requests the report.

---


## CLI Usage

```bash
# Phase 1: Ingestion smoke check
python -m src.ingest --offline

# Phase 2: Risk scoring
python -m src.engine --top 5 --details

# Phase 3: Scoring + NIST retrieval
python -m src.engine --top 5 --with-nist --nist-candidates 3

# Phase 4: LLM narration
python -m src.llm --top 5 --rate-limit-sleep 2

# Phase 5: Markdown report
python -m src.output --top 5 --output-dir reports --rate-limit-sleep 2

# Phase 6: Streamlit app
python -m streamlit run app.py --server.port 8501
```

PowerShell convenience scripts:

```powershell
.\scripts\run_phase1.ps1
.\scripts\run_phase2.ps1
.\scripts\run_phase3.ps1
.\scripts\run_phase4.ps1
.\scripts\run_phase5.ps1
.\scripts\run_app.ps1
.\scripts\run_tests.ps1
```

---

## Tests

```bash
python -m pytest -q
```

The current suite contains 34 tests across ingestion, scoring, retrieval, LLM prompt handling, report rendering, and Streamlit AppTest rendering. It includes checks for CVSS-not-dominant ranking, NIST control diversity, unsupported component retrieval, CitrixBleed access-control retrieval, Fortinet authentication retrieval, LLM call reduction, and markdown report structure.

---

