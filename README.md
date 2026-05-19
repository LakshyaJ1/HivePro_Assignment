# TawasolPay AI Cyber Risk Assistant

An evidence-first, automated cybersecurity risk assessment system that ingests multi-source security data, scores vulnerabilities with a deterministic 9-factor composite formula, retrieves NIST SP 800-53 controls via hybrid RAG, and generates CISO-level risk briefings through constrained LLM narration.

## Architecture

```
Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5 → Phase 6
Ingest     Score     Retrieve   Narrate    Report    UI
```

| Phase | Purpose | Key Technique |
|-------|---------|---------------|
| 1 - Ingest | Load CSVs, fetch KEV/EPSS/NIST, normalize, join | pandas joins, CVE normalization, 24h cache TTL |
| 2 - Score | 9-factor composite risk score | Weighted formula, CVSS capped at 20% |
| 3 - Retrieve | NIST SP 800-53 control mapping | MiniLM embeddings + ChromaDB + BM25 + deterministic priors |
| 4 - Narrate | LLM-generated risk explanations | Groq Llama 3.3 70B, evidence-first prompts |
| 5 - Report | Downloadable markdown report | Structured output with retrieval audit trail |
| 6 - UI | Interactive risk dashboard | Streamlit with run-on-demand pipeline |

## Quick Start

```powershell
# Install dependencies
pip install -r requirements.txt

# Provision the embedding model (once)
.\scripts\provision_phase3_model.ps1

# Run the Streamlit app
python -m streamlit run app.py --server.port 8501
```

## Phase 1: Data Ingestion

- Loads and validates all provided CSVs and the MDR threat report against expected schemas.
- Normalizes CVE/control identifiers with `str.strip().upper()`, booleans, criticality labels, dates, and numeric fields before joins.
- Enriches vulnerabilities with asset, business service, threat intelligence, CISA KEV, and EPSS fields via deterministic pandas joins.
- Preserves unmatched threat intelligence as a first-class output (15 of 40 records) instead of dropping noise silently.
- Downloads public CISA KEV, FIRST EPSS, and NIST SP 800-53 catalog data into `data/external/` at runtime with a 24-hour cache (`@st.cache_data(ttl=86400)`).
- Produces RAG-ready narrative documents for the MDR advisory and parsed NIST controls.

## Phase 2: Risk Scoring

The scoring engine ranks every open vulnerability with a deterministic Composite Risk Score (CRS). CVSS is intentionally capped at 20% of the final score; the rest comes from active exploitation (15%), ransomware association (20%), EPSS probability (15%), internet exposure (10%), business impact (10%), threat-intel matches (5%), finding age (3%), and missing EDR (2%).

This means a CVSS 10 on a dev server with no internet exposure and no ransomware campaign scores roughly 0.40, while a CVSS 8 on an internet-exposed payment gateway with a confirmed ransomware campaign and high EPSS scores 0.75+. This behavior matches what production security teams actually need.

Every ranked row includes normalized feature columns, weighted contribution columns, `composite_risk_score`, `risk_severity`, `risk_rank`, `score_drivers`, and a deterministic explanation. This keeps the later LLM layer evidence-first.

## Phase 3: Hybrid NIST Retrieval

The NIST retrieval engine uses `sentence-transformers/all-MiniLM-L6-v2` to embed every parsed NIST SP 800-53 Rev. 5 control into an in-memory ChromaDB collection. For each scored risk, it builds a rich remediation query from the asset, vulnerability, exposure, ransomware status, campaign, and business context.

Retrieval is hybrid: Chroma returns semantic candidates, BM25 re-ranks those candidates by keyword relevance, and deterministic control priors boost known mappings such as `SI-2` for flaw remediation, `RA-5` for vulnerability monitoring, `IR-4` for incident handling, `AC-2` for account management, and `SA-22` for unsupported components. The selected control includes semantic, BM25, prior, and final hybrid scores so the choice is auditable.

## Phase 4: LLM Narration

The LLM layer uses Groq's OpenAI-compatible Chat Completions API with `llama-3.3-70b-versatile`. It does not ask the model to infer risk from raw data. Instead, it passes pre-computed Phase 2 score evidence and Phase 3 NIST retrieval evidence into constrained prompts.

To protect the free tier and reduce failure surface, Phase 4 uses one consolidated LLM call per risk. That single response returns `WHY IT MATTERS`, `THREAT SUMMARY`, and `NIST APPLICATION`, then one final call produces the executive brief. For five risks this is six Groq calls total.

`GROQ_API_KEY` is loaded from the environment or `.env`. The key is never committed because `.env` is gitignored. If the key is missing, Phase 4 fails before doing expensive ingestion or embedding work.

## Phase 5: Markdown Report Output

The output layer generates a human-readable markdown report for the top risks. Each risk entry follows the assignment structure: asset, vulnerability, risk score, business impact, threat intelligence, CISA KEV/ransomware status, days open, why it ranks here, retrieved NIST control, NIST application note, threat context, score drivers, and retrieval audit evidence. Reports also include appendices for unmatched threat intelligence handling, LLM usage statistics, and retrieval candidates.

## Phase 6: Streamlit Application

The Streamlit app is the primary working interface. It runs the same production pipeline used by the CLI: ingestion, scoring, NIST retrieval, optional Groq narration, and markdown report download. Nothing is processed on page load — the pipeline starts only when you click "Run Risk Analysis". This prevents wasted compute and API calls during development.

---

## Data Split Justification

**Structured records (pandas):** All CSV data — assets, vulnerabilities, threat intelligence, business services, remediation guidance, CISA KEV JSON, and EPSS scores — is loaded into pandas DataFrames. These contain precise, filterable fields (CVE IDs, boolean flags like `internet_exposed`, numeric scores like CVSS, business criticality ratings) where SQL-style joins and filters are both correct and efficient. You don't embed `internet_exposed: true` — you filter it with `df[df['internet_exposed'] == True]`.

**Narrative documents (RAG):** NIST SP 800-53 Rev. 5 control descriptions and the MDR threat report are embedded for semantic retrieval. These are unstructured prose documents best retrieved by semantic similarity — you cannot know in advance which exact control or which threat narrative phrase will be most relevant to a given risk. For example, "patch management for externally exploitable OS vulnerability on internet-facing asset" should find SI-2 without hard-coding that mapping.

## Known Failure Modes

1. **CVE mismatch on KEV join:** A CVE in `vulnerabilities.csv` uses a slightly different format (e.g., `CVE-2021-44228 ` with trailing whitespace) and fails to join against CISA KEV JSON. The vulnerability will not be flagged as ransomware-associated even if it is. **Mitigation:** All CVE IDs are normalized with `str.strip().upper()` before any join, and a join-coverage validation logs unmatched CVEs.

2. **NIST control retrieval hallucination boundary:** The retrieved control may be topically correct (e.g., SI-2 for patching) but the LLM narration applies it too generically, missing the specific provision that applies to this CVE type. **Mitigation:** The full control prose (not just the name) is included in the prompt, and the LLM is instructed to reference the specific sub-clause it is applying.

3. **Stale CISA KEV cache:** The KEV JSON is downloaded once at startup. If the system runs continuously, a CVE's ransomware flag may have been updated by CISA without triggering a re-download. **Mitigation:** A 24-hour TTL on the cache forces daily refresh via `@st.cache_data(ttl=86400)`.

## One Thing to Improve

Implement **dependency-graph risk propagation** across assets. Currently, each asset is scored independently. But `business_services.csv` includes service dependencies (`depends_on` column) — if a payment gateway depends on an auth service, a vulnerability on the auth service should inherit elevated business criticality from the gateway. Building this adjacency propagation would make the top-N ranking materially more accurate by accounting for transitive business impact.

---

## CLI Usage

```powershell
# Phase 1: Ingestion smoke check (offline)
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

Convenience scripts are also available:

```powershell
.\scripts\run_phase1.ps1
.\scripts\run_phase2.ps1
.\scripts\run_phase3.ps1
.\scripts\run_phase4.ps1
.\scripts\run_phase5.ps1
.\scripts\run_app.ps1
.\scripts\run_tests.ps1
```

## Tests

```powershell
python -m pytest -q
```

The test suite covers all six phases with both unit and integration tests, including synthetic risk fixtures, mock LLM clients, retrieval quality assertions, and Streamlit AppTest rendering.

## Tech Stack

| Component | Choice | Reason |
|-----------|--------|--------|
| LLM | Groq (Llama 3.3 70B) | Fastest free tier, structured output, 1000 RPD |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | Local, CPU, no API cost |
| Vector Store | ChromaDB (in-memory) | Zero config, rebuilds in <10s from NIST catalog |
| Structured Data | pandas | Standard, fast joins across 5+ data sources |
| CISA KEV | requests + JSON | Live fetch at startup, 24h TTL |
| EPSS | requests + FIRST.org API | Free, batch CVE query |
| NIST 800-53 | openpyxl → ChromaDB | Direct from CSRC, parsed at startup |
| UI | Streamlit | Fastest deployment path to public URL |
| Hosting | Streamlit Community Cloud | Free, GitHub-connected, 5 min deploy |
