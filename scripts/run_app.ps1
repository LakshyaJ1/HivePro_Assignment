Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

python -m streamlit run app.py --server.port 8501 --server.headless true

