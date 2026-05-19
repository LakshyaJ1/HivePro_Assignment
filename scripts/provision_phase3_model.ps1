Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

python -m src.engine --top 1 --with-nist --allow-model-download --nist-candidates 1

