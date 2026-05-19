Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

python -m src.engine --top 5 --with-nist --nist-candidates 3

