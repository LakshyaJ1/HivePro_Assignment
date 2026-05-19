Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

python -m src.output --top 5 --output-dir reports --rate-limit-sleep 2

