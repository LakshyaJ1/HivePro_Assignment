Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

python -m src.llm --top 5 --rate-limit-sleep 2

