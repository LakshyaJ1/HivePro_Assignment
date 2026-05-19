from __future__ import annotations

import argparse
import sys

from .client import GroqAPIError, MissingGroqApiKey
from .config import LLMConfig
from .pipeline import run_phase4_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 4 live Groq narration.")
    parser.add_argument("--top", type=int, default=5, help="Number of risks to narrate.")
    parser.add_argument(
        "--rate-limit-sleep",
        type=float,
        default=2.0,
        help="Seconds to sleep between Groq calls.",
    )
    args = parser.parse_args()

    try:
        result = run_phase4_pipeline(
            top_n=args.top,
            llm_config=LLMConfig(rate_limit_sleep_seconds=args.rate_limit_sleep),
        )
    except MissingGroqApiKey as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)
    except GroqAPIError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(3)

    print(f"Phase 4 LLM narration completed for {result.top_risk_count} risks")
    print("\nEXECUTIVE BRIEF")
    print(result.report.executive_brief)
    for risk in result.report.risks:
        print(f"\nRISK #{risk.risk_rank} ({risk.vuln_id})")
        print(f"Why: {risk.risk_explanation}")
        print(f"NIST: {risk.nist_application}")
        print(f"Threat: {risk.threat_summary}")
    print(f"\nModel: {result.report.model}")
    print(f"Usage: {result.report.total_usage}")


if __name__ == "__main__":
    main()
