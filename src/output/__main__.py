from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.llm.client import GroqAPIError, MissingGroqApiKey
from src.llm.config import LLMConfig

from .pipeline import run_phase5_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the Phase 5 markdown risk report.")
    parser.add_argument("--top", type=int, default=5, help="Number of risks to include.")
    parser.add_argument("--output-dir", default="reports", help="Directory for generated markdown.")
    parser.add_argument(
        "--rate-limit-sleep",
        type=float,
        default=2.0,
        help="Seconds to sleep between Groq calls.",
    )
    args = parser.parse_args()

    try:
        result = run_phase5_pipeline(
            top_n=args.top,
            output_dir=Path(args.output_dir),
            llm_config=LLMConfig(rate_limit_sleep_seconds=args.rate_limit_sleep),
        )
    except MissingGroqApiKey as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)
    except GroqAPIError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(3)

    print(f"Phase 5 report generated: {result.report.output_path}")
    print(f"Risks included: {result.report.risk_count}")


if __name__ == "__main__":
    main()

