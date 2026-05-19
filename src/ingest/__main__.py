from __future__ import annotations

import argparse

from .pipeline import build_ingestion_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Phase 1 ingestion smoke check.")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip runtime public reference downloads and validate local data only.",
    )
    args = parser.parse_args()

    bundle = build_ingestion_bundle(include_external=not args.offline)
    print("Phase 1 ingestion completed")
    print(f"assets: {len(bundle.structured.assets)}")
    print(f"vulnerabilities: {len(bundle.structured.vulnerabilities)}")
    print(f"threat intelligence: {len(bundle.structured.threat_intelligence)}")
    print(f"business services: {len(bundle.structured.business_services)}")
    print(f"enriched vulnerabilities: {len(bundle.enriched_vulnerabilities)}")
    print(f"unmatched threat intel: {len(bundle.unmatched_threat_intelligence)}")
    print(f"rag documents: {len(bundle.rag_documents)}")
    if bundle.warnings:
        print("warnings:")
        for warning in bundle.warnings:
            print(f"- {warning}")


if __name__ == "__main__":
    main()

