from __future__ import annotations

import argparse

from src.engine.scoring import top_risks
from src.engine.scoring_config import ScoringConfig
from src.ingest.pipeline import build_ingestion_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 2 risk scoring.")
    parser.add_argument("--offline", action="store_true", help="Skip external data refresh.")
    parser.add_argument("--top", type=int, default=5, help="Number of top risks to print.")
    parser.add_argument(
        "--details",
        action="store_true",
        help="Print feature and contribution columns for the top risks.",
    )
    parser.add_argument(
        "--with-nist",
        action="store_true",
        help="Build the Phase 3 NIST hybrid retriever and print selected controls.",
    )
    parser.add_argument(
        "--nist-candidates",
        type=int,
        default=3,
        help="Number of NIST candidates to print per risk when --with-nist is used.",
    )
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Allow sentence-transformers to download the embedding model if it is not cached.",
    )
    args = parser.parse_args()

    bundle = build_ingestion_bundle(include_external=not args.offline)
    config = ScoringConfig(top_n=args.top)
    risks = top_risks(bundle.enriched_vulnerabilities, config=config, n=args.top)

    retriever = None
    if args.with_nist:
        from src.engine.nist_retriever import NistHybridRetriever
        from src.engine.retrieval_config import RetrievalConfig

        retriever = NistHybridRetriever(
            bundle.external.nist_controls,
            config=RetrievalConfig(
                embedding_model_local_files_only=not args.allow_model_download
            ),
        )

    phase = "Phase 2 scoring + Phase 3 NIST retrieval" if args.with_nist else "Phase 2 scoring"
    print(f"{phase} completed: top {args.top} risks")
    for _, row in risks.iterrows():
        print(
            f"#{int(row['risk_rank'])} {row['risk_severity']} "
            f"{row['composite_risk_score']:.3f} "
            f"{row['asset_name']} | {row['cve']} | {row['vulnerability_name']}"
        )
        print(f"   drivers: {row['score_drivers']}")
        print(f"   service: {row['business_service']} | threat intel: {row.get('campaign_names', '')}")
        if args.details:
            features = [
                "feature_cvss",
                "feature_active_exploitation",
                "feature_ransomware",
                "feature_epss",
                "feature_internet_exposed",
                "feature_business_impact",
                "feature_threat_intel_match",
                "feature_days_open",
                "feature_missing_edr",
            ]
            print("   features: " + ", ".join(f"{name}={row[name]:.2f}" for name in features))
        if retriever is not None:
            result = retriever.retrieve(row)
            best = result.best
            print(
                f"   NIST: {best.control_id} {best.name} "
                f"(hybrid={best.hybrid_score:.3f}, semantic={best.semantic_score:.3f}, "
                f"bm25={best.bm25_score:.3f}, prior={best.control_prior_score:.3f})"
            )
            if best.prior_reasons:
                print(f"   prior: {'; '.join(best.prior_reasons)}")
            if args.nist_candidates > 1:
                for candidate in result.candidates[1 : args.nist_candidates]:
                    print(
                        f"      candidate {candidate.rank}: {candidate.control_id} {candidate.name} "
                        f"(hybrid={candidate.hybrid_score:.3f})"
                    )


if __name__ == "__main__":
    main()
