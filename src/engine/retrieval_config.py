from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RetrievalWeights:
    semantic: float = 0.45
    bm25: float = 0.25
    control_prior: float = 0.30

    def validate(self) -> None:
        total = self.semantic + self.bm25 + self.control_prior
        if abs(total - 1.0) > 0.000001:
            raise ValueError(f"Retrieval weights must sum to 1.0, got {total:.6f}")
        if min(self.semantic, self.bm25, self.control_prior) < 0:
            raise ValueError("Retrieval weights cannot be negative")


@dataclass(frozen=True)
class RetrievalConfig:
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_model_local_files_only: bool = True
    chroma_collection_name: str = "nist_sp800_53_rev5_controls"
    chroma_persist_dir: Path | str = "/tmp/chroma_tawasolpay"
    semantic_candidate_count: int = 30
    final_candidate_count: int = 5
    batch_size: int = 64
    weights: RetrievalWeights = RetrievalWeights()

    def __post_init__(self) -> None:
        self.weights.validate()
        if self.semantic_candidate_count < self.final_candidate_count:
            raise ValueError("semantic_candidate_count must be >= final_candidate_count")
        if self.final_candidate_count < 1:
            raise ValueError("final_candidate_count must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
