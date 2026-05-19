from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

import chromadb
import pandas as pd
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from src.ingest.models import NistControl

from .bm25 import BM25Index, normalize_scores, tokenize
from .nist_priors import ControlPrior, infer_control_intent_terms, infer_control_priors, prior_score_for_control
from .retrieval_config import RetrievalConfig


class RetrievalIndexError(ValueError):
    """Raised when the NIST retrieval index cannot be built or queried."""


@dataclass(frozen=True)
class RetrievalCandidate:
    control_id: str
    name: str
    discussion: str
    semantic_score: float
    bm25_score: float
    control_prior_score: float
    hybrid_score: float
    rank: int
    prior_reasons: tuple[str, ...]

    @property
    def document_text(self) -> str:
        return f"[{self.control_id}] {self.name}: {self.discussion}".strip()


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    priors: tuple[ControlPrior, ...]
    candidates: tuple[RetrievalCandidate, ...]

    @property
    def best(self) -> RetrievalCandidate:
        if not self.candidates:
            raise RetrievalIndexError("No NIST candidates were retrieved")
        return self.candidates[0]


class NistHybridRetriever:
    """Hybrid RAG retriever for NIST SP 800-53 controls.

    Architecture:
    1. Embed every NIST control with sentence-transformers/all-MiniLM-L6-v2.
    2. Store embeddings in a persistent Chroma collection under /tmp.
    3. Query semantically from the scored risk context.
    4. Re-rank candidates with BM25 and deterministic control-family priors.
    """

    def __init__(
        self,
        controls: list[NistControl],
        config: RetrievalConfig | None = None,
    ) -> None:
        self.config = config or RetrievalConfig()
        if not controls:
            raise RetrievalIndexError("Cannot build NIST retriever with zero controls")

        self.controls = controls
        self.controls_by_id = {control.control_id: control for control in controls}
        self.model = SentenceTransformer(
            self.config.embedding_model_name,
            local_files_only=self.config.embedding_model_local_files_only,
        )
        self.chroma_persist_dir = _resolve_chroma_persist_dir(self.config.chroma_persist_dir)
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = self._new_chroma_client()
        self.collection = self._get_or_create_collection()
        self._ensure_collection_ready()
        logger.info(
            "NIST retriever initialized: %d controls indexed, model=%s, chroma_dir=%s",
            len(controls),
            self.config.embedding_model_name,
            self.chroma_persist_dir,
        )

    def retrieve(self, risk: pd.Series | dict[str, Any]) -> RetrievalResult:
        query = build_risk_query(risk)
        priors = infer_control_priors(risk)
        semantic_candidates = self._semantic_candidates(query)
        candidate_ids = dict.fromkeys(candidate["control_id"] for candidate in semantic_candidates)

        for prior in priors:
            if prior.control_id in self.controls_by_id:
                candidate_ids.setdefault(prior.control_id, None)

        candidates = [self.controls_by_id[control_id] for control_id in candidate_ids if control_id in self.controls_by_id]
        semantic_scores = {
            candidate["control_id"]: candidate["semantic_score"]
            for candidate in semantic_candidates
        }
        ranked = self._rerank(query=query, controls=candidates, semantic_scores=semantic_scores, priors=priors)

        return RetrievalResult(
            query=query,
            priors=tuple(priors),
            candidates=tuple(ranked[: self.config.final_candidate_count]),
        )

    def retrieve_for_top_risks(self, risks: pd.DataFrame) -> list[tuple[pd.Series, RetrievalResult]]:
        return [(row, self.retrieve(row)) for _, row in risks.iterrows()]

    def _build_collection(self) -> None:
        documents = [control.document_text for control in self.controls]
        ids = [control.control_id for control in self.controls]
        metadatas = [
            {
                "control_id": control.control_id,
                "name": control.name,
                "family": control.family or "",
            }
            for control in self.controls
        ]

        for start in range(0, len(documents), self.config.batch_size):
            end = start + self.config.batch_size
            batch_documents = documents[start:end]
            embeddings = self.model.encode(
                batch_documents,
                batch_size=self.config.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).tolist()
            self.collection.upsert(
                ids=ids[start:end],
                documents=batch_documents,
                metadatas=metadatas[start:end],
                embeddings=embeddings,
            )

    def _semantic_candidates(self, query: str) -> list[dict[str, float | str]]:
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0].tolist()
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=self.config.semantic_candidate_count,
                include=["distances", "metadatas"],
            )
        except Exception as exc:
            logger.warning(
                "Chroma query failed; rebuilding persistent NIST collection: %s",
                exc,
                exc_info=True,
            )
            self.collection = self._recreate_collection()
            self._build_collection()
            self._verify_collection_populated()
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=self.config.semantic_candidate_count,
                include=["distances", "metadatas"],
            )

        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        candidates: list[dict[str, float | str]] = []
        for control_id, distance in zip(ids, distances):
            semantic_score = max(0.0, min(1.0, 1.0 - float(distance)))
            candidates.append(
                {
                    "control_id": str(control_id),
                    "semantic_score": semantic_score,
                }
            )
        return candidates

    def _new_chroma_client(self):
        return chromadb.PersistentClient(
            path=str(self.chroma_persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )

    def _get_or_create_collection(self):
        return self.client.get_or_create_collection(
            name=self.config.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _ensure_collection_ready(self) -> None:
        try:
            count = self.collection.count()
        except Exception as exc:
            logger.warning(
                "Chroma collection health check failed; recreating collection: %s",
                exc,
                exc_info=True,
            )
            self.collection = self._recreate_collection()
            self._build_collection()
            self._verify_collection_populated()
            return

        expected_count = len(self.controls)
        if count == expected_count:
            return

        if count == 0:
            logger.info("Chroma collection is empty; building NIST collection")
        else:
            logger.warning(
                "Chroma collection has %d controls, expected %d; rebuilding collection",
                count,
                expected_count,
            )
            self.collection = self._recreate_collection()

        self._build_collection()
        self._verify_collection_populated()

    def _recreate_collection(self):
        try:
            self.client.delete_collection(name=self.config.chroma_collection_name)
        except Exception:
            logger.info("No existing Chroma collection to delete before rebuild", exc_info=True)
        return self._get_or_create_collection()

    def _verify_collection_populated(self) -> None:
        try:
            count = self.collection.count()
        except Exception as exc:
            raise RetrievalIndexError(
                f"Chroma collection verification failed after rebuild: {exc}"
            ) from exc
        if count == 0:
            raise RetrievalIndexError("Chroma collection rebuild completed with zero indexed controls")

    def _rerank(
        self,
        *,
        query: str,
        controls: list[NistControl],
        semantic_scores: dict[str, float],
        priors: list[ControlPrior],
    ) -> list[RetrievalCandidate]:
        if not controls:
            return []

        semantic_scores = dict(semantic_scores)
        missing_semantic_controls = [control for control in controls if control.control_id not in semantic_scores]
        if missing_semantic_controls:
            semantic_scores.update(self._direct_semantic_scores(query, missing_semantic_controls))

        bm25_index = BM25Index([tokenize(control.document_text) for control in controls])
        bm25_scores = normalize_scores(bm25_index.score(tokenize(query)))

        candidates: list[RetrievalCandidate] = []
        for index, control in enumerate(controls):
            prior_score, prior_reasons = prior_score_for_control(control.control_id, priors)
            semantic_score = semantic_scores.get(control.control_id, 0.0)
            bm25_score = bm25_scores[index] if index < len(bm25_scores) else 0.0
            hybrid_score = (
                semantic_score * self.config.weights.semantic
                + bm25_score * self.config.weights.bm25
                + prior_score * self.config.weights.control_prior
            )
            candidates.append(
                RetrievalCandidate(
                    control_id=control.control_id,
                    name=control.name,
                    discussion=control.discussion,
                    semantic_score=semantic_score,
                    bm25_score=bm25_score,
                    control_prior_score=prior_score,
                    hybrid_score=hybrid_score,
                    rank=0,
                    prior_reasons=tuple(prior_reasons),
                )
            )

        ranked = sorted(
            candidates,
            key=lambda item: (
                item.hybrid_score,
                item.control_prior_score,
                item.semantic_score,
                item.bm25_score,
                _base_control_order(item.control_id),
            ),
            reverse=True,
        )
        return [
            RetrievalCandidate(
                control_id=candidate.control_id,
                name=candidate.name,
                discussion=candidate.discussion,
                semantic_score=candidate.semantic_score,
                bm25_score=candidate.bm25_score,
                control_prior_score=candidate.control_prior_score,
                hybrid_score=candidate.hybrid_score,
                rank=index + 1,
                prior_reasons=candidate.prior_reasons,
            )
            for index, candidate in enumerate(ranked)
        ]

    def _direct_semantic_scores(self, query: str, controls: list[NistControl]) -> dict[str, float]:
        texts = [query, *(control.document_text for control in controls)]
        embeddings = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        query_embedding = embeddings[0]
        scores: dict[str, float] = {}
        for control, embedding in zip(controls, embeddings[1:]):
            score = float(sum(left * right for left, right in zip(query_embedding, embedding)))
            scores[control.control_id] = max(0.0, min(1.0, score))
        return scores


def build_risk_query(risk: pd.Series | dict[str, Any]) -> str:
    get = risk.get
    ransomware = "yes" if _truthy(get("feature_ransomware", get("threat_ransomware_bool", False))) else "no"
    exposure = "internet-facing" if _truthy(get("feature_internet_exposed", get("internet_exposed_bool", False))) else "internal"
    exploit_status = "actively exploited" if _truthy(get("feature_active_exploitation", False)) else "not confirmed actively exploited"
    control_intent = infer_control_intent_terms(risk)

    return " ".join(
        [
            f"NIST SP 800-53 control mapping for {get('vulnerability_name', '')}",
            f"affecting component {get('affected_component', '')}",
            f"on {get('asset_type', '')} asset {get('asset_name', '')}.",
            f"The asset is {exposure} and supports {get('business_service', '')}.",
            f"Exploit status: {exploit_status}.",
            f"Ransomware involved: {ransomware}.",
            f"Threat campaign: {get('campaign_names', '')}.",
            f"Threat summary: {get('threat_summaries', '')}.",
            f"Business impact: {get('business_impact', '')}.",
            f"Control intent terms: {control_intent}.",
        ]
    )


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value > 0
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "yes", "1", "known"}


def _base_control_order(control_id: str) -> int:
    preferred = ["AC-2", "IA-2", "IA-5", "CM-6", "IR-4", "SI-2", "RA-5", "SA-22", "SI-4", "CM-8"]
    base = str(control_id).split("(")[0]
    try:
        return len(preferred) - preferred.index(base)
    except ValueError:
        return 0


def _resolve_chroma_persist_dir(path: Path | str) -> Path:
    configured = Path(path)
    if os.name == "nt" and str(configured).replace("\\", "/") == "/tmp/chroma_tawasolpay":
        return Path.cwd() / ".chroma" / "chroma_tawasolpay"
    return configured
