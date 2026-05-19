from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass


TOKEN_PATTERN = re.compile(r"[a-z0-9][a-z0-9\-_/]*", re.IGNORECASE)

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
}


def tokenize(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_PATTERN.findall(text or "")
        if token.lower() not in STOPWORDS
    ]


@dataclass
class BM25Index:
    documents: list[list[str]]
    k1: float = 1.5
    b: float = 0.75

    def __post_init__(self) -> None:
        self.doc_count = len(self.documents)
        self.doc_lengths = [len(document) for document in self.documents]
        self.average_doc_length = (
            sum(self.doc_lengths) / self.doc_count if self.doc_count else 0.0
        )
        self.term_frequencies = [Counter(document) for document in self.documents]
        self.document_frequencies = Counter()
        for document in self.documents:
            self.document_frequencies.update(set(document))

    def score(self, query_tokens: list[str]) -> list[float]:
        if not self.documents:
            return []
        unique_query_terms = list(dict.fromkeys(query_tokens))
        return [self._score_document(index, unique_query_terms) for index in range(self.doc_count)]

    def _score_document(self, index: int, query_terms: list[str]) -> float:
        score = 0.0
        doc_length = self.doc_lengths[index] or 1
        frequencies = self.term_frequencies[index]

        for term in query_terms:
            term_frequency = frequencies.get(term, 0)
            if not term_frequency:
                continue
            document_frequency = self.document_frequencies.get(term, 0)
            inverse_document_frequency = math.log(
                1 + (self.doc_count - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            denominator = term_frequency + self.k1 * (
                1 - self.b + self.b * doc_length / (self.average_doc_length or 1)
            )
            score += inverse_document_frequency * (
                term_frequency * (self.k1 + 1) / denominator
            )

        return score


def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    minimum = min(scores)
    maximum = max(scores)
    if math.isclose(maximum, minimum):
        return [0.0 for _ in scores]
    return [(score - minimum) / (maximum - minimum) for score in scores]

