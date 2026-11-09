#!/usr/bin/env python3
"""BM25 lexical retrieval and Reciprocal Rank Fusion (RRF).

Assignment parameters:
    BM25 k1 = 1.5
    BM25 b  = 0.75
    RRF k   = 60

This script is self-contained and uses only the Python standard library.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> List[str]:
    """Lowercase tokenization that preserves identifiers containing '_'."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return TOKEN_PATTERN.findall(text.lower())


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str


class BM25:
    """Small, transparent BM25 index using k1=1.5 and b=0.75 by default."""

    def __init__(
        self,
        documents: Mapping[str, str] | Iterable[Tuple[str, str]],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")

        items = list(documents.items()) if isinstance(documents, Mapping) else list(documents)
        ids = [doc_id for doc_id, _ in items]
        if len(ids) != len(set(ids)):
            raise ValueError("document IDs must be unique")

        self.k1 = float(k1)
        self.b = float(b)
        self.doc_ids: List[str] = []
        self.doc_tokens: Dict[str, List[str]] = {}
        self.term_freqs: Dict[str, Counter[str]] = {}
        self.doc_lengths: Dict[str, int] = {}
        self.df: Counter[str] = Counter()

        for doc_id, text in items:
            if not isinstance(doc_id, str) or not doc_id:
                raise ValueError("each document ID must be a non-empty string")
            tokens = tokenize(text)
            self.doc_ids.append(doc_id)
            self.doc_tokens[doc_id] = tokens
            self.term_freqs[doc_id] = Counter(tokens)
            self.doc_lengths[doc_id] = len(tokens)
            self.df.update(set(tokens))

        self.N = len(self.doc_ids)
        self.avgdl = (
            sum(self.doc_lengths.values()) / self.N if self.N else 0.0
        )

    def _idf(self, term: str) -> float:
        """Positive Robertson–Spärck Jones IDF variant."""
        n = self.df.get(term, 0)
        if self.N == 0 or n == 0:
            return 0.0
        return math.log(1.0 + (self.N - n + 0.5) / (n + 0.5))

    def score(self, query: str, doc_id: str) -> float:
        if doc_id not in self.term_freqs:
            raise KeyError(f"unknown document ID: {doc_id}")
        terms = tokenize(query)
        if not terms:
            return 0.0

        tf_map = self.term_freqs[doc_id]
        dl = self.doc_lengths[doc_id]
        norm = self.k1 * (1.0 - self.b + self.b * dl / self.avgdl) if self.avgdl else self.k1
        total = 0.0
        for term in terms:
            tf = tf_map.get(term, 0)
            if tf == 0:
                continue
            numerator = tf * (self.k1 + 1.0)
            denominator = tf + norm
            total += self._idf(term) * numerator / denominator
        return total

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Return (document_id, score), sorted by score descending then ID."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        if self.N == 0 or not tokenize(query):
            return []

        ranked = [(doc_id, self.score(query, doc_id)) for doc_id in self.doc_ids]
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return ranked[: min(top_k, len(ranked))]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> List[Tuple[str, float]]:
    """Fuse ranked document IDs with RRF; duplicate IDs per list count once."""
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer")
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights and rankings must have equal lengths")
    if any(weight < 0 for weight in weights):
        raise ValueError("weights must be non-negative")

    scores: Dict[str, float] = defaultdict(float)
    for ranking, weight in zip(rankings, weights):
        seen = set()
        rank = 0
        for doc_id in ranking:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            rank += 1
            scores[doc_id] += float(weight) / (k + rank)

    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def hybrid_search(
    bm25: BM25,
    query: str,
    dense_ranking: Sequence[str],
    top_k: int = 10,
    rrf_k: int = 60,
    bm25_weight: float = 1.0,
    dense_weight: float = 1.0,
) -> List[Tuple[str, float]]:
    """Combine BM25 and a supplied dense ranking using weighted RRF."""
    lexical = [doc_id for doc_id, _ in bm25.search(query, top_k=top_k)]
    return reciprocal_rank_fusion(
        [lexical, list(dense_ranking)],
        k=rrf_k,
        weights=[bm25_weight, dense_weight],
    )[:top_k]


def run_smoke_tests() -> None:
    docs = {
        "DOC_001": "HNSW is an approximate nearest neighbor graph index.",
        "DOC_002": "BM25 is a lexical ranking function for information retrieval.",
        "DOC_003": "SERIAL_NO_AURA_9982X identifies the AURA console device.",
        "DOC_004": "GraphRAG builds a knowledge graph to support retrieval augmented generation.",
    }
    index = BM25(docs, k1=1.5, b=0.75)

    assert index.search("SERIAL_NO_AURA_9982X", top_k=1)[0][0] == "DOC_003"
    assert index.search("GraphRAG knowledge graph", top_k=2)[0][0] == "DOC_004"
    assert index.search("", top_k=5) == []
    assert BM25({}).search("anything") == []

    fused = reciprocal_rank_fusion(
        [["A", "B", "A"], ["B", "C"]], k=60
    )
    assert fused[0][0] == "B"
    assert len(dict(fused)) == 3

    hybrid = hybrid_search(
        index, "GraphRAG knowledge graph",
        ["DOC_004", "DOC_002", "DOC_001"], top_k=3
    )
    assert hybrid and hybrid[0][0] == "DOC_004"
    print("BM25/RRF smoke tests passed.")


def run_demonstration() -> None:
    documents = {
        "DOC_001": "HNSW is a graph-based approximate nearest-neighbour index.",
        "DOC_002": "BM25 is a lexical information retrieval ranking algorithm.",
        "DOC_003": "The serial identifier SERIAL_NO_AURA_9982X belongs to AURA.",
        "DOC_004": "GraphRAG uses knowledge graphs for retrieval augmented generation.",
    }
    index = BM25(documents, k1=1.5, b=0.75)

    scenarios = [
        ("Exact identifier lookup", "SERIAL_NO_AURA_9982X",
         ["DOC_001", "DOC_002", "DOC_004", "DOC_003"]),
        ("Conceptual retrieval", "GraphRAG knowledge graph synthesis",
         ["DOC_004", "DOC_002", "DOC_001", "DOC_003"]),
    ]

    print("\n=== BM25 + RRF component demonstration ===")
    print("Parameters: k1=1.5, b=0.75, RRF k=60")
    for label, query, dense_ranking in scenarios:
        lexical = index.search(query, top_k=4)
        hybrid = hybrid_search(index, query, dense_ranking, top_k=4, rrf_k=60)
        print(f"\n{label}: {query}")
        print("BM25 ranking:", [doc_id for doc_id, _ in lexical])
        print("Supplied dense ranking:", dense_ranking)
        print("Hybrid RRF ranking:", hybrid)
    print("\n[PASS] BM25 and RRF component demonstration completed.")
    print("Note: dense rankings above are illustrative inputs, not measured embeddings.")


if __name__ == "__main__":
    run_smoke_tests()
    run_demonstration()
