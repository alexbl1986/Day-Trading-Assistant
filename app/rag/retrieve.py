"""Hybrid retrieval pipeline: dense + BM25 fused with reciprocal rank fusion.

First-stage dense and lexical retrieval both run over the chunk corpus and are
fused with RRF; the fused chunks are returned verbatim (chunk size is bounded
at ingest, and each chunk carries its section heading in-text plus full
metadata for tracing). Cohere reranking (Task 6.1) layers on top later.

BM25 is rebuilt per query from the user's chunks in Qdrant — the corpus is
tiny (two reviews) and replace-on-upload keeps it fresh, so there is no separate
lexical index to fall out of sync. The tokenizer is Unicode-aware: the course's
``[a-z0-9]+`` would drop every Hebrew token, which is most of this corpus.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Iterable, Sequence

from rank_bm25 import BM25Okapi

from app.rag.index import CorpusIndex, SearchHit

_TOKEN = re.compile(r"\w+", re.UNICODE)  # \w matches Hebrew, Latin, and digits


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@dataclass(frozen=True)
class RetrievedDoc:
    id: str  # the chunk_id — traceable to docs/chunk_preview and the Qdrant point
    text: str
    score: float | None
    doc_type: str
    review_date: str | None
    source: str
    section: str | None
    pages: tuple[int, ...]


def reciprocal_rank_fusion(
    ranked_lists: Iterable[Sequence[RetrievedDoc]],
    *,
    limit: int,
    rrf_constant: int = 60,
) -> list[RetrievedDoc]:
    """Fuse ranked lists by summed reciprocal rank, keyed on document id."""
    scores: dict[str, float] = {}
    docs: dict[str, RetrievedDoc] = {}
    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list, start=1):
            docs.setdefault(doc.id, doc)
            scores[doc.id] = scores.get(doc.id, 0.0) + 1.0 / (rrf_constant + rank)
    ordered = sorted(scores, key=lambda doc_id: scores[doc_id], reverse=True)[:limit]
    return [replace(docs[doc_id], score=scores[doc_id]) for doc_id in ordered]


class HybridRetriever:
    """Dense + BM25 (RRF) over the user's chunk corpus."""

    def __init__(
        self,
        index: CorpusIndex,
        *,
        first_stage_k: int = 8,
        rrf_constant: int = 60,
    ) -> None:
        self._index = index
        self._first_stage_k = first_stage_k
        self._rrf_constant = rrf_constant

    def dense(self, query: str, *, user_id: str, k: int) -> list[RetrievedDoc]:
        return [_from_hit(hit) for hit in self._index.search(query, user_id=user_id, k=k)]

    def bm25(self, query: str, *, user_id: str, k: int) -> list[RetrievedDoc]:
        docs = self._index.all_chunks(user_id=user_id)
        if not docs:
            return []
        bm25 = BM25Okapi([tokenize(doc.text) for doc in docs])
        scores = bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)[:k]
        return [replace(_from_hit(docs[i]), score=float(scores[i])) for i in ranked]

    def retrieve(self, query: str, *, user_id: str, k: int = 5) -> list[RetrievedDoc]:
        """Full pipeline: top-``k`` RRF-fused chunks from dense + BM25."""
        return reciprocal_rank_fusion(
            [
                self.dense(query, user_id=user_id, k=self._first_stage_k),
                self.bm25(query, user_id=user_id, k=self._first_stage_k),
            ],
            limit=k,
            rrf_constant=self._rrf_constant,
        )


class SharedCorpusRetriever:
    """Serve ONE owner's corpus to every caller.

    The cert prototype bakes the desk reviews in for all users (ADR-0005
    amendment): callers keep passing their own ``user_id`` — the tools' per-call
    user binding stays intact for Demo Day's per-user corpora — but retrieval
    here always reads the shared owner's documents."""

    def __init__(self, inner, *, owner: str) -> None:
        self._inner = inner
        self._owner = owner

    def retrieve(self, query: str, *, user_id: str, k: int = 5) -> list[RetrievedDoc]:
        return self._inner.retrieve(query, user_id=self._owner, k=k)


def _from_hit(hit: SearchHit) -> RetrievedDoc:
    return RetrievedDoc(
        id=hit.chunk_id,
        text=hit.text,
        score=hit.score or None,
        doc_type=hit.doc_type,
        review_date=hit.review_date,
        source=hit.source,
        section=hit.section,
        pages=hit.pages,
    )
