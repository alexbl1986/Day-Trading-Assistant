"""Hybrid retrieval pipeline: dense + BM25 fused with reciprocal rank fusion.

First-stage dense and lexical retrieval both run over the chunk corpus and are
fused with RRF; the fused chunks are returned verbatim (chunk size is bounded
at ingest, and each chunk carries its section heading in-text plus full
metadata for tracing). ``RerankingRetriever`` (Task 6.1) layers Cohere
reranking over the same fused pool — only the final selection mechanism
changes, so eval deltas attribute cleanly to the reranker.

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

    def fused_pool(self, query: str, *, user_id: str) -> list[RetrievedDoc]:
        """The FULL RRF-fused candidate pool (up to 2×first_stage_k unique
        chunks, best-first) — what ``retrieve`` cuts to ``k``, and what a
        reranker consumes whole so chunks can move in and out of the top-k."""
        return reciprocal_rank_fusion(
            [
                self.dense(query, user_id=user_id, k=self._first_stage_k),
                self.bm25(query, user_id=user_id, k=self._first_stage_k),
            ],
            limit=2 * self._first_stage_k,
            rrf_constant=self._rrf_constant,
        )

    def retrieve(self, query: str, *, user_id: str, k: int = 5) -> list[RetrievedDoc]:
        """Full pipeline: top-``k`` RRF-fused chunks from dense + BM25."""
        return self.fused_pool(query, user_id=user_id)[:k]


class SharedCorpusRetriever:
    """Serve ONE owner's corpus to every caller.

    The cert prototype bakes the desk reviews in for all users: callers keep
    passing their own ``user_id`` — the tools' per-call
    user binding stays intact for Demo Day's per-user corpora — but retrieval
    here always reads the shared owner's documents."""

    def __init__(self, inner, *, owner: str) -> None:
        self._inner = inner
        self._owner = owner

    def retrieve(self, query: str, *, user_id: str, k: int = 5) -> list[RetrievedDoc]:
        return self._inner.retrieve(query, user_id=self._owner, k=k)


class RerankingRetriever:
    """Cohere rerank over the hybrid's full fused pool (Task 6.1).

    Same first stage as the baseline (dense + BM25 top-first_stage_k, RRF),
    but instead of cutting the fused list at ``k``, the whole pool goes to a
    cross-encoder that reads query and chunk together; the top-``k`` by rerank
    relevance come back (scores become Cohere relevance scores). The candidate
    pool is identical to the baseline's internal one — only the final
    selection mechanism changes. ``rerank`` is any callable with the
    ``cohere_rerank`` shape; tests pass a deterministic fake."""

    def __init__(self, inner: HybridRetriever, rerank) -> None:
        self._inner = inner
        self._rerank = rerank

    def retrieve(self, query: str, *, user_id: str, k: int = 5) -> list[RetrievedDoc]:
        pool = self._inner.fused_pool(query, user_id=user_id)
        if len(pool) <= k:
            return pool
        ranked = self._rerank(query, [doc.text for doc in pool], top_n=k)
        return [replace(pool[i], score=score) for i, score in ranked]


def cohere_rerank(model: str = "rerank-v3.5"):
    """Production reranker (Cohere v2; multilingual is required — Hebrew
    chunks scored against English queries). Returns a callable mapping
    ``(query, texts, top_n)`` to ``[(pool_index, relevance_score), ...]``
    best-first. Reads ``COHERE_API_KEY``."""
    import os

    import cohere

    client = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])

    def rerank(query: str, texts: Sequence[str], *, top_n: int) -> list[tuple[int, float]]:
        response = client.rerank(model=model, query=query, documents=list(texts), top_n=top_n)
        return [(r.index, r.relevance_score) for r in response.results]

    return rerank


RETRIEVAL_MODES = ("baseline", "rerank")


def apply_retrieval_mode(hybrid: HybridRetriever, mode: str):
    """One source of truth for what the retrieval-mode names mean — the
    dev/deploy wiring (``DESK_RETRIEVAL`` env var) and the eval harness's
    ``--variant`` both resolve modes here, so production and eval can't
    drift apart."""
    if mode == "baseline":
        return hybrid
    if mode == "rerank":
        return RerankingRetriever(hybrid, cohere_rerank())
    raise ValueError(f"unknown retrieval mode {mode!r} (expected one of {RETRIEVAL_MODES})")


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
