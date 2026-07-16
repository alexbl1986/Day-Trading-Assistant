"""Hybrid retrieval pipeline: tokenizer, RRF, and end-to-end chunk retrieval.

Offline throughout: in-memory Qdrant + a deterministic token-hashing embedder
whose tokenizer (\\w+) covers Hebrew, so dense retrieval works on the real
reviews without the network.
"""

import math
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from app.rag.chunk import Chunk, chunk_document
from app.rag.index import CorpusIndex
from app.rag.retrieve import (
    HybridRetriever,
    RetrievedDoc,
    reciprocal_rank_fusion,
    tokenize,
)

DIM = 256
REVIEWS = Path(__file__).resolve().parent.parent / "data" / "reviews"
DAILY = REVIEWS / "סקירת דסק יומית מידעפנים 08_07_26.pdf"


class FakeEmbedder:
    """Unicode-aware token-hashing bag-of-words -> L2-normalized vector."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * DIM
        for tok in tokenize(text):
            vec[hash(tok) % DIM] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm else vec


def _index() -> CorpusIndex:
    return CorpusIndex(QdrantClient(location=":memory:"), FakeEmbedder(), vector_size=DIM)


# --- tokenizer -----------------------------------------------------------


def test_tokenizer_covers_hebrew():
    # The whole point: the course's [a-z0-9]+ would return [] for Hebrew.
    assert tokenize("זיכרון HBM ו-SMR 25%") == ["זיכרון", "hbm", "ו", "smr", "25"]


# --- RRF -----------------------------------------------------------------


def _doc(doc_id):
    return RetrievedDoc(
        id=doc_id, text="x", score=None, doc_type="daily", review_date=None,
        source="s", section=None, pages=(1,),
    )


def test_rrf_rewards_agreement_across_lists():
    list_a = [_doc("1"), _doc("2"), _doc("3")]
    list_b = [_doc("2"), _doc("4"), _doc("1")]
    fused = reciprocal_rank_fusion([list_a, list_b], limit=4)
    # id2 (ranks 2,1) edges id1 (ranks 1,3); singletons follow.
    assert [d.id for d in fused] == ["2", "1", "4", "3"]
    assert [d.score for d in fused] == sorted((d.score for d in fused), reverse=True)


# --- end-to-end pipeline -------------------------------------------------


def _synthetic_corpus():
    return [
        _chunk("daily-d-s00-c00", "Memory HBM DRAM shortage Micron", "Memory"),
        _chunk("daily-d-s00-c01", "capex memory pricing power", "Memory"),
        _chunk("daily-d-s01-c00", "Iran oil Oman risk premium", "Oil"),
    ]


def _chunk(chunk_id, text, section):
    return Chunk(
        text=text, doc_type="daily", review_date="2026-07-08", source="d.pdf",
        chunk_id=chunk_id, section=section, pages=(1,),
    )


def test_retrieve_returns_ranked_deduped_chunks():
    idx = _index()
    idx.replace_document(_synthetic_corpus(), user_id="alex")

    results = HybridRetriever(idx).retrieve("DRAM shortage Micron", user_id="alex", k=2)

    assert results[0].id == "daily-d-s00-c00"  # the chunk itself, verbatim
    assert results[0].text == "Memory HBM DRAM shortage Micron"
    assert results[0].section == "Memory"
    assert len({r.id for r in results}) == len(results)  # each chunk once


def test_bm25_finds_exact_lexical_token():
    idx = _index()
    idx.replace_document(_synthetic_corpus(), user_id="alex")

    hits = HybridRetriever(idx).bm25("Oman", user_id="alex", k=1)
    assert hits and hits[0].id == "daily-d-s01-c00"


def test_retrieval_respects_user_isolation():
    idx = _index()
    idx.replace_document(_synthetic_corpus(), user_id="alex")
    idx.replace_document(
        [_chunk("daily-d-s09-c00", "Palladium squeeze demo only", "PD")],
        user_id="demo",
    )

    alex = HybridRetriever(idx).retrieve("palladium squeeze", user_id="alex", k=5)
    assert all("Palladium" not in r.text for r in alex)


@pytest.mark.parametrize("query,needle", [("TSMC PIC CPO אופטית צוואר בקבוק", "TSMC")])
def test_end_to_end_on_real_daily_review(query, needle):
    idx = _index()
    idx.replace_document(chunk_document(str(DAILY)), user_id="alex")
    results = HybridRetriever(idx).retrieve(query, user_id="alex", k=3)

    assert results
    assert all("-c" in r.id for r in results)  # chunk ids, traceable to preview
    assert any(needle in r.text for r in results)


def test_shared_corpus_retriever_serves_one_owner_to_every_caller():
    # Cert-prototype mode (ADR-0005 amendment): the baked-in reviews are shared,
    # so retrieval reads the shared owner's corpus no matter who is asking —
    # while callers keep passing their own user_id (the tools' binding intact).
    from app.rag.retrieve import SharedCorpusRetriever

    class _Recorder:
        def __init__(self):
            self.owners = []

        def retrieve(self, query, *, user_id, k=5):
            self.owners.append(user_id)
            return []

    inner = _Recorder()
    shared = SharedCorpusRetriever(inner, owner="baked-desk")
    shared.retrieve("palladium", user_id="alice")
    shared.retrieve("palladium", user_id="bob", k=3)
    assert inner.owners == ["baked-desk", "baked-desk"]
