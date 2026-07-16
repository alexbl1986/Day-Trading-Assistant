"""CorpusIndex over an in-memory Qdrant with a deterministic fake embedder.

No network: the fake embedder hashes tokens into a fixed-width bag-of-words
vector, so cosine similarity tracks token overlap and search is deterministic.
"""

import itertools
import math
import re

import pytest
from qdrant_client import QdrantClient

from app.rag.chunk import Chunk
from app.rag.index import CorpusIndex

DIM = 64


class FakeEmbedder:
    """Token-hashing bag-of-words -> L2-normalized vector (offline, stable)."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * DIM
        for tok in re.findall(r"[A-Za-z]+", text.lower()):
            vec[hash(tok) % DIM] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm else vec


def _index() -> CorpusIndex:
    return CorpusIndex(
        QdrantClient(location=":memory:"), FakeEmbedder(), vector_size=DIM
    )


_ids = itertools.count()


def _chunk(text, doc_type="daily", date="2026-07-08", section=None):
    n = next(_ids)  # globally unique so point ids never collide
    return Chunk(
        text=text,
        doc_type=doc_type,
        review_date=date,
        source=f"{doc_type}.pdf",
        chunk_id=f"{doc_type}-{date}-c{n:03d}",
        section=section,
        pages=(1,),
    )


DAILY_V1 = [
    _chunk("Micron memory HBM DRAM shortage capex", section="Memory"),
    _chunk("Iran oil Oman risk premium macro", section="Macro"),
    _chunk("AI infrastructure capex hyperscalers", section="AI"),
]


def test_search_returns_most_relevant_chunk_with_payload():
    idx = _index()
    idx.replace_document(DAILY_V1, user_id="alex")

    hits = idx.search("memory DRAM shortage", user_id="alex", k=1)

    assert len(hits) == 1
    top = hits[0]
    assert "Micron" in top.text
    assert top.section == "Memory"
    assert top.source == "daily.pdf"
    assert top.chunk_id.startswith("daily-")
    assert top.doc_type == "daily"
    assert top.review_date == "2026-07-08"
    assert top.pages == (1,)


def test_user_filter_isolates_corpora():
    # ADR-0005: a shared corpus would let one user's reviews poison another's.
    idx = _index()
    idx.replace_document(DAILY_V1, user_id="alex")
    idx.replace_document(
        [_chunk("Palladium squeeze synthetic demo note")],
        user_id="demo",
    )

    alex_hits = idx.search("palladium squeeze", user_id="alex", k=5)
    assert "Palladium" not in " ".join(h.text for h in alex_hits)

    demo_hits = idx.search("palladium squeeze", user_id="demo", k=5)
    assert len(demo_hits) == 1
    assert "Palladium" in demo_hits[0].text


def test_replace_on_upload_swaps_same_doc_type_only():
    idx = _index()
    idx.replace_document(DAILY_V1, user_id="alex")
    idx.replace_document(
        [_chunk("Weekly conviction map memory names", doc_type="weekly", date="2026-07-06")],
        user_id="alex",
    )

    # New daily upload replaces the old daily, leaves the weekly intact.
    result = idx.replace_document(
        [_chunk("Fresh daily note nvidia guidance", date="2026-07-09")],
        user_id="alex",
    )
    assert result.replaced is True
    assert result.warning is None

    dailies = idx.search("memory DRAM shortage nvidia guidance", user_id="alex", k=10)
    dailies = [h for h in dailies if h.doc_type == "daily"]
    assert all(h.review_date == "2026-07-09" for h in dailies)
    assert all("Micron" not in h.text for h in dailies)  # v1 chunks gone

    weekly = idx.search("conviction map", user_id="alex", k=10)
    assert any(h.doc_type == "weekly" for h in weekly)  # untouched


def test_replace_warns_when_upload_is_older():
    idx = _index()
    idx.replace_document([_chunk("newer daily", date="2026-07-09")], user_id="alex")

    result = idx.replace_document([_chunk("older daily", date="2026-07-07")], user_id="alex")

    assert result.replaced is True
    assert result.warning is not None
    assert "older" in result.warning


def test_empty_upload_is_rejected():
    with pytest.raises(ValueError):
        _index().replace_document([], user_id="alex")
