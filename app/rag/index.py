"""Qdrant indexing for the per-user desk-review corpus (ADR-0004 / ADR-0005).

The corpus holds only the latest daily + latest weekly review per user, so an
upload *replaces* the prior document of that type (replace-on-upload), and a
`user_id` payload filter scopes every read — the demo user's synthetic reviews
must never surface in the real trader's answers.

The embedder is an injected seam (like `quotes.fetch`): production wraps OpenAI
`text-embedding-3-large`; tests pass a deterministic fake and an in-memory
Qdrant client, so the suite never touches the network.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Protocol, Sequence

from qdrant_client import QdrantClient
from qdrant_client import models

from app.rag.chunk import Chunk

COLLECTION = "desk_reviews"
OPENAI_3_LARGE_DIM = 3072


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


def openai_embedder(model: str = "text-embedding-3-large") -> Embedder:
    """Production embedder (direct OpenAI). Satisfies ``Embedder`` as-is.

    Pair with ``vector_size=OPENAI_3_LARGE_DIM`` when building a CorpusIndex.
    """
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=model)


_POINT_NS = uuid.UUID("d5e6b3a2-1c4f-4a8b-9e2d-0f1a2b3c4d5e")  # stable id namespace


@dataclass(frozen=True)
class SearchHit:
    text: str
    doc_type: str
    review_date: str | None
    source: str
    chunk_id: str
    section: str | None
    pages: tuple[int, ...]
    score: float


@dataclass(frozen=True)
class IndexResult:
    doc_type: str
    review_date: str | None
    chunks_indexed: int
    replaced: bool
    warning: str | None


class CorpusIndex:
    """Thin wrapper over a Qdrant collection with per-user, per-doc-type scoping."""

    def __init__(
        self,
        client: QdrantClient,
        embedder: Embedder,
        *,
        vector_size: int,
        collection: str = COLLECTION,
    ) -> None:
        self._client = client
        self._embedder = embedder
        self._collection = collection
        if not client.collection_exists(collection):
            client.create_collection(
                collection_name=collection,
                vectors_config=models.VectorParams(
                    size=vector_size, distance=models.Distance.COSINE
                ),
            )

    def replace_document(self, chunks: Sequence[Chunk], *, user_id: str) -> IndexResult:
        """Replace this user's document of the incoming type.

        All chunks must share one ``doc_type``. Emits a warning if the upload's
        review date is older than the one it replaces — an explicit upload
        still wins, but the staleness is surfaced.
        """
        if not chunks:
            raise ValueError("replace_document requires at least one chunk")
        doc_type = chunks[0].doc_type
        if any(c.doc_type != doc_type for c in chunks):
            raise ValueError("all chunks must share one doc_type")
        review_date = chunks[0].review_date

        prior_date = self._current_review_date(user_id, doc_type)
        replaced = prior_date is not None
        warning = None
        if prior_date is not None and review_date is not None and review_date < prior_date:
            warning = (
                f"uploaded {doc_type} review is dated {review_date}, older than the "
                f"current {prior_date}; replaced anyway"
            )

        self._delete(user_id, doc_type)
        self._client.upsert(
            collection_name=self._collection, points=self._points(user_id, chunks)
        )

        return IndexResult(
            doc_type=doc_type,
            review_date=review_date,
            chunks_indexed=len(chunks),
            replaced=replaced,
            warning=warning,
        )

    def search(self, query: str, *, user_id: str, k: int = 5) -> list[SearchHit]:
        """Dense top-``k`` search over only this user's chunks."""
        vector = self._embedder.embed_query(query)
        result = self._client.query_points(
            collection_name=self._collection,
            query=vector,
            query_filter=self._user_filter(user_id),
            limit=k,
            with_payload=True,
        )
        return [_to_hit(p.payload, p.score) for p in result.points]

    def all_chunks(self, *, user_id: str) -> list[SearchHit]:
        """Every chunk for a user (score 0.0) — the BM25 corpus source."""
        docs: list[SearchHit] = []
        offset = None
        while True:
            points, offset = self._client.scroll(
                collection_name=self._collection,
                scroll_filter=self._user_filter(user_id),
                limit=256,
                offset=offset,
                with_payload=True,
            )
            docs.extend(_to_hit(p.payload, 0.0) for p in points)
            if offset is None:
                break
        return docs

    # -- internals --------------------------------------------------------

    def _points(
        self, user_id: str, chunks: Sequence[Chunk]
    ) -> list[models.PointStruct]:
        vectors = self._embedder.embed_documents([c.text for c in chunks])
        return [
            models.PointStruct(
                id=str(uuid.uuid5(_POINT_NS, f"{user_id}:{c.chunk_id}")),
                vector=vector,
                payload={
                    "user_id": user_id,
                    "text": c.text,
                    "doc_type": c.doc_type,
                    "review_date": c.review_date,
                    "source": c.source,
                    "chunk_id": c.chunk_id,
                    "section": c.section,
                    "pages": list(c.pages),
                },
            )
            for c, vector in zip(chunks, vectors)
        ]

    def _user_filter(
        self, user_id: str, doc_type: str | None = None
    ) -> models.Filter:
        must = [
            models.FieldCondition(
                key="user_id", match=models.MatchValue(value=user_id)
            )
        ]
        if doc_type is not None:
            must.append(
                models.FieldCondition(
                    key="doc_type", match=models.MatchValue(value=doc_type)
                )
            )
        return models.Filter(must=must)

    def _current_review_date(self, user_id: str, doc_type: str) -> str | None:
        points, _ = self._client.scroll(
            collection_name=self._collection,
            scroll_filter=self._user_filter(user_id, doc_type),
            limit=1,
            with_payload=True,
        )
        return points[0].payload.get("review_date") if points else None

    def _delete(self, user_id: str, doc_type: str) -> None:
        self._client.delete(
            collection_name=self._collection,
            points_selector=models.FilterSelector(
                filter=self._user_filter(user_id, doc_type)
            ),
        )


def _to_hit(payload: dict, score: float) -> SearchHit:
    return SearchHit(
        text=payload["text"],
        doc_type=payload["doc_type"],
        review_date=payload.get("review_date"),
        source=payload.get("source", ""),
        chunk_id=payload.get("chunk_id", ""),
        section=payload.get("section"),
        pages=tuple(payload.get("pages", [])),
        score=score,
    )
