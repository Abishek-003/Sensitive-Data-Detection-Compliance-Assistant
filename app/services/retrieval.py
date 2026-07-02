from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from app.models.schemas import Chunk
from app.services.embeddings import BGEEmbedder
from app.services.vector_store import VectorStore
from app.services.bm25_index import BM25Index


@dataclass
class SearchHit:
    chunk: Chunk
    vector_rank: int | None = None
    bm25_rank: int | None = None
    vector_score: float | None = None
    bm25_score: float | None = None
    rrf_score: float = 0.0
    rerank_score: float = 0.0


def _to_chunk(doc_id: str, text: str, metadata: dict) -> Chunk:
    return Chunk(
        chunk_id=str(metadata.get("chunk_id") or doc_id),
        session_id=str(metadata.get("session_id", "")),
        document_id=str(metadata.get("document_id", "")),
        page_no=int(metadata.get("page_no", 0) or 0),
        text=text,
        section=metadata.get("section"),
    )


def reciprocal_rank_fusion(
    vector_hits: list[Chunk],
    bm25_hits: list[tuple[Chunk, float]],
    c: int = 60,
) -> list[SearchHit]:
    fused: dict[str, SearchHit] = {}

    for rank, chunk in enumerate(vector_hits, start=1):
        hit = fused.setdefault(chunk.chunk_id, SearchHit(chunk=chunk))
        hit.vector_rank = rank
        hit.rrf_score += 1.0 / (c + rank)

    for rank, (chunk, score) in enumerate(bm25_hits, start=1):
        hit = fused.setdefault(chunk.chunk_id, SearchHit(chunk=chunk))
        hit.bm25_rank = rank
        hit.bm25_score = score
        hit.rrf_score += 1.0 / (c + rank)

    return sorted(fused.values(), key=lambda hit: hit.rrf_score, reverse=True)


class HybridRetriever:
    def __init__(
        self,
        embedder: BGEEmbedder,
        vector_store: VectorStore,
        bm25_index: BM25Index,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25_index = bm25_index

    def vector_search(
        self,
        session_id: str,
        query: str,
        k: int = 5,
        document_id: str | None = None,
    ) -> list[Chunk]:
        collection = self.vector_store.get_session_collection(session_id)
        query_embedding = self.embedder.embed_texts([query])[0]
        where = {"document_id": document_id} if document_id else None
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        ids = result.get("ids", [[]])[0]
        return [
            _to_chunk(doc_id, doc, {**meta, "chunk_id": doc_id})
            for doc_id, doc, meta in zip(ids, docs, metas)
        ]

    def bm25_search(
        self,
        session_id: str,
        query: str,
        k: int = 5,
        document_id: str | None = None,
    ) -> list[tuple[Chunk, float]]:
        try:
            hits = self.bm25_index.search(session_id, query, k=k)
            if document_id:
                hits = [(chunk, score) for chunk, score in hits if chunk.document_id == document_id]
            return hits
        except KeyError:
            return []

    def _rerank_score(self, query: str, text: str) -> float:
        query_terms = {token for token in query.lower().split() if token}
        text_terms = {token for token in text.lower().split() if token}
        if not query_terms:
            return 0.0
        overlap = len(query_terms & text_terms)
        coverage = overlap / max(len(query_terms), 1)
        density = overlap / max(len(text_terms), 1)
        return round((coverage * 0.7) + (density * 0.3), 4)

    def search(
        self,
        session_id: str,
        query: str,
        k: int = 5,
        candidate_pool: int = 10,
        document_id: str | None = None,
    ) -> list[SearchHit]:
        vector_hits = self.vector_search(session_id, query, k=candidate_pool, document_id=document_id)
        bm25_hits = self.bm25_search(session_id, query, k=candidate_pool, document_id=document_id)
        fused = reciprocal_rank_fusion(vector_hits, bm25_hits)

        for hit in fused:
            hit.rerank_score = self._rerank_score(query, hit.chunk.text)

        fused.sort(key=lambda hit: (hit.rerank_score, hit.rrf_score), reverse=True)
        return fused[:k]


def format_context(hits: Iterable[SearchHit]) -> str:
    parts = []
    for i, hit in enumerate(hits, start=1):
        parts.append(
            f"[{i}] doc={hit.chunk.document_id} page={hit.chunk.page_no} chunk={hit.chunk.chunk_id}\n{hit.chunk.text}"
        )
    return "\n\n".join(parts)
