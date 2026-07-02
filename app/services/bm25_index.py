from __future__ import annotations
from typing import Dict, List, Tuple

from rank_bm25 import BM25Okapi

from app.models.schemas import Chunk


class BM25Index:
    def __init__(self):
        self._indices: Dict[str, BM25Okapi] = {}
        self._corpora: Dict[str, List[List[str]]] = {}
        self._chunks: Dict[str, List[Chunk]] = {}

    def build_index(self, session_id: str, chunks: List[Chunk]) -> None:
        corpus = [c.text.lower().split() for c in chunks]
        bm25 = BM25Okapi(corpus)

        self._indices[session_id] = bm25
        self._corpora[session_id] = corpus
        self._chunks[session_id] = chunks

    def search(
        self,
        session_id: str,
        query: str,
        k: int = 10,
    ) -> List[Tuple[Chunk, float]]:
        bm25 = self._indices[session_id]
        chunks = self._chunks[session_id]

        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)

        ranked_idx = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:k]

        return [(chunks[i], float(scores[i])) for i in ranked_idx]

    def clear(self, session_id: str) -> None:
        self._indices.pop(session_id, None)
        self._corpora.pop(session_id, None)
        self._chunks.pop(session_id, None)
