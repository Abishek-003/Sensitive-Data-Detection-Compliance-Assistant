from typing import List, Tuple

from app.models.schemas import ParsedDocument, Chunk
from app.services.chunking import chunk_parsed_document
from app.services.embeddings import BGEEmbedder
from app.services.vector_store import VectorStore
from app.services.bm25_index import BM25Index


class RAGIndexer:
    def __init__(
        self,
        embedder: BGEEmbedder,
        vector_store: VectorStore,
        bm25_index: BM25Index,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25_index = bm25_index

    def index_parsed_document(
        self,
        session_id: str,
        parsed: ParsedDocument,
        max_chars: int = 1000,
        overlap_paragraphs: int = 1,
    ) -> Tuple[List[Chunk], int]:
        chunks = chunk_parsed_document(
            parsed,
            session_id=session_id,
            max_chars=max_chars,
            overlap_paragraphs=overlap_paragraphs,
        )

        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed_texts(texts)

        self.vector_store.index_chunks(session_id, chunks, embeddings)

        self.bm25_index.build_index(session_id, chunks)

        return chunks, len(chunks)
