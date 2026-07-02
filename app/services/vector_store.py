from pathlib import Path
from typing import List
from threading import RLock
import shutil

import chromadb
from chromadb.config import Settings

from app.models.schemas import Chunk
from app.config import CHROMA_DIR


class VectorStore:
    def __init__(self, base_dir: Path = CHROMA_DIR):
        self.base_dir = base_dir
        self._lock = RLock()
        self.client = self._create_client()

    def _create_client(self):
        return chromadb.PersistentClient(
            path=str(self.base_dir),
            settings=Settings(anonymized_telemetry=False),
        )

    def _rebuild_storage(self) -> None:
        try:
            if self.client is not None:
                self.client.close()
        except Exception:
            pass

        try:
            shutil.rmtree(self.base_dir, ignore_errors=True)
        except Exception:
            pass

        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.client = self._create_client()

    def _collection_name(self, session_id: str) -> str:
        return f"session_{session_id}"

    def _clean_metadata(self, metadata: dict) -> dict:
        cleaned = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                cleaned[key] = value
            else:
                cleaned[key] = str(value)
        return cleaned

    def get_session_collection(self, session_id: str):
        with self._lock:
            if self.client is None:
                self.client = self._create_client()
            name = self._collection_name(session_id)
            try:
                return self.client.get_or_create_collection(
                    name=name,
                    metadata={"session_id": str(session_id)},
                )
            except Exception:
                self._rebuild_storage()
                return self.client.get_or_create_collection(
                    name=name,
                    metadata={"session_id": str(session_id)},
                )

    def index_chunks(
        self,
        session_id: str,
        chunks: List[Chunk],
        embeddings: List[list[float]],
    ) -> None:
        ids = [c.chunk_id for c in chunks]
        metadatas = [
            self._clean_metadata(
                {
                    "session_id": c.session_id,
                    "document_id": c.document_id,
                    "page_no": c.page_no,
                    "section": c.section,
                }
            )
            for c in chunks
        ]
        texts = [c.text for c in chunks]

        with self._lock:
            collection = self.get_session_collection(session_id)
            try:
                collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    documents=texts,
                )
            except Exception:
                self._rebuild_storage()
                collection = self.get_session_collection(session_id)
                collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    documents=texts,
                )

    def reset_storage(self) -> None:
        with self._lock:
            self._rebuild_storage()
