from __future__ import annotations

import hashlib
import math
from typing import List

from sentence_transformers import SentenceTransformer


class BGEEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model = None
        self.dim = 384
        try:
            self.model = SentenceTransformer(model_name)
        except Exception:
            self.model = None

    def embed_texts(self, texts: List[str]) -> List[list[float]]:
        if self.model is None:
            return [self._fallback_embed(text) for text in texts]
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def _fallback_embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        tokens = [token for token in text.lower().split() if token]
        if not tokens:
            return vector
        for token in tokens:
            digest = hashlib.sha1(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self.dim
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]
