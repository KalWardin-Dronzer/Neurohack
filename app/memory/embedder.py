from __future__ import annotations

import hashlib
import json
import re
from typing import Optional

import numpy as np


class EmbeddingModel:
    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self._st_model = self._load_sentence_transformer()

    @staticmethod
    def _load_sentence_transformer() -> Optional[object]:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        except Exception:
            return None

    def embed(self, text: str) -> np.ndarray:
        text = text.strip()
        if not text:
            return np.zeros(self.dim, dtype=np.float32)

        if self._st_model is not None:
            try:
                vector = self._st_model.encode(text, normalize_embeddings=True)
                return np.asarray(vector, dtype=np.float32)
            except Exception:
                pass

        return self._hashed_embedding(text)

    def _hashed_embedding(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = re.findall(r"[a-zA-Z0-9_]+", text.lower())
        if not tokens:
            return vec

        for token in tokens:
            idx = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % self.dim
            vec[idx] += 1.0

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    @staticmethod
    def to_payload(vector: np.ndarray) -> str:
        return json.dumps(vector.astype(float).tolist())

    @staticmethod
    def from_payload(payload: str) -> np.ndarray:
        return np.asarray(json.loads(payload), dtype=np.float32)
