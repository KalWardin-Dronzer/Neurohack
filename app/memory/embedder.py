from __future__ import annotations

import hashlib
import re
from typing import Sequence

import numpy as np


class EmbeddingModel:
    """Sentence embeddings for memories and queries.

    sentence-transformers is the default backend. The hashed bag-of-words backend exists for
    offline tests and must be chosen explicitly; v1 fell back to it silently whenever
    sentence-transformers was missing, which was the default install.
    """

    def __init__(
        self,
        backend: str = "sentence-transformers",
        model_name: str = "BAAI/bge-small-en-v1.5",
        dim: int = 256,
        query_prefix: str = "",
    ) -> None:
        self.backend = backend
        self.query_prefix = query_prefix
        if backend == "sentence-transformers":
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is not installed: run `pip install -r requirements.txt`, "
                    "or set EMBEDDING_BACKEND=hashed for an offline test setup"
                ) from exc
            self._model = SentenceTransformer(model_name)
            # sentence-transformers 5 renamed the method; keep working on 3.x and 4.x as well.
            dimension_of = getattr(self._model, "get_embedding_dimension", None) or self._model.get_sentence_embedding_dimension
            self.dim = int(dimension_of())
            self.model_id = f"st:{model_name}"
        elif backend == "hashed":
            self._model = None
            self.dim = dim
            self.model_id = f"hashed:{dim}"
        else:
            raise ValueError(f"unknown embedding backend {backend!r}; use 'sentence-transformers' or 'hashed'")

    def embed(self, text: str) -> np.ndarray:
        return self.embed_many([text])[0]

    def embed_query(self, text: str) -> np.ndarray:
        # Asymmetric models such as BGE expect an instruction prefix on queries, never on documents.
        prefix = self.query_prefix if self._model is not None else ""
        return self.embed(prefix + text)

    def embed_many(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if self._model is not None:
            vectors = self._model.encode(
                list(texts), normalize_embeddings=True, batch_size=64, show_progress_bar=False
            )
            return np.asarray(vectors, dtype=np.float32)
        return np.stack([self._hashed_embedding(text) for text in texts])

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
    def to_bytes(vector: np.ndarray) -> bytes:
        return np.asarray(vector, dtype=np.float32).tobytes()

    @staticmethod
    def from_bytes(payload: bytes, dim: int) -> np.ndarray:
        vector = np.frombuffer(payload, dtype=np.float32)
        if vector.shape[0] != dim:
            raise ValueError(f"stored vector has {vector.shape[0]} dimensions, expected {dim}")
        return vector


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    # v1 truncated mismatched vectors to a common length, which silently compared unrelated spaces.
    if left.shape != right.shape:
        raise ValueError(f"cannot compare vectors of shapes {left.shape} and {right.shape}")
    left_norm = np.linalg.norm(left)
    right_norm = np.linalg.norm(right)
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))
