from __future__ import annotations

import math
import threading
from typing import Any, Protocol

from qlora_lab.retrieval.settings import RetrievalSettings

EMBEDDING_MODEL_ID = "BAAI/bge-m3"
RERANK_MODEL_ID = "BAAI/bge-reranker-v2-m3"


class RetrievalBackend(Protocol):
    @property
    def loaded(self) -> bool: ...

    def load(self) -> None: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...

    def rerank(self, query: str, documents: list[str]) -> list[float]: ...

    def health(self) -> dict[str, Any]: ...


class SentenceTransformersBackend:
    """CPU-first backend with imports deferred until service startup."""

    def __init__(self, settings: RetrievalSettings) -> None:
        self.settings = settings
        self._embedding_model: Any | None = None
        self._reranker: Any | None = None
        self._load_lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._embedding_model is not None and self._reranker is not None

    def load(self) -> None:
        if self.loaded:
            return
        with self._load_lock:
            if self.loaded:
                return
            try:
                from sentence_transformers import CrossEncoder, SentenceTransformer
                from torch.nn import Sigmoid
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is required to run the retrieval model service."
                ) from exc

            # The model IDs, device, and remote-code policy are deliberately not
            # caller-controlled. Retrieval requests can never select local paths
            # or execute model-repository Python code.
            embedding_model = SentenceTransformer(
                EMBEDDING_MODEL_ID,
                device=self.settings.device,
                trust_remote_code=False,
            )
            reranker = CrossEncoder(
                RERANK_MODEL_ID,
                device=self.settings.device,
                trust_remote_code=False,
                activation_fn=Sigmoid(),
            )
            self._embedding_model = embedding_model
            self._reranker = reranker

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not self.loaded:
            raise RuntimeError("Retrieval models are not loaded.")
        vectors = self._embedding_model.encode(
            texts,
            batch_size=min(self.settings.embedding_batch_size, len(texts)),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        result = vectors.tolist()
        if any(len(vector) != self.settings.embedding_dimensions for vector in result):
            raise RuntimeError("Embedding model returned an unexpected vector dimension.")
        return result

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        if not self.loaded:
            raise RuntimeError("Retrieval models are not loaded.")
        pairs = [(query, document) for document in documents]
        scores = self._reranker.predict(
            pairs,
            batch_size=min(self.settings.rerank_batch_size, len(pairs)),
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        raw_scores = scores.tolist()
        if isinstance(raw_scores, float):
            return [raw_scores]
        return [float(score) for score in raw_scores]

    def health(self) -> dict[str, Any]:
        return {
            "status": "ready" if self.loaded else "starting",
            "models_loaded": self.loaded,
            "device": self.settings.device,
            "embedding_model": EMBEDDING_MODEL_ID,
            "rerank_model": RERANK_MODEL_ID,
            "embedding_dimensions": self.settings.embedding_dimensions,
        }


def validate_embeddings(
    vectors: list[list[float]],
    *,
    expected_count: int,
    expected_dimensions: int,
) -> None:
    if len(vectors) != expected_count:
        raise RuntimeError("Embedding backend returned an unexpected number of vectors.")
    if any(len(vector) != expected_dimensions for vector in vectors):
        raise RuntimeError("Embedding backend returned an unexpected vector dimension.")
    if any(not math.isfinite(value) for vector in vectors for value in vector):
        raise RuntimeError("Embedding backend returned a non-finite value.")


def validate_rerank_scores(scores: list[float], *, expected_count: int) -> None:
    if len(scores) != expected_count:
        raise RuntimeError("Rerank backend returned an unexpected number of scores.")
    if any(not math.isfinite(score) for score in scores):
        raise RuntimeError("Rerank backend returned a non-finite score.")
    if any(score < 0 or score > 1 for score in scores):
        raise RuntimeError("Rerank backend scores must be normalized to the [0, 1] range.")
