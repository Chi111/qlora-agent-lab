from __future__ import annotations

import argparse
import asyncio
import threading
import uuid
from collections.abc import AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, TypeVar

from fastapi import FastAPI

from qlora_lab.common.errors import AppError
from qlora_lab.common.http import install_http_support
from qlora_lab.common.logging import configure_logging
from qlora_lab.retrieval.backend import (
    EMBEDDING_MODEL_ID,
    RERANK_MODEL_ID,
    RetrievalBackend,
    SentenceTransformersBackend,
    validate_embeddings,
    validate_rerank_scores,
)
from qlora_lab.retrieval.schemas import EmbeddingsRequest, RerankRequest
from qlora_lab.retrieval.settings import RetrievalSettings

ResultT = TypeVar("ResultT")


class RetrievalGate:
    """Bound both active CPU work and queued work without blocking the event loop."""

    def __init__(self, max_concurrency: int, queue_size: int, timeout_seconds: float) -> None:
        self._capacity = max_concurrency + queue_size
        self._timeout_seconds = timeout_seconds
        self._executor = ThreadPoolExecutor(
            max_workers=max_concurrency,
            thread_name_prefix="retrieval-model",
        )
        self._in_flight = 0
        self._counter_lock = threading.Lock()

    async def run(self, function: Callable[..., ResultT], *args: Any) -> ResultT:
        with self._counter_lock:
            if self._in_flight >= self._capacity:
                raise AppError(
                    "MODEL_BUSY",
                    "Retrieval model queue is full.",
                    429,
                    True,
                )
            self._in_flight += 1

        try:
            native_future = self._executor.submit(function, *args)
        except Exception:
            self._release_slot()
            raise
        native_future.add_done_callback(lambda _: self._release_slot())
        future = asyncio.wrap_future(native_future)
        try:
            return await asyncio.wait_for(
                asyncio.shield(future),
                timeout=self._timeout_seconds,
            )
        except TimeoutError as exc:
            native_future.cancel()
            raise AppError(
                "MODEL_TIMEOUT",
                "Retrieval model operation timed out.",
                504,
                True,
            ) from exc

    def _release_slot(self) -> None:
        with self._counter_lock:
            self._in_flight -= 1

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)


def _reject_blank_or_oversized_texts(
    texts: list[str],
    *,
    max_batch: int,
    max_text_chars: int,
    max_total_chars: int,
    subject: str,
) -> None:
    if len(texts) > max_batch:
        raise AppError(
            "BATCH_TOO_LARGE",
            f"{subject} batch exceeds the limit of {max_batch}.",
            400,
            False,
        )
    if any(not text.strip() for text in texts):
        raise AppError("INPUT_EMPTY", f"{subject} text cannot be blank.", 400, False)
    if any(len(text) > max_text_chars for text in texts):
        raise AppError(
            "INPUT_TOO_LARGE",
            f"One {subject.lower()} text exceeds {max_text_chars} characters.",
            400,
            False,
        )
    if sum(len(text) for text in texts) > max_total_chars:
        raise AppError(
            "INPUT_TOO_LARGE",
            f"Total {subject.lower()} text exceeds {max_total_chars} characters.",
            400,
            False,
        )


def _estimated_tokens(texts: list[str]) -> int:
    return sum(max(1, (len(text) + 3) // 4) for text in texts)


def create_app(
    backend: RetrievalBackend | None = None,
    settings: RetrievalSettings | None = None,
    *,
    load_on_startup: bool = True,
) -> FastAPI:
    settings = settings or RetrievalSettings()
    backend = backend or SentenceTransformersBackend(settings)
    gate = RetrievalGate(
        settings.max_concurrency,
        settings.queue_size,
        settings.operation_timeout_seconds,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            if load_on_startup and not backend.loaded:
                await asyncio.to_thread(backend.load)
            yield
        finally:
            gate.close()

    app = FastAPI(
        title="QLoRA Lab Retrieval Model API",
        version="0.1.0",
        lifespan=lifespan,
    )
    install_http_support(app)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return backend.health()

    @app.post("/v1/embeddings")
    async def embeddings(request: EmbeddingsRequest) -> dict[str, Any]:
        if request.model != EMBEDDING_MODEL_ID:
            raise AppError("MODEL_NOT_FOUND", f"Unknown model: {request.model}", 404, False)
        if not backend.loaded:
            raise AppError("MODEL_NOT_READY", "Retrieval models are not loaded.", 503, True)
        if request.dimensions not in {None, settings.embedding_dimensions}:
            raise AppError(
                "DIMENSIONS_NOT_SUPPORTED",
                f"{EMBEDDING_MODEL_ID} only supports {settings.embedding_dimensions} dimensions.",
                400,
                False,
            )
        texts = request.texts
        _reject_blank_or_oversized_texts(
            texts,
            max_batch=settings.max_embedding_batch,
            max_text_chars=settings.max_embedding_text_chars,
            max_total_chars=settings.max_embedding_total_chars,
            subject="Embedding",
        )
        vectors = await gate.run(backend.embed, texts)
        validate_embeddings(
            vectors,
            expected_count=len(texts),
            expected_dimensions=settings.embedding_dimensions,
        )
        prompt_tokens = _estimated_tokens(texts)
        return {
            "object": "list",
            "data": [
                {"object": "embedding", "index": index, "embedding": vector}
                for index, vector in enumerate(vectors)
            ],
            "model": EMBEDDING_MODEL_ID,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "total_tokens": prompt_tokens,
            },
        }

    @app.post("/v1/rerank")
    async def rerank(request: RerankRequest) -> dict[str, Any]:
        if request.model != RERANK_MODEL_ID:
            raise AppError("MODEL_NOT_FOUND", f"Unknown model: {request.model}", 404, False)
        if not backend.loaded:
            raise AppError("MODEL_NOT_READY", "Retrieval models are not loaded.", 503, True)
        if len(request.documents) > settings.max_rerank_documents:
            raise AppError(
                "BATCH_TOO_LARGE",
                f"Rerank batch exceeds the limit of {settings.max_rerank_documents}.",
                400,
                False,
            )
        _reject_blank_or_oversized_texts(
            [request.query],
            max_batch=1,
            max_text_chars=settings.max_rerank_query_chars,
            max_total_chars=settings.max_rerank_query_chars,
            subject="Rerank query",
        )
        _reject_blank_or_oversized_texts(
            request.documents,
            max_batch=settings.max_rerank_documents,
            max_text_chars=settings.max_rerank_document_chars,
            max_total_chars=settings.max_rerank_total_chars,
            subject="Rerank document",
        )
        scores = await gate.run(backend.rerank, request.query, request.documents)
        validate_rerank_scores(scores, expected_count=len(request.documents))
        ranked = sorted(enumerate(scores), key=lambda item: (-item[1], item[0]))
        top_n = request.top_n or len(ranked)
        results: list[dict[str, Any]] = []
        for index, score in ranked[:top_n]:
            item: dict[str, Any] = {
                "index": index,
                "relevance_score": score,
            }
            if request.return_documents:
                item["document"] = {"text": request.documents[index]}
            results.append(item)
        return {
            "id": f"rerank-{uuid.uuid4().hex}",
            "model": RERANK_MODEL_ID,
            "results": results,
        }

    return app


app = create_app()


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Run the local embedding and rerank service.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8003)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(
        "qlora_lab.retrieval.app:app",
        host=args.host,
        port=args.port,
        workers=1,
    )


if __name__ == "__main__":
    main()
