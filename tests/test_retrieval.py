from __future__ import annotations

import math
import time
from typing import Any

from fastapi.testclient import TestClient

from qlora_lab.retrieval.app import create_app
from qlora_lab.retrieval.backend import EMBEDDING_MODEL_ID, RERANK_MODEL_ID
from qlora_lab.retrieval.settings import RetrievalSettings


class FakeRetrievalBackend:
    loaded = True

    def load(self) -> None:
        pass

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text))] + [0.0] * 1023 for text in texts]

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        return [min(float(document.count(query)) / 2, 1.0) for document in documents]

    def health(self) -> dict[str, Any]:
        return {"status": "ready", "models_loaded": True, "device": "fake"}


class SlowRetrievalBackend(FakeRetrievalBackend):
    def embed(self, texts: list[str]) -> list[list[float]]:
        time.sleep(0.08)
        return super().embed(texts)


class InvalidRetrievalBackend(FakeRetrievalBackend):
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[math.nan] * 1024 for _ in texts]


def test_health_uses_injected_backend_without_model_download() -> None:
    app = create_app(FakeRetrievalBackend(), RetrievalSettings())

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "models_loaded": True,
        "device": "fake",
    }
    assert response.headers["X-Request-ID"]


def test_openai_compatible_embeddings() -> None:
    app = create_app(FakeRetrievalBackend(), RetrievalSettings())

    with TestClient(app) as client:
        response = client.post(
            "/v1/embeddings",
            json={
                "model": EMBEDDING_MODEL_ID,
                "input": ["冰箱不制冷", "显示器黑屏"],
                "encoding_format": "float",
                "dimensions": 1024,
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert body["model"] == EMBEDDING_MODEL_ID
    assert [item["index"] for item in body["data"]] == [0, 1]
    assert all(len(item["embedding"]) == 1024 for item in body["data"])
    assert body["usage"]["total_tokens"] == body["usage"]["prompt_tokens"]


def test_rerank_orders_scores_and_optionally_returns_documents() -> None:
    app = create_app(FakeRetrievalBackend(), RetrievalSettings())

    with TestClient(app) as client:
        response = client.post(
            "/v1/rerank",
            json={
                "model": RERANK_MODEL_ID,
                "query": "冰箱",
                "documents": ["电视无声", "冰箱冰箱不制冷", "冰箱结霜"],
                "top_n": 2,
                "return_documents": True,
            },
        )

    assert response.status_code == 200
    results = response.json()["results"]
    assert [item["index"] for item in results] == [1, 2]
    assert results[0]["relevance_score"] == 1.0
    assert results[0]["document"]["text"] == "冰箱冰箱不制冷"


def test_model_ids_are_fixed() -> None:
    app = create_app(FakeRetrievalBackend(), RetrievalSettings())

    with TestClient(app) as client:
        embedding_response = client.post(
            "/v1/embeddings",
            json={"model": "local/path", "input": "test"},
        )
        rerank_response = client.post(
            "/v1/rerank",
            json={"model": "other-reranker", "query": "q", "documents": ["d"]},
        )

    assert embedding_response.status_code == 404
    assert embedding_response.json()["error"]["code"] == "MODEL_NOT_FOUND"
    assert rerank_response.status_code == 404
    assert rerank_response.json()["error"]["code"] == "MODEL_NOT_FOUND"


def test_embedding_batch_and_total_length_are_bounded() -> None:
    settings = RetrievalSettings(
        max_embedding_batch=2,
        max_embedding_text_chars=100,
        max_embedding_total_chars=100,
    )
    app = create_app(FakeRetrievalBackend(), settings)

    with TestClient(app) as client:
        batch_response = client.post(
            "/v1/embeddings",
            json={"model": EMBEDDING_MODEL_ID, "input": ["a", "b", "c"]},
        )
        total_response = client.post(
            "/v1/embeddings",
            json={"model": EMBEDDING_MODEL_ID, "input": ["a" * 60, "b" * 60]},
        )

    assert batch_response.status_code == 400
    assert batch_response.json()["error"]["code"] == "BATCH_TOO_LARGE"
    assert total_response.status_code == 400
    assert total_response.json()["error"]["code"] == "INPUT_TOO_LARGE"


def test_rerank_rejects_blank_and_invalid_top_n() -> None:
    app = create_app(FakeRetrievalBackend(), RetrievalSettings())

    with TestClient(app) as client:
        blank_response = client.post(
            "/v1/rerank",
            json={"model": RERANK_MODEL_ID, "query": " ", "documents": ["document"]},
        )
        top_n_response = client.post(
            "/v1/rerank",
            json={
                "model": RERANK_MODEL_ID,
                "query": "query",
                "documents": ["document"],
                "top_n": 2,
            },
        )

    assert blank_response.status_code == 400
    assert blank_response.json()["error"]["code"] == "INPUT_EMPTY"
    assert top_n_response.status_code == 422
    assert top_n_response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_schema_forbids_unknown_fields() -> None:
    app = create_app(FakeRetrievalBackend(), RetrievalSettings())

    with TestClient(app) as client:
        response = client.post(
            "/v1/embeddings",
            json={
                "model": EMBEDDING_MODEL_ID,
                "input": "test",
                "user_supplied_path": "/tmp/model",
            },
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_operation_timeout_and_queue_capacity_are_enforced() -> None:
    settings = RetrievalSettings(
        max_concurrency=1,
        queue_size=0,
        operation_timeout_seconds=0.01,
    )
    app = create_app(SlowRetrievalBackend(), settings)

    with TestClient(app) as client:
        timeout_response = client.post(
            "/v1/embeddings",
            json={"model": EMBEDDING_MODEL_ID, "input": "slow"},
        )
        busy_response = client.post(
            "/v1/embeddings",
            json={"model": EMBEDDING_MODEL_ID, "input": "still busy"},
        )

    assert timeout_response.status_code == 504
    assert timeout_response.json()["error"]["code"] == "MODEL_TIMEOUT"
    assert timeout_response.json()["error"]["retryable"] is True
    assert busy_response.status_code == 429
    assert busy_response.json()["error"]["code"] == "MODEL_BUSY"


def test_backend_output_is_validated() -> None:
    app = create_app(InvalidRetrievalBackend(), RetrievalSettings())

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/v1/embeddings",
            json={"model": EMBEDDING_MODEL_ID, "input": "test"},
        )

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "INTERNAL_ERROR"
