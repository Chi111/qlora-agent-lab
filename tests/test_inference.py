from typing import Any

import pytest
from fastapi.testclient import TestClient

from qlora_lab.inference.app import create_app
from qlora_lab.inference.generator import (
    GenerationResult,
    _model_context_window,
    parse_tool_calls,
)
from qlora_lab.inference.schemas import ChatCompletionRequest
from qlora_lab.inference.settings import InferenceSettings


class FakeGenerator:
    loaded = True

    def load(self) -> None:
        pass

    def generate(self, request: ChatCompletionRequest) -> GenerationResult:
        return GenerationResult(
            content=f"收到：{request.messages[-1].content}",
            tool_calls=[],
            prompt_tokens=5,
            completion_tokens=3,
        )

    def health(self) -> dict[str, Any]:
        return {"status": "ready", "model_loaded": True, "device": "fake"}


def test_parse_qwen_tool_call() -> None:
    content, calls = parse_tool_calls(
        '正在查询。\n<tool_call>{"name":"get_order","arguments":{"order_id":"A100"}}</tool_call>'
    )

    assert content == "正在查询。"
    assert calls[0]["function"]["name"] == "get_order"
    assert '"A100"' in calls[0]["function"]["arguments"]


@pytest.mark.parametrize("payload", ["null", "[]", '"get_order"', '{"function":[]}'])
def test_parse_tool_call_ignores_non_object_payload(payload: str) -> None:
    content, calls = parse_tool_calls(f"<tool_call>{payload}</tool_call>")

    assert content is None
    assert calls == []


def test_model_context_window_prefers_model_config() -> None:
    model = type("Model", (), {"config": type("Config", (), {"max_position_embeddings": 4096})()})
    tokenizer = type("Tokenizer", (), {"model_max_length": 8192})()

    assert _model_context_window(model, tokenizer) == 4096


def test_openai_compatible_completion() -> None:
    settings = InferenceSettings(model_name="local-qlora")
    app = create_app(FakeGenerator(), settings)

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "local-qlora",
                "messages": [{"role": "user", "content": "你好"}],
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "收到：你好"
    assert body["usage"]["total_tokens"] == 8
    assert response.headers["X-Request-ID"]


def test_streaming_is_rejected_with_structured_error() -> None:
    app = create_app(FakeGenerator(), InferenceSettings())

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "你好"}],
                "stream": True,
            },
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_unknown_model_is_rejected() -> None:
    app = create_app(FakeGenerator(), InferenceSettings(model_name="expected"))

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "arbitrary-path",
                "messages": [{"role": "user", "content": "你好"}],
            },
        )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "MODEL_NOT_FOUND"


def test_serialized_tool_schema_counts_toward_request_limit() -> None:
    app = create_app(FakeGenerator(), InferenceSettings(max_input_chars=1000))
    large_description = "x" * 1200

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "你好"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "large_tool",
                            "description": large_description,
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            },
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INPUT_TOO_LARGE"
