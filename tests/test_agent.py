from dataclasses import dataclass
from typing import Any

from fastapi.testclient import TestClient

from qlora_lab.agent.app import create_app
from qlora_lab.agent.settings import AgentSettings


@dataclass
class FakeMessage:
    content: str
    type: str = "ai"


class FakeRuntime:
    def invoke(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        assert messages[-1]["role"] == "user"
        return {"messages": [FakeMessage("订单 A100 已发货。")]}


def test_agent_endpoint_with_injected_runtime() -> None:
    app = create_app(FakeRuntime(), AgentSettings())

    with TestClient(app) as client:
        response = client.post(
            "/agent/invoke",
            json={
                "messages": [{"role": "user", "content": "查询 A100"}],
                "debug": True,
            },
        )

    assert response.status_code == 200
    assert response.json()["content"] == "订单 A100 已发货。"
    assert response.json()["trace"][0]["type"] == "ai"


def test_agent_requires_user_message() -> None:
    app = create_app(FakeRuntime(), AgentSettings())

    with TestClient(app) as client:
        response = client.post(
            "/agent/invoke",
            json={"messages": [{"role": "assistant", "content": "hello"}]},
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"
