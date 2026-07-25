from dataclasses import dataclass
from typing import Any

from fastapi.testclient import TestClient

from qlora_lab.agent.app import create_app, requests_human_handoff
from qlora_lab.agent.conversations import ConversationStore
from qlora_lab.agent.runtime import SYSTEM_PROMPT
from qlora_lab.agent.settings import AgentSettings


@dataclass
class FakeMessage:
    content: str
    type: str = "ai"


class FakeRuntime:
    def invoke(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        assert messages[-1]["role"] == "user"
        return {"messages": [FakeMessage("订单 A100 已发货。")]}


def test_agent_endpoint_with_injected_runtime(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = AgentSettings(conversation_db_path=tmp_path / "conversations.sqlite3")
    app = create_app(FakeRuntime(), settings)

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


def test_agent_requires_user_message(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = AgentSettings(conversation_db_path=tmp_path / "conversations.sqlite3")
    app = create_app(FakeRuntime(), settings)

    with TestClient(app) as client:
        response = client.post(
            "/agent/invoke",
            json={"messages": [{"role": "assistant", "content": "hello"}]},
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_chat_persists_multi_turn_transcript(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store = ConversationStore(tmp_path / "conversations.sqlite3")
    settings = AgentSettings(conversation_db_path=store.path)
    app = create_app(FakeRuntime(), settings, store)

    with TestClient(app) as client:
        first = client.post("/agent/chat", json={"message": "查询 A100"})
        session_id = first.json()["session_id"]
        second = client.post(
            "/agent/chat",
            json={"session_id": session_id, "message": "再说一次"},
        )
        transcript = client.get(f"/agent/sessions/{session_id}")

    assert first.status_code == 200
    assert second.status_code == 200
    assert transcript.status_code == 200
    assert [item["role"] for item in transcript.json()["messages"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]


def test_explicit_handoff_is_deterministic(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store = ConversationStore(tmp_path / "conversations.sqlite3")
    app = create_app(
        FakeRuntime(),
        AgentSettings(conversation_db_path=store.path),
        store,
    )

    with TestClient(app) as client:
        response = client.post("/agent/chat", json={"message": "我要转人工客服"})
        session_id = response.json()["session_id"]
        follow_up = client.post(
            "/agent/chat",
            json={"session_id": session_id, "message": "我补充一条信息"},
        )

    assert response.status_code == 200
    assert response.json()["status"] == "waiting_human"
    assert follow_up.json()["status"] == "waiting_human"


def test_web_ui_is_served(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = AgentSettings(conversation_db_path=tmp_path / "conversations.sqlite3")
    app = create_app(FakeRuntime(), settings)

    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "澄心客服" in response.text
    assert "restoreSession" in response.text
    assert "/agent/sessions/${sessionId}" in response.text


def test_handoff_endpoint_is_saved_in_transcript(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store = ConversationStore(tmp_path / "conversations.sqlite3")
    session_id = store.ensure_session()
    app = create_app(
        FakeRuntime(),
        AgentSettings(conversation_db_path=store.path),
        store,
    )

    with TestClient(app) as client:
        handoff = client.post(
            f"/agent/sessions/{session_id}/handoff",
            json={"reason": "用户点击转人工"},
        )
        transcript = client.get(f"/agent/sessions/{session_id}")

    assert handoff.status_code == 200
    assert transcript.json()["messages"][-1]["content"].startswith("已申请转接")


def test_handoff_intent_requires_explicit_human_transfer_action() -> None:
    assert requests_human_handoff("我受够了，马上转人工客服")
    assert requests_human_handoff("请联系真人客服")
    assert not requests_human_handoff("我想了解人工客服政策")
    assert not requests_human_handoff("我不要转人工客服")


def test_system_prompt_defines_tone_and_topic_boundary() -> None:
    assert "非客服问题应礼貌拒绝" in SYSTEM_PROMPT
    assert "混合问题只处理客服部分" in SYSTEM_PROMPT
    assert "专业、耐心、克制" in SYSTEM_PROMPT
    assert "不承诺无法保证的结果" in SYSTEM_PROMPT
