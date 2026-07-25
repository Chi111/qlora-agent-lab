import json
import time
from typing import Any

import httpx

from qlora_lab.agent.tools import (
    TICKET_WRITE_AUTHORIZATION,
    TICKET_WRITE_DEADLINE,
    MockBackendClient,
    TicketAuthorization,
    build_tools,
    ticket_creation_authorization,
)


class RecordingClient:
    def __init__(self) -> None:
        self.created: list[dict[str, str]] = []

    def get_order(self, order_id: str) -> dict[str, Any]:
        return {"ok": True, "data": {"order_id": order_id}}

    def create_ticket(self, order_id: str, reason: str, priority: str) -> dict[str, Any]:
        self.created.append(
            {"order_id": order_id, "reason": reason, "priority": priority}
        )
        return {"ok": True, "data": {"ticket_id": "T1000"}}


def test_ticket_write_requires_programmatic_authorization() -> None:
    client = RecordingClient()
    ticket_tool = next(tool for tool in build_tools(client) if tool.name == "create_ticket")
    arguments = {"order_id": "A101", "reason": "订单延迟", "priority": "normal"}

    denied = json.loads(ticket_tool.invoke(arguments))
    authorization_token = TICKET_WRITE_AUTHORIZATION.set(
        TicketAuthorization("A101", "normal")
    )
    deadline_token = TICKET_WRITE_DEADLINE.set(time.monotonic() + 10)
    try:
        allowed = json.loads(ticket_tool.invoke(arguments))
    finally:
        TICKET_WRITE_DEADLINE.reset(deadline_token)
        TICKET_WRITE_AUTHORIZATION.reset(authorization_token)

    assert denied["error"]["code"] == "WRITE_NOT_AUTHORIZED"
    assert allowed["ok"] is True
    assert len(client.created) == 1


def test_ticket_authorization_binds_explicit_and_follow_up_consent() -> None:
    assert ticket_creation_authorization(
        [{"role": "user", "content": "A101 延迟了，帮我建一个高优先级工单"}]
    ) == TicketAuthorization("A101", "high")
    assert ticket_creation_authorization(
        [
            {"role": "assistant", "content": "订单 A101 延迟，需要我为你创建工单吗？"},
            {"role": "user", "content": "需要，普通优先级就行"},
        ]
    ) == TicketAuthorization("A101", "normal")
    assert ticket_creation_authorization(
        [{"role": "user", "content": "不要帮我创建工单"}]
    ) is None
    assert ticket_creation_authorization(
        [{"role": "user", "content": "帮我创建工单"}]
    ) is None
    assert ticket_creation_authorization(
        [
            {"role": "assistant", "content": "订单 A101 延迟，需要我为你创建工单吗？"},
            {"role": "user", "content": "是什么工单？"},
        ]
    ) is None


def test_ticket_write_rejects_parameter_mismatch_and_expired_deadline() -> None:
    client = RecordingClient()
    ticket_tool = next(tool for tool in build_tools(client) if tool.name == "create_ticket")
    authorization_token = TICKET_WRITE_AUTHORIZATION.set(
        TicketAuthorization("A101", "high")
    )
    deadline_token = TICKET_WRITE_DEADLINE.set(time.monotonic() + 10)
    try:
        wrong_order = json.loads(
            ticket_tool.invoke(
                {"order_id": "A102", "reason": "订单延迟", "priority": "high"}
            )
        )
        wrong_priority = json.loads(
            ticket_tool.invoke(
                {"order_id": "A101", "reason": "订单延迟", "priority": "normal"}
            )
        )
        expired_token = TICKET_WRITE_DEADLINE.set(time.monotonic() - 1)
        try:
            expired = json.loads(
                ticket_tool.invoke(
                    {"order_id": "A101", "reason": "订单延迟", "priority": "high"}
                )
            )
        finally:
            TICKET_WRITE_DEADLINE.reset(expired_token)
    finally:
        TICKET_WRITE_DEADLINE.reset(deadline_token)
        TICKET_WRITE_AUTHORIZATION.reset(authorization_token)

    assert wrong_order["error"]["code"] == "WRITE_PARAMETERS_NOT_AUTHORIZED"
    assert wrong_priority["error"]["code"] == "WRITE_PARAMETERS_NOT_AUTHORIZED"
    assert expired["error"]["code"] == "WRITE_DEADLINE_EXCEEDED"
    assert client.created == []


def test_order_tool_minimizes_personal_data(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    class FakeResponse:
        is_success = True
        status_code = 200

        @staticmethod
        def json() -> dict[str, Any]:
            return {
                "order_id": "A100",
                "status": "shipped",
                "item": "机械键盘",
                "customer_name": "张三",
                "amount_cny": 499,
            }

    monkeypatch.setattr(httpx, "request", lambda *args, **kwargs: FakeResponse())

    result = MockBackendClient("http://mock", 1).get_order("A100")

    assert result["ok"] is True
    assert "customer_name" not in result["data"]
    assert "amount_cny" not in result["data"]
