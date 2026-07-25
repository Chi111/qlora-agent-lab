from __future__ import annotations

import hashlib
import json
import re
import time
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field

from qlora_lab.agent.knowledge import KnowledgeBase

ORDER_ID_PATTERN = re.compile(r"\b(?=[A-Za-z0-9_-]*\d)[A-Za-z][A-Za-z0-9_-]{2,63}\b")


@dataclass(frozen=True, slots=True)
class TicketAuthorization:
    order_id: str
    priority: Literal["low", "normal", "high"]


TICKET_WRITE_AUTHORIZATION: ContextVar[TicketAuthorization | None] = ContextVar(
    "ticket_write_authorized",
    default=None,
)
TICKET_WRITE_DEADLINE: ContextVar[float | None] = ContextVar(
    "ticket_write_deadline",
    default=None,
)
TICKET_ACTION_PATTERN = re.compile(
    r"(创建|新建|提交|开|建).{0,16}(工单|售后单)|(工单|售后单).{0,16}(创建|新建|提交|开|建)"
)
TICKET_NEGATION_PATTERN = re.compile(
    r"(不|不要|不用|无需|别|取消).{0,12}(创建|新建|提交|开|建)?.{0,8}(工单|售后单)"
)
CONFIRMATION_PATTERN = re.compile(
    r"^(需要|同意|可以|确认|好的|好|是|行|麻烦了)"
    r"([，, ]*(低|普通|正常|高)(优先级)?(就行|即可)?)?[。！! ]*$"
)
TICKET_OFFER_PATTERN = re.compile(
    r"(需要|是否|要不要).{0,16}(创建|新建|提交|开|建).{0,8}(工单|售后单)"
)


def _latest_order_id(messages: list[dict[str, str]]) -> str | None:
    for message in reversed(messages):
        matches = ORDER_ID_PATTERN.findall(message.get("content", ""))
        if matches:
            return matches[-1].upper()
    return None


def _requested_priority(content: str) -> Literal["low", "normal", "high"]:
    if "高" in content:
        return "high"
    if "低" in content:
        return "low"
    return "normal"


def ticket_creation_authorization(
    messages: list[dict[str, str]],
) -> TicketAuthorization | None:
    user_indexes = [
        index for index, message in enumerate(messages) if message.get("role") == "user"
    ]
    if not user_indexes:
        return None
    last_index = user_indexes[-1]
    content = messages[last_index].get("content", "").strip()
    if TICKET_NEGATION_PATTERN.search(content):
        return None
    order_id = _latest_order_id(messages[: last_index + 1])
    if order_id is None:
        return None
    if TICKET_ACTION_PATTERN.search(content):
        return TicketAuthorization(order_id, _requested_priority(content))
    if CONFIRMATION_PATTERN.match(content) and last_index > 0:
        previous = messages[last_index - 1]
        previous_content = previous.get("content", "")
        if previous.get("role") == "assistant" and TICKET_OFFER_PATTERN.search(
            previous_content
        ):
            return TicketAuthorization(order_id, _requested_priority(content))
    return None


class GetOrderInput(BaseModel):
    order_id: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9_-]+$",
        description="订单号，例如 A100。",
    )


class CreateTicketInput(BaseModel):
    order_id: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9_-]+$",
        description="需要处理的订单号。",
    )
    reason: str = Field(min_length=3, max_length=500, description="创建工单的明确原因。")
    priority: Literal["low", "normal", "high"] = Field(
        default="normal",
        description="工单优先级。",
    )


class MockBackendClient:
    def __init__(self, base_url: str, timeout_seconds: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def get_order(self, order_id: str) -> dict[str, Any]:
        result = self._request("GET", f"/api/orders/{order_id.upper()}")
        if result.get("ok") and isinstance(result.get("data"), dict):
            data = result["data"]
            result["data"] = {
                key: data[key]
                for key in ("order_id", "status", "item", "updated_at")
                if key in data
            }
        return result

    def create_ticket(
        self,
        order_id: str,
        reason: str,
        priority: str,
    ) -> dict[str, Any]:
        digest = hashlib.sha256(f"{order_id.upper()}\0{reason}\0{priority}".encode()).hexdigest()
        return self._request(
            "POST",
            "/api/tickets",
            json={
                "order_id": order_id.upper(),
                "reason": reason,
                "priority": priority,
            },
            headers={"Idempotency-Key": f"agent-{digest}"},
        )

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = httpx.request(
                method,
                f"{self.base_url}{path}",
                timeout=self.timeout_seconds,
                **kwargs,
            )
        except httpx.TimeoutException:
            return {
                "ok": False,
                "error": {
                    "code": "BACKEND_TIMEOUT",
                    "message": "Mock backend timed out.",
                    "retryable": True,
                },
            }
        except httpx.HTTPError as exc:
            return {
                "ok": False,
                "error": {
                    "code": "BACKEND_UNAVAILABLE",
                    "message": str(exc),
                    "retryable": True,
                },
            }

        try:
            payload = response.json()
        except ValueError:
            payload = {
                "error": {
                    "code": "INVALID_BACKEND_RESPONSE",
                    "message": "Mock backend returned non-JSON content.",
                    "retryable": False,
                }
            }
        if response.is_success:
            return {"ok": True, "data": payload}
        return {
            "ok": False,
            "status_code": response.status_code,
            "error": payload.get("error", payload),
        }


def build_tools(
    client: MockBackendClient,
    knowledge: KnowledgeBase | None = None,
    *,
    rag_top_k: int = 3,
):  # type: ignore[no-untyped-def]
    try:
        from langchain.tools import tool
    except ImportError as exc:
        raise RuntimeError('Install agent dependencies with: pip install -e ".[agent]"') from exc

    @tool("get_order", args_schema=GetOrderInput)
    def get_order(order_id: str) -> str:
        """查询一个订单的当前状态。用户询问订单信息时使用；不要猜测订单状态。"""
        return json.dumps(client.get_order(order_id), ensure_ascii=False)

    @tool("create_ticket", args_schema=CreateTicketInput)
    def create_ticket(
        order_id: str,
        reason: str,
        priority: Literal["low", "normal", "high"] = "normal",
    ) -> str:
        """为已存在的异常订单创建客服工单。只有用户明确要求或同意创建时才能使用。"""
        authorization = TICKET_WRITE_AUTHORIZATION.get()
        deadline = TICKET_WRITE_DEADLINE.get()
        if authorization is None:
            return json.dumps(
                {
                    "ok": False,
                    "error": {
                        "code": "WRITE_NOT_AUTHORIZED",
                        "message": "用户尚未明确授权创建工单。",
                        "retryable": False,
                    },
                },
                ensure_ascii=False,
            )
        if deadline is None or time.monotonic() >= deadline:
            return json.dumps(
                {
                    "ok": False,
                    "error": {
                        "code": "WRITE_DEADLINE_EXCEEDED",
                        "message": "本次 Agent 执行已超过安全写入时限。",
                        "retryable": False,
                    },
                },
                ensure_ascii=False,
            )
        if (
            order_id.upper() != authorization.order_id
            or priority != authorization.priority
        ):
            return json.dumps(
                {
                    "ok": False,
                    "error": {
                        "code": "WRITE_PARAMETERS_NOT_AUTHORIZED",
                        "message": "工单参数与用户授权不一致。",
                        "retryable": False,
                    },
                },
                ensure_ascii=False,
            )
        return json.dumps(
            client.create_ticket(order_id, reason, priority),
            ensure_ascii=False,
        )

    tools = [get_order, create_ticket]
    if knowledge is not None and knowledge.ready:

        @tool("search_knowledge")
        def search_knowledge(query: str) -> str:
            """查询产品说明、配送和售后政策等非实时知识；不要用于查询具体订单状态。"""
            results = knowledge.search(query, top_k=rag_top_k)
            return json.dumps(
                {
                    "ok": bool(results),
                    "results": results,
                    "message": None if results else "知识库中没有找到相关内容。",
                },
                ensure_ascii=False,
            )

        tools.append(search_knowledge)
    return tools
