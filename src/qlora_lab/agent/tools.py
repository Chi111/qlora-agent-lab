from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field


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
        return self._request("GET", f"/api/orders/{order_id.upper()}")

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


def build_tools(client: MockBackendClient):  # type: ignore[no-untyped-def]
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
        return json.dumps(
            client.create_ticket(order_id, reason, priority),
            ensure_ascii=False,
        )

    return [get_order, create_ticket]
