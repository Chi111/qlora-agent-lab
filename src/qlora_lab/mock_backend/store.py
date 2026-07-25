from __future__ import annotations

import json
import threading
from datetime import UTC, datetime

from qlora_lab.mock_backend.schemas import CreateTicketRequest, Order, Ticket


class IdempotencyConflictError(ValueError):
    """Raised when an idempotency key is reused for a different operation."""


class MockStore:
    def __init__(self) -> None:
        now = datetime.now(UTC)
        self.orders = {
            "A100": Order(
                order_id="A100",
                customer_name="张三",
                status="shipped",
                item="机械键盘",
                amount_cny=499,
                updated_at=now,
            ),
            "A101": Order(
                order_id="A101",
                customer_name="李四",
                status="delayed",
                item="显示器",
                amount_cny=1899,
                updated_at=now,
            ),
            "A102": Order(
                order_id="A102",
                customer_name="王五",
                status="processing",
                item="鼠标",
                amount_cny=299,
                updated_at=now,
            ),
        }
        self.tickets: dict[str, Ticket] = {}
        self.idempotency_keys: dict[str, tuple[str, str]] = {}
        self._lock = threading.Lock()

    def get_order(self, order_id: str) -> Order | None:
        return self.orders.get(order_id.upper())

    def create_ticket(
        self,
        request: CreateTicketRequest,
        idempotency_key: str,
    ) -> tuple[Ticket, bool]:
        normalized = request.model_dump()
        normalized["order_id"] = request.order_id.upper()
        fingerprint = json.dumps(normalized, ensure_ascii=False, sort_keys=True)
        with self._lock:
            existing = self.idempotency_keys.get(idempotency_key)
            if existing:
                existing_id, existing_fingerprint = existing
                if existing_fingerprint != fingerprint:
                    raise IdempotencyConflictError(
                        "Idempotency key was already used for a different payload."
                    )
                return self.tickets[existing_id], False
            ticket_id = f"T{len(self.tickets) + 1000}"
            ticket = Ticket(
                ticket_id=ticket_id,
                order_id=request.order_id.upper(),
                reason=request.reason,
                priority=request.priority,
                created_at=datetime.now(UTC),
            )
            self.tickets[ticket_id] = ticket
            self.idempotency_keys[idempotency_key] = (ticket_id, fingerprint)
            return ticket, True
