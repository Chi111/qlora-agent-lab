from __future__ import annotations

import argparse
import asyncio
from typing import Literal

from fastapi import FastAPI, Header, Query, Response

from qlora_lab.common.errors import AppError
from qlora_lab.common.http import install_http_support
from qlora_lab.common.logging import configure_logging
from qlora_lab.mock_backend.schemas import CreateTicketRequest, Order, Ticket
from qlora_lab.mock_backend.store import IdempotencyConflictError, MockStore


def create_app(store: MockStore | None = None) -> FastAPI:
    store = store or MockStore()
    app = FastAPI(title="QLoRA Lab Mock Backend", version="0.1.0")
    install_http_support(app)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ready"}

    @app.get("/api/orders/{order_id}", response_model=Order)
    async def get_order(
        order_id: str,
        simulate: Literal["none", "error", "timeout"] = Query(default="none"),
        delay_ms: int = Query(default=0, ge=0, le=5000),
    ) -> Order:
        if simulate == "error":
            raise AppError("MOCK_UPSTREAM_ERROR", "Simulated backend failure.", 500, True)
        if simulate == "timeout":
            await asyncio.sleep(5)
        elif delay_ms:
            await asyncio.sleep(delay_ms / 1000)
        order = store.get_order(order_id)
        if order is None:
            raise AppError(
                "ORDER_NOT_FOUND",
                f"Order {order_id!r} was not found.",
                404,
                False,
            )
        return order

    @app.post("/api/tickets", response_model=Ticket)
    async def create_ticket(
        payload: CreateTicketRequest,
        response: Response,
        idempotency_key: str = Header(
            min_length=8,
            max_length=128,
            alias="Idempotency-Key",
        ),
    ) -> Ticket:
        if store.get_order(payload.order_id) is None:
            raise AppError(
                "ORDER_NOT_FOUND",
                f"Order {payload.order_id!r} was not found.",
                404,
                False,
            )
        try:
            ticket, created = store.create_ticket(payload, idempotency_key)
        except IdempotencyConflictError as exc:
            raise AppError(
                "IDEMPOTENCY_CONFLICT",
                str(exc),
                409,
                False,
            ) from exc
        response.status_code = 201 if created else 200
        response.headers["X-Idempotent-Replay"] = "false" if created else "true"
        return ticket

    return app


app = create_app()


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Run the mock business backend.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(
        "qlora_lab.mock_backend.app:app",
        host=args.host,
        port=args.port,
        workers=1,
    )


if __name__ == "__main__":
    main()
