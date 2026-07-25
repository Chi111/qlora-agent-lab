from fastapi.testclient import TestClient

from qlora_lab.mock_backend.app import create_app
from qlora_lab.mock_backend.store import MockStore


def test_get_known_and_unknown_order() -> None:
    with TestClient(create_app(MockStore())) as client:
        known = client.get("/api/orders/A100")
        missing = client.get("/api/orders/UNKNOWN")

    assert known.status_code == 200
    assert known.json()["status"] == "shipped"
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "ORDER_NOT_FOUND"


def test_ticket_creation_is_idempotent() -> None:
    payload = {
        "order_id": "A101",
        "reason": "订单延迟",
        "priority": "high",
    }
    headers = {"Idempotency-Key": "same-operation-123"}

    with TestClient(create_app(MockStore())) as client:
        first = client.post("/api/tickets", json=payload, headers=headers)
        replay = client.post("/api/tickets", json=payload, headers=headers)

    assert first.status_code == 201
    assert replay.status_code == 200
    assert first.json()["ticket_id"] == replay.json()["ticket_id"]
    assert replay.headers["X-Idempotent-Replay"] == "true"


def test_ticket_requires_idempotency_key() -> None:
    with TestClient(create_app(MockStore())) as client:
        response = client.post(
            "/api/tickets",
            json={"order_id": "A101", "reason": "订单延迟"},
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_idempotency_key_rejects_different_payload() -> None:
    headers = {"Idempotency-Key": "same-operation-123"}

    with TestClient(create_app(MockStore())) as client:
        first = client.post(
            "/api/tickets",
            json={"order_id": "A101", "reason": "订单延迟"},
            headers=headers,
        )
        conflict = client.post(
            "/api/tickets",
            json={"order_id": "A101", "reason": "用户修改了原因"},
            headers=headers,
        )

    assert first.status_code == 201
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "IDEMPOTENCY_CONFLICT"
