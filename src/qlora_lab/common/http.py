from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from qlora_lab.common.errors import AppError

LOGGER = logging.getLogger(__name__)
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,80}$")


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", str(uuid.uuid4()))


def error_body(
    *,
    code: str,
    message: str,
    request_id: str,
    retryable: bool,
    details: Any | None = None,
) -> dict[str, Any]:
    return {
        "error": {
            "code": code,
            "message": message,
            "request_id": request_id,
            "retryable": retryable,
            "details": details,
        }
    }


def install_http_support(app: FastAPI) -> None:
    @app.middleware("http")
    async def add_request_id(request: Request, call_next):  # type: ignore[no-untyped-def]
        candidate = request.headers.get("X-Request-ID", "")
        request.state.request_id = (
            candidate if REQUEST_ID_PATTERN.fullmatch(candidate) else str(uuid.uuid4())
        )
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=error_body(
                code=exc.code,
                message=exc.message,
                request_id=_request_id(request),
                retryable=exc.retryable,
                details=exc.details,
            ),
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=error_body(
                code="VALIDATION_ERROR",
                message="Request validation failed.",
                request_id=_request_id(request),
                retryable=False,
                details=json.loads(json.dumps(exc.errors(), default=str)),
            ),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        LOGGER.exception("Unhandled request error", exc_info=exc)
        return JSONResponse(
            status_code=500,
            content=error_body(
                code="INTERNAL_ERROR",
                message="An unexpected server error occurred.",
                request_id=_request_id(request),
                retryable=False,
            ),
        )
