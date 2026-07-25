from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from qlora_lab.common.errors import AppError
from qlora_lab.common.http import install_http_support
from qlora_lab.common.logging import configure_logging
from qlora_lab.inference.generator import Generator, TransformersGenerator
from qlora_lab.inference.schemas import ChatCompletionRequest
from qlora_lab.inference.settings import InferenceSettings


class GenerationGate:
    def __init__(self, queue_size: int, timeout_seconds: float) -> None:
        self._semaphore = asyncio.Semaphore(1)
        self._queue_size = queue_size
        self._timeout_seconds = timeout_seconds
        self._waiting = 0
        self._counter_lock = asyncio.Lock()

    async def run(self, generator: Generator, request: ChatCompletionRequest):  # type: ignore[no-untyped-def]
        async with self._counter_lock:
            if self._semaphore.locked() and self._waiting >= self._queue_size:
                raise AppError(
                    "MODEL_BUSY",
                    "GPU generation queue is full.",
                    429,
                    True,
                )
            self._waiting += 1
        try:
            try:
                await asyncio.wait_for(
                    self._semaphore.acquire(),
                    timeout=self._timeout_seconds,
                )
            except TimeoutError as exc:
                raise AppError(
                    "MODEL_QUEUE_TIMEOUT",
                    "Timed out waiting for the GPU.",
                    429,
                    True,
                ) from exc
        finally:
            async with self._counter_lock:
                self._waiting -= 1

        try:
            return await asyncio.to_thread(generator.generate, request)
        finally:
            self._semaphore.release()


def create_app(
    generator: Generator | None = None,
    settings: InferenceSettings | None = None,
    *,
    load_on_startup: bool = True,
) -> FastAPI:
    settings = settings or InferenceSettings()
    generator = generator or TransformersGenerator(settings)
    gate = GenerationGate(settings.queue_size, settings.queue_timeout_seconds)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if load_on_startup and not generator.loaded:
            await asyncio.to_thread(generator.load)
        yield

    app = FastAPI(
        title="QLoRA Inference API",
        version="0.1.0",
        lifespan=lifespan,
    )
    install_http_support(app)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return generator.health()

    @app.post("/v1/chat/completions")
    async def chat_completion(request: ChatCompletionRequest) -> Any:
        request_chars = len(request.model_dump_json())
        if request_chars > settings.max_input_chars:
            raise AppError(
                "INPUT_TOO_LARGE",
                f"Serialized request exceeds {settings.max_input_chars} characters.",
                400,
                False,
            )
        if request.model not in {settings.model_name, "local-qlora"}:
            raise AppError(
                "MODEL_NOT_FOUND",
                f"Unknown model: {request.model}",
                404,
                False,
            )

        result = await gate.run(generator, request)
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created_at = int(time.time())
        finish_reason = "tool_calls" if result.tool_calls else "stop"
        usage = {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.prompt_tokens + result.completion_tokens,
        }

        if request.stream:
            async def event_stream() -> AsyncIterator[str]:
                delta: dict[str, Any] = {"role": "assistant"}
                if result.content is not None:
                    delta["content"] = result.content
                if result.tool_calls:
                    delta["tool_calls"] = [
                        {"index": index, **tool_call}
                        for index, tool_call in enumerate(result.tool_calls)
                    ]
                first_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created_at,
                    "model": settings.model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": delta,
                            "finish_reason": None,
                        }
                    ],
                }
                final_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created_at,
                    "model": settings.model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": finish_reason,
                        }
                    ],
                    "usage": usage,
                }
                for chunk in (first_chunk, final_chunk):
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                event_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        message: dict[str, Any] = {"role": "assistant", "content": result.content}
        if result.tool_calls:
            message["tool_calls"] = result.tool_calls
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created_at,
            "model": settings.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": usage,
        }

    return app


app = create_app()


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Run the QLoRA inference service.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(
        "qlora_lab.inference.app:app",
        host=args.host,
        port=args.port,
        workers=1,
    )


if __name__ == "__main__":
    main()
