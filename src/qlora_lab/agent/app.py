from __future__ import annotations

import argparse
import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from qlora_lab.agent.runtime import (
    AgentRuntime,
    LangChainRuntime,
    message_content,
    safe_trace,
)
from qlora_lab.agent.schemas import AgentRequest, AgentResponse
from qlora_lab.agent.settings import AgentSettings
from qlora_lab.common.errors import AppError
from qlora_lab.common.http import install_http_support
from qlora_lab.common.logging import configure_logging

LOGGER = logging.getLogger(__name__)


def create_app(
    runtime: AgentRuntime | None = None,
    settings: AgentSettings | None = None,
) -> FastAPI:
    settings = settings or AgentSettings()
    runtime_holder: dict[str, AgentRuntime | None] = {"runtime": runtime}

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if runtime_holder["runtime"] is None:
            runtime_holder["runtime"] = await asyncio.to_thread(LangChainRuntime, settings)
        yield

    app = FastAPI(title="QLoRA LangChain Agent", version="0.1.0", lifespan=lifespan)
    install_http_support(app)

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {
            "status": "ready" if runtime_holder["runtime"] is not None else "starting",
            "model": settings.model_name,
            "model_base_url": settings.model_base_url,
            "mock_api_url": settings.mock_api_url,
        }

    @app.post("/agent/invoke", response_model=AgentResponse)
    async def invoke(request: AgentRequest) -> AgentResponse:
        current_runtime = runtime_holder["runtime"]
        if current_runtime is None:
            raise AppError("AGENT_NOT_READY", "Agent is not initialized.", 503, True)
        messages = [message.model_dump() for message in request.messages]
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(current_runtime.invoke, messages),
                timeout=settings.timeout_seconds * 2,
            )
        except TimeoutError as exc:
            raise AppError("AGENT_TIMEOUT", "Agent execution timed out.", 504, True) from exc
        except Exception as exc:
            LOGGER.exception("Agent execution failed")
            raise AppError(
                "AGENT_EXECUTION_FAILED",
                "Agent execution failed.",
                502,
                True,
            ) from exc
        result_messages = result.get("messages", [])
        if not result_messages:
            raise AppError("EMPTY_AGENT_RESPONSE", "Agent returned no messages.", 502, False)
        content = message_content(result_messages[-1]).strip()
        if not content:
            raise AppError("EMPTY_AGENT_RESPONSE", "Agent returned empty content.", 502, False)
        return AgentResponse(
            content=content,
            trace=safe_trace(result) if request.debug else None,
        )

    return app


app = create_app()


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Run the LangChain agent service.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(
        "qlora_lab.agent.app:app",
        host=args.host,
        port=args.port,
        workers=1,
    )


if __name__ == "__main__":
    main()
