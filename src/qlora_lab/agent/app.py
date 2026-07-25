from __future__ import annotations

import argparse
import asyncio
import logging
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi import Path as ApiPath
from fastapi.responses import FileResponse

from qlora_lab.agent.conversations import ConversationStore
from qlora_lab.agent.runtime import (
    AgentRuntime,
    LangChainRuntime,
    message_content,
    safe_trace,
)
from qlora_lab.agent.schemas import (
    AgentRequest,
    AgentResponse,
    ChatRequest,
    ChatResponse,
    HandoffRequest,
    SessionTranscript,
)
from qlora_lab.agent.settings import AgentSettings
from qlora_lab.common.errors import AppError
from qlora_lab.common.http import install_http_support
from qlora_lab.common.logging import configure_logging

LOGGER = logging.getLogger(__name__)
HANDOFF_PATTERN = re.compile(
    r"(转接|转|接入|联系|找|要|需要|呼叫).{0,6}(人工客服|真人客服|人工|真人)"
    r"|(人工客服|真人客服).{0,4}(转接|接入|联系)"
)
HANDOFF_NEGATION_PATTERN = re.compile(
    r"(不|不用|无需|别|不要).{0,5}(转接|转|接入|联系|找|要|需要|呼叫)"
    r".{0,6}(人工客服|真人客服|人工|真人)"
)


def requests_human_handoff(message: str) -> bool:
    return bool(HANDOFF_PATTERN.search(message)) and not HANDOFF_NEGATION_PATTERN.search(message)


def create_app(
    runtime: AgentRuntime | None = None,
    settings: AgentSettings | None = None,
    conversation_store: ConversationStore | None = None,
) -> FastAPI:
    settings = settings or AgentSettings()
    runtime_holder: dict[str, AgentRuntime | None] = {"runtime": runtime}
    store = conversation_store or ConversationStore(settings.conversation_db_path)
    chat_lock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if runtime_holder["runtime"] is None:
            runtime_holder["runtime"] = await asyncio.to_thread(LangChainRuntime, settings)
        yield

    app = FastAPI(title="QLoRA LangChain Agent", version="0.1.0", lifespan=lifespan)
    install_http_support(app)
    web_index = Path(__file__).with_name("web") / "index.html"

    async def run_agent(
        messages: list[dict[str, str]],
        *,
        debug: bool,
    ) -> tuple[str, list[dict[str, object]] | None]:
        current_runtime = runtime_holder["runtime"]
        if current_runtime is None:
            raise AppError("AGENT_NOT_READY", "Agent is not initialized.", 503, True)
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
        return content, safe_trace(result) if debug else None

    @app.get("/", include_in_schema=False)
    async def index():  # type: ignore[no-untyped-def]
        if not web_index.is_file():
            raise AppError("WEB_UI_NOT_FOUND", "Web UI is not installed.", 404, False)
        return FileResponse(web_index)

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
        messages = [message.model_dump() for message in request.messages]
        content, trace = await run_agent(messages, debug=request.debug)
        return AgentResponse(content=content, trace=trace)

    @app.post("/agent/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest) -> ChatResponse:
        async with chat_lock:
            session_id = await asyncio.to_thread(store.ensure_session, request.session_id)
            session = await asyncio.to_thread(store.get_session, session_id)
            if session is None:
                raise AppError("SESSION_NOT_FOUND", "Conversation session was not found.", 404)

            if session["status"] == "waiting_human":
                content = "该会话正在等待人工客服接管，我已保留你的新消息。"
                await asyncio.to_thread(
                    store.append_exchange,
                    session_id,
                    request.message,
                    content,
                )
                return ChatResponse(
                    session_id=session_id,
                    content=content,
                    status="waiting_human",
                )

            if requests_human_handoff(request.message):
                content = "已为你申请转接人工客服，会话记录会一并保留。"
                await asyncio.to_thread(
                    store.append_exchange,
                    session_id,
                    request.message,
                    content,
                )
                await asyncio.to_thread(
                    store.request_handoff,
                    session_id,
                    "用户在对话中明确请求人工客服",
                )
                return ChatResponse(
                    session_id=session_id,
                    content=content,
                    status="waiting_human",
                )

            history = await asyncio.to_thread(
                store.history,
                session_id,
                settings.max_history_messages,
            )
            messages = [*history, {"role": "user", "content": request.message}]
            content, trace = await run_agent(messages, debug=request.debug)
            await asyncio.to_thread(
                store.append_exchange,
                session_id,
                request.message,
                content,
            )
            return ChatResponse(
                session_id=session_id,
                content=content,
                status="active",
                trace=trace,
            )

    @app.get("/agent/sessions/{session_id}", response_model=SessionTranscript)
    async def get_session(
        session_id: str = ApiPath(
            min_length=8,
            max_length=64,
            pattern=r"^[A-Za-z0-9_-]+$",
        ),
    ) -> SessionTranscript:
        transcript = await asyncio.to_thread(store.transcript, session_id)
        if transcript is None:
            raise AppError("SESSION_NOT_FOUND", "Conversation session was not found.", 404)
        return SessionTranscript.model_validate(transcript)

    @app.post("/agent/sessions/{session_id}/handoff", response_model=ChatResponse)
    async def request_handoff(
        request: HandoffRequest,
        session_id: str = ApiPath(
            min_length=8,
            max_length=64,
            pattern=r"^[A-Za-z0-9_-]+$",
        ),
    ) -> ChatResponse:
        content = "已申请转接人工客服，会话记录会一并保留。"
        async with chat_lock:
            try:
                session = await asyncio.to_thread(
                    store.request_handoff,
                    session_id,
                    request.reason,
                )
            except KeyError as exc:
                raise AppError(
                    "SESSION_NOT_FOUND",
                    "Conversation session was not found.",
                    404,
                ) from exc
            await asyncio.to_thread(store.append_message, session_id, "assistant", content)
        return ChatResponse(
            session_id=session_id,
            content=content,
            status=session["status"],
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
