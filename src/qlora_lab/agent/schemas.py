from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class AgentMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=16000)


class AgentRequest(BaseModel):
    messages: list[AgentMessage] = Field(min_length=1, max_length=20)
    debug: bool = False

    @model_validator(mode="after")
    def validate_total_size(self) -> "AgentRequest":
        if sum(len(item.content) for item in self.messages) > 32000:
            raise ValueError("Total message content exceeds 32000 characters.")
        if not any(item.role == "user" for item in self.messages):
            raise ValueError("At least one user message is required.")
        return self


class AgentResponse(BaseModel):
    content: str
    trace: list[dict[str, Any]] | None = None


class ChatRequest(BaseModel):
    session_id: str | None = Field(
        default=None,
        min_length=8,
        max_length=64,
        pattern=r"^[A-Za-z0-9_-]+$",
    )
    message: str = Field(min_length=1, max_length=8000)
    debug: bool = False


class ChatResponse(BaseModel):
    session_id: str
    content: str
    status: Literal["active", "waiting_human"]
    trace: list[dict[str, Any]] | None = None


class HandoffRequest(BaseModel):
    reason: str = Field(default="用户请求人工客服", min_length=2, max_length=500)


class SessionTranscript(BaseModel):
    session_id: str
    status: Literal["active", "waiting_human"]
    handoff_reason: str | None = None
    created_at: str
    updated_at: str
    messages: list[AgentMessage]
