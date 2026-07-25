from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FunctionDefinition(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    description: str | None = Field(default=None, max_length=2000)
    parameters: dict[str, Any] = Field(default_factory=dict)


class ToolDefinition(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionDefinition


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = Field(default=None, max_length=16000)
    name: str | None = Field(default=None, max_length=128)
    tool_call_id: str | None = Field(default=None, max_length=128)
    tool_calls: list[dict[str, Any]] | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str = Field(default="local-qlora", min_length=1, max_length=128)
    messages: list[ChatMessage] = Field(min_length=1, max_length=32)
    temperature: float = Field(default=0.2, ge=0, le=2)
    top_p: float = Field(default=0.9, gt=0, le=1)
    max_tokens: int = Field(default=256, ge=1, le=2048)
    stream: Literal[False] = False
    stop: str | list[str] | None = None
    tools: list[ToolDefinition] | None = Field(default=None, max_length=32)
    tool_choice: Any | None = None

    @model_validator(mode="after")
    def require_message_content(self) -> ChatCompletionRequest:
        for message in self.messages:
            if message.content is None and not message.tool_calls:
                raise ValueError("Each message needs content or tool_calls.")
        return self
