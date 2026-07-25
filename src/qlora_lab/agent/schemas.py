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
