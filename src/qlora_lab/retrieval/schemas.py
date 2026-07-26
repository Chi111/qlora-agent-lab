from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

InputText = Annotated[str, Field(min_length=1, max_length=100_000)]
InputList = Annotated[list[InputText], Field(min_length=1, max_length=128)]


class EmbeddingsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1, max_length=128)
    input: InputText | InputList
    encoding_format: str = Field(default="float", pattern=r"^float$")
    dimensions: int | None = Field(default=None, ge=1, le=4096)

    @property
    def texts(self) -> list[str]:
        return [self.input] if isinstance(self.input, str) else self.input


class RerankRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1, max_length=128)
    query: InputText
    documents: InputList
    top_n: int | None = Field(default=None, ge=1, le=256)
    return_documents: bool = False

    @model_validator(mode="after")
    def top_n_cannot_exceed_document_count(self) -> RerankRequest:
        if self.top_n is not None and self.top_n > len(self.documents):
            raise ValueError("top_n cannot exceed the number of documents.")
        return self
