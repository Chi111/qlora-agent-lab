from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RetrievalSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RETRIEVAL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    device: Literal["cpu"] = "cpu"
    embedding_dimensions: Literal[1024] = 1024
    embedding_batch_size: int = Field(default=8, ge=1, le=32)
    rerank_batch_size: int = Field(default=8, ge=1, le=32)

    max_embedding_batch: int = Field(default=32, ge=1, le=128)
    max_embedding_text_chars: int = Field(default=12_000, ge=100, le=100_000)
    max_embedding_total_chars: int = Field(default=48_000, ge=100, le=500_000)

    max_rerank_documents: int = Field(default=64, ge=1, le=256)
    max_rerank_query_chars: int = Field(default=2_000, ge=10, le=20_000)
    max_rerank_document_chars: int = Field(default=12_000, ge=100, le=100_000)
    max_rerank_total_chars: int = Field(default=96_000, ge=100, le=1_000_000)

    max_concurrency: int = Field(default=1, ge=1, le=8)
    queue_size: int = Field(default=2, ge=0, le=32)
    operation_timeout_seconds: float = Field(default=30, ge=0.01, le=600)
