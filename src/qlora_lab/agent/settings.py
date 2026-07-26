from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AGENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    model_base_url: str = "http://127.0.0.1:8000/v1"
    model_name: str = "local-qlora"
    model_api_key: str = "local-not-secret"
    mock_api_url: str = "http://127.0.0.1:8001"
    timeout_seconds: float = Field(default=30, ge=1, le=300)
    recursion_limit: int = Field(default=10, ge=2, le=30)
    rag_base_url: str = "http://127.0.0.1:4111/api"
    rag_internal_api_key: str = "change-me-local-rag-key"
    conversation_db_path: Path = Path("artifacts/conversations.sqlite3")
    max_history_messages: int = Field(default=20, ge=2, le=60)
    rag_top_k: int = Field(default=4, ge=1, le=8)
