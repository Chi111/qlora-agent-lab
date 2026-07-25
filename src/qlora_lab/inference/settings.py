from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class InferenceSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="INFERENCE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    base_model_id: str = "Qwen/Qwen3-1.7B"
    adapter_path: Path = Path("artifacts/qlora-adapter")
    model_name: str = "local-qlora"
    max_input_chars: int = Field(default=32000, ge=1000, le=200000)
    max_new_tokens: int = Field(default=512, ge=1, le=4096)
    queue_size: int = Field(default=2, ge=0, le=32)
    queue_timeout_seconds: float = Field(default=60, ge=1, le=600)
    allow_adapter_model_mismatch: bool = False
