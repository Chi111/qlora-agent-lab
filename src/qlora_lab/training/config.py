from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TrainConfig:
    base_model_id: str
    train_file: Path
    eval_file: Path
    output_dir: Path
    max_length: int = 512
    epochs: float = 2.0
    learning_rate: float = 2e-4
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    logging_steps: int = 5
    save_steps: int = 50
    eval_steps: int = 50
    seed: int = 42
    resume_from_checkpoint: str | None = None

    @classmethod
    def from_yaml(cls, path: Path) -> TrainConfig:
        try:
            import yaml
        except ImportError as exc:
            message = 'Install training dependencies with: pip install -e ".[ml]"'
            raise RuntimeError(message) from exc

        payload: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Training config must be a mapping: {path}")

        known = {field.name for field in fields(cls)}
        unknown = sorted(set(payload) - known)
        if unknown:
            raise ValueError(f"Unknown training config keys: {', '.join(unknown)}")

        for key in ("train_file", "eval_file", "output_dir"):
            payload[key] = Path(payload[key])
        config = cls(**payload)
        config.validate()
        return config

    def validate(self) -> None:
        if not self.base_model_id.strip():
            raise ValueError("base_model_id cannot be empty")
        if not 64 <= self.max_length <= 8192:
            raise ValueError("max_length must be between 64 and 8192")
        if self.batch_size < 1 or self.gradient_accumulation_steps < 1:
            raise ValueError("batch_size and gradient_accumulation_steps must be positive")
        if self.lora_r < 1 or self.lora_alpha < 1:
            raise ValueError("LoRA rank and alpha must be positive")
        if not 0 <= self.lora_dropout < 1:
            raise ValueError("lora_dropout must be in [0, 1)")
        if self.epochs <= 0 or self.learning_rate <= 0:
            raise ValueError("epochs and learning_rate must be positive")
        if not self.train_file.is_file():
            raise FileNotFoundError(f"Training dataset not found: {self.train_file}")
        if not self.eval_file.is_file():
            raise FileNotFoundError(f"Evaluation dataset not found: {self.eval_file}")

    def as_serializable_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for key in ("train_file", "eval_file", "output_dir"):
            result[key] = str(result[key])
        return result
