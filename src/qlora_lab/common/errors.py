from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class AppError(Exception):
    code: str
    message: str
    status_code: int = 500
    retryable: bool = False
    details: Any | None = None

    def __str__(self) -> str:
        return self.message
