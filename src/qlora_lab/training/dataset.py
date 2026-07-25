from __future__ import annotations

import json
from pathlib import Path
from typing import Any

VALID_ROLES = {"system", "user", "assistant", "tool"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    default_tools_path = path.parent / "tools.json"
    default_tools = (
        json.loads(default_tools_path.read_text(encoding="utf-8"))
        if default_tools_path.is_file()
        else None
    )
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            _validate_row(row, path, line_number)
            if row.get("tools") is None and default_tools is not None:
                row["tools"] = default_tools
            if isinstance(row.get("tools"), list):
                row["tools"] = json.dumps(row["tools"], ensure_ascii=False)
            rows.append(row)
    if not rows:
        raise ValueError(f"Dataset is empty: {path}")
    return rows


def _validate_row(row: Any, path: Path, line_number: int) -> None:
    location = f"{path}:{line_number}"
    if not isinstance(row, dict):
        raise ValueError(f"{location}: each row must be a JSON object")
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError(f"{location}: messages must contain at least two items")
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"{location}: message {index} must be an object")
        if message.get("role") not in VALID_ROLES:
            raise ValueError(f"{location}: message {index} has an invalid role")
        if "content" not in message and "tool_calls" not in message:
            raise ValueError(f"{location}: message {index} needs content or tool_calls")
    tools = row.get("tools")
    if tools is not None and not isinstance(tools, (list, str)):
        raise ValueError(f"{location}: tools must be a JSON schema list or JSON string")


def load_training_dataset(path: Path):  # type: ignore[no-untyped-def]
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise RuntimeError('Install training dependencies with: pip install -e ".[ml]"') from exc
    return Dataset.from_list(read_jsonl(path))
