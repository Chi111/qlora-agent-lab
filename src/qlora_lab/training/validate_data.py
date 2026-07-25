from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from qlora_lab.training.config import TrainConfig
from qlora_lab.training.dataset import read_jsonl


def _fingerprint(row: dict[str, Any]) -> str:
    normalized = json.dumps(
        row["messages"],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _declared_tools(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    tools = row.get("tools")
    if isinstance(tools, str):
        tools = json.loads(tools)
    if not isinstance(tools, list):
        return {}
    return {
        function["name"]: function
        for tool in tools
        if isinstance(tool, dict)
        and isinstance((function := tool.get("function")), dict)
        and isinstance(function.get("name"), str)
    }


def _validate_tool_arguments(
    arguments: Any,
    definition: dict[str, Any],
    location: str,
) -> None:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{location}: tool arguments are not valid JSON.") from exc
    if not isinstance(arguments, dict):
        raise ValueError(f"{location}: tool arguments must be an object.")

    parameters = definition.get("parameters", {})
    properties = parameters.get("properties", {}) if isinstance(parameters, dict) else {}
    required = parameters.get("required", []) if isinstance(parameters, dict) else []
    missing = [name for name in required if name not in arguments]
    if missing:
        raise ValueError(f"{location}: missing required tool arguments: {', '.join(missing)}")
    unknown = sorted(set(arguments) - set(properties))
    if unknown:
        raise ValueError(f"{location}: unknown tool arguments: {', '.join(unknown)}")

    python_types = {"string": str, "integer": int, "number": (int, float), "boolean": bool}
    for name, value in arguments.items():
        specification = properties.get(name, {})
        expected_type = python_types.get(specification.get("type"))
        if expected_type is not None and not isinstance(value, expected_type):
            raise ValueError(f"{location}: tool argument {name!r} has the wrong type.")
        choices = specification.get("enum")
        if isinstance(choices, list) and value not in choices:
            raise ValueError(f"{location}: tool argument {name!r} is outside its enum.")


def dataset_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    roles: Counter[str] = Counter()
    tool_calls: Counter[str] = Counter()
    characters = 0
    for row_index, row in enumerate(rows, start=1):
        declared = _declared_tools(row)
        pending_tools: list[str] = []
        for message_index, message in enumerate(row["messages"]):
            location = f"sample {row_index}, message {message_index}"
            roles[message["role"]] += 1
            characters += len(str(message.get("content") or ""))
            calls = message.get("tool_calls") or []
            if calls and message["role"] != "assistant":
                raise ValueError(f"{location}: only assistant messages may call tools.")
            if calls and pending_tools:
                raise ValueError(f"{location}: previous tool calls are missing responses.")
            for call_index, call in enumerate(calls):
                function = call.get("function", {}) if isinstance(call, dict) else {}
                name = function.get("name")
                call_location = f"{location}, tool call {call_index}"
                if not isinstance(name, str) or not name:
                    raise ValueError(f"{call_location}: tool name is missing.")
                tool_calls[name] += 1
                if name not in declared:
                    raise ValueError(f"{call_location}: tool {name!r} is not declared.")
                _validate_tool_arguments(
                    function.get("arguments", {}),
                    declared[name],
                    call_location,
                )
                pending_tools.append(name)

            if message["role"] == "tool":
                name = message.get("name")
                if not isinstance(name, str) or name not in pending_tools:
                    raise ValueError(f"{location}: tool response has no matching call.")
                pending_tools.remove(name)
            elif pending_tools and not calls:
                raise ValueError(f"{location}: tool responses must follow tool calls.")
        if pending_tools:
            raise ValueError(f"sample {row_index}: tool calls are missing responses.")
    return {
        "rows": len(rows),
        "messages": sum(roles.values()),
        "roles": dict(sorted(roles.items())),
        "tool_calls": dict(sorted(tool_calls.items())),
        "content_characters": characters,
    }


def validate_dataset_pair(train_file: Path, eval_file: Path) -> dict[str, Any]:
    train_rows = read_jsonl(train_file)
    eval_rows = read_jsonl(eval_file)
    train_fingerprints = [_fingerprint(row) for row in train_rows]
    eval_fingerprints = [_fingerprint(row) for row in eval_rows]

    duplicate_train = len(train_fingerprints) - len(set(train_fingerprints))
    duplicate_eval = len(eval_fingerprints) - len(set(eval_fingerprints))
    overlap = set(train_fingerprints) & set(eval_fingerprints)
    if duplicate_train or duplicate_eval or overlap:
        raise ValueError(
            "Dataset split validation failed: "
            f"duplicate_train={duplicate_train}, duplicate_eval={duplicate_eval}, "
            f"train_eval_overlap={len(overlap)}"
        )

    return {
        "status": "ok",
        "train": dataset_stats(train_rows),
        "eval": dataset_stats(eval_rows),
        "train_eval_overlap": 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate QLoRA train/eval datasets.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train_8gb.yaml"),
    )
    return parser.parse_args()


def main() -> None:
    config = TrainConfig.from_yaml(parse_args().config)
    report = validate_dataset_pair(config.train_file, config.eval_file)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
