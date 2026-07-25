import json
from pathlib import Path

import pytest

from qlora_lab.training.config import TrainConfig
from qlora_lab.training.dataset import read_jsonl
from qlora_lab.training.preflight import estimated_minimum_vram_gb
from qlora_lab.training.validate_data import validate_dataset_pair


def test_example_datasets_are_valid() -> None:
    train_rows = read_jsonl(Path("data/train.jsonl"))
    eval_rows = read_jsonl(Path("data/eval.jsonl"))

    assert len(train_rows) >= 60
    assert len(eval_rows) >= 15
    assert all(isinstance(json.loads(row["tools"]), list) for row in train_rows)


def test_dataset_reports_invalid_json_line(tmp_path: Path) -> None:
    dataset = tmp_path / "bad.jsonl"
    dataset.write_text('{"messages": []}\nnot-json\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"messages must contain"):
        read_jsonl(dataset)


def test_8gb_training_profile_is_conservative() -> None:
    config = TrainConfig.from_yaml(Path("configs/train_8gb.yaml"))

    assert config.batch_size == 1
    assert config.max_length <= 512
    assert "9B" not in config.base_model_id.upper()
    assert estimated_minimum_vram_gb(config.base_model_id) <= 8


def test_train_and_eval_splits_are_distinct() -> None:
    report = validate_dataset_pair(Path("data/train.jsonl"), Path("data/eval.jsonl"))

    assert report["status"] == "ok"
    assert report["train_eval_overlap"] == 0
    assert report["train"]["tool_calls"]["get_order"] >= 20
    assert report["train"]["tool_calls"]["search_knowledge"] >= 10
    assert report["train"]["tool_calls"]["create_ticket"] >= 6


def test_dataset_validator_rejects_split_leakage(tmp_path: Path) -> None:
    row = '{"messages":[{"role":"user","content":"你好"},{"role":"assistant","content":"你好"}]}\n'
    train = tmp_path / "train.jsonl"
    evaluation = tmp_path / "eval.jsonl"
    train.write_text(row, encoding="utf-8")
    evaluation.write_text(row, encoding="utf-8")

    with pytest.raises(ValueError, match="train_eval_overlap=1"):
        validate_dataset_pair(train, evaluation)


def test_dataset_validator_rejects_invalid_tool_arguments(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    evaluation = tmp_path / "eval.jsonl"
    train.write_text(
        json.dumps(
            {
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_order",
                            "parameters": {
                                "type": "object",
                                "properties": {"order_id": {"type": "string"}},
                                "required": ["order_id"],
                            },
                        },
                    }
                ],
                "messages": [
                    {"role": "user", "content": "查询订单"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {"name": "get_order", "arguments": {}},
                            }
                        ],
                    },
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    evaluation.write_text(
        '{"messages":[{"role":"user","content":"你好"},'
        '{"role":"assistant","content":"你好"}]}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing required tool arguments"):
        validate_dataset_pair(train, evaluation)
