import json
from pathlib import Path

import pytest

from qlora_lab.training.config import TrainConfig
from qlora_lab.training.dataset import read_jsonl


def test_example_datasets_are_valid() -> None:
    train_rows = read_jsonl(Path("data/train.jsonl"))
    eval_rows = read_jsonl(Path("data/eval.jsonl"))

    assert len(train_rows) >= 10
    assert len(eval_rows) >= 4
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
