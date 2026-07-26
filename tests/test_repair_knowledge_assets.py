from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

KNOWLEDGE_DIRECTORY = Path("data/knowledge")
GOLDEN_SET = KNOWLEDGE_DIRECTORY / "eval" / "repair-rag-golden.jsonl"
NEGATIVE_SET = KNOWLEDGE_DIRECTORY / "eval" / "repair-rag-negative.jsonl"
EXPERIENCE_DOCUMENT = (
    KNOWLEDGE_DIRECTORY / "experience" / "approved" / "vector-database-experience.md"
)

REPAIR_DOCUMENTS = {
    "refrigerator": KNOWLEDGE_DIRECTORY / "refrigerator-repair.md",
    "television": KNOWLEDGE_DIRECTORY / "television-repair.md",
    "monitor": KNOWLEDGE_DIRECTORY / "monitor-repair.md",
}

REQUIRED_METADATA = {
    "document_id",
    "title",
    "source",
    "knowledge_scope",
    "appliance_type",
    "document_version",
    "updated_at",
    "safety_level",
}

REQUIRED_SECTIONS = {
    "refrigerator": {
        "安全红线",
        "不制冷或制冷弱",
        "完全不启动",
        "噪声或振动异常",
        "漏水或积水",
        "结霜过厚或冷藏室结冰",
        "报修信息清单",
    },
    "television": {
        "安全红线",
        "无电或无法开机",
        "有声音但黑屏",
        "有画面但无声音",
        "花屏条纹或闪烁",
        "HDMI 无信号",
        "报修信息清单",
    },
    "monitor": {
        "安全红线",
        "无电或无法开机",
        "有电但无信号",
        "闪烁黑屏或间歇断连",
        "固定亮点暗点或残影",
        "分辨率或刷新率不正确",
        "报修信息清单",
    },
}

GOLDEN_FIELDS = {
    "case_id",
    "query",
    "appliance_type",
    "expected_source",
    "expected_section",
    "must_include",
    "must_not_include",
}


def _read_markdown(path: Path) -> tuple[dict[str, str], str]:
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n"), f"{path} must start with YAML frontmatter"
    frontmatter, separator, body = text[4:].partition("\n---\n")
    assert separator, f"{path} has no closing frontmatter delimiter"

    metadata: dict[str, str] = {}
    for line in frontmatter.splitlines():
        key, delimiter, value = line.partition(":")
        assert delimiter and key.strip() and value.strip(), f"invalid frontmatter line: {line!r}"
        metadata[key.strip()] = value.strip().strip("\"'")
    return metadata, body


def _sections(body: str) -> dict[str, str]:
    matches = list(re.finditer(r"^## (.+)$", body, flags=re.MULTILINE))
    return {
        match.group(1).strip(): body[
            match.end() : matches[index + 1].start() if index + 1 < len(matches) else None
        ]
        for index, match in enumerate(matches)
    }


def _load_golden_cases() -> list[dict[str, Any]]:
    lines = GOLDEN_SET.read_text(encoding="utf-8").splitlines()
    assert lines and all(line.strip() for line in lines)
    return [json.loads(line) for line in lines]


def test_repair_markdown_contract_and_safety_sections() -> None:
    document_ids: set[str] = set()

    for appliance_type, path in REPAIR_DOCUMENTS.items():
        metadata, body = _read_markdown(path)

        assert metadata.keys() >= REQUIRED_METADATA
        assert metadata["document_id"] == path.stem
        assert metadata["source"] == path.name
        assert metadata["knowledge_scope"] == "repair"
        assert metadata["appliance_type"] == appliance_type
        assert metadata["document_id"] not in document_ids
        document_ids.add(metadata["document_id"])

        assert body.lstrip().startswith(f"# {metadata['title']}\n")
        section_names = _sections(body).keys()
        assert REQUIRED_SECTIONS[appliance_type] <= section_names
        assert "禁止拆" in _sections(body)["安全红线"]
        assert "报修" in body

    assert "制冷剂" in _read_markdown(REPAIR_DOCUMENTS["refrigerator"])[1]
    assert "高压" in _read_markdown(REPAIR_DOCUMENTS["television"])[1]
    assert "高压" in _read_markdown(REPAIR_DOCUMENTS["monitor"])[1]


def test_approved_vector_database_experience_contract() -> None:
    metadata, body = _read_markdown(EXPERIENCE_DOCUMENT)

    assert metadata.keys() >= REQUIRED_METADATA
    assert metadata["knowledge_scope"] == "experience"
    assert metadata["appliance_type"] == "general"
    assert metadata["review_status"] == "approved"
    assert metadata["source"] == "experience/approved/vector-database-experience.md"
    assert {
        "Markdown 文档契约",
        "向量模型选择",
        "切片与元数据",
        "Qdrant 集合版本",
        "两阶段检索与 Rerank",
        "安全与隐私经验",
        "评测与回归",
    } <= _sections(body).keys()


def test_repair_rag_golden_set_schema_and_coverage() -> None:
    cases = _load_golden_cases()

    assert len(cases) == 30
    assert Counter(case["appliance_type"] for case in cases) == {
        "refrigerator": 10,
        "television": 10,
        "monitor": 10,
    }
    assert len({case["case_id"] for case in cases}) == len(cases)
    assert len({case["query"] for case in cases}) == len(cases)

    for case in cases:
        assert set(case) == GOLDEN_FIELDS
        assert case["case_id"].startswith(f"{case['appliance_type']}-")
        assert case["query"].strip()
        assert case["expected_source"] == REPAIR_DOCUMENTS[case["appliance_type"]].name
        assert case["expected_section"].strip()
        assert case["must_include"] and case["must_not_include"]
        assert all(isinstance(value, str) and value.strip() for value in case["must_include"])
        assert all(isinstance(value, str) and value.strip() for value in case["must_not_include"])
        assert not set(case["must_include"]) & set(case["must_not_include"])


def test_golden_expectations_resolve_to_document_sections() -> None:
    documents = {
        path.name: _sections(_read_markdown(path)[1]) for path in REPAIR_DOCUMENTS.values()
    }

    for case in _load_golden_cases():
        sections = documents[case["expected_source"]]
        assert case["expected_section"] in sections
        expected_content = sections[case["expected_section"]]
        for phrase in case["must_include"]:
            assert phrase in expected_content, (
                f"{case['case_id']} requires {phrase!r}, but it is missing from "
                f"{case['expected_source']}#{case['expected_section']}"
            )


def test_negative_rag_set_covers_no_answer_behavior() -> None:
    cases = [
        json.loads(line)
        for line in NEGATIVE_SET.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert len(cases) >= 6
    assert len({case["case_id"] for case in cases}) == len(cases)
    assert all(case["must_return_no_answer"] is True for case in cases)
    assert all(case["query"].strip() for case in cases)
