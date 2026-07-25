from pathlib import Path

from qlora_lab.agent.knowledge import KnowledgeBase


def test_knowledge_search_returns_relevant_document() -> None:
    knowledge = KnowledgeBase(Path("data/knowledge"))

    results = knowledge.search("显示器有固定亮点怎么排查", top_k=1)

    assert knowledge.ready is True
    assert results[0]["source"] == "products.md"
    assert "视频线" in str(results[0]["content"])


def test_knowledge_search_returns_empty_for_unrelated_query() -> None:
    knowledge = KnowledgeBase(Path("data/knowledge"))

    assert knowledge.search("量子引力弦理论") == []
