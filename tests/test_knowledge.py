from pathlib import Path

import httpx

from qlora_lab.agent.knowledge import KnowledgeBase, RagKnowledgeClient


def test_knowledge_search_returns_relevant_document() -> None:
    knowledge = KnowledgeBase(Path("data/knowledge"))

    results = knowledge.search("显示器有固定亮点怎么排查", top_k=1)

    assert knowledge.ready is True
    assert results[0]["source"] in {"products.md", "monitor-repair.md"}
    assert "视频线" in str(results[0]["content"])


def test_knowledge_search_returns_empty_for_unrelated_query() -> None:
    knowledge = KnowledgeBase(Path("data/knowledge"))

    assert knowledge.search("量子引力弦理论") == []


def test_rag_client_only_accepts_reranked_results() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["x-internal-api-key"] == "test-key"
        assert b'"knowledgeScope":"repair"' in request.content
        return httpx.Response(
            200,
            json={
                "ok": True,
                "reranked": True,
                "results": [
                    {
                        "document_id": "fridge",
                        "title": "冰箱维修",
                        "content": "先断电，再检查电源。",
                        "rank": 1,
                        "vector_score": 0.7,
                        "rerank_score": 0.9,
                    }
                ],
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        knowledge = RagKnowledgeClient("http://rag.test/api", "test-key", 1, client)
        results = knowledge.search("冰箱不制冷")

    assert results[0]["rerank_score"] == 0.9


def test_rag_client_fails_closed_without_rerank() -> None:
    transport = httpx.MockTransport(
        lambda _: httpx.Response(
            200,
            json={
                "ok": True,
                "reranked": False,
                "results": [{"content": "未经重排的文本"}],
            },
        )
    )
    with httpx.Client(transport=transport) as client:
        knowledge = RagKnowledgeClient("http://rag.test/api", "test-key", 1, client)
        assert knowledge.search("测试") == []
