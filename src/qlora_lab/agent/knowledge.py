from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

WORD_PATTERN = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]")


@dataclass(frozen=True, slots=True)
class KnowledgeDocument:
    document_id: str
    title: str
    content: str
    source: str


def _tokens(text: str) -> list[str]:
    raw = [item.lower() for item in WORD_PATTERN.findall(text)]
    chinese = [item for item in raw if "\u4e00" <= item <= "\u9fff"]
    bigrams = [f"{left}{right}" for left, right in zip(chinese, chinese[1:], strict=False)]
    return raw + bigrams


class KnowledgeBase:
    """Small, dependency-free BM25 index for trusted local Markdown documents."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.documents = self._load_documents(directory)
        self._term_counts = [Counter(_tokens(item.content)) for item in self.documents]
        self._lengths = [sum(counts.values()) for counts in self._term_counts]
        self._average_length = (
            sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
        )
        self._document_frequency: Counter[str] = Counter()
        for counts in self._term_counts:
            self._document_frequency.update(counts.keys())

    @staticmethod
    def _load_documents(directory: Path) -> list[KnowledgeDocument]:
        if not directory.is_dir():
            return []
        documents: list[KnowledgeDocument] = []
        for path in sorted(directory.glob("*.md")):
            content = path.read_text(encoding="utf-8").strip()
            if not content:
                continue
            first_line = content.splitlines()[0].strip()
            title = first_line.lstrip("#").strip() if first_line.startswith("#") else path.stem
            documents.append(
                KnowledgeDocument(
                    document_id=path.stem,
                    title=title,
                    content=content,
                    source=path.name,
                )
            )
        return documents

    @property
    def ready(self) -> bool:
        return bool(self.documents)

    def search(self, query: str, top_k: int = 3) -> list[dict[str, object]]:
        query_terms = Counter(_tokens(query))
        if not query_terms or not self.documents:
            return []

        scored: list[tuple[float, int]] = []
        document_count = len(self.documents)
        for index, term_counts in enumerate(self._term_counts):
            if not any(len(term) > 1 and term in term_counts for term in query_terms):
                continue
            score = 0.0
            length = self._lengths[index]
            for term, query_frequency in query_terms.items():
                frequency = term_counts.get(term, 0)
                if not frequency:
                    continue
                document_frequency = self._document_frequency[term]
                inverse_document_frequency = math.log(
                    1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                denominator = frequency + 1.5 * (
                    1 - 0.75 + 0.75 * length / max(self._average_length, 1)
                )
                score += (
                    inverse_document_frequency
                    * frequency
                    * 2.5
                    / denominator
                    * query_frequency
                )
            if score > 0:
                scored.append((score, index))

        results: list[dict[str, object]] = []
        for score, index in sorted(scored, reverse=True)[: max(1, top_k)]:
            document = self.documents[index]
            results.append(
                {
                    "document_id": document.document_id,
                    "title": document.title,
                    "source": document.source,
                    "score": round(score, 4),
                    "content": document.content[:4000],
                }
            )
        return results


class RagKnowledgeClient:
    """Read-only client for Mastra's vector retrieval + mandatory rerank route."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout_seconds: float,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self._client = client

    @property
    def ready(self) -> bool:
        return bool(self.base_url and self.api_key)

    def search(self, query: str, top_k: int = 4) -> list[dict[str, object]]:
        request = self._client.post if self._client is not None else httpx.post
        try:
            response = request(
                f"{self.base_url}/internal/knowledge/search",
                headers={"x-internal-api-key": self.api_key},
                json={"query": query, "topK": top_k, "knowledgeScope": "repair"},
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload: Any = response.json()
        except (httpx.HTTPError, ValueError):
            return []

        if (
            not isinstance(payload, dict)
            or payload.get("ok") is not True
            or payload.get("reranked") is not True
            or not isinstance(payload.get("results"), list)
        ):
            return []

        results: list[dict[str, object]] = []
        for item in payload["results"][:top_k]:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            results.append(
                {
                    key: item[key]
                    for key in (
                        "document_id",
                        "title",
                        "source",
                        "section",
                        "content",
                        "rank",
                        "score",
                        "vector_score",
                        "rerank_score",
                    )
                    if key in item
                }
            )
        return results
