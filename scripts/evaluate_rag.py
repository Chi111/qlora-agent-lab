from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POSITIVE_SET = PROJECT_ROOT / "data/knowledge/eval/repair-rag-golden.jsonl"
NEGATIVE_SET = PROJECT_ROOT / "data/knowledge/eval/repair-rag-negative.jsonl"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def evaluate(base_url: str, api_key: str, timeout: float) -> dict[str, Any]:
    positives = load_jsonl(POSITIVE_SET)
    negatives = load_jsonl(NEGATIVE_SET)
    hit_at_4 = 0
    top_1 = 0
    negative_passes = 0
    failures: list[dict[str, str]] = []

    with httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout) as client:
        for case in positives:
            response = client.post(
                "/internal/knowledge/search",
                headers={"x-internal-api-key": api_key},
                json={
                    "query": case["query"],
                    "topK": 4,
                    "knowledgeScope": "repair",
                },
            )
            response.raise_for_status()
            results = response.json().get("results", [])
            matches = [
                index
                for index, result in enumerate(results)
                if result.get("source") == case["expected_source"]
                and result.get("section") == case["expected_section"]
            ]
            if matches:
                hit_at_4 += 1
                if matches[0] == 0:
                    top_1 += 1
            else:
                failures.append(
                    {"case_id": case["case_id"], "reason": "expected chunk not in Top 4"}
                )

        for case in negatives:
            response = client.post(
                "/internal/knowledge/search",
                headers={"x-internal-api-key": api_key},
                json={
                    "query": case["query"],
                    "topK": 4,
                    "knowledgeScope": "repair",
                },
            )
            response.raise_for_status()
            if not response.json().get("results"):
                negative_passes += 1
            else:
                failures.append({"case_id": case["case_id"], "reason": "unexpected answer"})

    return {
        "positive_cases": len(positives),
        "hit_at_4": hit_at_4 / len(positives),
        "top_1_accuracy": top_1 / len(positives),
        "negative_cases": len(negatives),
        "no_answer_accuracy": negative_passes / len(negatives),
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the live BGE/Qdrant/Rerank pipeline.")
    parser.add_argument("--base-url", default="http://127.0.0.1:4111/api")
    parser.add_argument("--api-key", default=os.getenv("AGENT_RAG_INTERNAL_API_KEY"))
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    if not args.api_key:
        raise SystemExit("Pass --api-key or set AGENT_RAG_INTERNAL_API_KEY.")

    report = evaluate(args.base_url, args.api_key, args.timeout)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.strict and (
        report["hit_at_4"] < 0.9
        or report["top_1_accuracy"] < 0.8
        or report["no_answer_accuracy"] < 0.8
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
