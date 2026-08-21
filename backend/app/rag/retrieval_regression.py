from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


DEFAULT_REGRESSION_CASES = [
    {
        "id": "section-retrieval",
        "category": "章节检索",
        "query": "PN结形成时空间电荷区、扩散运动和漂移运动之间有什么关系？",
        "expected_terms": ["PN结", "空间电荷区", "扩散", "漂移"],
    },
    {
        "id": "formula-question",
        "category": "公式问答",
        "query": "二极管伏安特性方程表示什么，各参数有什么含义？",
        "expected_terms": ["二极管", "伏安特性", "正向"],
    },
    {
        "id": "table-question",
        "category": "表格问答",
        "query": "N型半导体与P型半导体的多数载流子和少数载流子如何对比？",
        "expected_terms": ["N型半导体", "P型半导体", "多数载流子", "少数载流子"],
    },
    {
        "id": "course-image-question",
        "category": "课程图片",
        "query": "共射放大电路图中基极、集电极和发射极周围元件分别起什么作用？",
        "expected_terms": ["共射", "基极", "集电极", "发射极"],
    },
    {
        "id": "alias-question",
        "category": "别名检索",
        "query": "p-n结在正向偏置时为什么能够导通？",
        "expected_terms": ["PN结", "正向偏置", "导通"],
    },
    {
        "id": "cross-section-relation",
        "category": "跨章节关系",
        "query": "静态工作点如何影响晶体管放大电路的非线性失真？",
        "expected_terms": ["静态工作点", "晶体管", "放大", "失真"],
    },
]


def load_regression_cases(path: Path | None = None) -> list[dict[str, Any]]:
    if path and path.is_file():
        value = json.loads(path.read_text(encoding="utf-8"))
        cases = value.get("cases", []) if isinstance(value, dict) else []
        if isinstance(cases, list) and cases:
            return [dict(item) for item in cases if isinstance(item, dict)]
    return [dict(item) for item in DEFAULT_REGRESSION_CASES]


def _normalized(value: str) -> str:
    return re.sub(r"[^\w\u3400-\u9fff]+", "", value.casefold())


def _evaluate(retriever: Any, case: dict[str, Any], *, k: int) -> dict[str, Any]:
    hits = retriever.search(str(case["query"]), k=k)
    combined = _normalized(" ".join(
        " ".join(filter(None, [
            hit.chunk.chapter,
            hit.chunk.section,
            " ".join(hit.chunk.knowledge_tags),
            hit.chunk.text,
        ]))
        for hit in hits
    ))
    terms = [str(value) for value in case.get("expected_terms", []) if str(value).strip()]
    matched = [term for term in terms if _normalized(term) in combined]
    return {
        "hits": len(hits),
        "term_coverage": len(matched) / max(1, len(terms)),
        "matched_terms": matched,
        "top_sources": [
            {
                "chunk_id": hit.chunk.id,
                "source": hit.chunk.source,
                "section": hit.chunk.section,
                "page_start": hit.chunk.page_start,
                "score": round(float(hit.score), 6),
            }
            for hit in hits[:3]
        ],
    }


def audit_retrieval_regression(
    candidate: Any,
    baseline: Any | None,
    *,
    cases: list[dict[str, Any]] | None = None,
    k: int = 12,
) -> dict[str, Any]:
    values = cases or load_regression_cases()
    results: list[dict[str, Any]] = []
    regressions = 0
    for case in values:
        candidate_result = _evaluate(candidate, case, k=k)
        baseline_result = _evaluate(baseline, case, k=k) if baseline is not None else None
        regressed = bool(
            candidate_result["hits"] == 0
            or (
                baseline_result is not None
                and candidate_result["term_coverage"] + 1e-9
                < baseline_result["term_coverage"]
            )
        )
        regressions += int(regressed)
        results.append({
            "id": case.get("id"),
            "category": case.get("category"),
            "query": case.get("query"),
            "expected_terms": case.get("expected_terms", []),
            "candidate": candidate_result,
            "baseline": baseline_result,
            "regressed": regressed,
        })
    candidate_average = sum(
        float(item["candidate"]["term_coverage"]) for item in results
    ) / max(1, len(results))
    baseline_average = (
        sum(float(item["baseline"]["term_coverage"]) for item in results) / len(results)
        if baseline is not None and results else None
    )
    return {
        "schema_version": "1.0-fixed-course-retrieval-ab",
        "status": "passed" if regressions == 0 else "failed",
        "cases": len(results),
        "regressions": regressions,
        "candidate_average_term_coverage": candidate_average,
        "baseline_average_term_coverage": baseline_average,
        "results": results,
    }
