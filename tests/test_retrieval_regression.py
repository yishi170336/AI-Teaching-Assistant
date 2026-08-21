from __future__ import annotations

from types import SimpleNamespace

from backend.app.rag.retrieval_regression import audit_retrieval_regression


class _Retriever:
    def __init__(self, text: str):
        self.text = text

    def search(self, _query: str, k: int = 12):
        if not self.text:
            return []
        return [SimpleNamespace(
            chunk=SimpleNamespace(
                id="chunk-1",
                chapter="第一章",
                section="1.1 PN结",
                knowledge_tags=["PN结"],
                text=self.text,
                source="book.pdf",
                page_start=18,
            ),
            score=0.8,
        )][:k]


def _cases():
    return [{
        "id": "pn",
        "category": "章节检索",
        "query": "PN结如何形成？",
        "expected_terms": ["PN结", "空间电荷区", "扩散"],
    }]


def test_candidate_retrieval_cannot_regress_below_active_index():
    audit = audit_retrieval_regression(
        _Retriever("PN结"),
        _Retriever("PN结通过扩散形成空间电荷区"),
        cases=_cases(),
    )

    assert audit["status"] == "failed"
    assert audit["regressions"] == 1
    assert audit["results"][0]["regressed"] is True


def test_candidate_retrieval_passes_when_term_coverage_is_not_lower():
    audit = audit_retrieval_regression(
        _Retriever("PN结通过扩散形成空间电荷区"),
        _Retriever("PN结与空间电荷区"),
        cases=_cases(),
    )

    assert audit["status"] == "passed"
    assert audit["candidate_average_term_coverage"] >= audit["baseline_average_term_coverage"]


def test_first_index_still_requires_each_regression_query_to_return_hits():
    audit = audit_retrieval_regression(_Retriever(""), None, cases=_cases())

    assert audit["status"] == "failed"
    assert audit["regressions"] == 1

