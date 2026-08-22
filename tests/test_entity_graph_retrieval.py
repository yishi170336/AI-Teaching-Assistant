from __future__ import annotations

import numpy as np

from backend.app.rag.retriever import Schema4Retriever


def test_entity_alias_is_exact_and_strong_one_hop_relation_is_available():
    import faiss

    retriever = object.__new__(Schema4Retriever)
    retriever.entities = {
        "entity:pn": {"id": "entity:pn", "name": "PN结", "aliases": ["p-n 结"]},
        "entity:barrier": {"id": "entity:barrier", "name": "势垒区", "aliases": []},
    }
    retriever.entity_items = [
        {"entity_id": "entity:pn"}, {"entity_id": "entity:barrier"},
    ]
    index = faiss.IndexFlatIP(2)
    index.add(np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    retriever.entity_index = index
    retriever.relationships = [{
        "source": "entity:pn", "target": "entity:barrier",
        "type": "concept_relation", "relation": "包含",
        "strength": 8.0, "confidence": 0.9,
    }]

    entities = retriever._entity_candidates(
        "p-n 结是什么", np.asarray([[-1.0, 0.0]], dtype=np.float32)
    )
    relations = retriever._relationship_candidates(entities, 4)

    assert entities[0]["id"] == "entity:pn"
    assert entities[0]["exact_match"] is True
    assert relations[0]["target"] == "entity:barrier"


def test_relationship_ranking_uses_query_meaning_within_recalled_graph():
    retriever = object.__new__(Schema4Retriever)
    retriever.entities = {
        "entity:feedback": {"id": "entity:feedback", "name": "负反馈"},
        "entity:gain": {"id": "entity:gain", "name": "增益稳定性"},
        "entity:bias": {"id": "entity:bias", "name": "静态工作点"},
    }
    retriever.relationships = [
        {
            "source": "entity:feedback", "target": "entity:bias",
            "relation": "关联", "description": "负反馈可以稳定静态工作点",
            "strength": 9.0, "confidence": 0.9,
        },
        {
            "source": "entity:feedback", "target": "entity:gain",
            "relation": "关联", "description": "负反馈提高放大电路的增益稳定性",
            "strength": 9.0, "confidence": 0.9,
        },
    ]
    entities = [
        {**retriever.entities["entity:feedback"], "retrieval_score": 1.0, "aliases": []},
        {**retriever.entities["entity:gain"], "retrieval_score": 0.9, "aliases": ["放大倍数稳定性"]},
        {**retriever.entities["entity:bias"], "retrieval_score": 0.9, "aliases": []},
    ]

    relationships = retriever._relationship_candidates(
        entities, 2, "为什么负反馈能够提高放大倍数稳定性"
    )

    assert relationships[0]["target"] == "entity:gain"
