from __future__ import annotations

import numpy as np

from backend.app.rag.models import TextChunk
from backend.app.rag.retriever import HybridRetriever


def _chunk(chunk_id: str, section_id: str) -> TextChunk:
    return TextChunk(
        id=chunk_id,
        text="教材正文",
        source="book.pdf",
        chapter="第一章",
        section="1.1 PN结",
        page_start=1,
        page_end=1,
        doc_type="textbook",
        knowledge_tags=[],
        multimodal={"section_id": section_id},
    )


def test_entity_vector_retrieval_honors_alias_and_expands_strong_one_hop():
    import faiss

    retriever = object.__new__(HybridRetriever)
    retriever.chunks = [_chunk("chunk-a", "section:a:1"), _chunk("chunk-b", "section:a:2")]
    vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    index = faiss.IndexFlatIP(2)
    index.add(vectors)
    retriever._entity_index = index
    retriever._entity_items = [
        {
            "entity_id": "entity:pn",
            "name": "PN结",
            "aliases": ["p-n 结"],
            "section_ids": ["section:a:1"],
        },
        {
            "entity_id": "entity:barrier",
            "name": "势垒区",
            "aliases": [],
            "section_ids": ["section:a:2"],
        },
    ]
    retriever._entity_relations = [{
        "source": "entity:pn",
        "target": "entity:barrier",
        "type": "concept_relation",
        "relation": "包含",
        "strength": 8.0,
        "confidence": 0.9,
    }]

    scores = retriever._entity_graph_scores(
        "p-n 结是什么", np.asarray([[-1.0, 0.0]], dtype=np.float32)
    )

    assert scores[0] == 1.0
    assert scores[1] >= 0.85
