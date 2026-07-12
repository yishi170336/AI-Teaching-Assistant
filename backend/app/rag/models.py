from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class PageDocument:
    text: str
    source: str
    page: int
    chapter: str
    section: str
    doc_type: str = "textbook"
    bbox: list[float] | None = None
    element_type: str = "text"
    parent_id: str | None = None
    image_path: str | None = None
    content_hash: str | None = None
    extra: dict[str, Any] | None = None


@dataclass
class TextChunk:
    id: str
    text: str
    source: str
    chapter: str
    section: str
    page_start: int | None
    page_end: int | None
    doc_type: str
    knowledge_tags: list[str]
    element_type: str = "text"
    bbox: list[float] | None = None
    parent_id: str | None = None
    image_path: str | None = None
    content_hash: str | None = None
    multimodal: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalHit:
    chunk: TextChunk
    score: float
    vector_score: float
    bm25_score: float
    rerank_score: float
    graph_score: float = 0.0
    cross_encoder_score: float = 0.0
    image_score: float = 0.0

    def source_dict(self) -> dict[str, Any]:
        return {
            "id": self.chunk.id,
            "source": self.chunk.source,
            "chapter": self.chunk.chapter,
            "section": self.chunk.section,
            "page_start": self.chunk.page_start,
            "page_end": self.chunk.page_end,
            "score": round(self.score, 4),
            "doc_type": self.chunk.doc_type,
            "excerpt": self.chunk.text[:360],
            "knowledge_tags": self.chunk.knowledge_tags[:8],
            "element_type": self.chunk.element_type,
            "bbox": self.chunk.bbox,
            "image_path": self.chunk.image_path,
            "parent_id": self.chunk.parent_id,
            "vector_score": round(self.vector_score, 4),
            "bm25_score": round(self.bm25_score, 4),
            "graph_score": round(self.graph_score, 4),
            "image_score": round(self.image_score, 4),
            "rerank_score": round(self.rerank_score, 4),
        }

