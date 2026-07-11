from __future__ import annotations

import io

import fitz
from PIL import Image, ImageDraw

from backend.app.rag.models import PageDocument, TextChunk
from backend.app.rag.manager import KnowledgeBaseManager
from backend.app.rag.multimodal import (
    _normalize_circuit_result,
    build_local_knowledge_graph,
    enhance_pdf,
)


def _diagram_png() -> bytes:
    image = Image.new("RGB", (420, 220), "white")
    draw = ImageDraw.Draw(image)
    draw.line((30, 110, 120, 110), fill="black", width=3)
    draw.rectangle((120, 80, 220, 140), outline="black", width=3)
    draw.line((220, 110, 390, 110), fill="black", width=3)
    draw.line((30, 110, 30, 190, 390, 190, 390, 110), fill="black", width=3)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_pdf_fallback_preserves_layout_and_image_metadata(tmp_path):
    pdf_path = tmp_path / "lesson.pdf"
    pdf = fitz.open()
    page = pdf.new_page(width=500, height=700)
    page.insert_text((40, 50), "R1 + R2 = 3 ohm")
    page.insert_image(fitz.Rect(40, 100, 460, 320), stream=_diagram_png())
    pdf.save(pdf_path)
    pdf.close()

    docs = [PageDocument("R1 + R2 = 3 ohm", pdf_path.name, 1, "chapter", "section")]
    kept, elements, audit = enhance_pdf(pdf_path, docs, tmp_path / "index")

    assert kept == docs
    assert audit[0]["keep"] is True
    assert all(len(element.bbox) == 4 for element in elements)
    image = next(element for element in elements if element.image_path)
    assert (tmp_path / "index" / image.image_path).exists()
    assert image.page == 1
    assert image.content_hash


def test_multimodal_chunk_and_graph_keep_circuit_relationships():
    chunk = TextChunk(
        id="circuit-1",
        text="R1 连接节点 n1 与 n2",
        source="lesson.pdf",
        chapter="第一章",
        section="串联电路",
        page_start=8,
        page_end=8,
        doc_type="multimodal",
        knowledge_tags=["电阻", "串联"],
        element_type="circuit",
        multimodal={
            "components": [{"id": "R1", "type": "resistor", "terminals": ["n1", "n2"]}],
            "nets": [{"id": "n1", "terminals": ["R1.1"]}],
        },
    )
    graph = build_local_knowledge_graph([chunk])

    assert any(node["type"] == "component" for node in graph["nodes"])
    assert any(edge["type"] == "MENTIONS" for edge in graph["edges"])
    assert any(edge["type"] == "CONTAINS" for edge in graph["edges"])
    assert any(edge["type"] == "CONNECTED_TO" for edge in graph["edges"])


def test_malformed_vision_json_is_safely_normalized():
    value = _normalize_circuit_result({
        "is_circuit": "false",
        "components": ["R1", {"id": "R2", "type": "resistor"}],
        "nets": [None, {"id": "n1"}],
    })
    assert value["is_circuit"] is False
    assert value["components"] == [{"id": "R2", "type": "resistor"}]
    assert value["nets"] == [{"id": "n1"}]


def test_index_activation_replaces_complete_directory(tmp_path):
    final = tmp_path / "default"
    staging = tmp_path / ".default.building-test"
    final.mkdir()
    staging.mkdir()
    (final / "version.txt").write_text("old", encoding="utf-8")
    (staging / "version.txt").write_text("new", encoding="utf-8")

    KnowledgeBaseManager._activate_index(final, staging)

    assert (final / "version.txt").read_text(encoding="utf-8") == "new"
    assert not staging.exists()
