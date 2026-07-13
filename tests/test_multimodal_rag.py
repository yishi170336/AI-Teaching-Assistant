from __future__ import annotations

import asyncio
import io
from dataclasses import replace

import fitz
from PIL import Image, ImageDraw

from backend.app.rag import manager as manager_module
from backend.app.rag.models import PageDocument, TextChunk
from backend.app.rag.manager import KnowledgeBaseManager
from backend.app.rag.pdf_extract_kit import DetectedRegion, PDFExtractKitAdapter
from backend.app.rag.pipeline import KnowledgeBaseBuildCancelled
from backend.app.rag.multimodal import (
    LayoutElement,
    _analyze_image,
    _formula_latex_from_pdf_geometry,
    _indexable_pdfkit_regions,
    _normalize_circuit_result,
    _normalize_formula_result,
    _safe_partial_noise_fragment,
    build_local_knowledge_graph,
    enhance_pdf,
    project_student_knowledge_graph,
)
from backend.app.services.qwen_multimodal_client import QwenMultimodalAPIError
from backend.app.rag.ontology import (
    extract_formula_concepts,
    meaningful_section,
    normalize_concept_name,
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


def test_pdf_fallback_preserves_layout_and_image_metadata(tmp_path, monkeypatch):
    # Unit tests exercise the auditable fallback without loading 400 MB GPU models.
    monkeypatch.setattr(PDFExtractKitAdapter, "detect", lambda _self, _image: [])
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


def test_graph_separates_documents_pages_concepts_and_components():
    chunk = TextChunk(
        id="circuit-pages",
        text="第101页共射放大电路，Rb设置静态工作点。",
        source="analog_electronics_pages_101_103.pdf",
        chapter="analog_electronics_pages_101_103",
        section="analog_electronics_pages_101_103",
        page_start=101,
        page_end=101,
        doc_type="multimodal",
        knowledge_tags=["analog_electronics_pages_101_103", "formula", "晶体管", "静态工作点"],
        element_type="circuit",
        multimodal={
            "components": [{"id": "Rb", "type": "resistor", "terminals": ["n1", "n2"]}],
            "nets": [{"id": "n1", "terminals": ["Rb.1"]}],
        },
    )
    graph = build_local_knowledge_graph([chunk])
    names_by_type = {
        kind: {node["name"] for node in graph["nodes"] if node["type"] == kind}
        for kind in ("document", "page", "concept", "component")
    }
    assert "第 101–103 页教材节选" in names_by_type["document"]
    assert "第 101 页" in names_by_type["page"]
    assert {"晶体管", "静态工作点", "电阻"}.issubset(names_by_type["concept"])
    assert "formula" not in names_by_type["concept"]
    assert "analog_electronics_pages_101_103" not in names_by_type["concept"]
    assert "Rb" in names_by_type["component"]


def test_pdf_section_number_prefixes_are_removed_from_concepts():
    assert normalize_concept_name(". 1. 3PN结") == "PN结"
    assert meaningful_section("1. 2. 4 二极管的等效电路") == "二极管的等效电路"

    chunk = TextChunk(
        id="section-prefix",
        text="PN结具有单向导电性。",
        source="lesson.pdf",
        chapter="第一章",
        section="1. 1. 3PN结",
        page_start=31,
        page_end=31,
        doc_type="textbook",
        knowledge_tags=[". 1. 3PN结", "PN结"],
        element_type="text",
    )
    graph = build_local_knowledge_graph([chunk])
    concept_names = [
        node["name"] for node in graph["nodes"] if node["type"] == "concept"
    ]

    assert concept_names == ["PN结"]


def test_student_projection_merges_legacy_numbered_concept_aliases():
    legacy_graph = {
        "nodes": [
            {"id": "document:1", "type": "document", "name": "教材"},
            {"id": "page:1", "type": "page", "name": "第 31 页", "page": 31},
            {"id": "chunk:1", "type": "chunk", "name": "正文"},
            {"id": "chunk:2", "type": "chunk", "name": "正文"},
            {"id": "concept:dirty", "type": "concept", "name": ". 1. 3PN结"},
            {"id": "concept:clean", "type": "concept", "name": "PN结"},
        ],
        "edges": [
            {"source": "document:1", "type": "HAS_PAGE", "target": "page:1"},
            {"source": "page:1", "type": "HAS_CHUNK", "target": "chunk:1"},
            {"source": "page:1", "type": "HAS_CHUNK", "target": "chunk:2"},
            {"source": "chunk:1", "type": "MENTIONS", "target": "concept:dirty"},
            {"source": "chunk:2", "type": "MENTIONS", "target": "concept:clean"},
        ],
    }

    projected = project_student_knowledge_graph(legacy_graph)
    concepts = [node for node in projected["nodes"] if node["type"] == "concept"]
    covers = [edge for edge in projected["edges"] if edge["type"] == "COVERS"]

    assert [node["name"] for node in concepts] == ["PN结"]
    assert concepts[0]["evidence_count"] == 2
    assert len(covers) == 1
    assert covers[0]["evidence_count"] == 2


def test_malformed_vision_json_is_safely_normalized():
    value = _normalize_circuit_result({
        "is_circuit": "false",
        "components": ["R1", {"id": "R2", "type": "resistor"}],
        "nets": [None, {"id": "n1"}],
    })
    assert value["is_circuit"] is False
    assert value["components"] == [{"id": "R2", "type": "resistor"}]
    assert value["nets"] == [{"id": "n1"}]


def test_missing_netlist_is_synthesized_without_inventing_values():
    value = _normalize_circuit_result({
        "is_circuit": True,
        "components": [
            {
                "id": "R1",
                "type": "resistor",
                "value": None,
                "terminals": ["n1", "n2"],
            }
        ],
        "nets": [{"id": "n1", "terminals": ["R1.1"]}],
        "netlist": "",
    })

    assert value["netlist"].startswith("* Generated from Qwen3-VL")
    assert "R1 n1 n2 UNKNOWN" in value["netlist"]


def test_partial_cleaning_only_accepts_explicit_publishing_noise():
    assert _safe_partial_noise_fragment("版权所有，扫码关注公众号") is True
    assert _safe_partial_noise_fragment("Q 是英文 Quiescent 的字头") is False
    assert _safe_partial_noise_fragment("2.2 基本共射放大电路的工作原理") is False


def test_waveform_figure_is_not_promoted_to_circuit_when_vision_fails(monkeypatch):
    class FailedVision:
        model = "qwen3-vl-flash"

        def complete_json(self, *_args, **_kwargs):
            raise QwenMultimodalAPIError("invalid json")

    monkeypatch.setattr(
        "backend.app.rag.multimodal._circuit_image_heuristic",
        lambda _image: (True, 0.85),
    )
    element = LayoutElement(
        id="wave",
        source="lesson.pdf",
        page=103,
        element_type="image",
        bbox=[0, 0, 100, 100],
        nearby_text="图2.2.3 基本共射放大电路的波形分析",
    )
    _analyze_image(element, _diagram_png(), FailedVision())
    assert element.element_type == "image"
    assert element.components == []


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


def test_background_build_reports_progress_and_cleans_cache_on_cancel(tmp_path, monkeypatch):
    resources = tmp_path / "resources"
    indexes = tmp_path / "indexes"
    resources.mkdir()
    indexes.mkdir()
    manager = KnowledgeBaseManager()
    cleaned_qdrant: list[str] = []
    monkeypatch.setattr(manager, "resource_dir", lambda _knowledge_base: resources)
    monkeypatch.setattr(manager, "index_dir", lambda knowledge_base: indexes / knowledge_base)
    monkeypatch.setattr(
        "backend.app.rag.manager.delete_qdrant_indexes",
        lambda path: cleaned_qdrant.append(path.name),
    )

    async def fake_worker(knowledge_base, _job_path, _progress_path, _result_path, _api_key):
        manager._update_progress(knowledge_base, 42, "embedding", "正在生成向量")
        while manager._states[knowledge_base]["state"] != "cancelling":
            await asyncio.sleep(0.01)
        raise KnowledgeBaseBuildCancelled("cancelled")

    monkeypatch.setattr(manager, "_run_build_subprocess", fake_worker)

    async def scenario():
        started = manager.start_build("cancel-me")
        assert started["state"] == "building"
        await asyncio.sleep(0.05)
        assert manager.statuses()[0]["progress"] == 42

        cancelling = manager.cancel_build("cancel-me")
        assert cancelling["state"] == "cancelling"
        await manager._tasks["cancel-me"]

        status = manager.statuses()[0]
        assert status["state"] == "cancelled"
        assert "缓存已清理" in status["message"]
        assert not list(indexes.glob(".cancel-me.building-*"))
        assert any(name.startswith(".cancel-me.building-") for name in cleaned_qdrant)

    asyncio.run(scenario())


def test_delete_knowledge_base_removes_index_and_resources(tmp_path, monkeypatch):
    resources_root = tmp_path / "resources"
    indexes_root = tmp_path / "indexes"
    resource_dir = resources_root / "knowledge_bases" / "deletable"
    index_dir = indexes_root / "deletable"
    resource_dir.mkdir(parents=True)
    index_dir.mkdir(parents=True)
    (resource_dir / "lesson.md").write_text("lesson", encoding="utf-8")
    (index_dir / "index_meta.json").write_text("{}", encoding="utf-8")

    manager = KnowledgeBaseManager()
    manager._states["deletable"] = {
        "id": "deletable", "state": "ready", "documents": 1, "chunks": 1,
    }
    monkeypatch.setattr(manager, "resource_dir", lambda _knowledge_base: resource_dir)
    monkeypatch.setattr(manager, "index_dir", lambda _knowledge_base: index_dir)

    asyncio.run(manager.delete("deletable"))

    assert not resource_dir.exists()
    assert not index_dir.exists()
    assert manager.statuses() == []


def test_system_default_delete_preserves_custom_knowledge_base_resources(tmp_path, monkeypatch):
    resources_root = tmp_path / "resources"
    custom_resource = resources_root / "knowledge_bases" / "keep-me" / "lesson.md"
    index_dir = tmp_path / "indexes" / "default"
    custom_resource.parent.mkdir(parents=True)
    index_dir.mkdir(parents=True)
    custom_resource.write_text("custom lesson", encoding="utf-8")
    (resources_root / "default.pdf").write_text("default lesson", encoding="utf-8")
    (resources_root / "default.xlsx").write_text("default questions", encoding="utf-8")
    (index_dir / "index_meta.json").write_text("{}", encoding="utf-8")

    manager = KnowledgeBaseManager()
    manager._states["default"] = {
        "id": "default", "state": "ready", "documents": 1, "chunks": 1,
    }
    monkeypatch.setattr(manager, "resource_dir", lambda _knowledge_base: resources_root)
    monkeypatch.setattr(manager, "index_dir", lambda _knowledge_base: index_dir)

    asyncio.run(manager.delete("default"))

    assert resources_root.exists()
    assert custom_resource.read_text(encoding="utf-8") == "custom lesson"
    assert not (resources_root / "default.pdf").exists()
    assert not (resources_root / "default.xlsx").exists()
    assert not index_dir.exists()
    assert manager.statuses() == []


def test_load_existing_does_not_recreate_deleted_default(tmp_path, monkeypatch):
    resources_root = tmp_path / "resources"
    indexes_root = tmp_path / "indexes"
    custom_resource = resources_root / "knowledge_bases" / "keep-me" / "lesson.md"
    custom_resource.parent.mkdir(parents=True)
    indexes_root.mkdir(parents=True)
    custom_resource.write_text("custom lesson", encoding="utf-8")
    monkeypatch.setattr(manager_module, "settings", replace(
        manager_module.settings,
        resources_dir=resources_root,
        vector_stores_dir=indexes_root,
    ))

    manager = KnowledgeBaseManager()
    manager.load_existing()

    assert [item["id"] for item in manager.statuses()] == ["keep-me"]


def test_load_existing_keeps_default_when_source_files_remain(tmp_path, monkeypatch):
    resources_root = tmp_path / "resources"
    indexes_root = tmp_path / "indexes"
    resources_root.mkdir(parents=True)
    indexes_root.mkdir(parents=True)
    (resources_root / "lesson.md").write_text("default lesson", encoding="utf-8")
    monkeypatch.setattr(manager_module, "settings", replace(
        manager_module.settings,
        resources_dir=resources_root,
        vector_stores_dir=indexes_root,
    ))

    manager = KnowledgeBaseManager()
    manager.load_existing()

    assert manager.statuses() == [{
        "id": "default",
        "state": "missing",
        "documents": 1,
        "chunks": 0,
        "message": "资料已保留，尚未完成知识库构建",
        "progress": 0,
        "stage": "missing",
        "cancellable": False,
        "available": False,
    }]


def test_inline_formula_regions_remain_text_evidence_only():
    regions = [
        DetectedRegion("inline", [0, 0, 20, 10], 0.9, "pdf-extract-kit:formula"),
        DetectedRegion("isolated", [0, 20, 100, 60], 0.8, "pdf-extract-kit:formula"),
        DetectedRegion("isolate_formula", [0, 22, 45, 40], 0.95, "pdf-extract-kit:layout"),
        DetectedRegion("figure", [0, 70, 100, 170], 0.9, "pdf-extract-kit:layout"),
    ]

    selected = _indexable_pdfkit_regions(regions)

    assert [region.category for region in selected] == ["isolate_formula", "figure"]


def test_formula_recognition_normalizes_latex_and_rejects_prose():
    formula = _normalize_formula_result(
        {
            "is_formula": True,
            "latex": r"I_{BQ}=\frac{V_{BB}-U_{BEQ}}{R_b}",
            "plain_text": "IBQ=(VBB-UBEQ)/Rb",
            "confidence": 0.96,
        },
        "",
    )
    prose = _normalize_formula_result(
        {"is_formula": True, "plain_text": "这只是普通正文，没有数学表达式"},
        "",
    )

    assert formula["is_formula"] is True
    assert formula["latex"].startswith("I_{BQ}")
    assert prose["is_formula"] is False


def test_native_pdf_geometry_recovers_subscripts_and_fraction():
    pdf = fitz.open()
    page = pdf.new_page(width=240, height=120)
    page.insert_text((10, 55), "I", fontsize=14)
    page.insert_text((18, 56), "BQ", fontsize=7)
    page.insert_text((30, 55), "=", fontsize=14)
    page.insert_text((55, 44), "V", fontsize=14)
    page.insert_text((64, 45), "BB", fontsize=7)
    page.insert_text((76, 44), "-U", fontsize=14)
    page.insert_text((92, 45), "BEQ", fontsize=7)
    page.insert_text((75, 64), "R", fontsize=14)
    page.insert_text((84, 65), "b", fontsize=7)

    latex = _formula_latex_from_pdf_geometry(page, [5, 20, 130, 80])
    pdf.close()

    assert latex == r"I_{BQ}=\frac{V_{BB}-U_{BEQ}}{R_{b}}"


def test_formula_symbols_map_to_course_concepts():
    concepts = extract_formula_concepts(
        r"I_{BQ}=\frac{V_{BB}-U_{BEQ}}{R_b},\quad I_{CQ}=\beta I_{BQ}"
    )

    assert {"静态工作点", "电流放大", "直流电源", "电阻"}.issubset(concepts)


def test_student_graph_projection_hides_chunks_formulas_and_nets():
    chunks = [
        TextChunk(
            id="text-1", text="共射放大电路使用晶体管", source="lesson.pdf",
            chapter="chapter", section="section", page_start=101, page_end=101,
            doc_type="textbook", knowledge_tags=["晶体管"], element_type="text",
        ),
        TextChunk(
            id="formula-1", text=r"I_{CQ}=\beta I_{BQ}", source="lesson.pdf",
            chapter="chapter", section="section", page_start=101, page_end=101,
            doc_type="multimodal", knowledge_tags=["晶体管"], element_type="formula",
        ),
        TextChunk(
            id="circuit-1", text="Rb 与晶体管相连", source="lesson.pdf",
            chapter="chapter", section="section", page_start=101, page_end=101,
            doc_type="multimodal", knowledge_tags=["晶体管", "电阻"], element_type="circuit",
            multimodal={
                "components": [{"id": "Rb", "type": "resistor", "terminals": ["n1", "n2"]}],
                "nets": [{"id": "n1", "terminals": ["Rb.1"]}],
            },
        ),
    ]

    projected = project_student_knowledge_graph(build_local_knowledge_graph(chunks))
    node_types = {node["type"] for node in projected["nodes"]}

    assert "chunk" not in node_types
    assert "net" not in node_types
    assert "formula" not in node_types
    assert {"document", "page", "concept", "circuit", "component"}.issubset(node_types)
    assert sum(node["type"] == "component" for node in projected["nodes"]) == 1
    assert any(edge["type"] == "COVERS" for edge in projected["edges"])
