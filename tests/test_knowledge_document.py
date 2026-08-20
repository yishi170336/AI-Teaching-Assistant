from __future__ import annotations

import json

from backend.app.rag.knowledge_document import (
    KnowledgeUnit,
    compile_knowledge_document,
    enrich_formula_knowledge,
    knowledge_units_to_chunks,
    write_knowledge_document,
)
from backend.app.rag.models import PageDocument
from backend.app.rag.multimodal import LayoutElement


def test_compiler_fuses_consecutive_ocr_pages_and_verified_visual_knowledge(tmp_path):
    documents = [
        PageDocument(
            text="2.6 电流源电路及其应用\n镜像电流源利用匹配晶体管复制电流。",
            source="教材.pdf",
            page=95,
            source_page=95,
            chapter="第二章 基本放大电路",
            section="2.6 电流源电路及其应用",
            extra={
                "text_blocks": [
                    {"type": "section_heading", "text": "2.6 电流源电路及其应用"},
                    {"type": "paragraph", "text": "镜像电流源利用匹配晶体管复制电流。"},
                ]
            },
        ),
        PageDocument(
            text="基极电流会带来复制误差。",
            source="教材.pdf",
            page=96,
            source_page=96,
            chapter="第二章 基本放大电路",
            section="2.6 电流源电路及其应用",
            extra={"text_blocks": [{"type": "paragraph", "text": "基极电流会带来复制误差。"}]},
        ),
    ]
    elements = [
        LayoutElement(
            id="formula:p95:2.6.1",
            source="教材.pdf",
            page=95,
            source_page=95,
            element_type="formula",
            bbox=[10, 10, 100, 40],
            text="I_C1 = I_C2",
            description="两管参数和温度一致时，两个集电极电流近似相等",
            confidence=0.96,
        ),
        LayoutElement(
            id="formula:p96:bad",
            source="教材.pdf",
            page=96,
            source_page=96,
            element_type="formula",
            bbox=[10, 10, 100, 40],
            text="broken latex",
            description="无法确认的公式",
            confidence=0.2,
            uncertain=True,
        ),
        LayoutElement(
            id="circuit:p96:mirror",
            source="教材.pdf",
            page=96,
            source_page=96,
            element_type="circuit",
            bbox=[20, 100, 300, 400],
            description="共用基极电位使匹配晶体管建立镜像电流",
            components=[{"ref": "T1", "type": "晶体管", "role": "参考支路"}],
            confidence=0.94,
        ),
    ]

    units = compile_knowledge_document(documents, elements)

    assert len(units) == 1
    assert units[0].page_start == 95 and units[0].page_end == 96
    assert "公式所表达的知识" in units[0].text
    assert "电路图所表达的知识" in units[0].text
    assert "broken latex" not in units[0].text
    assert units[0].quality["status"] == "review"
    assert {item["id"] for item in units[0].knowledge_elements} == {
        "formula:p95:2.6.1",
        "formula:p96:bad",
        "circuit:p96:mirror",
    }

    chunks = knowledge_units_to_chunks(units, max_chars=300)
    assert chunks
    assert all(chunk.parent_id == units[0].id for chunk in chunks)
    assert all(chunk.page_start == 95 and chunk.page_end == 96 for chunk in chunks)

    write_knowledge_document(units, tmp_path)
    payload = json.loads((tmp_path / "book_knowledge_document.json").read_text(encoding="utf-8"))
    assert payload["units"][0]["knowledge_elements"][1]["uncertain"] is True
    assert "镜像电流源" in (tmp_path / "book_knowledge_document.md").read_text(encoding="utf-8")


def test_formula_enrichment_adds_only_grounded_natural_language_knowledge(tmp_path):
    unit = KnowledgeUnit(
        id="knowledge-unit:formula",
        source="教材.pdf",
        title_path=["第二章", "2.6 电流源"],
        chapter="第二章",
        section="2.6 电流源",
        page_start=96,
        page_end=96,
        text="两个匹配晶体管共用基极电位。",
        source_text="OCR",
        knowledge_elements=[{
            "id": "formula:1",
            "type": "formula",
            "raw_text": "I_C1 = I_C2",
            "meaning": '变量：[{"symbol":"I_C1","meaning":"集电极电流"}]',
            "nearby_text": "两管参数一致时，两管集电极电流近似相等。",
            "uncertain": False,
            "included_in_graph": False,
        }],
        quality={"status": "verified", "warnings": []},
    )

    class FakeClient:
        class Config:
            enabled = True
            model = "qwen3.7-flash"

        config = Config()

        def complete_json(self, _prompt):
            return {"items": [{
                "knowledge_unit_id": unit.id,
                "formulas": [{
                    "id": "formula:1",
                    "knowledge": "两管参数与温度一致时，两个集电极电流近似相等。",
                    "confidence": 0.96,
                }],
            }]}

    enriched = enrich_formula_knowledge(
        [unit], FakeClient(), cache_path=tmp_path / "formula-cache.jsonl"
    )

    assert "公式所表达的知识" in enriched[0].text
    assert enriched[0].knowledge_elements[0]["included_in_graph"] is True
    assert enriched[0].quality["formula_knowledge_enriched"] == ["formula:1"]
    assert (tmp_path / "formula-cache.jsonl").exists()
