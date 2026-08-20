from __future__ import annotations

import json

from backend.app.rag.knowledge_document import (
    KnowledgeUnit,
    compile_knowledge_document,
    enrich_formula_knowledge,
    enrich_knowledge_statements,
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


def test_statement_enrichment_keeps_attributes_conditions_modalities_and_page_evidence(tmp_path):
    unit = KnowledgeUnit(
        id="knowledge-unit:statements",
        source="教材.pdf",
        title_path=["2.6 电流源"],
        chapter="第二章",
        section="2.6 电流源",
        page_start=95,
        page_end=96,
        text="镜像电流源用于复制参考电流。\n\n忽略基极电流时，输出电流约等于参考电流。",
        source_text=(
            "[第 95 页]\n镜像电流源用于复制参考电流。\n\n"
            "[第 96 页]\n忽略基极电流时，输出电流约等于参考电流。"
        ),
        evidence_ids=["ocr:教材.pdf:p95", "ocr:教材.pdf:p96"],
        quality={"status": "verified", "warnings": []},
    )

    class FakeClient:
        class Config:
            enabled = True
            model = "qwen3.7-flash"

        config = Config()

        def complete_json(self, _prompt):
            return {"statements": [{
                "statement_type": "relation",
                "subject": "镜像电流源",
                "subject_type": "电路",
                "predicate_original": "用于复制",
                "predicate_normalized": "COPIES",
                "object": "参考电流",
                "object_type": "电路参数",
                "qualifiers": [{"key": "condition", "value": "晶体管匹配时"}],
                "evidence_text": "镜像电流源用于复制参考电流。",
                "modality": "text",
                "confidence": 0.96,
            }, {
                "statement_type": "attribute",
                "subject": "镜像电流源",
                "subject_type": "电路",
                "predicate_original": "输出电流关系",
                "predicate_normalized": "HAS_FORMULA",
                "value": "输出电流约等于参考电流",
                "value_type": "formula",
                "qualifiers": ["忽略基极电流时"],
                "evidence_text": "忽略基极电流时，输出电流约等于参考电流。",
                "modality": "formula",
                "confidence": 0.94,
            }]}

    enriched = enrich_knowledge_statements(
        [unit], FakeClient(), cache_path=tmp_path / "statement-cache.jsonl"
    )

    assert len(enriched[0].statements) == 2
    assert enriched[0].statements[0].qualifiers == ["条件：晶体管匹配时"]
    formula = enriched[0].statements[1]
    assert formula.statement_type == "attribute"
    assert formula.qualifiers == ["忽略基极电流时"]
    assert formula.evidence_id == "ocr:教材.pdf:p96"
    assert formula.source_page == 96
    assert formula.modality == "text"
    assert enriched[0].quality["knowledge_statement_count"] == 2


def test_uncovered_verified_image_gets_grounded_attribute_fallback(tmp_path):
    image_id = "image:p96:curve"
    meaning = "该图为双极型晶体管输出特性曲线，展示集电极电流随集电极-发射极电压的变化。"
    unit = KnowledgeUnit(
        id="knowledge-unit:image",
        source="教材.pdf",
        title_path=["2.6 电流源"],
        chapter="第二章",
        section="2.6 电流源",
        page_start=96,
        page_end=96,
        text="晶体管的输出特性体现恒流特点。",
        source_text="[第 96 页]\n晶体管的输出特性体现恒流特点。",
        evidence_ids=["ocr:教材.pdf:p96", image_id],
        knowledge_elements=[{
            "id": image_id, "type": "image", "page": 96,
            "meaning": meaning, "raw_text": "", "caption": "",
            "confidence": 0.95, "included_in_graph": True,
        }],
        quality={"status": "verified", "warnings": []},
    )

    class EmptyClient:
        class Config:
            enabled = True
            model = "qwen3.7-flash"

        config = Config()

        def complete_json(self, _prompt):
            return {}

    enriched = enrich_knowledge_statements(
        [unit], EmptyClient(), cache_path=tmp_path / "empty-statement-cache.jsonl"
    )

    assert len(enriched[0].statements) == 1
    fallback = enriched[0].statements[0]
    assert fallback.statement_type == "attribute"
    assert fallback.subject == "双极型晶体管输出特性曲线"
    assert fallback.evidence_id == image_id
    assert fallback.modality == "image"


def test_semantic_chunking_does_not_cut_an_oversized_formula_statement():
    atomic_sentence = "在晶体管参数与温度一致时，" + "集电极电流保持近似相等" * 30 + "。"
    unit = KnowledgeUnit(
        id="knowledge-unit:long",
        source="教材.pdf",
        title_path=["2.6 电流源"],
        chapter="第二章",
        section="2.6 电流源",
        page_start=95,
        page_end=95,
        text=atomic_sentence,
        source_text=atomic_sentence,
    )

    chunks = knowledge_units_to_chunks([unit], max_chars=120, overlap_chars=20)

    assert len(chunks) == 1
    assert chunks[0].text == atomic_sentence
