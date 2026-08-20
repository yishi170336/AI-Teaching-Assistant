from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from backend.app.rag.models import PageDocument, TextChunk
from backend.app.rag.multimodal import LayoutElement
from backend.app.rag.ontology import extract_course_concepts


KNOWLEDGE_DOCUMENT_SCHEMA_VERSION = "1.0-textbook-multimodal"
GRAPH_BLOCK_TYPES = {"paragraph", "list_item"}
IGNORED_GRAPH_BLOCK_TYPES = {
    "chapter_heading",
    "section_heading",
    "figure_caption",
    "formula",
    "table",
    "exercise",
    "page_header",
    "page_footer",
    "noise",
}


@dataclass
class KnowledgeUnit:
    id: str
    source: str
    title_path: list[str]
    chapter: str
    section: str
    page_start: int
    page_end: int
    text: str
    source_text: str
    evidence_ids: list[str] = field(default_factory=list)
    knowledge_elements: list[dict[str, Any]] = field(default_factory=list)
    quality: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _stable_id(*values: object) -> str:
    raw = "|".join(str(value) for value in values)
    return "knowledge-unit:" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def _compact(value: str) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def _page_body(document: PageDocument) -> str:
    """Return graph-ready prose, leaving identifiers and raw math as evidence."""

    extra = document.extra if isinstance(document.extra, dict) else {}
    blocks = extra.get("text_blocks")
    if isinstance(blocks, list) and blocks:
        prose = [
            _compact(str(block.get("text", "")))
            for block in blocks
            if isinstance(block, dict)
            and str(block.get("type", "paragraph")) in GRAPH_BLOCK_TYPES
            and _compact(str(block.get("text", "")))
        ]
        if prose:
            return "\n\n".join(prose)

    kept: list[str] = []
    for raw_line in document.text.splitlines():
        line = _compact(raw_line)
        if not line:
            continue
        if re.fullmatch(r"(?:图|表|式)\s*[（(]?\d+(?:\.\d+)+[）)]?.*", line):
            continue
        if re.fullmatch(r"[（(]?\d+(?:\.\d+){1,3}[a-z]?[）)]?", line, re.I):
            continue
        kept.append(line)
    return "\n\n".join(kept)


def _component_summary(components: Iterable[dict[str, Any]]) -> str:
    values: list[str] = []
    for component in components:
        if not isinstance(component, dict):
            continue
        component_type = _compact(component.get("type", ""))
        role = _compact(component.get("role", "") or component.get("function", ""))
        phrase = "，".join(part for part in (component_type, role) if part)
        if phrase and phrase not in values:
            values.append(phrase)
    return "；".join(values[:16])


def _element_record(element: LayoutElement) -> tuple[dict[str, Any], str]:
    raw_text = _compact(element.text)
    description = _compact(element.description)
    caption = _compact(element.caption)
    element_type = str(element.element_type).strip().lower()
    confidence = float(element.confidence or 0.0)
    graph_safe = not element.uncertain and confidence >= 0.45
    semantic = ""

    if element_type == "formula":
        # Raw LaTeX and equation numbers remain evidence. Only a verified
        # natural-language interpretation is allowed into entity extraction.
        graph_safe = (
            graph_safe
            and len(description) >= 8
            and not description.startswith(("变量：", "变量:", "[{", "{"))
        )
        if graph_safe:
            semantic = f"公式所表达的知识：{description}"
    elif element_type == "table":
        meaning = description or raw_text
        graph_safe = graph_safe and len(meaning) >= 8
        if graph_safe:
            semantic = f"表格给出的知识：{meaning}"
    elif element_type == "circuit":
        meaning = description
        component_summary = _component_summary(element.components)
        graph_safe = graph_safe and len(meaning) >= 8
        if graph_safe:
            semantic = f"电路图所表达的知识：{meaning}"
            if component_summary:
                semantic += f"。其中元件类型及作用包括：{component_summary}"
    elif element_type in {"figure", "image", "diagram", "curve"}:
        meaning = description or raw_text
        graph_safe = graph_safe and len(meaning) >= 8
        if graph_safe:
            semantic = f"插图所表达的知识：{meaning}"

    record = {
        "id": element.id,
        "type": element_type,
        "page": int(element.source_page or element.page),
        "caption": caption,
        "raw_text": raw_text,
        "meaning": description,
        "nearby_text": _compact(element.nearby_text)[:1600],
        "image_path": element.image_path,
        "bbox": list(element.bbox),
        "components": element.components,
        "nets": element.nets,
        "confidence": confidence,
        "uncertain": bool(element.uncertain),
        "included_in_graph": bool(semantic),
    }
    return record, semantic


def compile_knowledge_document(
    documents: Iterable[PageDocument],
    elements: Iterable[LayoutElement],
    *,
    target_chars: int = 2600,
) -> list[KnowledgeUnit]:
    """Compile OCR prose and verified visual semantics into coherent units."""

    element_map: dict[tuple[str, int], list[LayoutElement]] = {}
    for element in elements:
        page = int(element.source_page or element.page)
        element_map.setdefault((element.source, page), []).append(element)

    page_records: list[dict[str, Any]] = []
    for document in sorted(
        documents,
        key=lambda item: (item.source, item.source_page or item.page, item.page),
    ):
        if document.doc_type in {"question", "exercise"}:
            continue
        source_page = int(document.source_page or document.page)
        knowledge_elements: list[dict[str, Any]] = []
        supplements: list[str] = []
        warnings: list[str] = []
        for element in sorted(
            element_map.get((document.source, source_page), []),
            key=lambda item: (item.reading_order, item.bbox[1] if len(item.bbox) == 4 else 0),
        ):
            if element.element_type == "text":
                continue
            record, semantic = _element_record(element)
            knowledge_elements.append(record)
            if semantic:
                supplements.append(semantic)
            elif element.uncertain:
                warnings.append(f"{element.id}: uncertain {element.element_type}")

        body = _page_body(document)
        graph_text = "\n\n".join(part for part in [body, *supplements] if part).strip()
        if not graph_text:
            continue
        page_records.append({
            "source": document.source,
            "chapter": document.chapter,
            "section": document.section,
            "page": source_page,
            "graph_text": graph_text,
            "source_text": document.text.strip(),
            "evidence_ids": [
                f"ocr:{document.source}:p{source_page}",
                *[str(item["id"]) for item in knowledge_elements],
            ],
            "knowledge_elements": knowledge_elements,
            "warnings": warnings,
        })

    units: list[KnowledgeUnit] = []
    current: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal current
        if not current:
            return
        first, last = current[0], current[-1]
        text = "\n\n".join(str(item["graph_text"]) for item in current).strip()
        source_text = "\n\n".join(
            f"[第 {item['page']} 页]\n{item['source_text']}" for item in current
        ).strip()
        evidence_ids = list(dict.fromkeys(
            evidence_id
            for item in current
            for evidence_id in item["evidence_ids"]
        ))
        element_values = [
            element
            for item in current
            for element in item["knowledge_elements"]
        ]
        warnings = [warning for item in current for warning in item["warnings"]]
        chapter = str(first["chapter"])
        section = str(first["section"])
        units.append(KnowledgeUnit(
            id=_stable_id(
                first["source"], chapter, section, first["page"], last["page"], text
            ),
            source=str(first["source"]),
            title_path=list(dict.fromkeys(part for part in (chapter, section) if part)),
            chapter=chapter,
            section=section,
            page_start=int(first["page"]),
            page_end=int(last["page"]),
            text=text,
            source_text=source_text,
            evidence_ids=evidence_ids,
            knowledge_elements=element_values,
            quality={
                "status": "review" if warnings else "verified",
                "warnings": warnings,
            },
        ))
        current = []

    for item in page_records:
        same_scope = bool(current) and (
            current[-1]["source"] == item["source"]
            and current[-1]["chapter"] == item["chapter"]
            and current[-1]["section"] == item["section"]
            and int(current[-1]["page"]) + 1 == int(item["page"])
        )
        projected = sum(len(str(value["graph_text"])) for value in current) + len(
            str(item["graph_text"])
        )
        if current and (not same_scope or projected > target_chars):
            flush()
        current.append(item)
    flush()
    return units


def enrich_formula_knowledge(
    units: Iterable[KnowledgeUnit],
    client: Any | None,
    *,
    cache_path: Path | None = None,
    batch_size: int = 4,
) -> list[KnowledgeUnit]:
    """Turn verified formula evidence into natural-language knowledge with Qwen."""

    values = list(units)
    if client is None or not getattr(getattr(client, "config", None), "enabled", False):
        return values
    model = str(getattr(client.config, "model", ""))
    cache: dict[str, dict[str, Any]] = {}
    if cache_path and cache_path.exists():
        try:
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                item = json.loads(line)
                if isinstance(item, dict) and item.get("knowledge_unit_id"):
                    cache[str(item["knowledge_unit_id"])] = item
        except (OSError, ValueError, json.JSONDecodeError):
            cache = {}

    def candidates(unit: KnowledgeUnit) -> list[dict[str, Any]]:
        return [
            {
                "id": str(element.get("id", "")),
                "raw_formula": str(element.get("raw_text", "")),
                "known_variables": str(element.get("meaning", "")),
                "nearby_text": str(element.get("nearby_text", ""))[:1000],
            }
            for element in unit.knowledge_elements
            if element.get("type") == "formula"
            and not element.get("uncertain")
            and not element.get("included_in_graph")
            and str(element.get("raw_text", "")).strip()
        ]

    pending: list[tuple[KnowledgeUnit, str, list[dict[str, Any]]]] = []
    results: dict[str, list[dict[str, Any]]] = {}
    for unit in values:
        formula_values = candidates(unit)
        if not formula_values:
            continue
        content_hash = hashlib.sha256(json.dumps(
            {"text": unit.text, "formulas": formula_values},
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")).hexdigest()
        cached = cache.get(unit.id)
        if (
            cached
            and cached.get("content_hash") == content_hash
            and cached.get("model") == model
            and cached.get("schema_version") == "1.0-formula-knowledge"
        ):
            results[unit.id] = cached.get("formulas", [])
        else:
            pending.append((unit, content_hash, formula_values))

    prompt = """你是电子电路教材公式知识整理器。输入包含已经通过视觉识别的公式、变量说明、邻近正文和本知识单元正文。
只依据输入，把公式表达的成立条件、变量关系及可直接推出的结论写成简洁中文知识陈述。公式编号、图号、变量符号和局部元件编号不是知识实体；不得仅复述符号表，不得补充教材没有说明的结论。看不清、上下文不足或无法确定含义时返回空 knowledge。
只返回 JSON：{"items":[{"knowledge_unit_id":"...","formulas":[{"id":"...","knowledge":"...","confidence":0.0}]}]}。
输入："""
    for start in range(0, len(pending), max(1, batch_size)):
        batch = pending[start : start + max(1, batch_size)]
        payload = [
            {
                "knowledge_unit_id": unit.id,
                "chapter": unit.chapter,
                "section": unit.section,
                "context": unit.text[:2600],
                "formulas": formula_values,
            }
            for unit, _content_hash, formula_values in batch
        ]
        response = client.complete_json(prompt + json.dumps(payload, ensure_ascii=False))
        items = response.get("items", []) if isinstance(response, dict) else []
        returned = {
            str(item.get("knowledge_unit_id", "")): item.get("formulas", [])
            for item in items
            if isinstance(item, dict) and item.get("knowledge_unit_id")
        }
        for unit, content_hash, _formula_values in batch:
            formula_results = returned.get(unit.id, [])
            results[unit.id] = formula_results if isinstance(formula_results, list) else []
            cache[unit.id] = {
                "schema_version": "1.0-formula-knowledge",
                "knowledge_unit_id": unit.id,
                "content_hash": content_hash,
                "model": model,
                "formulas": results[unit.id],
            }

    for unit in values:
        by_id = {
            str(item.get("id", "")): item
            for item in results.get(unit.id, [])
            if isinstance(item, dict)
        }
        supplements: list[str] = []
        enriched_ids: list[str] = []
        for element in unit.knowledge_elements:
            result = by_id.get(str(element.get("id", "")))
            if result is None:
                continue
            knowledge = _compact(result.get("knowledge", ""))
            try:
                confidence = float(result.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            if (
                confidence < 0.65
                or len(knowledge) < 8
                or any(marker in knowledge for marker in ("无法确定", "无法识别", "上下文不足"))
            ):
                continue
            element["meaning"] = knowledge
            element["included_in_graph"] = True
            element["knowledge_confidence"] = confidence
            supplements.append(f"公式所表达的知识：{knowledge}")
            enriched_ids.append(str(element.get("id", "")))
        if supplements:
            unit.text = "\n\n".join([unit.text, *supplements])
            unit.quality["formula_knowledge_enriched"] = enriched_ids

    if cache_path:
        temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
        temporary.write_text(
            "\n".join(
                json.dumps(cache[key], ensure_ascii=False) for key in sorted(cache)
            ),
            encoding="utf-8",
        )
        temporary.replace(cache_path)
    return values


def _split_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    pieces: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidates = [paragraph]
        if len(paragraph) > max_chars:
            candidates = [
                paragraph[index : index + max_chars]
                for index in range(0, len(paragraph), max_chars)
            ]
        for candidate in candidates:
            if current and len(current) + len(candidate) + 2 > max_chars:
                pieces.append(current)
                overlap = current[-overlap_chars:] if overlap_chars else ""
                current = overlap
            current = f"{current}\n\n{candidate}".strip()
    if current:
        pieces.append(current)
    return pieces


def knowledge_units_to_chunks(
    units: Iterable[KnowledgeUnit],
    *,
    max_chars: int = 1600,
    overlap_chars: int = 120,
) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    for unit in units:
        pieces = _split_text(unit.text, max_chars, overlap_chars)
        for index, text in enumerate(pieces, 1):
            chunk_id = hashlib.sha1(
                f"{unit.id}|{index}|{text}".encode("utf-8")
            ).hexdigest()[:16]
            chunks.append(TextChunk(
                id=chunk_id,
                text=text,
                source=unit.source,
                chapter=unit.chapter,
                section=unit.section,
                page_start=unit.page_start,
                page_end=unit.page_end,
                doc_type="textbook",
                knowledge_tags=extract_course_concepts(text, unit.section),
                element_type="knowledge_unit",
                parent_id=unit.id,
                multimodal={
                    "knowledge_unit_id": unit.id,
                    "evidence_ids": unit.evidence_ids,
                    "knowledge_elements": unit.knowledge_elements,
                    "quality": unit.quality,
                },
            ))
    return chunks


def write_knowledge_document(units: Iterable[KnowledgeUnit], output_dir: Path) -> None:
    values = list(units)
    payload = {
        "schema_version": KNOWLEDGE_DOCUMENT_SCHEMA_VERSION,
        "units": [unit.to_dict() for unit in values],
    }
    (output_dir / "book_knowledge_document.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = ["# 教材完整知识文档", ""]
    previous_path: list[str] = []
    for unit in values:
        if unit.title_path != previous_path:
            for level, title in enumerate(unit.title_path, 2):
                if level - 2 >= len(previous_path) or previous_path[level - 2] != title:
                    lines.extend([f"{'#' * level} {title}", ""])
            previous_path = unit.title_path
        lines.extend([
            f"> 来源：{unit.source}，页码：{unit.page_start}-{unit.page_end}，证据：{', '.join(unit.evidence_ids)}",
            "",
            unit.text,
            "",
        ])
        for element in unit.knowledge_elements:
            meaning = str(element.get("meaning", "")).strip()
            if meaning:
                lines.extend([
                    f"- {element.get('type', 'element')} `{element.get('id', '')}`：{meaning}",
                    "",
                ])
    (output_dir / "book_knowledge_document.md").write_text(
        "\n".join(lines).strip() + "\n",
        encoding="utf-8",
    )
