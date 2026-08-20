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


KNOWLEDGE_DOCUMENT_SCHEMA_VERSION = "2.0-textbook-atomic-statements"
KNOWLEDGE_STATEMENT_SCHEMA_VERSION = "1.2-multimodal-coverage-facts"
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
class KnowledgeStatement:
    id: str
    knowledge_unit_id: str
    statement_type: str
    subject: str
    subject_type: str
    predicate_original: str
    predicate_normalized: str
    object: str = ""
    object_type: str = ""
    value: str = ""
    value_type: str = ""
    qualifiers: list[str] = field(default_factory=list)
    evidence_text: str = ""
    evidence_id: str = ""
    source_page: int = 0
    modality: str = "text"
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def graph_text(self) -> str:
        condition = "；适用条件：" + "；".join(self.qualifiers) if self.qualifiers else ""
        if self.statement_type == "attribute":
            return (
                f"{self.subject}{self.predicate_original}{self.value}{condition}。"
                f"\n证据定位：第{self.source_page}页 {self.evidence_id}"
            )
        return (
            f"{self.subject}{self.predicate_original}{self.object}{condition}。"
            f"\n证据定位：第{self.source_page}页 {self.evidence_id}"
        )


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
    statements: list[KnowledgeStatement] = field(default_factory=list)
    quality: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _stable_id(*values: object) -> str:
    raw = "|".join(str(value) for value in values)
    return "knowledge-unit:" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def _compact(value: str) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def _atomic_paragraphs(text: str, *, max_chars: int = 700) -> list[str]:
    """Split only at semantic boundaries; never cut a formula or word by characters."""

    values: list[str] = []
    for paragraph in (part.strip() for part in re.split(r"\n\s*\n", text)):
        if not paragraph:
            continue
        if len(paragraph) <= max_chars:
            values.append(paragraph)
            continue
        sentences = [
            item.strip()
            for item in re.split(r"(?<=[。！？；;])\s*", paragraph)
            if item.strip()
        ]
        current = ""
        for sentence in sentences or [paragraph]:
            if current and len(current) + len(sentence) > max_chars:
                values.append(current)
                current = ""
            current = f"{current}{sentence}".strip()
        if current:
            values.append(current)
    return values


_ENTITY_TYPES = {
    "课程概念",
    "电路",
    "器件与元件",
    "电路参数",
    "物理过程与效应",
    "方法与模型",
}


def _statement_entity_type(value: Any) -> str:
    raw = _compact(value)
    aliases = {
        "物理过程与性质": "物理过程与效应",
        "方法": "方法与模型",
        "模型": "方法与模型",
        "参数": "电路参数",
        "器件": "器件与元件",
    }
    value = aliases.get(raw, raw)
    return value if value in _ENTITY_TYPES else "课程概念"


def _concept_name(value: Any) -> str:
    name = _compact(value).strip(" “”\"'，。；：")
    if not name or len(name) > 48:
        return ""
    if re.match(r"^(?:图|表|式|第?\d+页)", name):
        return ""
    if re.fullmatch(r"[A-Za-zΑ-Ωα-ω0-9_{}()\[\].+\-/\s]+", name):
        return ""
    if re.fullmatch(r"[TRCQLDU]_?\d+(?:的.*)?", name, re.I):
        return ""
    local_symbols = re.findall(r"(?:[TRCQLDU]_?\d+|[IV]_[A-Za-z0-9]+|I[RO0])", name, re.I)
    if "晶体管" in name and local_symbols:
        return ""
    for property_name in (
        "参考电流", "输出电流", "集电极电流", "基极电流",
        "发射极电流", "输出电阻", "限流电阻", "发射极电阻",
    ):
        name = re.sub(
            re.escape(property_name) + r"(?:[_ ]?[A-Za-z0-9{}]+)?",
            property_name,
            name,
            flags=re.I,
        )
    if re.search(r"[TRCQLDU]_?\d+", name, re.I):
        return ""
    if re.match(r"^节点\s*[A-Za-z]?\d*$", name, re.I):
        return ""
    if re.search(r"(?:图中|本式|其中|这时|可得|是基本|等于|导致).{4,}", name):
        return ""
    return name


def _element_fact_subject(
    source: dict[str, Any],
    *,
    section: str,
) -> str:
    caption = re.sub(
        r"^(?:图|表|式)\s*[\d.]+(?:\s*[（(][a-zA-Z0-9]+[）)])?\s*",
        "",
        _compact(source.get("caption", "")),
    ).strip("：: ")
    if caption:
        candidate = re.split(r"[，,。；;：:]", caption, maxsplit=1)[0]
        candidate = re.sub(r"^所示的?(?:是)?", "", candidate).strip()
        if value := _concept_name(candidate):
            return value
    meaning = _compact(source.get("meaning", ""))
    match = re.search(
        r"(?:电路类型[：:]|(?:该图|图中|本图)为|(?:该表|本表)总结)([^，,。；;]{2,28})",
        meaning,
    )
    if match:
        candidate = re.sub(r"[（(][^）)]*[）)]", "", match.group(1)).strip()
        if value := _concept_name(candidate):
            return value
    section_name = re.sub(r"^\d+(?:\.\d+)+\s*", "", section).strip()
    label = {
        "formula": "公式知识",
        "table": "表格知识",
        "circuit": "电路图知识",
        "image": "视觉知识",
    }.get(str(source.get("modality", "")), "多模态知识")
    return _concept_name(f"{section_name}{label}") or label


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


def _statement_response_items(response: Any) -> list[Any]:
    if not isinstance(response, dict):
        return []
    direct = response.get("statements")
    if isinstance(direct, list):
        return direct
    for key in ("result", "data", "output"):
        nested = response.get(key)
        if isinstance(nested, dict) and isinstance(nested.get("statements"), list):
            return nested["statements"]
    items = response.get("items")
    if isinstance(items, list):
        if all(isinstance(item, dict) and item.get("subject") for item in items):
            return items
        flattened = [
            statement
            for item in items
            if isinstance(item, dict) and isinstance(item.get("statements"), list)
            for statement in item["statements"]
        ]
        if flattened:
            return flattened
    return []


def _statement_qualifiers(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    labels = {
        "condition": "条件",
        "affected_object": "受影响对象",
        "assumption": "假设",
        "scope": "范围",
    }
    normalized: list[str] = []
    for item in value:
        if isinstance(item, dict):
            raw_value = _compact(item.get("value", item.get("text", "")))
            raw_key = _compact(item.get("key", item.get("type", "")))
            text = (
                f"{labels.get(raw_key, raw_key)}：{raw_value}"
                if raw_key and raw_value else raw_value
            )
        else:
            text = _compact(item)
        if text and text not in normalized:
            normalized.append(text)
    return normalized


def enrich_knowledge_statements(
    units: Iterable[KnowledgeUnit],
    client: Any | None,
    *,
    cache_path: Path | None = None,
) -> list[KnowledgeUnit]:
    """Extract auditable concept-level facts before Microsoft GraphRAG runs."""

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

    prompt = """你是电子电路教材的原子知识陈述编译器。输入 evidence_sources 含 OCR 正文和已验证的公式、表格、电路图语义。
尽可能完整抽取教材明确陈述的核心事实，包括定义、组成、作用、条件、优缺点、因果、参数关系、设计目标和表格单元格知识。
实体必须是脱离页面仍可理解的简短中文名词概念（建议不超过18字）。T1、晶体管T1、T1和T2、R_o、参考电流IR、节点A、图号、式号不得作为关系实体；必须改写为“参考支路晶体管”、“匹配晶体管对”、“参考电流”等概念。object 不得是整句命题，predicate_original 只保留谓词。
公式若表达可概念化的关系，输出 relation；若只能保留等式或数值，输出 attribute，其 subject 仍必须是中文概念。
电路类型只能使用正文或图题明确支持的名称；忽略局部节点和元件编号。
每条事实必须填它所在的 evidence_source_id。evidence_text 必须逐字复制该 evidence_source 中能独立证明事实的最短连续片段；qualifiers 保留“当…时”、“若忽略…”等适用条件。对每个含独立知识的 formula/table/circuit/image 证据源，至少抽取1条事实。
实体类型只能为：课程概念、电路、器件与元件、电路参数、物理过程与效应、方法与模型。
返回 JSON：{"statements":[{"statement_type":"relation|attribute","subject":"...","subject_type":"...","predicate_original":"...","predicate_normalized":"...","object":"...","object_type":"...","value":"...","value_type":"formula|quantity|text","qualifiers":["..."],"evidence_source_id":"...","evidence_text":"...","confidence":0.0}]}。
对每个知识单元返回 6-14 条不重复的高价值陈述；原文不足时宁可少输出，不得补充常识。
    输入："""

    for unit in values:
        page_sources = [
            {
                "id": f"ocr:{unit.source}:p{int(match.group(1))}",
                "page": int(match.group(1)),
                "modality": "text",
                "text": match.group(2),
            }
            for match in re.finditer(
                r"\[第\s*(\d+)\s*页\]\s*\n(.*?)(?=\n\n\[第\s*\d+\s*页\]|\Z)",
                unit.source_text,
                flags=re.S,
            )
        ]
        if not page_sources:
            page_sources = [{
                "id": unit.evidence_ids[0] if unit.evidence_ids else unit.id,
                "page": unit.page_start,
                "modality": "text",
                "text": unit.text,
            }]
        elements = [
            {
                "id": str(element.get("id", "")),
                "page": int(element.get("page", unit.page_start) or unit.page_start),
                "modality": str(element.get("type", "text")),
                "meaning": str(element.get("meaning", "")),
                "text": "\n".join(filter(None, (
                    str(element.get("meaning", "")),
                    str(element.get("raw_text", ""))[:1800],
                ))),
                "caption": str(element.get("caption", "")),
                "confidence": float(element.get("knowledge_confidence", element.get("confidence", 0)) or 0),
            }
            for element in unit.knowledge_elements
            if element.get("included_in_graph")
        ]
        content_hash = hashlib.sha256(json.dumps(
            {
                "schema": KNOWLEDGE_STATEMENT_SCHEMA_VERSION,
                "evidence_sources": [*page_sources, *elements],
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")).hexdigest()
        cached = cache.get(unit.id)
        if (
            cached
            and cached.get("content_hash") == content_hash
            and cached.get("model") == model
            and cached.get("schema_version") == KNOWLEDGE_STATEMENT_SCHEMA_VERSION
        ):
            raw_statements = cached.get("statements", [])
        else:
            payload = {
                "knowledge_unit_id": unit.id,
                "chapter": unit.chapter,
                "section": unit.section,
                "page_start": unit.page_start,
                "page_end": unit.page_end,
                "evidence_sources": [*page_sources, *elements],
            }
            response: dict[str, Any] = {}
            raw_statements: list[Any] = []
            for attempt in range(2):
                response = client.complete_json(
                    prompt
                    + json.dumps(payload, ensure_ascii=False)
                    + (
                        "\n上次未返回可解析的 statements 数组，"
                        "请仅返回紧凑合法 JSON。"
                        if attempt else ""
                    )
                )
                raw_statements = _statement_response_items(response)
                if raw_statements:
                    break
            coverage_response: dict[str, Any] = {}
            if elements:
                coverage_response = client.complete_json(
                    """你是教材多模态事实补全器。对输入的每个 evidence_source 分别输出 1-2 条不重复、有教学价值的原子事实，不得跳过任何证据源。
主体和客体必须是脱离图号仍可理解的中文概念，不得使用 T1、R_o、I_0、节点名或完整句子作实体。
evidence_source_id 必须逐字复制对应 id；evidence_text 必须是该 source.text 中的最短连续原文。公式值或表格数值关系可用 attribute。
仅返回 JSON：{"statements":[{"statement_type":"relation|attribute","subject":"...","subject_type":"课程概念|电路|器件与元件|电路参数|物理过程与效应|方法与模型","predicate_original":"...","predicate_normalized":"...","object":"...","object_type":"...","value":"...","value_type":"formula|quantity|text","qualifiers":[],"evidence_source_id":"...","evidence_text":"...","confidence":0.0}]}。
输入："""
                    + json.dumps({"evidence_sources": elements}, ensure_ascii=False)
                )
                raw_statements = [
                    *raw_statements,
                    *_statement_response_items(coverage_response),
                ]
            cache[unit.id] = {
                "schema_version": KNOWLEDGE_STATEMENT_SCHEMA_VERSION,
                "knowledge_unit_id": unit.id,
                "content_hash": content_hash,
                "model": model,
                "statements": raw_statements,
                "response_keys": sorted(response) if isinstance(response, dict) else [],
                "coverage_response_keys": (
                    sorted(coverage_response)
                    if isinstance(coverage_response, dict) else []
                ),
            }

        evidence_sources = [
            (
                str(source.get("text", "")),
                str(source.get("id", "")),
                int(source.get("page", unit.page_start) or unit.page_start),
                str(source.get("modality", "text")),
            )
            for source in [*page_sources, *elements]
            if str(source.get("id", "")) and str(source.get("text", "")).strip()
        ]
        evidence_by_id = {source_id: item for item in evidence_sources for source_id in [item[1]]}
        statements: list[KnowledgeStatement] = []
        seen: set[tuple[str, str, str, str]] = set()
        for item in raw_statements if isinstance(raw_statements, list) else []:
            if not isinstance(item, dict):
                continue
            statement_type = str(item.get("statement_type", "relation")).lower()
            if statement_type not in {"relation", "attribute"}:
                continue
            subject = _concept_name(item.get("subject", ""))
            predicate_original = _compact(item.get("predicate_original", ""))
            predicate_normalized = _compact(
                item.get("predicate_normalized", predicate_original)
            )
            object_name = _concept_name(item.get("object", "")) if statement_type == "relation" else ""
            value = _compact(item.get("value", "")) if statement_type == "attribute" else ""
            if (
                not subject
                or not predicate_original
                or (statement_type == "relation" and not object_name)
                or (statement_type == "attribute" and not value)
            ):
                continue
            evidence_text = _compact(item.get("evidence_text", ""))
            if not evidence_text:
                continue
            requested_source = evidence_by_id.get(
                str(item.get("evidence_source_id", "")).strip()
            )
            matched_source = (
                requested_source
                if requested_source and evidence_text in _compact(requested_source[0])
                else next(
                (
                    source
                    for source in evidence_sources
                    for source_text in [source[0]]
                    if evidence_text and evidence_text in _compact(source_text)
                ),
                None,
                )
            )
            if matched_source is None:
                continue
            try:
                confidence = float(item.get("confidence", 0))
            except (TypeError, ValueError):
                confidence = 0.0
            if confidence < 0.65:
                continue
            key = (statement_type, subject, predicate_normalized, object_name or value)
            if key in seen:
                continue
            seen.add(key)
            _, source_id, source_page, source_modality = matched_source
            qualifiers = _statement_qualifiers(item.get("qualifiers", []))
            statements.append(KnowledgeStatement(
                id="knowledge-statement:" + hashlib.sha1(
                    f"{unit.id}|{key}|{evidence_text}".encode("utf-8")
                ).hexdigest()[:20],
                knowledge_unit_id=unit.id,
                statement_type=statement_type,
                subject=subject,
                subject_type=_statement_entity_type(item.get("subject_type", "")),
                predicate_original=predicate_original,
                predicate_normalized=predicate_normalized or predicate_original,
                object=object_name,
                object_type=(
                    _statement_entity_type(item.get("object_type", ""))
                    if statement_type == "relation" else ""
                ),
                value=value,
                value_type=(
                    str(item.get("value_type", "text"))
                    if statement_type == "attribute" else ""
                ),
                qualifiers=qualifiers,
                evidence_text=evidence_text,
                evidence_id=source_id,
                source_page=source_page,
                modality=source_modality,
                confidence=confidence,
            ))
        covered_element_ids = {
            statement.evidence_id for statement in statements
        }
        for source in elements:
            source_id = str(source.get("id", ""))
            if not source_id or source_id in covered_element_ids:
                continue
            evidence_source = _compact(
                source.get("meaning", "") or source.get("text", "")
            )
            if len(evidence_source) < 8:
                continue
            summary_match = re.match(r".*?[。；;]", evidence_source)
            summary = (
                summary_match.group(0) if summary_match else evidence_source
            )[:600].strip()
            if len(summary) < 8:
                continue
            modality = str(source.get("modality", "text"))
            subject = _element_fact_subject(source, section=unit.section)
            predicate = {
                "formula": "表达的公式知识",
                "table": "总结的表格知识",
                "circuit": "表达的电路图知识",
                "image": "表达的视觉知识",
            }.get(modality, "表达的多模态知识")
            fallback_key = ("attribute", subject, "HAS_GROUNDED_SEMANTICS", summary)
            if fallback_key in seen:
                continue
            seen.add(fallback_key)
            try:
                fallback_confidence = float(source.get("confidence", 0) or 0)
            except (TypeError, ValueError):
                fallback_confidence = 0.0
            statements.append(KnowledgeStatement(
                id="knowledge-statement:" + hashlib.sha1(
                    f"{unit.id}|{source_id}|{summary}".encode("utf-8")
                ).hexdigest()[:20],
                knowledge_unit_id=unit.id,
                statement_type="attribute",
                subject=subject,
                subject_type=("电路" if modality == "circuit" else "课程概念"),
                predicate_original=predicate,
                predicate_normalized="HAS_GROUNDED_SEMANTICS",
                value=summary,
                value_type="text",
                evidence_text=summary,
                evidence_id=source_id,
                source_page=int(source.get("page", unit.page_start) or unit.page_start),
                modality=modality,
                confidence=max(0.65, min(1.0, fallback_confidence)),
            ))
            covered_element_ids.add(source_id)
        unit.statements = statements
        unit.quality["knowledge_statement_count"] = len(statements)
        unit.quality["knowledge_statement_candidates"] = len(raw_statements)
        if not statements:
            unit.quality["status"] = "review"
            warnings = unit.quality.setdefault("warnings", [])
            if "no grounded atomic statements" not in warnings:
                warnings.append("no grounded atomic statements")

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
    paragraphs = _atomic_paragraphs(text, max_chars=max_chars)
    pieces: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if current and len(current) + len(paragraph) + 2 > max_chars:
            pieces.append(current)
            overlap = ""
            if overlap_chars:
                sentences = [
                    item.strip()
                    for item in re.split(r"(?<=[。！？；;])\s*", current)
                    if item.strip()
                ]
                if sentences and len(sentences[-1]) <= overlap_chars:
                    overlap = sentences[-1]
            current = overlap
        current = f"{current}\n\n{paragraph}".strip()
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
        if unit.statements:
            lines.extend(["### 原子知识陈述", ""])
            for statement in unit.statements:
                lines.extend([
                    f"- {statement.graph_text().replace(chr(10), ' ')}",
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
