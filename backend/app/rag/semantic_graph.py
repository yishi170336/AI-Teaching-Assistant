from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import networkx as nx

from backend.app.rag.models import PageDocument
from backend.app.rag.ontology import COURSE_CONCEPTS, component_role


logger = logging.getLogger(__name__)

SEMANTIC_GRAPH_SCHEMA_VERSION = "3.0-graphrag-semantic"
TERMINAL_PUNCTUATION = ("。", "！", "？", "!", "?", "；", ";")
EXCLUDED_SECTION_PATTERN = re.compile(
    r"(?:目录|前言|绪论|习题|复习题|思考题|自测题|参考答案|答案索引|版权|内容简介)"
)
HEADING_PATTERN = re.compile(
    r"^(?:第[零〇一二三四五六七八九十百两0-9]+章|"
    r"\*?\d+(?:\.\d+){1,3}\s*\S+|本章小结|习题|复习题|思考题|自测题)"
)
FIGURE_CAPTION_PATTERN = re.compile(r"^(?:图|表)(?:题)?\s*\d")
QUESTION_PATTERN = re.compile(
    r"(?:试求|试画|试分析|试证明|回答下列|判断下列|是否正确|求出|计算下列|为什么[？?]?$)"
)
PROVENANCE_ENTITY_PATTERN = re.compile(
    r"^(?:"
    r".*\.(?:pdf|docx?|md|txt)|"
    r"(?:教材)?第\s*\d+\s*页(?:的)?(?:电路图|插图|内容)?|"
    r"(?:本页|该页)(?:电路图|插图|内容)?|"
    r"(?:图|图题|表)\s*[0-9一二三四五六七八九十]+(?:\.[0-9]+)*.*"
    r")$",
    re.I,
)
RELATION_MARKERS = (
    "是", "为", "称为", "又称", "简称", "具有", "包含", "包括", "组成", "构成",
    "产生", "形成", "导致", "使", "提高", "降低", "增大", "减小", "增强", "减弱",
    "阻碍", "促进", "抑制", "决定", "影响", "放大", "衰减", "转换", "实现", "用于",
    "适用于", "依赖", "等于", "近似为", "正比于", "反比于", "连接", "串联", "并联",
)
TECHNICAL_ENTITY_PATTERN = re.compile(
    r"[A-Za-z0-9_βλΔΦφΩμ\u4e00-\u9fff-]{1,24}"
    r"(?:PN结|半导体|载流子|电场|电流|电压|电阻|电容|电感|二极管|晶体管|"
    r"场效应管|放大电路|反馈电路|电路|信号|增益|放大倍数|工作点|特性|模型|"
    r"效应|区域|运动|过程|方法|定律|定理|导电性|稳定性)"
)


@dataclass
class SemanticTextUnit:
    id: str
    text: str
    source: str
    page_start: int
    page_end: int
    chapter: str
    section: str
    block_ids: list[str] = field(default_factory=list)
    modality: str = "text"
    image_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _stable_id(prefix: str, *values: object) -> str:
    raw = "|".join(str(value) for value in values)
    return f"{prefix}:" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def _compact(value: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(value))).lower()


def _normalized_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value)).replace("\u200b", "")
    value = re.sub(r"[ \t]+", " ", value)
    return value.strip()


def _valid_entity_name(value: str) -> bool:
    name = _normalized_text(value).strip("，。；：、")
    if not 2 <= len(name) <= 60:
        return False
    if PROVENANCE_ENTITY_PATTERN.match(name):
        return False
    return name not in {"教材", "页面", "页码", "电路图", "插图", "文件名"}


def _join_visual_lines(value: str) -> str:
    lines = [_normalized_text(line) for line in str(value).splitlines() if line.strip()]
    if not lines:
        return ""
    result = lines[0]
    for line in lines[1:]:
        separator = " " if result[-1:].isascii() and line[:1].isascii() else ""
        result += separator + line
    return result.strip()


def _block_type(text: str, declared: str = "") -> str:
    declared = str(declared).strip().lower()
    if declared:
        return declared
    if HEADING_PATTERN.match(text):
        return "section_heading"
    if FIGURE_CAPTION_PATTERN.match(text):
        return "figure_caption"
    if re.match(r"^(?:[一二三四五六七八九十]+、|\(?\d+[.)）])", text):
        return "list_item"
    return "paragraph"


def _document_blocks(document: PageDocument) -> list[dict[str, Any]]:
    raw_blocks = (
        document.extra.get("text_blocks", [])
        if isinstance(document.extra, dict)
        else []
    )
    blocks: list[dict[str, Any]] = []
    if isinstance(raw_blocks, list):
        for index, raw in enumerate(raw_blocks, 1):
            if not isinstance(raw, dict):
                continue
            text = _join_visual_lines(str(raw.get("text", "")))
            if not text:
                continue
            blocks.append({
                "id": str(raw.get("id") or f"block-{index}"),
                "type": _block_type(text, str(raw.get("type", ""))),
                "text": text,
                "bbox": raw.get("bbox", []),
                "reading_order": int(raw.get("reading_order", index) or index),
            })
    if blocks:
        return sorted(blocks, key=lambda item: int(item["reading_order"]))
    paragraphs = [item.strip() for item in re.split(r"\n\s*\n", document.text) if item.strip()]
    return [
        {
            "id": f"paragraph-{index}",
            "type": _block_type(paragraph),
            "text": _join_visual_lines(paragraph),
            "bbox": [],
            "reading_order": index,
        }
        for index, paragraph in enumerate(paragraphs, 1)
    ]


def _sentence_parts(text: str) -> list[str]:
    return [
        part.strip()
        for part in re.split(r"(?<=[。！？!?；;])", text)
        if part.strip()
    ]


def _split_semantic_text(text: str, max_chars: int = 800) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    sentences = _sentence_parts(text)
    if len(sentences) <= 1:
        return [text[index : index + max_chars] for index in range(0, len(text), max_chars)]
    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        if current and len(current) + len(sentence) > max_chars:
            pieces.append(current)
            current = ""
        current += sentence
    if current:
        pieces.append(current)
    return pieces


def _content_is_extractable(text: str, section: str, block_type: str) -> bool:
    if block_type in {
        "chapter_heading", "section_heading", "figure_caption", "page_header",
        "page_footer", "noise", "exercise", "table",
    }:
        return False
    if EXCLUDED_SECTION_PATTERN.search(section):
        return False
    compact = re.sub(r"\s+", "", text)
    if len(compact) < 4 or (len(compact) < 12 and text.endswith(TERMINAL_PUNCTUATION)):
        return False
    if QUESTION_PATTERN.search(compact):
        return False
    if compact.startswith("[本页主要包含"):
        return False
    return True


def build_semantic_text_units(
    documents: Iterable[PageDocument],
    circuit_elements: Iterable[Any] = (),
) -> list[SemanticTextUnit]:
    """Create layout-aware GraphRAG TextUnits without exposing pages as graph nodes."""

    units: list[SemanticTextUnit] = []
    ordered_documents = sorted(
        documents,
        key=lambda item: (item.source, item.source_page or item.page, item.page),
    )
    for document in ordered_documents:
        if document.doc_type in {"question", "exercise"}:
            continue
        page = int(document.source_page or document.page)
        current_chapter = document.chapter
        current_section = document.section
        blocks = _document_blocks(document)
        extractable = [
            block for block in blocks
            if block.get("type") not in {"page_header", "page_footer", "noise"}
        ]
        for block_index, block in enumerate(extractable):
            text = str(block.get("text", "")).strip()
            block_type = str(block.get("type", "paragraph"))
            if block_type == "chapter_heading":
                current_chapter = text
                current_section = text
                continue
            if block_type == "section_heading":
                current_section = text
                continue
            block_id = f"{document.source}:p{page}:{block.get('id', block_index + 1)}"
            can_continue_previous = (
                bool(units)
                and block_index == 0
                and units[-1].source == document.source
                and units[-1].page_end + 1 == page
                and units[-1].section == current_section
                and not units[-1].text.endswith(TERMINAL_PUNCTUATION)
                and block_type in {"paragraph", "list_item"}
            )
            if can_continue_previous:
                units[-1].text += text
                units[-1].page_end = page
                units[-1].block_ids.append(block_id)
                continue
            if not _content_is_extractable(text, current_section, block_type):
                continue
            for piece_index, piece in enumerate(_split_semantic_text(text), 1):
                unit_id = _stable_id(
                    "text-unit", document.source, page, current_section, block_id, piece_index, piece
                )
                units.append(SemanticTextUnit(
                    id=unit_id,
                    text=piece,
                    source=document.source,
                    page_start=page,
                    page_end=page,
                    chapter=current_chapter,
                    section=current_section,
                    block_ids=[block_id],
                ))

    # A short block at the bottom of one page may only become meaningful after it
    # is joined with the first block on the next page. Remove unresolved fragments
    # only after all textbook pages have been considered.
    units = [unit for unit in units if len(re.sub(r"\s+", "", unit.text)) >= 12]

    for element in circuit_elements:
        if str(getattr(element, "element_type", "")) != "circuit":
            continue
        page = int(getattr(element, "source_page", None) or getattr(element, "page", 0) or 0)
        text = "\n".join(
            value.strip()
            for value in (
                str(getattr(element, "caption", "")),
                str(getattr(element, "description", "")),
                str(getattr(element, "nearby_text", ""))[:1800],
            )
            if value.strip()
        )
        if len(re.sub(r"\s+", "", text)) < 12:
            continue
        element_id = str(getattr(element, "id", ""))
        units.append(SemanticTextUnit(
            id=_stable_id("text-unit", "circuit", element_id, text),
            text=text,
            source=str(getattr(element, "source", "")),
            page_start=page,
            page_end=page,
            chapter=str(getattr(element, "chapter", "")),
            section=str(getattr(element, "section", "")),
            block_ids=[f"circuit:{element_id}"],
            modality="circuit",
            image_path=str(getattr(element, "image_path", "")) or None,
        ))
    return units


def _entity_type(name: str) -> str:
    if name.endswith(("电路", "放大器")):
        return "电路"
    if name.endswith(("二极管", "晶体管", "场效应管", "电阻", "电容", "电感")):
        return "器件或元件"
    if name.endswith(("电流", "电压", "增益", "放大倍数", "工作点")):
        return "电路参数"
    if name.endswith(("运动", "过程", "效应", "导电性", "稳定性", "特性")):
        return "物理过程或性质"
    return "课程概念"


def _candidate_entities(sentence: str) -> list[tuple[int, int, str]]:
    candidates: dict[tuple[int, int], str] = {}
    for concept in COURSE_CONCEPTS:
        for match in re.finditer(re.escape(concept), sentence, re.I):
            candidates[(match.start(), match.end())] = sentence[match.start() : match.end()]
    for match in TECHNICAL_ENTITY_PATTERN.finditer(sentence):
        value = match.group(0).strip("，。；：、()（）")
        if 2 <= len(value) <= 28:
            start = match.start() + match.group(0).find(value)
            candidates[(start, start + len(value))] = value
    ordered = sorted((start, end, value) for (start, end), value in candidates.items())
    # Prefer the most specific non-overlapping span at a given location.
    result: list[tuple[int, int, str]] = []
    for item in ordered:
        if result and item[0] < result[-1][1]:
            previous = result[-1]
            if item[1] - item[0] > previous[1] - previous[0]:
                result[-1] = item
            continue
        result.append(item)
    return result


def _fallback_extract(unit: SemanticTextUnit) -> dict[str, Any]:
    entities: dict[str, dict[str, str]] = {}
    relationships: list[dict[str, Any]] = []
    for sentence in _sentence_parts(unit.text):
        spans = _candidate_entities(sentence)
        for _, _, name in spans:
            entities.setdefault(name, {
                "name": name,
                "type": _entity_type(name),
                "description": "",
            })
        for left, right in zip(spans, spans[1:]):
            relation = sentence[left[1] : right[0]].strip(" ，、：:（）()")
            if not relation or len(relation) > 24:
                continue
            if not any(marker in relation for marker in RELATION_MARKERS):
                continue
            relationships.append({
                "source": left[2],
                "target": right[2],
                "relation_original": relation,
                "evidence_text": sentence,
                "strength": 5,
            })
    return {"entities": list(entities.values()), "relationships": relationships}


GRAPH_EXTRACTION_PROMPT = """你是教材 GraphRAG 实体与关系抽取器。输入是一组已经恢复自然段结构的 TextUnit。
对每个 TextUnit 同时抽取课程知识实体和实体之间明确陈述的关系。实体类型根据教材内容自由填写，不使用预设关系词表。

硬性规则：
1. relation_original 必须逐字复制 evidence_text 中连续出现的关系短语，不得改写、概括、规范化或补充关系。
2. evidence_text 必须逐字来自对应 TextUnit；source、target 必须在正文或章节上下文中明确出现。
3. 保留条件、方向、否定、近似和大小变化；不同关系分别输出。
4. 目录、页码、图号、表号、习题要求、文件名和“第几页电路图”不是知识实体。
5. 对 circuit 模态，只抽取图题、视觉描述和邻近正文明确支持的知识；不得凭常识猜测电路功能。
6. 没有可靠关系时返回空 relationships，不要为了连图而制造关系。

仅返回 JSON：
{"items":[{"text_unit_id":"...","entities":[{"name":"...","type":"...","description":"..."}],"relationships":[{"source":"...","target":"...","relation_original":"...","evidence_text":"...","strength":1}]}]}

TextUnits：
"""


def _valid_evidence_span(evidence: str, unit_text: str) -> bool:
    return bool(evidence) and _compact(evidence) in _compact(unit_text)


def _normalize_extraction(raw: Any, unit: SemanticTextUnit) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return _fallback_extract(unit)
    entity_values = raw.get("entities", [])
    relationship_values = raw.get("relationships", [])
    entities: list[dict[str, str]] = []
    names: set[str] = set()
    if isinstance(entity_values, list):
        for item in entity_values:
            if not isinstance(item, dict):
                continue
            name = _normalized_text(str(item.get("name", ""))).strip("，。；：、")
            if not _valid_entity_name(name):
                continue
            if _compact(name) not in _compact(unit.text + unit.section):
                continue
            entities.append({
                "name": name,
                "type": _normalized_text(str(item.get("type", "课程概念")))[:32] or "课程概念",
                "description": _normalized_text(str(item.get("description", "")))[:500],
            })
            names.add(_compact(name))
    relationships: list[dict[str, Any]] = []
    if isinstance(relationship_values, list):
        for item in relationship_values:
            if not isinstance(item, dict):
                continue
            source = _normalized_text(str(item.get("source", ""))).strip("，。；：、")
            target = _normalized_text(str(item.get("target", ""))).strip("，。；：、")
            relation = _normalized_text(str(item.get("relation_original", ""))).strip("，。；：、")
            evidence = _normalized_text(str(item.get("evidence_text", "")))
            if (
                not _valid_entity_name(source)
                or not _valid_entity_name(target)
                or source == target
                or not relation
            ):
                continue
            if not _valid_evidence_span(evidence, unit.text):
                continue
            if _compact(relation) not in _compact(evidence):
                continue
            if _compact(source) not in _compact(unit.text + unit.section):
                continue
            if _compact(target) not in _compact(unit.text + unit.section):
                continue
            try:
                strength = max(1.0, min(10.0, float(item.get("strength", 5))))
            except (TypeError, ValueError):
                strength = 5.0
            relationships.append({
                "source": source,
                "target": target,
                "relation_original": relation,
                "evidence_text": evidence,
                "strength": strength,
            })
            for name in (source, target):
                if _compact(name) not in names:
                    entities.append({
                        "name": name,
                        "type": _entity_type(name),
                        "description": "",
                    })
                    names.add(_compact(name))
    if not relationships and not entities:
        return _fallback_extract(unit)
    return {"entities": entities, "relationships": relationships}


def _read_extraction_cache(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    result: dict[str, dict[str, Any]] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            if isinstance(item, dict) and item.get("text_unit_id"):
                result[str(item["text_unit_id"])] = item
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    return result


def _write_extraction_cache(path: Path | None, values: dict[str, dict[str, Any]]) -> None:
    if path is None:
        return
    path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in values.values()),
        encoding="utf-8",
    )


def extract_text_unit_graphs(
    units: Iterable[SemanticTextUnit],
    client: Any | None = None,
    *,
    cache_path: Path | None = None,
    batch_size: int = 6,
    max_workers: int = 3,
) -> dict[str, dict[str, Any]]:
    """Extract GraphRAG subgraphs while retaining verbatim relationship mentions."""

    unit_items = list(units)
    cache = _read_extraction_cache(cache_path)
    results: dict[str, dict[str, Any]] = {}
    pending: list[SemanticTextUnit] = []
    model_name = str(getattr(getattr(client, "config", None), "model", "rule"))
    for unit in unit_items:
        content_hash = hashlib.sha256(unit.text.encode("utf-8")).hexdigest()
        cached = cache.get(unit.id)
        if (
            cached
            and cached.get("content_hash") == content_hash
            and cached.get("model") == model_name
        ):
            results[unit.id] = _normalize_extraction(cached.get("result"), unit)
        else:
            pending.append(unit)

    batches = [
        pending[start : start + max(1, batch_size)]
        for start in range(0, len(pending), max(1, batch_size))
    ]

    def extract_batch(batch: list[SemanticTextUnit]) -> list[tuple[SemanticTextUnit, dict[str, Any]]]:
        raw_by_id: dict[str, dict[str, Any]] = {}
        if client is not None:
            payload = [
                {
                    "text_unit_id": unit.id,
                    "modality": unit.modality,
                    "chapter": unit.chapter,
                    "section": unit.section,
                    "text": unit.text,
                }
                for unit in batch
            ]
            response = client.complete_json(
                GRAPH_EXTRACTION_PROMPT + json.dumps(payload, ensure_ascii=False)
            )
            items = response.get("items", []) if isinstance(response, dict) else []
            if isinstance(items, list):
                raw_by_id = {
                    str(item.get("text_unit_id")): item
                    for item in items
                    if isinstance(item, dict) and item.get("text_unit_id")
                }
        normalized_batch: list[tuple[SemanticTextUnit, dict[str, Any]]] = []
        for unit in batch:
            normalized = _normalize_extraction(raw_by_id.get(unit.id), unit)
            normalized_batch.append((unit, normalized))
        return normalized_batch

    worker_count = min(max(1, max_workers), len(batches)) if batches else 0
    if worker_count == 1:
        completed_batches = (extract_batch(batch) for batch in batches)
        for normalized_batch in completed_batches:
            for unit, normalized in normalized_batch:
                results[unit.id] = normalized
                cache[unit.id] = {
                    "text_unit_id": unit.id,
                    "content_hash": hashlib.sha256(unit.text.encode("utf-8")).hexdigest(),
                    "model": model_name,
                    "result": normalized,
                }
            _write_extraction_cache(cache_path, cache)
    elif worker_count > 1:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(extract_batch, batch) for batch in batches]
            for future in as_completed(futures):
                for unit, normalized in future.result():
                    results[unit.id] = normalized
                    cache[unit.id] = {
                        "text_unit_id": unit.id,
                        "content_hash": hashlib.sha256(unit.text.encode("utf-8")).hexdigest(),
                        "model": model_name,
                        "result": normalized,
                    }
                _write_extraction_cache(cache_path, cache)
    return {unit.id: results[unit.id] for unit in unit_items}


def _circuit_name(element: Any) -> str:
    candidates = [
        str(getattr(element, "caption", "")),
        str(getattr(element, "description", ""))[:300],
        str(getattr(element, "section", "")),
    ]
    for candidate in candidates:
        match = re.search(r"([\u4e00-\u9fffA-Za-z0-9-]{2,28}电路)", candidate)
        if match:
            value = re.sub(r"^(?:图|图题)\s*\d+(?:\.\d+)*\s*", "", match.group(1))
            if 2 <= len(value) <= 32:
                return value
    return ""


def _circuit_visual_records(elements: Iterable[Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    evidence: list[dict[str, Any]] = []
    mentions: list[dict[str, Any]] = []
    for element in elements:
        if str(getattr(element, "element_type", "")) != "circuit":
            continue
        element_id = str(getattr(element, "id", ""))
        circuit_name = _circuit_name(element)
        page = int(getattr(element, "source_page", None) or getattr(element, "page", 0) or 0)
        evidence_id = _stable_id("evidence", "circuit", element_id)
        components = [
            item for item in getattr(element, "components", []) if isinstance(item, dict)
        ]
        evidence.append({
            "id": evidence_id,
            "modality": "circuit",
            "source": str(getattr(element, "source", "")),
            "page": page,
            "image_path": str(getattr(element, "image_path", "")),
            "caption": str(getattr(element, "caption", "")),
            "description": str(getattr(element, "description", "")),
            "components": components,
            "nets": getattr(element, "nets", []),
            "netlist": str(getattr(element, "netlist", "")),
        })
        if not circuit_name:
            continue
        component_records: list[tuple[str, str, list[str]]] = []
        for component in components:
            ref = str(component.get("id") or component.get("ref") or "")
            component_type = str(component.get("type", "unknown"))
            role = component_role(
                ref,
                component_type,
                explicit_role=str(component.get("role") or component.get("function") or ""),
                context="\n".join((
                    str(getattr(element, "caption", "")),
                    str(getattr(element, "description", "")),
                    str(getattr(element, "nearby_text", "")),
                )),
            )
            if not role or role == "电路元件":
                continue
            terminals = [
                str(value).strip() for value in component.get("terminals", []) if str(value).strip()
            ]
            component_records.append((ref, role, terminals))
            mentions.append({
                "source": circuit_name,
                "target": role,
                "relation_original": "包含",
                "evidence_text": f"{circuit_name}包含{role}",
                "strength": 6.0,
                "evidence_id": evidence_id,
                "text_unit_id": None,
                "source_modality": "circuit_visual",
                "source_page": page,
            })
        for left, right in combinations(component_records, 2):
            if left[1] == right[1] or not set(left[2]).intersection(right[2]):
                continue
            mentions.append({
                "source": left[1],
                "target": right[1],
                "relation_original": "连接",
                "evidence_text": f"{left[0]}与{right[0]}共享网络节点",
                "strength": 7.0,
                "evidence_id": evidence_id,
                "text_unit_id": None,
                "source_modality": "circuit_visual",
                "source_page": page,
            })
    return evidence, mentions


def _merge_graph_records(
    units: list[SemanticTextUnit],
    extractions: dict[str, dict[str, Any]],
    circuit_mentions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    unit_map = {unit.id: unit for unit in units}
    entities: dict[str, dict[str, Any]] = {}
    raw_mentions: list[dict[str, Any]] = []

    def ensure_entity(name: str, entity_type: str = "课程概念", description: str = "") -> str:
        if not _valid_entity_name(name):
            return ""
        key = _compact(name)
        entity_id = _stable_id("entity", key)
        entity = entities.setdefault(entity_id, {
            "id": entity_id,
            "type": "entity",
            "entity_type": entity_type or "课程概念",
            "name": name,
            "description": description,
            "aliases": [],
            "text_unit_ids": [],
            "evidence_count": 0,
            "pages": [],
        })
        if name != entity["name"] and name not in entity["aliases"]:
            entity["aliases"].append(name)
        if description and not entity.get("description"):
            entity["description"] = description
        return entity_id

    for unit_id, extraction in extractions.items():
        unit = unit_map[unit_id]
        for item in extraction.get("entities", []):
            if not isinstance(item, dict) or not item.get("name"):
                continue
            entity_id = ensure_entity(
                str(item["name"]), str(item.get("type", "课程概念")), str(item.get("description", ""))
            )
            if not entity_id:
                continue
            entity = entities[entity_id]
            if unit_id not in entity["text_unit_ids"]:
                entity["text_unit_ids"].append(unit_id)
            for page in range(unit.page_start, unit.page_end + 1):
                if page not in entity["pages"]:
                    entity["pages"].append(page)
        for position, item in enumerate(extraction.get("relationships", []), 1):
            if not isinstance(item, dict):
                continue
            source_name = str(item.get("source", "")).strip()
            target_name = str(item.get("target", "")).strip()
            relation = str(item.get("relation_original", "")).strip()
            if not source_name or not target_name or not relation:
                continue
            source_id = ensure_entity(source_name, _entity_type(source_name))
            target_id = ensure_entity(target_name, _entity_type(target_name))
            if not source_id or not target_id:
                continue
            mention_id = _stable_id("relationship-mention", unit_id, position, source_id, relation, target_id)
            raw_mentions.append({
                "id": mention_id,
                "source": source_id,
                "target": target_id,
                "relation": relation,
                "evidence_text": str(item.get("evidence_text", "")),
                "strength": float(item.get("strength", 5) or 5),
                "text_unit_id": unit_id,
                "evidence_id": unit_id,
                "source_modality": unit.modality,
                "source_page": unit.page_start,
            })

    for position, item in enumerate(circuit_mentions, 1):
        source_name = str(item.get("source", "")).strip()
        target_name = str(item.get("target", "")).strip()
        relation = str(item.get("relation_original", "")).strip()
        if not source_name or not target_name or not relation:
            continue
        source_id = ensure_entity(source_name, _entity_type(source_name))
        target_id = ensure_entity(target_name, _entity_type(target_name))
        if not source_id or not target_id:
            continue
        mention_id = _stable_id(
            "relationship-mention", item.get("evidence_id"), position, source_id, relation, target_id
        )
        raw_mentions.append({
            "id": mention_id,
            "source": source_id,
            "target": target_id,
            "relation": relation,
            "evidence_text": str(item.get("evidence_text", "")),
            "strength": float(item.get("strength", 5) or 5),
            "text_unit_id": item.get("text_unit_id"),
            "evidence_id": item.get("evidence_id"),
            "source_modality": "circuit_visual",
            "source_page": item.get("source_page"),
        })

    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    links: dict[tuple[str, str], dict[str, Any]] = {}
    for mention in raw_mentions:
        edge_key = (mention["source"], _compact(mention["relation"]), mention["target"])
        edge = grouped.setdefault(edge_key, {
            "id": _stable_id("relationship", *edge_key),
            "source": mention["source"],
            "target": mention["target"],
            "type": mention["relation"],
            "relation": mention["relation"],
            "mention_ids": [],
            "text_unit_ids": [],
            "evidence_ids": [],
            "evidence_count": 0,
            "weight": 0.0,
        })
        edge["mention_ids"].append(mention["id"])
        if mention.get("text_unit_id") and mention["text_unit_id"] not in edge["text_unit_ids"]:
            edge["text_unit_ids"].append(mention["text_unit_id"])
        if mention.get("evidence_id") and mention["evidence_id"] not in edge["evidence_ids"]:
            edge["evidence_ids"].append(mention["evidence_id"])
        edge["evidence_count"] += 1
        edge["weight"] += float(mention.get("strength", 5))

        pair = tuple(sorted((mention["source"], mention["target"])))
        link = links.setdefault(pair, {
            "id": _stable_id("relationship-link", *pair),
            "source": pair[0],
            "target": pair[1],
            "mention_ids": [],
            "relationship_ids": [],
            "text_unit_ids": [],
            "weight": 0.0,
        })
        link["mention_ids"].append(mention["id"])
        if edge["id"] not in link["relationship_ids"]:
            link["relationship_ids"].append(edge["id"])
        if mention.get("text_unit_id") and mention["text_unit_id"] not in link["text_unit_ids"]:
            link["text_unit_ids"].append(mention["text_unit_id"])
        link["weight"] += float(mention.get("strength", 5))

    connected_ids = {
        endpoint for edge in grouped.values() for endpoint in (edge["source"], edge["target"])
    }
    for entity_id in list(entities):
        if entity_id not in connected_ids:
            del entities[entity_id]
            continue
        entity = entities[entity_id]
        entity["evidence_count"] = len(entity["text_unit_ids"])
        entity["pages"] = sorted(entity["pages"])
    return list(entities.values()), list(grouped.values()), raw_mentions, list(links.values())


def build_graph_communities(
    nodes: list[dict[str, Any]],
    links: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    graph = nx.Graph()
    graph.add_nodes_from(str(node["id"]) for node in nodes)
    for link in links:
        graph.add_edge(str(link["source"]), str(link["target"]), weight=float(link.get("weight", 1)))
    if graph.number_of_nodes() == 0:
        return []

    partitions: list[set[str]]
    algorithm = "hierarchical-leiden"
    try:
        from graspologic.partition import hierarchical_leiden

        records = hierarchical_leiden(graph, max_cluster_size=32, random_seed=42)
        final_by_cluster: dict[int, set[str]] = defaultdict(set)
        for record in records:
            if bool(record.is_final_cluster):
                final_by_cluster[int(record.cluster)].add(str(record.node))
        partitions = [members for members in final_by_cluster.values() if members]
    except Exception as exc:
        logger.info("Hierarchical Leiden unavailable; using deterministic Louvain fallback: %s", exc)
        algorithm = "louvain-fallback"
        partitions = [
            set(map(str, members))
            for members in nx.community.louvain_communities(graph, weight="weight", seed=42)
        ]

    node_map = {str(node["id"]): node for node in nodes}
    communities: list[dict[str, Any]] = []
    for index, members in enumerate(sorted(partitions, key=lambda item: (-len(item), sorted(item)[0])), 1):
        member_links = [
            link for link in links
            if str(link["source"]) in members and str(link["target"]) in members
        ]
        ranked = sorted(
            members,
            key=lambda node_id: (-graph.degree(node_id, weight="weight"), str(node_map[node_id].get("name", ""))),
        )
        names = [str(node_map[node_id].get("name", "")) for node_id in ranked[:4]]
        communities.append({
            "id": f"community:0:{index}",
            "community": index,
            "parent": None,
            "children": [],
            "level": 0,
            "title": "、".join(name for name in names if name) or f"知识社区 {index}",
            "entity_ids": sorted(members),
            "relationship_ids": sorted({
                relationship_id
                for link in member_links
                for relationship_id in link.get("relationship_ids", [])
            }),
            "text_unit_ids": sorted({
                unit_id for link in member_links for unit_id in link.get("text_unit_ids", [])
            }),
            "size": len(members),
            "algorithm": algorithm,
        })
    return communities


def build_community_reports(
    communities: list[dict[str, Any]],
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Produce grounded reports without writing generated summaries back into relations."""

    node_map = {str(node["id"]): node for node in nodes}
    edge_map = {str(edge["id"]): edge for edge in edges}
    reports: list[dict[str, Any]] = []
    for community in communities:
        entity_names = [
            str(node_map[entity_id].get("name", ""))
            for entity_id in community.get("entity_ids", [])
            if entity_id in node_map
        ]
        relation_lines = [
            f"{node_map[edge['source']]['name']}—{edge['relation']}→{node_map[edge['target']]['name']}"
            for relationship_id in community.get("relationship_ids", [])
            if (edge := edge_map.get(relationship_id))
            and edge.get("source") in node_map
            and edge.get("target") in node_map
        ]
        reports.append({
            "id": _stable_id("community-report", community["id"]),
            "community": community["community"],
            "level": community["level"],
            "title": community["title"],
            "summary": "本社区包含：" + "、".join(entity_names[:12]),
            "full_content": "\n".join(relation_lines[:80]),
            "findings": relation_lines[:10],
            "entity_ids": community.get("entity_ids", []),
            "relationship_ids": community.get("relationship_ids", []),
            "text_unit_ids": community.get("text_unit_ids", []),
        })
    return reports


def build_semantic_knowledge_graph(
    documents: Iterable[PageDocument],
    circuit_elements: Iterable[Any] = (),
    *,
    client: Any | None = None,
    extraction_cache_path: Path | None = None,
) -> dict[str, Any]:
    document_items = list(documents)
    element_items = list(circuit_elements)
    units = build_semantic_text_units(document_items, element_items)
    extractions = extract_text_unit_graphs(units, client, cache_path=extraction_cache_path)
    circuit_evidence, circuit_mentions = _circuit_visual_records(element_items)
    nodes, edges, mentions, links = _merge_graph_records(units, extractions, circuit_mentions)
    communities = build_graph_communities(nodes, links)
    reports = build_community_reports(communities, nodes, edges)
    text_evidence = [
        {
            "id": unit.id,
            "modality": unit.modality,
            "source": unit.source,
            "page_start": unit.page_start,
            "page_end": unit.page_end,
            "chapter": unit.chapter,
            "section": unit.section,
            "block_ids": unit.block_ids,
            "text": unit.text,
            "image_path": unit.image_path,
        }
        for unit in units
    ]
    return {
        "schema_version": SEMANTIC_GRAPH_SCHEMA_VERSION,
        "nodes": nodes,
        "edges": edges,
        "relationship_mentions": mentions,
        "relationship_links": links,
        "text_units": [unit.to_dict() for unit in units],
        "evidence": [*text_evidence, *circuit_evidence],
        "communities": communities,
        "community_reports": reports,
        "stats": {
            "entities": len(nodes),
            "relationships": len(edges),
            "relationship_mentions": len(mentions),
            "text_units": len(units),
            "communities": len(communities),
            "circuit_evidence": len(circuit_evidence),
        },
    }


def is_semantic_graph(graph: dict[str, Any]) -> bool:
    return str(graph.get("schema_version", "")).startswith("3.")
