from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import networkx as nx

from backend.app.rag.models import PageDocument
from backend.app.rag.ontology import COURSE_CONCEPTS, component_role


logger = logging.getLogger(__name__)

SEMANTIC_GRAPH_SCHEMA_VERSION = "3.3-graphrag-semantic"
GRAPH_EXTRACTION_VERSION = "2026-08-text-llm-v12-graphrag-json"
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
ENTITY_PREFIX_PATTERN = re.compile(
    r"^(?:其|它|该|这种|这些|这类|此类|上述|本页|该页|其中|因此|所以|同时|从而|由于|当|若|如果|在|对|由|将)"
)
ENTITY_CLAUSE_PATTERN = re.compile(
    r"(?:称为|简称|叫做|是指|可以|能够|用于|使得|导致|产生|形成|构成|包括|包含|具有|"
    r"增大|减小|增多|减少|增强|减弱|提高|降低|大于|小于|高于|低于|等于|近似为|正比于|反比于)"
)
ENTITY_CONTEXT_CLAUSE_PATTERN = re.compile(
    r"(?:作用下|情况下|工作情况|发生变化时|从而|因此).*(?:过程|运动|状态|规律|情况)$"
)
EXPRESSION_VALUE_PATTERN = re.compile(r"(?:的倒数|之和|之差|的乘积|的比值|的平方)$")
NOMINALIZED_ENTITY_END_PATTERN = re.compile(
    r"的(?:物质|材料|器件|元件|电路|结构|区域|过程|现象|运动|性能|能力|特性|参数)$"
)
ANAPHORIC_ENTITY_PATTERN = re.compile(
    r"(?:这一|这种|该|上述|其)(?:性质|特性|作用|状态|数值|变化|效应|情况)"
)
COREFERENCE_MENTION_PATTERN = re.compile(
    r"^(?:它|其|该(?:器件|元件|电路|结构|材料|参数|物质|区域|过程|现象)?|"
    r"这种(?:器件|元件|电路|结构|材料|参数|物质|区域|过程|现象)?|"
    r"上述(?:器件|元件|电路|结构|材料|参数|物质|区域|过程|现象)?)$"
)
GENERIC_ENTITY_NAMES = {
    "教材", "页面", "页码", "电路图", "插图", "文件名", "公式", "曲线", "图形", "数值",
    "加强", "减弱", "降低", "提高", "增大", "减小", "增强", "产生", "形成", "作用", "允许值",
    "增多", "减少", "参数", "优点", "缺点", "情况", "条件", "方向", "位置",
    "左侧", "右侧", "上方", "下方", "一侧", "另一侧", "图中",
    "主要作用", "背离耗尽层",
}
VALUE_LITERAL_PATTERN = re.compile(
    r"^[≈≃≅=<>≤≥±+\-]?\s*\d+(?:\.\d+)?\s*(?:V|A|mA|μA|uA|Ω|kΩ|MΩ|Hz|kHz|MHz|℃|°C|%)?$",
    re.I,
)
QUANTITY_ENTITY_PATTERN = re.compile(
    r"^(?:约|大约|近似为|≈|=|[<>≤≥])?\s*[+\-]?\d+(?:\.\d+)?\s*"
    r"(?:个(?:电子|空穴|载流子|原子)|倍|%|V|A|mA|μA|uA|Ω|kΩ|MΩ|Hz|kHz|MHz|℃|°C)$",
    re.I,
)
FORMULA_REFERENCE_PATTERN = re.compile(r"^式\s*[（(]?\d+(?:\.\d+)+[）)]?$", re.I)
RELATION_CLAUSE_PATTERN = re.compile(r"[。！？!?；;，,：:]|(?:由于|因此|所以|从而|其中|这时|此时|当.+时)")
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
FORMULA_SYMBOL_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])"
    r"[A-Za-zΑ-Ωα-ω][A-Za-zΑ-Ωα-ω0-9]*"
    r"(?:_(?:[A-Za-zΑ-Ωα-ω0-9]+|\([A-Za-zΑ-Ωα-ω0-9]+\))|"
    r"\([A-Za-zΑ-Ωα-ω0-9]+\))?"
    r"(?![A-Za-z0-9])"
)
DISPLAY_NAME_SUFFIXES = (
    "电压当量", "阈值电压", "死区电压", "开启电压", "接触电位差", "电位差",
    "结电容", "电压", "电流", "电阻", "电容", "电感", "电荷量", "浓度",
    "电场", "电势", "功率", "频率", "增益", "放大倍数", "工作点", "系数",
    "参数", "当量", "常数", "电导率", "迁移率", "温度", "电荷",
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


def _entity_key(value: str) -> str:
    """Return an identity key while treating formula typography as aliases."""

    normalized = unicodedata.normalize("NFKC", str(value)).strip()
    if re.fullmatch(r"[A-Za-zΑ-Ωα-ω0-9_{}()（）\[\].+\-/\s]+", normalized):
        return re.sub(r"[\s_{}()（）\[\]]+", "", normalized)
    return re.sub(r"\s+", "", normalized)


def _display_name_score(value: str) -> tuple[int, int, str]:
    """Prefer cleaner source-observed typography when aliases share an ID."""

    name = unicodedata.normalize("NFKC", str(value)).strip()
    penalty = len(re.findall(r"\s", name)) * 10 + name.count("_(") * 6
    if re.fullmatch(r"[A-Za-zΑ-Ωα-ω0-9_{}()（）\[\].+\-/\s]+", name):
        if "_" in name and "_(" not in name:
            penalty -= 3
    return penalty, len(name), name.lower()


def _normalized_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value)).replace("\u200b", "")
    value = re.sub(r"[ \t]+", " ", value)
    return value.strip()


def _canonical_entity_name(value: str) -> tuple[str, str]:
    """Remove non-identifying contextual modifiers while retaining the surface mention."""

    surface = _normalized_text(value).strip("，。；：、")
    canonical = re.sub(r"^(?:常用的|常见的|典型的|通常所说的)", "", surface).strip()
    if len(canonical) < 2:
        canonical = surface
    return canonical, surface


def _entity_rejection_reasons(value: str) -> list[str]:
    name = _normalized_text(value).strip("，。；：、")
    reasons: list[str] = []
    if not 2 <= len(name) <= 36:
        reasons.append("entity_length")
    if PROVENANCE_ENTITY_PATTERN.match(name) or FORMULA_REFERENCE_PATTERN.match(name):
        reasons.append("provenance_or_formula_reference")
    if name in GENERIC_ENTITY_NAMES:
        reasons.append("generic_word")
    if ENTITY_PREFIX_PATTERN.match(name) or name.startswith("本章"):
        reasons.append("pronoun_or_clause_prefix")
    if ENTITY_CLAUSE_PATTERN.search(name) and not NOMINALIZED_ENTITY_END_PATTERN.search(name):
        reasons.append("predicate_in_entity")
    if ENTITY_CONTEXT_CLAUSE_PATTERN.search(name):
        reasons.append("context_clause_entity")
    if EXPRESSION_VALUE_PATTERN.search(name):
        reasons.append("expression_value")
    if ANAPHORIC_ENTITY_PATTERN.search(name):
        reasons.append("anaphoric_entity")
    if re.search(r"(?:或|以及|和|与)", name):
        reasons.append("combined_entities")
    if re.search(r"[。！？!?；;，,：:、]", name):
        reasons.append("sentence_fragment")
    if VALUE_LITERAL_PATTERN.match(name):
        reasons.append("literal_value")
    if QUANTITY_ENTITY_PATTERN.match(name):
        reasons.append("quantity_value")
    if re.search(r"(?:<=|>=|[=<>≤≥≈≃≅])", name):
        reasons.append("formula_comparison")
    if re.search(r"(?:的|了|着|过|并且|而且)$", name):
        reasons.append("incomplete_phrase")
    if re.search(r"(?:时|的情况)$", name):
        reasons.append("condition_phrase")
    if re.search(r"(?:左侧|右侧|上方|下方)$", name):
        reasons.append("relative_location")
    if re.search(r"(?:能|会)(?:够|使|产生|形成|越过|通过|提高|降低|增大|减小|增多|减少)", name):
        reasons.append("verbal_clause")
    return reasons


def _valid_entity_name(value: str) -> bool:
    return not _entity_rejection_reasons(value)


def _relation_rejection_reasons(
    value: str,
    *,
    allow_formula_operator: bool = False,
) -> list[str]:
    relation = _normalized_text(value).strip("，。；：、")
    reasons: list[str] = []
    if not relation or len(relation) > 24:
        reasons.append("relation_length")
    if RELATION_CLAUSE_PATTERN.search(relation):
        reasons.append("relation_is_clause")
    if "..." in relation or "…" in relation:
        reasons.append("relation_has_placeholder")
    if VALUE_LITERAL_PATTERN.match(relation):
        reasons.append("relation_is_literal")
    if not allow_formula_operator and re.fullmatch(r"[≈≃≅=<>≤≥±+\-/*—–]+", relation):
        reasons.append("relation_is_formula_operator")
    if "几几乎" in relation:
        reasons.append("relation_has_ocr_corruption")
    if relation in {"工作时", "在靠近", "因", "来表征其"}:
        reasons.append("relation_is_incomplete_condition")
    if relation in {
        "的", "和", "与", "及", "及其", "或", "其", "它", "这", "如", "式",
        "不断", "逐渐", "也", "均", "都", "会", "能", "将", "起",
    }:
        reasons.append("relation_is_function_word")
    return reasons


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

    # OCR caches created before layout-block persistence contain only visual
    # lines. Reconstruct headings and prose instead of classifying the entire
    # page by its first line (which would discard a page beginning with a title).
    lines = [_normalized_text(line) for line in document.text.splitlines() if line.strip()]
    fallback_blocks: list[dict[str, Any]] = []
    paragraph_lines: list[str] = []

    def append_block(text: str, block_type: str) -> None:
        fallback_blocks.append({
            "id": f"ocr-block-{len(fallback_blocks) + 1}",
            "type": block_type,
            "text": text,
            "bbox": [],
            "reading_order": len(fallback_blocks) + 1,
        })

    def flush_paragraph() -> None:
        if not paragraph_lines:
            return
        append_block(_join_visual_lines("\n".join(paragraph_lines)), "paragraph")
        paragraph_lines.clear()

    for line in lines:
        if re.fullmatch(r"\d{1,4}", line):
            flush_paragraph()
            append_block(line, "page_header")
            continue
        if re.match(r"^第[零〇一二三四五六七八九十百两0-9]+章(?:\s|$)", line):
            flush_paragraph()
            append_block(line, "chapter_heading")
            continue
        if HEADING_PATTERN.match(line):
            flush_paragraph()
            append_block(line, "section_heading")
            continue
        if re.match(r"^[一二三四五六七八九十]+、\S+", line) and len(line) <= 40:
            flush_paragraph()
            append_block(line, "section_heading")
            continue
        if FIGURE_CAPTION_PATTERN.match(line):
            flush_paragraph()
            append_block(line, "figure_caption")
            continue
        if paragraph_lines and paragraph_lines[-1].endswith(TERMINAL_PUNCTUATION):
            flush_paragraph()
        if re.match(r"^(?:\(?\d+[.)）]|[一二三四五六七八九十]+、)", line):
            flush_paragraph()
            append_block(line, "list_item")
            continue
        paragraph_lines.append(line)
    flush_paragraph()
    return fallback_blocks


def _sentence_parts(text: str) -> list[str]:
    return [
        part.strip()
        for part in re.split(r"(?<=[。！？!?；;])", text)
        if part.strip()
    ]


def _split_semantic_text(text: str, max_chars: int = 420) -> list[str]:
    """Keep sentence boundaries while giving the text LLM paragraph context.

    Layout blocks already represent OCR-restored paragraphs. Sentences are used
    as safe split points only when a paragraph exceeds the model-sized window;
    ordinary neighbouring sentences stay together so the model can resolve
    references such as “它”“该器件” without inventing a relationship.
    """

    def split_overlong(sentence: str) -> list[str]:
        if len(sentence) <= max_chars:
            return [sentence]
        clauses = [
            item.strip()
            for item in re.split(r"(?<=[，,：:])", sentence)
            if item.strip()
        ]
        if len(clauses) <= 1:
            return [
                sentence[index : index + max_chars]
                for index in range(0, len(sentence), max_chars)
            ]
        chunks: list[str] = []
        current = ""
        for clause in clauses:
            if current and len(current) + len(clause) > max_chars:
                chunks.append(current)
                current = ""
            current += clause
        if current:
            chunks.append(current)
        return chunks

    pieces: list[str] = []
    current = ""
    for sentence in _sentence_parts(text) or [text]:
        for chunk in split_overlong(sentence):
            if current and len(current) + len(chunk) > max_chars:
                pieces.append(current)
                current = ""
            current += chunk
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
    if len(compact) < 6:
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
    units = [
        unit for unit in units
        if (
            len(re.sub(r"\s+", "", unit.text)) >= 6
            and (
                unit.text.endswith(TERMINAL_PUNCTUATION)
                or len(re.sub(r"\s+", "", unit.text)) >= 12
            )
        )
    ]

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


def _stable_entity_type(name: str, raw_type: str = "") -> str:
    """Map free-form extraction labels to a small, stable entity type set.

    This only stabilizes node categories. It never rewrites relationship text.
    The extractor's original type is retained separately on the node.
    """

    value = f"{name} {raw_type}"
    if re.search(r"(?:电路|放大器|振荡器|滤波器|电源)$", name) or "电路" in raw_type:
        return "电路"
    if re.search(r"(?:二极管|晶体管|场效应管|电阻|电容|电感|器件|元件|PN结)$", name):
        return "器件与元件"
    if re.search(r"(?:半导体|导体|绝缘体|硅|锗|材料|结构|区域|耗尽层|耗尽区)", value):
        return "材料与结构"
    if re.search(r"(?:电流|电压|电阻|电容|电感|频率|增益|放大倍数|工作点|阈值|参数|浓度)$", name):
        return "物理量与参数"
    if re.search(r"(?:运动|过程|效应|漂移|扩散|复合|激发|击穿|导电)", value):
        return "物理过程与效应"
    if re.search(r"(?:特性|性质|规律|定律|定理|关系|方程|条件|状态)", value):
        return "特性与规律"
    if re.search(r"(?:模型|方法|分析|等效|近似|变换)", value):
        return "方法与模型"
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
            if not _valid_entity_name(name):
                continue
            entities.setdefault(name, {
                "name": name,
                "type": _entity_type(name),
                "description": "",
            })
        for left, right in zip(spans, spans[1:]):
            if not _valid_entity_name(left[2]) or not _valid_entity_name(right[2]):
                continue
            relation = sentence[left[1] : right[0]].strip(" ，、：:（）()")
            if _relation_rejection_reasons(relation):
                continue
            if not any(marker in relation for marker in RELATION_MARKERS):
                continue
            if not _valid_surface_triple(left[2], relation, right[2], sentence):
                continue
            relationships.append({
                "source": left[2],
                "target": right[2],
                "relation_original": relation,
                "evidence_text": sentence,
                "strength": 5,
            })
    return {
        "entities": list(entities.values()),
        "relationships": relationships,
        "attribute_facts": [],
        "rejected_candidates": [],
    }


GRAPH_EXTRACTION_PROMPT = """你是教材 GraphRAG 三元组抽取专家。输入是一组依据 OCR 版面、自然段和句末符号恢复的 TextUnit。
请利用完整段落的语言上下文，直接抽取正文明确陈述的“实体—原文关系—实体”三元组和数值/公式属性。关系完全来自原文，不使用预设关系词表。

硬性规则：
1. 先理解整段含义，再识别实体和关系。实体直接写在 relationships 的 source/target 或 attribute_facts 的 subject 中，不要另外枚举实体。实体必须是可独立理解的完整名词或技术术语；代词、数值、公式右值、动作词、整句/从句和残缺短语都不是实体。
2. relation_original 必须逐字复制 evidence_text 中连续出现的最短谓词短语，不得改写、概括、规范化、补充或生成另一种关系。
3. source、relation_original、target 的语义方向必须与原文一致。普通三元组的三个字段都逐字来自 evidence_text；如果段内出现“它、该器件、这种结构”等指代，可将 source/target 写成同段中明确出现的完整实体名，并用 source_mention/target_mention 保存事实句中逐字出现的代词。无法唯一确定指代就不输出。
   若原文实体带有“常用的、这种”等上下文修饰，source/target 写核心术语，source_mention/target_mention 写原文完整提及。例如 source=半导体器件、source_mention=常用的半导体器件。
4. 只有数值、单位和完整公式右值放入 attribute_facts.value，不得作为关系的实体节点。attribute_facts 不存普通文字定义。例如“硅管开启电压约为0.5 V”输出 subject=硅管开启电压、relation_original=约为、value=0.5 V。
5. evidence_text 必须逐字复制对应 TextUnit 中能够独立证明该事实的最短连续文本；解决段内指代时可以包含相邻两句。若事实带有条件或适用范围，将原文连续条件短语逐字放入 qualifier_text；没有则为空。保留否定、方向、近似和大小变化；不同事实分别输出。
6. 目录、页码、图号、表号、公式编号、习题要求、文件名和“第几页电路图”不是知识实体。
7. 对 circuit 模态，只抽取图中器件、连接和邻近正文明确支持的知识；不得凭常识猜测电路功能。
8. 不要输出独立 entities 数组；图实体将严格由通过校验的三元组端点和属性主语生成。
9. 输出前逐条自检实体完整性、三元组方向、指代唯一性和原文跨度。没有可靠事实时返回空数组，不要为了连图制造关系。
10. V_T、C_j、V_th 等公式符号、公式编号和公式右值不得作为关系实体；数值或公式只进入 attribute_facts。只有正文明确给出中文物理含义时，才抽取对应的中文概念实体。

抽取示例：
- “半导体器件包括半导体二极管、双极型晶体管。”应输出“半导体器件—包括—半导体二极管”和“半导体器件—包括—双极型晶体管”，不得反向。
- “负反馈能够提高放大电路的稳定性。”应输出“负反馈—能够提高—放大电路的稳定性”。
- “PN结具有单向导电性。它允许电流沿一个方向流动。”第二句可输出 source=PN结、source_mention=它、relation_original=允许、target=电流，evidence_text 保留这两句；不得把“它”建成实体。
- “导电性能良好的物质称为导体。”可输出“导电性能良好的物质—称为—导体”，这是原文明示的定义关系，不得反转。
- “R_D=V_DQ/I_DQ。”只输出 attribute_fact：subject=R_D、relation_original==、value=V_DQ/I_DQ、value_type=formula。

禁止示例：P区—简称—体电阻（主语残缺）；反向击穿—称为—V(BR)（实体边界错误）；R_D—=—V_D（公式被截断）；其电阻（代词残片）；关系中使用“...”占位符；用“的、及其、如”单独作为关系。

仅返回 JSON：
{"items":[{"text_unit_id":"...","relationships":[{"source":"...","source_mention":"...","target":"...","target_mention":"...","relation_original":"...","qualifier_text":"...","evidence_text":"...","strength":1}],"attribute_facts":[{"subject":"...","relation_original":"...","value":"...","value_type":"quantity|formula","evidence_text":"..."}]}]}

TextUnits：
"""


GRAPH_REPAIR_PROMPT = """你是教材三元组抽取结果纠错器。下面给出原始 TextUnit、已经通过校验的事实，以及未通过程序校验的候选和原因。
请重新阅读原文并利用语言理解修正实体边界、原文谓词、方向、指代和证据范围。只返回能够由原文直接证明、且尚未出现在 accepted 中的修正事实；不要解释，不要生成原文不存在的关系，不要为了增加连线而猜测。

要求：
1. relation_original 必须是原文连续出现的最短谓词，不得同义改写。
2. source 和 target 必须是完整知识实体。段内指代可以解析为明确先行实体，但必须填写逐字出现的 source_mention/target_mention。
3. evidence_text 必须是 TextUnit 中的连续原文，普通事实尽量使用单句，指代事实可以使用相邻两句。
4. 同时检查原文是否还有首次遗漏的明确三元组或数值/公式属性；不能可靠修复或补充的事实直接舍弃。
5. 不要输出独立 entities 数组，实体只通过三元组端点或属性主语表达。
6. 只返回单个对象，不再套 items：
{"relationships":[{"source":"...","source_mention":"...","target":"...","target_mention":"...","relation_original":"...","qualifier_text":"...","evidence_text":"...","strength":1}],"attribute_facts":[{"subject":"...","relation_original":"...","value":"...","value_type":"quantity|formula","evidence_text":"..."}]}

方向复核示例：
- “二极管由PN结构成”可修正为“PN结—构成—二极管”，因为“构成”是原文谓词且语义施事是PN结。
- “当PN结反偏时，势垒电压增大”只陈述“势垒电压增大”，不得输出“PN结—增大—势垒电压”；无法形成两个实体之间的明确关系时应舍弃。

待纠错内容：
"""


def _valid_evidence_span(evidence: str, unit_text: str) -> bool:
    return bool(evidence) and _compact(evidence) in _compact(unit_text)


def _text_span(value: str, evidence: str) -> list[int] | None:
    start = evidence.find(value)
    if start < 0:
        return None
    return [start, start + len(value)]


def _ordered_surface_spans(
    source: str,
    relation: str,
    target: str,
    evidence: str,
) -> tuple[list[int], list[int], list[int]] | None:
    def occurrences(value: str) -> list[list[int]]:
        if not value:
            return []
        return [
            [match.start(), match.end()]
            for match in re.finditer(re.escape(value), evidence)
        ]

    candidates: list[tuple[int, list[int], list[int], list[int]]] = []
    harmless_source = re.compile(r"^[\s的也就则可均都]*$")
    harmless_target = re.compile(r"^[\s:：的也就则可均都]*$")
    list_relation = relation in {"包括", "包含", "分为", "分成"}
    definition_relation = relation in {"是指", "属于"}
    for source_span in occurrences(source):
        clause_start = max(
            [evidence.rfind(mark, 0, source_span[0]) for mark in "，,。！？!?；;：:"]
            or [-1]
        )
        source_prefix = evidence[clause_start + 1 : source_span[0]].strip()
        if re.search(r"(?:和|与|及|、|几个|若干)$", source_prefix):
            continue
        for relation_span in occurrences(relation):
            if relation_span[0] < source_span[1]:
                continue
            source_gap = evidence[source_span[1] : relation_span[0]]
            if not harmless_source.match(source_gap):
                continue
            for target_span in occurrences(target):
                if target_span[0] < relation_span[1]:
                    continue
                target_gap = evidence[relation_span[1] : target_span[0]]
                valid_list_gap = (
                    list_relation
                    and not re.search(r"[。！？!?；;]", target_gap)
                    and len(target_gap) <= 100
                )
                valid_definition_gap = (
                    definition_relation
                    and target_gap.endswith("的")
                    and not re.search(r"[。！？!?；;，,：:]", target_gap)
                    and len(target_gap) <= 24
                )
                if not harmless_target.match(target_gap) and not valid_list_gap and not valid_definition_gap:
                    continue
                suffix_end_values = [
                    position for mark in "，,。！？!?；;"
                    if (position := evidence.find(mark, target_span[1])) >= 0
                ]
                suffix_end = min(suffix_end_values) if suffix_end_values else len(evidence)
                target_suffix = evidence[target_span[1] : suffix_end].strip()
                if target_suffix.startswith("的") and len(target_suffix) > 1:
                    continue
                if target_suffix.startswith(("（", "(")) and re.search(
                    r"[）)].*[\u4e00-\u9fff]", target_suffix
                ):
                    continue
                distance = target_span[1] - source_span[0]
                candidates.append((distance, source_span, relation_span, target_span))
    if not candidates:
        return None
    _, source_span, relation_span, target_span = min(candidates, key=lambda item: item[0])
    return source_span, relation_span, target_span


def _relationship_qualifier(
    evidence: str,
    source_span: list[int],
    supplied: str = "",
) -> tuple[str, list[int] | None]:
    supplied_value = _normalized_text(supplied).strip("，,；;。")
    if supplied_value:
        supplied_span = _text_span(supplied_value, evidence)
        if supplied_span:
            return supplied_value, supplied_span
    before_source = evidence[: source_span[0]]
    leading = re.search(
        r"(?:^|[。；;])\s*((?:当|若|如果|在).{1,36}?(?:时|下|情况下))[，,]",
        before_source,
    )
    if leading:
        value = leading.group(1).strip()
        return value, _text_span(value, evidence)
    local = re.search(r"(?:[（(]?)(这时|此时)的?\s*$", before_source)
    if local:
        value = local.group(1)
        return value, _text_span(value, evidence)
    return "", None


def _surface_triple_spans(
    source: str,
    relation: str,
    target: str,
    evidence: str,
    *,
    source_mention: str = "",
    target_mention: str = "",
    allow_semantic_order: bool = False,
) -> tuple[list[int], list[int], list[int]] | None:
    """Locate an auditable triple, allowing explicit paragraph-local coreference."""

    source_surface = _normalized_text(source_mention) or source
    target_surface = _normalized_text(target_mention) or target

    def valid_mention(entity: str, mention: str) -> bool:
        if mention == entity:
            return True
        if COREFERENCE_MENTION_PATTERN.fullmatch(mention):
            return True
        return (
            _compact(entity) in _compact(mention)
            and len(mention) <= len(entity) + 12
            and not re.search(r"[。！？!?；;，,：:]", mention)
        )

    if not valid_mention(source, source_surface):
        return None
    if not valid_mention(target, target_surface):
        return None
    ordered = _ordered_surface_spans(source_surface, relation, target_surface, evidence)
    if ordered is not None or not allow_semantic_order:
        return ordered

    def occurrences(value: str) -> list[list[int]]:
        return [
            [match.start(), match.end()]
            for match in re.finditer(re.escape(value), evidence)
        ]

    candidates: list[tuple[int, list[int], list[int], list[int]]] = []
    for source_span in occurrences(source_surface):
        for relation_span in occurrences(relation):
            for target_span in occurrences(target_surface):
                spans = [source_span, relation_span, target_span]
                if any(
                    left[0] < right[1] and right[0] < left[1]
                    for left, right in combinations(spans, 2)
                ):
                    continue
                window = max(span[1] for span in spans) - min(span[0] for span in spans)
                candidates.append((window, source_span, relation_span, target_span))
    if not candidates:
        return None
    _, source_span, relation_span, target_span = min(candidates, key=lambda item: item[0])
    order = "".join(
        label
        for _, label in sorted((
            (source_span[0], "S"),
            (relation_span[0], "R"),
            (target_span[0], "T"),
        ))
    )
    if order in {"STR", "TRS"}:
        return None
    if order == "TSR":
        passive_gap = evidence[target_span[1] : source_span[0]]
        if not re.search(r"(?:由|被|受)", passive_gap):
            return None
    return source_span, relation_span, target_span


def _valid_surface_triple(
    source: str,
    relation: str,
    target: str,
    evidence: str,
    *,
    source_mention: str = "",
    target_mention: str = "",
    allow_semantic_order: bool = False,
) -> bool:
    """Accept source-grounded triples without requiring a pronoun to become a node."""

    return _surface_triple_spans(
        source,
        relation,
        target,
        evidence,
        source_mention=source_mention,
        target_mention=target_mention,
        allow_semantic_order=allow_semantic_order,
    ) is not None


def _normalize_extraction(
    raw: Any,
    unit: SemanticTextUnit,
    *,
    allow_fallback: bool = False,
    allow_semantic_order: bool = False,
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        if allow_fallback:
            return _fallback_extract(unit)
        return {"entities": [], "relationships": [], "attribute_facts": [], "rejected_candidates": []}
    entity_values = raw.get("entities", [])
    relationship_values = raw.get("relationships", [])
    attribute_values = raw.get("attribute_facts", [])
    entities: list[dict[str, str]] = []
    names: set[str] = set()
    rejected: list[dict[str, Any]] = []
    if isinstance(entity_values, list):
        for item in entity_values:
            if not isinstance(item, dict):
                continue
            name, _surface_name = _canonical_entity_name(str(item.get("name", "")))
            reasons = _entity_rejection_reasons(name)
            if reasons:
                rejected.append({"kind": "entity", "reasons": reasons})
                continue
            if _compact(name) not in _compact(unit.text + unit.section):
                rejected.append({"kind": "entity", "reasons": ["not_in_text"]})
                continue
            raw_type = _normalized_text(str(item.get("type", "课程概念")))[:64] or "课程概念"
            entities.append({
                "name": name,
                "type": _stable_entity_type(name, raw_type),
                "raw_type": raw_type,
                "description": _normalized_text(str(item.get("description", "")))[:500],
            })
            names.add(_compact(name))
    relationships: list[dict[str, Any]] = []
    if isinstance(relationship_values, list):
        for item in relationship_values:
            if not isinstance(item, dict):
                continue
            source, source_surface = _canonical_entity_name(str(item.get("source", "")))
            target, target_surface = _canonical_entity_name(str(item.get("target", "")))
            source_mention = _normalized_text(str(item.get("source_mention", ""))).strip(
                "，。；：、"
            )
            target_mention = _normalized_text(str(item.get("target_mention", ""))).strip(
                "，。；：、"
            )
            if source_surface != source and not source_mention:
                source_mention = source_surface
            if target_surface != target and not target_mention:
                target_mention = target_surface
            relation = _normalized_text(str(item.get("relation_original", ""))).strip("，。；：、")
            evidence = _normalized_text(str(item.get("evidence_text", "")))
            candidate = {
                "source": source,
                "source_mention": source_mention,
                "target": target,
                "target_mention": target_mention,
                "relation_original": relation,
                "qualifier_text": _normalized_text(str(item.get("qualifier_text", ""))),
                "evidence_text": evidence,
            }
            reasons = [
                *[f"source_{reason}" for reason in _entity_rejection_reasons(source)],
                *[f"target_{reason}" for reason in _entity_rejection_reasons(target)],
                *_relation_rejection_reasons(relation),
            ]
            context_text = unit.text + unit.section
            if _compact(source) not in _compact(context_text):
                reasons.append("source_not_in_context")
            if _compact(target) not in _compact(context_text):
                reasons.append("target_not_in_context")
            if _entity_key(source) == _entity_key(target):
                reasons.append("self_relation")
            if _compact(source) in _compact(relation) or _compact(target) in _compact(relation):
                reasons.append("relation_contains_endpoint")
            if reasons:
                rejected.append({
                    "kind": "relationship", "reasons": sorted(set(reasons)),
                    "candidate": candidate,
                })
                continue
            if not _valid_evidence_span(evidence, unit.text):
                rejected.append({
                    "kind": "relationship", "reasons": ["evidence_not_in_text"],
                    "candidate": candidate,
                })
                continue
            direct_spans = _surface_triple_spans(
                source,
                relation,
                target,
                evidence,
                source_mention=source_mention,
                target_mention=target_mention,
            )
            grounded_spans = direct_spans or _surface_triple_spans(
                source,
                relation,
                target,
                evidence,
                source_mention=source_mention,
                target_mention=target_mention,
                allow_semantic_order=True,
            )
            if grounded_spans is None:
                rejected.append({
                    "kind": "relationship", "reasons": ["invalid_surface_order_or_boundary"],
                    "candidate": candidate,
                })
                continue
            surface_order_is_srt = (
                grounded_spans[0][0] < grounded_spans[1][0] < grounded_spans[2][0]
            )
            if not surface_order_is_srt and not allow_semantic_order:
                rejected.append({
                    "kind": "relationship",
                    "reasons": ["requires_semantic_direction_review"],
                    "candidate": candidate,
                })
                continue
            try:
                strength = max(1.0, min(10.0, float(item.get("strength", 5))))
            except (TypeError, ValueError):
                strength = 5.0
            spans = grounded_spans
            if spans is None:
                continue
            qualifier_text, qualifier_span = _relationship_qualifier(
                evidence, spans[0], str(item.get("qualifier_text", ""))
            )
            relationships.append({
                "source": source,
                "source_mention": source_mention or source,
                "target": target,
                "target_mention": target_mention or target,
                "relation_original": relation,
                "evidence_text": evidence,
                "strength": strength,
                "source_span": spans[0],
                "relation_span": spans[1],
                "target_span": spans[2],
                "qualifier_text": qualifier_text,
                "qualifier_span": qualifier_span,
                "coreference_resolved": bool(
                    (source_mention and COREFERENCE_MENTION_PATTERN.fullmatch(source_mention))
                    or (target_mention and COREFERENCE_MENTION_PATTERN.fullmatch(target_mention))
                ),
                "surface_form_resolved": bool(
                    (source_mention and source_mention != source)
                    or (target_mention and target_mention != target)
                ),
                "semantic_direction_reviewed": bool(
                    allow_semantic_order and not surface_order_is_srt
                ),
            })
            for name in (source, target):
                if _compact(name) not in names:
                    entities.append({
                        "name": name,
                        "type": _stable_entity_type(name, _entity_type(name)),
                        "raw_type": _entity_type(name),
                        "description": "",
                    })
                    names.add(_compact(name))
    attribute_facts: list[dict[str, Any]] = []
    if isinstance(attribute_values, list):
        for item in attribute_values:
            if not isinstance(item, dict):
                continue
            subject, _subject_surface = _canonical_entity_name(str(item.get("subject", "")))
            relation = _normalized_text(str(item.get("relation_original", ""))).strip("，。；：、")
            value = _normalized_text(str(item.get("value", ""))).strip("，。；：、")
            value_type = str(item.get("value_type", "text")).strip().lower()
            evidence = _normalized_text(str(item.get("evidence_text", "")))
            reasons = [
                *[f"subject_{reason}" for reason in _entity_rejection_reasons(subject)],
                *_relation_rejection_reasons(relation, allow_formula_operator=True),
            ]
            if not value or len(value) > 120:
                reasons.append("invalid_attribute_value")
            if value_type not in {"quantity", "formula"}:
                reasons.append("unsupported_attribute_type")
            if not _valid_evidence_span(evidence, unit.text):
                reasons.append("evidence_not_in_text")
            attribute_spans = _ordered_surface_spans(subject, relation, value, evidence)
            if not attribute_spans:
                attribute_spans = _surface_triple_spans(
                    subject,
                    relation,
                    value,
                    evidence,
                    allow_semantic_order=True,
                )
            if not attribute_spans:
                reasons.append("attribute_span_not_found")
            if reasons:
                rejected.append({
                    "kind": "attribute_fact", "reasons": sorted(set(reasons)),
                    "candidate": {
                        "subject": subject,
                        "relation_original": relation,
                        "value": value,
                        "value_type": value_type,
                        "evidence_text": evidence,
                    },
                })
                continue
            attribute_facts.append({
                "subject": subject,
                "relation_original": relation,
                "value": value,
                "value_type": value_type,
                "evidence_text": evidence,
                "subject_span": attribute_spans[0],
                "relation_span": attribute_spans[1],
                "value_span": attribute_spans[2],
            })
            if _compact(subject) not in names:
                raw_type = _entity_type(subject)
                entities.append({
                    "name": subject,
                    "type": _stable_entity_type(subject, raw_type),
                    "raw_type": raw_type,
                    "description": "",
                })
                names.add(_compact(subject))
    return {
        "entities": entities,
        "relationships": relationships,
        "attribute_facts": attribute_facts,
        "rejected_candidates": rejected,
    }


def _merge_llm_extractions(
    primary: dict[str, Any],
    repaired: dict[str, Any] | None,
) -> dict[str, Any]:
    """Combine two grounded LLM passes without rewriting any relation text."""

    def rejection_summary(item: Any) -> dict[str, Any] | None:
        if not isinstance(item, dict):
            return None
        return {
            "kind": str(item.get("kind", "")),
            "reasons": list(item.get("reasons", [])),
        }

    result = {
        "entities": list(primary.get("entities", [])),
        "relationships": list(primary.get("relationships", [])),
        "attribute_facts": list(primary.get("attribute_facts", [])),
        "rejected_candidates": [
            summary
            for item in primary.get("rejected_candidates", [])
            if (summary := rejection_summary(item)) is not None
        ],
        "repair_attempted": repaired is not None,
        "repair_recovered_relationships": 0,
        "repair_recovered_attribute_facts": 0,
    }
    if repaired is None:
        return result

    entity_keys = {
        _entity_key(str(item.get("name", "")))
        for item in result["entities"]
        if isinstance(item, dict) and item.get("name")
    }
    for item in repaired.get("entities", []):
        if not isinstance(item, dict) or not item.get("name"):
            continue
        key = _entity_key(str(item["name"]))
        if key not in entity_keys:
            result["entities"].append(item)
            entity_keys.add(key)

    relationship_keys = {
        (
            _entity_key(str(item.get("source", ""))),
            str(item.get("relation_original", "")),
            _entity_key(str(item.get("target", ""))),
            str(item.get("evidence_text", "")),
        )
        for item in result["relationships"]
        if isinstance(item, dict)
    }
    for item in repaired.get("relationships", []):
        if not isinstance(item, dict):
            continue
        key = (
            _entity_key(str(item.get("source", ""))),
            str(item.get("relation_original", "")),
            _entity_key(str(item.get("target", ""))),
            str(item.get("evidence_text", "")),
        )
        if key not in relationship_keys:
            result["relationships"].append(item)
            relationship_keys.add(key)
            result["repair_recovered_relationships"] += 1

    attribute_keys = {
        (
            _entity_key(str(item.get("subject", ""))),
            str(item.get("relation_original", "")),
            str(item.get("value", "")),
            str(item.get("evidence_text", "")),
        )
        for item in result["attribute_facts"]
        if isinstance(item, dict)
    }
    for item in repaired.get("attribute_facts", []):
        if not isinstance(item, dict):
            continue
        key = (
            _entity_key(str(item.get("subject", ""))),
            str(item.get("relation_original", "")),
            str(item.get("value", "")),
            str(item.get("evidence_text", "")),
        )
        if key not in attribute_keys:
            result["attribute_facts"].append(item)
            attribute_keys.add(key)
            result["repair_recovered_attribute_facts"] += 1

    result["rejected_candidates"].extend(
        summary
        for item in repaired.get("rejected_candidates", [])
        if (summary := rejection_summary(item)) is not None
    )
    return result


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
    batch_size: int = 1,
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
            and cached.get("extraction_version") == GRAPH_EXTRACTION_VERSION
        ):
            cached_result = cached.get("result")
            if isinstance(cached_result, dict) and {
                "entities", "relationships", "attribute_facts", "rejected_candidates"
            }.issubset(cached_result):
                results[unit.id] = cached_result
            else:
                results[unit.id] = _normalize_extraction(cached_result, unit)
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
            for attempt in range(1, 4):
                response = client.complete_json(
                    GRAPH_EXTRACTION_PROMPT + json.dumps(payload, ensure_ascii=False)
                )
                if (
                    len(batch) == 1
                    and isinstance(response, dict)
                    and ("relationships" in response or "attribute_facts" in response)
                ):
                    raw_by_id[batch[0].id] = response
                items = response.get("items", []) if isinstance(response, dict) else []
                if isinstance(items, list) and items:
                    raw_by_id.update({
                        str(item.get("text_unit_id")): item
                        for item in items
                        if isinstance(item, dict)
                        and item.get("text_unit_id")
                        and any(unit.id == str(item.get("text_unit_id")) for unit in batch)
                    })
                    if len(raw_by_id) == len(batch):
                        break
                logger.warning(
                    "Semantic graph extraction missed %s TextUnits (attempt %s/3)",
                    len(batch) - len(raw_by_id),
                    attempt,
                )
        primary_by_id: dict[str, dict[str, Any]] = {}
        for unit in batch:
            primary_by_id[unit.id] = _normalize_extraction(
                raw_by_id.get(unit.id),
                unit,
                allow_fallback=client is None,
                allow_semantic_order=False,
            )

        repaired_by_id: dict[str, dict[str, Any]] = {}
        repair_attempted_ids: set[str] = set()
        if client is not None:
            repair_payload: list[dict[str, Any]] = []
            for unit in batch:
                primary = primary_by_id[unit.id]
                rejected_facts = [
                    item for item in primary.get("rejected_candidates", [])
                    if isinstance(item, dict)
                    and item.get("kind") in {"relationship", "attribute_fact"}
                ]
                if not rejected_facts:
                    continue
                compact_rejections = []
                for rejected in rejected_facts[:16]:
                    candidate = rejected.get("candidate", {})
                    compact_rejections.append({
                        "kind": rejected.get("kind"),
                        "reasons": rejected.get("reasons", []),
                        "candidate": {
                            key: value
                            for key, value in candidate.items()
                            if key not in {"evidence_text", "qualifier_text"}
                        } if isinstance(candidate, dict) else {},
                    })
                repair_payload.append({
                    "text_unit_id": unit.id,
                    "modality": unit.modality,
                    "chapter": unit.chapter,
                    "section": unit.section,
                    "text": unit.text,
                    "accepted": {
                        "relationships": [
                            {
                                "source": item.get("source"),
                                "relation_original": item.get("relation_original"),
                                "target": item.get("target"),
                            }
                            for item in primary.get("relationships", [])
                            if isinstance(item, dict)
                        ],
                        "attribute_facts": [
                            {
                                "subject": item.get("subject"),
                                "relation_original": item.get("relation_original"),
                                "value": item.get("value"),
                            }
                            for item in primary.get("attribute_facts", [])
                            if isinstance(item, dict)
                        ],
                    },
                    "rejected_candidates": compact_rejections,
                })
            if repair_payload:
                repair_attempted_ids = {
                    str(item["text_unit_id"]) for item in repair_payload
                }
                for attempt in range(1, 2):
                    response = client.complete_json(
                        GRAPH_REPAIR_PROMPT + json.dumps(repair_payload, ensure_ascii=False)
                    )
                    if (
                        len(repair_payload) == 1
                        and isinstance(response, dict)
                        and ("relationships" in response or "attribute_facts" in response)
                    ):
                        repaired_by_id[str(repair_payload[0]["text_unit_id"])] = response
                    items = response.get("items", []) if isinstance(response, dict) else []
                    if isinstance(items, list):
                        repaired_by_id.update({
                            str(item.get("text_unit_id")): item
                            for item in items
                            if isinstance(item, dict)
                            and str(item.get("text_unit_id")) in repair_attempted_ids
                        })
                    if len(repaired_by_id) == len(repair_attempted_ids):
                        break
                    logger.warning(
                        "Semantic graph repair missed %s TextUnits (attempt %s/1)",
                        len(repair_attempted_ids) - len(repaired_by_id),
                        attempt,
                    )

        normalized_batch: list[tuple[SemanticTextUnit, dict[str, Any]]] = []
        for unit in batch:
            repaired = (
                _normalize_extraction(
                    repaired_by_id.get(unit.id),
                    unit,
                    allow_semantic_order=True,
                )
                if unit.id in repair_attempted_ids
                else None
            )
            normalized_batch.append(
                (unit, _merge_llm_extractions(primary_by_id[unit.id], repaired))
            )
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
                    "extraction_version": GRAPH_EXTRACTION_VERSION,
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
                        "extraction_version": GRAPH_EXTRACTION_VERSION,
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
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    unit_map = {unit.id: unit for unit in units}
    entities: dict[str, dict[str, Any]] = {}
    raw_mentions: list[dict[str, Any]] = []
    attribute_facts: list[dict[str, Any]] = []

    def ensure_entity(
        name: str,
        entity_type: str = "课程概念",
        description: str = "",
        raw_type: str = "",
    ) -> str:
        name, _surface_name = _canonical_entity_name(name)
        if not _valid_entity_name(name):
            return ""
        key = _entity_key(name)
        entity_id = _stable_id("entity", key)
        stable_type = _stable_entity_type(name, entity_type or raw_type)
        entity = entities.setdefault(entity_id, {
            "id": entity_id,
            "type": "entity",
            "entity_type": stable_type,
            "name": name,
            "description": description,
            "aliases": [],
            "raw_entity_types": [],
            "text_unit_ids": [],
            "evidence_count": 0,
            "pages": [],
        })
        if name != entity["name"]:
            if _display_name_score(name) < _display_name_score(str(entity["name"])):
                previous_name = str(entity["name"])
                entity["name"] = name
                if name in entity["aliases"]:
                    entity["aliases"].remove(name)
                if previous_name not in entity["aliases"]:
                    entity["aliases"].append(previous_name)
            elif name not in entity["aliases"]:
                entity["aliases"].append(name)
        observed_type = raw_type or entity_type
        if observed_type and observed_type not in entity["raw_entity_types"]:
            entity["raw_entity_types"].append(observed_type)
        if description and not entity.get("description"):
            entity["description"] = description
        return entity_id

    for unit_id, extraction in extractions.items():
        unit = unit_map[unit_id]
        for item in extraction.get("entities", []):
            if not isinstance(item, dict) or not item.get("name"):
                continue
            entity_id = ensure_entity(
                str(item["name"]),
                str(item.get("type", "课程概念")),
                str(item.get("description", "")),
                str(item.get("raw_type", item.get("type", "课程概念"))),
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
            source_name, source_surface = _canonical_entity_name(str(item.get("source", "")))
            target_name, target_surface = _canonical_entity_name(str(item.get("target", "")))
            relation = str(item.get("relation_original", "")).strip()
            evidence_text = str(item.get("evidence_text", ""))
            source_mention = str(item.get("source_mention", "")).strip()
            target_mention = str(item.get("target_mention", "")).strip()
            if source_surface != source_name and not source_mention:
                source_mention = source_surface
            if target_surface != target_name and not target_mention:
                target_mention = target_surface
            if (
                not source_name
                or not target_name
                or not relation
                or _relation_rejection_reasons(relation)
                or _compact(source_name) in _compact(relation)
                or _compact(target_name) in _compact(relation)
                or not _valid_surface_triple(
                    source_name,
                    relation,
                    target_name,
                    evidence_text,
                    source_mention=source_mention,
                    target_mention=target_mention,
                    allow_semantic_order=True,
                )
            ):
                continue
            source_id = ensure_entity(source_name, _entity_type(source_name))
            target_id = ensure_entity(target_name, _entity_type(target_name))
            if not source_id or not target_id or source_id == target_id:
                continue
            for entity_id, canonical_name, mention in (
                (source_id, source_name, source_mention),
                (target_id, target_name, target_mention),
            ):
                if (
                    mention
                    and mention != canonical_name
                    and not COREFERENCE_MENTION_PATTERN.fullmatch(mention)
                    and _compact(canonical_name) in _compact(mention)
                    and mention not in entities[entity_id]["aliases"]
                ):
                    entities[entity_id]["aliases"].append(mention)
            for entity_id in (source_id, target_id):
                if evidence_text and not str(entities[entity_id].get("description", "")).strip():
                    entities[entity_id]["description"] = evidence_text[:500]
            source_span = item.get("source_span") or _text_span(source_name, evidence_text)
            qualifier_text, qualifier_span = (
                _relationship_qualifier(
                    evidence_text,
                    source_span,
                    str(item.get("qualifier_text", "")),
                )
                if source_span else ("", None)
            )
            mention_id = _stable_id("relationship-mention", unit_id, position, source_id, relation, target_id)
            raw_mentions.append({
                "id": mention_id,
                "source": source_id,
                "target": target_id,
                "source_name": source_name,
                "source_mention": source_mention or source_name,
                "target_name": target_name,
                "target_mention": target_mention or target_name,
                "relation": relation,
                "evidence_text": evidence_text,
                "strength": float(item.get("strength", 5) or 5),
                "text_unit_id": unit_id,
                "evidence_id": unit_id,
                "source_modality": unit.modality,
                "source_page": unit.page_start,
                "source_span": source_span,
                "relation_span": item.get("relation_span"),
                "target_span": item.get("target_span"),
                "qualifier_text": qualifier_text,
                "qualifier_span": qualifier_span,
                "coreference_resolved": bool(item.get("coreference_resolved", False)),
                "surface_form_resolved": bool(item.get("surface_form_resolved", False)),
                "semantic_direction_reviewed": bool(
                    item.get("semantic_direction_reviewed", False)
                ),
            })
        for position, item in enumerate(extraction.get("attribute_facts", []), 1):
            if not isinstance(item, dict):
                continue
            subject_name, _subject_surface = _canonical_entity_name(str(item.get("subject", "")))
            relation = str(item.get("relation_original", "")).strip()
            value = str(item.get("value", "")).strip()
            evidence_text = str(item.get("evidence_text", ""))
            if (
                _relation_rejection_reasons(relation, allow_formula_operator=True)
                or str(item.get("value_type", "")) not in {"quantity", "formula"}
                or not _valid_evidence_span(evidence_text, unit.text)
                or not _valid_surface_triple(
                    subject_name,
                    relation,
                    value,
                    evidence_text,
                    allow_semantic_order=True,
                )
            ):
                continue
            subject_id = ensure_entity(subject_name, _entity_type(subject_name))
            if not subject_id or not relation or not value:
                continue
            entity = entities[subject_id]
            if evidence_text and not str(entity.get("description", "")).strip():
                entity["description"] = evidence_text[:500]
            if unit_id not in entity["text_unit_ids"]:
                entity["text_unit_ids"].append(unit_id)
            for page in range(unit.page_start, unit.page_end + 1):
                if page not in entity["pages"]:
                    entity["pages"].append(page)
            attribute_facts.append({
                "id": _stable_id(
                    "attribute-fact", unit_id, position, subject_id, relation, value
                ),
                "subject": subject_id,
                "relation": relation,
                "value": value,
                "value_type": str(item.get("value_type", "text")),
                "evidence_text": evidence_text,
                "text_unit_id": unit_id,
                "evidence_id": unit_id,
                "source_page": unit.page_start,
                "subject_span": item.get("subject_span"),
                "relation_span": item.get("relation_span"),
                "value_span": item.get("value_span"),
            })

    for position, item in enumerate(circuit_mentions, 1):
        source_name = str(item.get("source", "")).strip()
        target_name = str(item.get("target", "")).strip()
        relation = str(item.get("relation_original", "")).strip()
        if not source_name or not target_name or not relation:
            continue
        source_id = ensure_entity(source_name, _entity_type(source_name))
        target_id = ensure_entity(target_name, _entity_type(target_name))
        if not source_id or not target_id or source_id == target_id:
            continue
        mention_id = _stable_id(
            "relationship-mention", item.get("evidence_id"), position, source_id, relation, target_id
        )
        raw_mentions.append({
            "id": mention_id,
            "source": source_id,
            "target": target_id,
            "source_name": source_name,
            "target_name": target_name,
            "relation": relation,
            "evidence_text": str(item.get("evidence_text", "")),
            "strength": float(item.get("strength", 5) or 5),
            "text_unit_id": item.get("text_unit_id"),
            "evidence_id": item.get("evidence_id"),
            "source_modality": "circuit_visual",
            "source_page": item.get("source_page"),
            "qualifier_text": "",
            "qualifier_span": None,
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
            "qualifier_texts": [],
        })
        edge["mention_ids"].append(mention["id"])
        if mention.get("text_unit_id") and mention["text_unit_id"] not in edge["text_unit_ids"]:
            edge["text_unit_ids"].append(mention["text_unit_id"])
        if mention.get("evidence_id") and mention["evidence_id"] not in edge["evidence_ids"]:
            edge["evidence_ids"].append(mention["evidence_id"])
        edge["evidence_count"] += 1
        edge["weight"] += float(mention.get("strength", 5))
        if mention.get("qualifier_text") and mention["qualifier_text"] not in edge["qualifier_texts"]:
            edge["qualifier_texts"].append(mention["qualifier_text"])

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
    connected_ids.update(str(item["subject"]) for item in attribute_facts)
    relationship_counts: dict[str, int] = defaultdict(int)
    for edge in grouped.values():
        relationship_counts[str(edge["source"])] += 1
        relationship_counts[str(edge["target"])] += 1
    for entity_id in list(entities):
        entity = entities[entity_id]
        if entity_id not in connected_ids:
            del entities[entity_id]
            continue
        entity["evidence_count"] = len(entity["text_unit_ids"])
        entity["relationship_count"] = relationship_counts.get(entity_id, 0)
        entity["standalone"] = entity_id not in connected_ids
        entity["pages"] = sorted(entity["pages"])
    return (
        list(entities.values()),
        list(grouped.values()),
        raw_mentions,
        list(links.values()),
        attribute_facts,
    )


def _graph_display_name(node: dict[str, Any]) -> str:
    return str(node.get("display_name") or node.get("name") or "").strip()


def _symbol_tokens(value: str) -> list[str]:
    return list(dict.fromkeys(
        match.group(0) for match in FORMULA_SYMBOL_PATTERN.finditer(str(value))
    ))


def _clean_chinese_display_name(value: str) -> str:
    label = _normalized_text(value).strip(" ，,。；;：:（）()的")
    for marker in ("称为", "又称", "简称", "叫做"):
        if marker in label:
            label = label.rsplit(marker, 1)[-1].strip()
    label = re.sub(r"^(?:一个|某个|所谓)", "", label).strip()
    label = re.sub(r"\s+(?=[\u4e00-\u9fff])", "", label)
    if not re.search(r"[\u4e00-\u9fff]", label):
        return ""
    if not 2 <= len(label) <= 24 or re.search(r"[，,。；;：:]", label):
        return ""
    if re.match(r"^(?:当|以|使|若|在|由|且|并|可|对|将|从|和|与|为了|保证)", label):
        return ""
    return label


def enrich_semantic_display_names(graph: dict[str, Any]) -> dict[str, Any]:
    """Add source-grounded Chinese labels while preserving textbook symbols.

    Entity ``name`` and every relationship remain untouched.  ``display_name``
    is only derived when the same TextUnit explicitly defines or apposes a
    formula symbol with a Chinese technical term.  This also upgrades existing
    semantic graph files at read time without rewriting their source data.
    """

    nodes = [item for item in graph.get("nodes", []) if isinstance(item, dict)]
    nodes_by_id = {str(item.get("id")): item for item in nodes if item.get("id")}
    symbol_forms: dict[str, set[str]] = defaultdict(set)
    for node in nodes:
        for value in [str(node.get("name", "")), *map(str, node.get("aliases", []))]:
            for symbol in _symbol_tokens(value):
                symbol_forms[_entity_key(symbol)].add(symbol)

    candidates: dict[str, list[tuple[int, int, str, str, str]]] = defaultdict(list)

    def add_candidate(
        symbol: str,
        label: str,
        *,
        score: int,
        evidence_text: str = "",
        text_unit_id: str = "",
    ) -> None:
        key = _entity_key(symbol)
        if key not in symbol_forms:
            return
        cleaned = _clean_chinese_display_name(label)
        if not cleaned or not cleaned.endswith(DISPLAY_NAME_SUFFIXES):
            return
        candidates[key].append((score, len(cleaned), cleaned, evidence_text, text_unit_id))

    text_units = [
        item for item in graph.get("text_units", [])
        if isinstance(item, dict) and str(item.get("text", "")).strip()
    ]
    for _key, forms in symbol_forms.items():
        for symbol in sorted(forms, key=lambda item: (-len(item), item)):
            escaped = re.escape(symbol)
            symbol_span = rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])"
            for unit in text_units:
                text = _normalized_text(str(unit.get("text", "")))
                if not re.search(symbol_span, text):
                    continue
                text_unit_id = str(unit.get("id", ""))
                patterns = (
                    (
                        130,
                        rf"{symbol_span}\s*[，,:：]?\s*(?:为|是|称为|又称|简称|叫做)\s*"
                        rf"(?P<label>[\u4e00-\u9fff][^，,。；;：:（）()]{{1,23}})",
                    ),
                    (
                        130,
                        rf"(?:称为|又称|简称|叫做)\s*"
                        rf"(?P<label>[A-Za-z\u4e00-\u9fff][^，,。；;：:（）()]{{1,23}})"
                        rf"\s*[）)]\s*{symbol_span}",
                    ),
                    (
                        120,
                        rf"(?P<label>[^，,。；;：:（）()]{{2,24}})[，,]\s*"
                        rf"(?:用|以)\s*{symbol_span}\s*(?:来)?(?:表示|记作)",
                    ),
                    (
                        80,
                        rf"(?P<label>[\u4e00-\u9fffA-Za-z]{{2,12}})\s+{symbol_span}",
                    ),
                )
                for score, pattern in patterns:
                    for match in re.finditer(pattern, text):
                        label = _clean_chinese_display_name(match.group("label"))
                        if not label:
                            continue
                        if score == 80 and (
                            len(symbol) == 1
                            or not re.search(r"[_()0-9Α-Ωα-ω]", symbol)
                            or not label.endswith(DISPLAY_NAME_SUFFIXES)
                        ):
                            continue
                        add_candidate(
                            symbol,
                            label,
                            score=score,
                            evidence_text=match.group(0),
                            text_unit_id=text_unit_id,
                        )

    definitions: dict[str, tuple[str, str, str]] = {}
    for key, values in candidates.items():
        if not values:
            continue
        _score, _length, label, evidence_text, text_unit_id = sorted(
            values,
            key=lambda item: (-item[0], item[1], item[2]),
        )[0]
        definitions[key] = (label, evidence_text, text_unit_id)

    localized = 0
    definition_records: list[dict[str, Any]] = []
    for key, (label, evidence_text, text_unit_id) in sorted(definitions.items()):
        representative = sorted(symbol_forms[key], key=lambda item: (-len(item), item))[0]
        definition_records.append({
            "symbol": representative,
            "chinese_name": label,
            "evidence_text": evidence_text,
            "text_unit_id": text_unit_id,
        })

    for node in nodes:
        raw_name = str(node.get("name", "")).strip()
        display_name = raw_name
        matched_symbols: list[str] = []
        for symbol in sorted(_symbol_tokens(raw_name), key=lambda item: -len(item)):
            definition = definitions.get(_entity_key(symbol))
            if not definition:
                continue
            label = definition[0]
            matched_symbols.append(symbol)
            if label in display_name:
                display_name = display_name.replace(symbol, "")
            else:
                display_name = display_name.replace(symbol, label)
        display_name = re.sub(r"\s*的\s*", "的", display_name)
        display_name = re.sub(r"(?<=[A-Za-z])\s+(?=[\u4e00-\u9fff])", "", display_name)
        display_name = re.sub(r"\s+", " ", display_name).strip(" ·•，,")
        if matched_symbols and display_name and display_name != raw_name:
            node["display_name"] = display_name
            node["symbols"] = list(dict.fromkeys(matched_symbols))
            node["display_name_source"] = "explicit_text_definition"
            localized += 1

    for chapter in graph.get("chapters", []):
        if not isinstance(chapter, dict):
            continue
        for concept in chapter.get("concepts", []):
            if not isinstance(concept, dict):
                continue
            entity_id = str(concept.get("entity_id") or concept.get("id") or "")
            node = nodes_by_id.get(entity_id)
            if not node or not node.get("display_name"):
                continue
            concept["source_entity_name"] = str(node.get("name", ""))
            concept["name"] = _graph_display_name(node)

    graph["symbol_definitions"] = definition_records
    graph.setdefault("stats", {})["localized_entity_names"] = localized
    return graph


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
    covered = set().union(*partitions) if partitions else set()
    partitions.extend({node_id} for node_id in graph.nodes if str(node_id) not in covered)

    node_map = {str(node["id"]): node for node in nodes}
    communities: list[dict[str, Any]] = []
    for index, members in enumerate(sorted(partitions, key=lambda item: (-len(item), sorted(item)[0])), 1):
        member_links = [
            link for link in links
            if str(link["source"]) in members and str(link["target"]) in members
        ]
        ranked = sorted(
            members,
            key=lambda node_id: (-graph.degree(node_id, weight="weight"), _graph_display_name(node_map[node_id])),
        )
        names = [_graph_display_name(node_map[node_id]) for node_id in ranked[:4]]
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
            _graph_display_name(node_map[entity_id])
            for entity_id in community.get("entity_ids", [])
            if entity_id in node_map
        ]
        relation_lines = [
            f"{_graph_display_name(node_map[edge['source']])}—{edge['relation']}→{_graph_display_name(node_map[edge['target']])}"
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


def bind_chapter_knowledge_points(
    chapters: Iterable[dict[str, Any]],
    nodes: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Bind chapter browse entries to final semantic entities.

    Unresolved legacy/tag concepts are recorded for audit but are not exposed as
    fake graph nodes. Every item left in ``chapter.concepts`` therefore points to
    a real semantic entity ID.
    """

    node_items = [item for item in nodes if item.get("id") and item.get("name")]
    exact: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for node in node_items:
        for label in [
            str(node.get("name", "")),
            str(node.get("display_name", "")),
            *map(str, node.get("aliases", [])),
        ]:
            if not label:
                continue
            values = exact[_entity_key(label)]
            if not any(str(item.get("id")) == str(node.get("id")) for item in values):
                values.append(node)

    resolved_total = 0
    source_total = 0
    unresolved_names: list[str] = []
    bound_chapters: list[dict[str, Any]] = []
    for chapter in chapters:
        chapter_copy = {key: value for key, value in chapter.items() if key != "concepts"}
        chapter_pages = {
            int(page) for page in chapter.get("pages", [])
            if isinstance(page, int) and page > 0
        }
        bound: list[dict[str, Any]] = []
        unresolved: list[str] = []
        seen_ids: set[str] = set()
        for concept in chapter.get("concepts", []):
            if not isinstance(concept, dict):
                continue
            source_total += 1
            name = str(concept.get("name", "")).strip()
            candidates = list(exact.get(_entity_key(name), []))
            match_method = "exact"
            if chapter_pages:
                candidates = [
                    node for node in candidates
                    if not node.get("pages") or chapter_pages.intersection(node.get("pages", []))
                ] or candidates
            if not candidates and len(_compact(name)) >= 2:
                contained = [
                    node for node in node_items
                    if (
                        any(
                            _compact(name) in _compact(label)
                            or _compact(label) in _compact(name)
                            for label in (
                                str(node.get("name", "")),
                                str(node.get("display_name", "")),
                            )
                            if label
                        )
                    )
                    and (
                        not chapter_pages
                        or not node.get("pages")
                        or chapter_pages.intersection(node.get("pages", []))
                    )
                ]
                if len(contained) == 1:
                    candidates = contained
                    match_method = "contained"
            if not candidates and len(_compact(name)) >= 3:
                ranked: list[tuple[float, dict[str, Any]]] = []
                for node in node_items:
                    if (
                        chapter_pages
                        and node.get("pages")
                        and not chapter_pages.intersection(node.get("pages", []))
                    ):
                        continue
                    labels = [
                        str(node.get("name", "")),
                        str(node.get("display_name", "")),
                        *map(str, node.get("aliases", [])),
                    ]
                    score = max(
                        SequenceMatcher(None, _compact(name), _compact(label)).ratio()
                        for label in labels if label
                    )
                    ranked.append((score, node))
                ranked.sort(key=lambda item: (-item[0], str(item[1].get("name", ""))))
                if ranked:
                    best_score, best_node = ranked[0]
                    second_score = ranked[1][0] if len(ranked) > 1 else 0.0
                    if best_score >= 0.84 or (
                        best_score >= 0.72 and best_score - second_score >= 0.08
                    ):
                        candidates = [best_node]
                        match_method = "fuzzy"
            if len(candidates) != 1:
                unresolved.append(name)
                unresolved_names.append(name)
                continue
            node = candidates[0]
            entity_id = str(node["id"])
            if entity_id in seen_ids:
                continue
            seen_ids.add(entity_id)
            resolved_total += 1
            bound.append({
                **concept,
                "id": entity_id,
                "entity_id": entity_id,
                "name": _graph_display_name(node),
                "source_entity_name": str(node["name"]),
                "source_concept_name": name,
                "match_method": match_method,
            })
        chapter_copy["concepts"] = bound
        chapter_copy["concept_count"] = len(bound)
        chapter_copy["unresolved_concepts"] = unresolved
        bound_chapters.append(chapter_copy)
    return bound_chapters, {
        "source_concepts": source_total,
        "resolved_concepts": resolved_total,
        "unresolved_concepts": source_total - resolved_total,
        "resolution_rate": round(resolved_total / source_total, 4) if source_total else 1.0,
        "unresolved_names": sorted(set(unresolved_names)),
    }


def audit_semantic_graph_quality(graph: dict[str, Any]) -> dict[str, Any]:
    """Audit graph format, provenance and conservative semantic invariants."""

    nodes = [item for item in graph.get("nodes", []) if isinstance(item, dict)]
    edges = [item for item in graph.get("edges", []) if isinstance(item, dict)]
    mentions = [
        item for item in graph.get("relationship_mentions", []) if isinstance(item, dict)
    ]
    attributes = [item for item in graph.get("attribute_facts", []) if isinstance(item, dict)]
    evidence = [item for item in graph.get("evidence", []) if isinstance(item, dict)]
    node_map = {str(item.get("id")): item for item in nodes if item.get("id")}
    evidence_map = {str(item.get("id")): item for item in evidence if item.get("id")}
    issues: list[dict[str, Any]] = []

    def add_issue(code: str, message: str, *, severity: str = "critical", count: int = 1) -> None:
        issues.append({"code": code, "severity": severity, "count": count, "message": message})

    node_ids = [str(item.get("id", "")) for item in nodes]
    edge_ids = [str(item.get("id", "")) for item in edges]
    if len(node_ids) != len(set(node_ids)):
        add_issue("duplicate_node_ids", "存在重复实体 ID")
    if len(edge_ids) != len(set(edge_ids)):
        add_issue("duplicate_edge_ids", "存在重复关系 ID")
    dangling_edges = [
        item for item in edges
        if str(item.get("source")) not in node_map or str(item.get("target")) not in node_map
    ]
    if dangling_edges:
        add_issue("dangling_edges", "存在指向缺失实体的关系", count=len(dangling_edges))

    invalid_nodes = [item for item in nodes if _entity_rejection_reasons(str(item.get("name", "")))]
    if invalid_nodes:
        add_issue("invalid_entity_names", "存在代词、动作词、位置或句子残片实体", count=len(invalid_nodes))
    alias_groups: dict[str, list[str]] = defaultdict(list)
    for node in nodes:
        alias_groups[_entity_key(str(node.get("name", "")))].append(str(node.get("id", "")))
    duplicate_alias_groups = [values for values in alias_groups.values() if len(values) > 1]
    if duplicate_alias_groups:
        add_issue("unmerged_aliases", "存在未合并的公式书写变体实体", count=len(duplicate_alias_groups))

    invalid_relations = [
        item for item in edges if _relation_rejection_reasons(str(item.get("relation", "")))
    ]
    if invalid_relations:
        add_issue("invalid_relations", "存在整句、字面量或功能词关系", count=len(invalid_relations))
    bad_mentions = 0
    for mention in mentions:
        source_node = node_map.get(str(mention.get("source")))
        target_node = node_map.get(str(mention.get("target")))
        evidence_item = evidence_map.get(str(mention.get("evidence_id")))
        if not source_node or not target_node or not evidence_item:
            bad_mentions += 1
            continue
        evidence_text = str(mention.get("evidence_text", ""))
        stored_text = str(evidence_item.get("text", ""))
        if mention.get("source_modality") == "circuit_visual":
            if not evidence_text:
                bad_mentions += 1
            continue
        if (
            not _valid_evidence_span(evidence_text, stored_text)
            or not _valid_surface_triple(
                str(mention.get("source_name") or source_node.get("name", "")),
                str(mention.get("relation", "")),
                str(mention.get("target_name") or target_node.get("name", "")),
                evidence_text,
                source_mention=str(mention.get("source_mention", "")),
                target_mention=str(mention.get("target_mention", "")),
                allow_semantic_order=True,
            )
        ):
            bad_mentions += 1
    if bad_mentions:
        add_issue("invalid_relationship_provenance", "关系原文或三元组跨度无法回溯", count=bad_mentions)

    bad_attributes = 0
    for fact in attributes:
        subject = node_map.get(str(fact.get("subject")))
        evidence_item = evidence_map.get(str(fact.get("evidence_id")))
        evidence_text = str(fact.get("evidence_text", ""))
        if (
            not subject
            or not evidence_item
            or not _valid_evidence_span(evidence_text, str(evidence_item.get("text", "")))
            or not _text_span(str(fact.get("relation", "")), evidence_text)
            or not _text_span(str(fact.get("value", "")), evidence_text)
        ):
            bad_attributes += 1
    if bad_attributes:
        add_issue("invalid_attribute_provenance", "公式或数值属性无法回溯原文", count=bad_attributes)

    graph_view = nx.Graph()
    graph_view.add_nodes_from(node_map)
    graph_view.add_edges_from(
        (str(item.get("source")), str(item.get("target"))) for item in edges
        if str(item.get("source")) in node_map and str(item.get("target")) in node_map
    )
    components = nx.number_connected_components(graph_view) if graph_view.number_of_nodes() else 0
    degree_one = sum(1 for _, degree in graph_view.degree() if degree == 1)
    isolated = sum(1 for _, degree in graph_view.degree() if degree == 0)
    relation_counts: dict[str, int] = defaultdict(int)
    for edge in edges:
        relation_counts[str(edge.get("relation", ""))] += 1
    singleton_relations = sum(1 for count in relation_counts.values() if count == 1)
    text_unit_count = len(graph.get("text_units", []))
    relationship_unit_count = len({
        str(item.get("text_unit_id")) for item in mentions if item.get("text_unit_id")
    })
    fact_unit_count = len({
        str(item.get("text_unit_id"))
        for item in [*mentions, *attributes]
        if item.get("text_unit_id")
    })
    if text_unit_count >= 20 and fact_unit_count < max(5, round(text_unit_count * 0.12)):
        add_issue(
            "low_fact_coverage",
            "有效 TextUnit 中抽取到关系或属性事实的比例过低",
            count=max(5, round(text_unit_count * 0.12)) - fact_unit_count,
        )
    largest_component = (
        max((len(component) for component in nx.connected_components(graph_view)), default=0)
        if graph_view.number_of_nodes() else 0
    )
    if len(nodes) >= 20 and largest_component / max(1, len(nodes)) < 0.35:
        add_issue(
            "fragmented_graph",
            "图谱连通性偏低，建议检查同义实体和跨句知识衔接",
            severity="warning",
        )
    empty_descriptions = sum(1 for item in nodes if not str(item.get("description", "")).strip())
    if nodes and empty_descriptions / len(nodes) > 0.5:
        add_issue(
            "sparse_entity_descriptions",
            "超过一半实体缺少可回溯的简短说明",
            severity="warning",
            count=empty_descriptions,
        )

    chapter_concepts = [
        concept
        for chapter in graph.get("chapters", [])
        if isinstance(chapter, dict)
        for concept in chapter.get("concepts", [])
        if isinstance(concept, dict)
    ]
    unresolved_chapter_refs = [
        item for item in chapter_concepts if str(item.get("id", "")) not in node_map
    ]
    if unresolved_chapter_refs:
        add_issue(
            "unresolved_chapter_concepts",
            "章节知识点没有引用最终语义实体",
            count=len(unresolved_chapter_refs),
        )
    omitted_concepts = sum(
        len(chapter.get("unresolved_concepts", []))
        for chapter in graph.get("chapters", [])
        if isinstance(chapter, dict)
    )
    if omitted_concepts:
        add_issue(
            "omitted_chapter_concepts",
            "部分旧知识标签未能与最终实体可靠对齐，已从显示列表移除",
            severity="warning",
            count=omitted_concepts,
        )

    critical_count = sum(item["count"] for item in issues if item["severity"] == "critical")
    return {
        "schema_version": "1.0-semantic-quality",
        "status": "passed" if critical_count == 0 else "failed",
        "critical_issues": critical_count,
        "warning_issues": sum(item["count"] for item in issues if item["severity"] == "warning"),
        "metrics": {
            "entities": len(nodes),
            "relationships": len(edges),
            "relationship_mentions": len(mentions),
            "attribute_facts": len(attributes),
            "text_units": text_unit_count,
            "relationship_text_units": relationship_unit_count,
            "fact_text_units": fact_unit_count,
            "fact_coverage": round(fact_unit_count / text_unit_count, 4) if text_unit_count else 1.0,
            "components": components,
            "largest_component": largest_component,
            "isolated_entities": isolated,
            "degree_one_entities": degree_one,
            "unique_relation_mentions": len(relation_counts),
            "singleton_relation_mentions": singleton_relations,
            "entity_types": len({str(item.get("entity_type", "")) for item in nodes}),
            "chapter_concepts": len(chapter_concepts),
        },
        "issues": issues,
    }


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
    nodes, edges, mentions, links, attribute_facts = _merge_graph_records(
        units, extractions, circuit_mentions
    )
    text_units = [unit.to_dict() for unit in units]
    display_metadata: dict[str, Any] = {
        "nodes": nodes,
        "edges": edges,
        "text_units": text_units,
        "stats": {},
    }
    enrich_semantic_display_names(display_metadata)
    communities = build_graph_communities(nodes, links)
    reports = build_community_reports(communities, nodes, edges)
    rejected_candidates = [
        candidate
        for extraction in extractions.values()
        for candidate in extraction.get("rejected_candidates", [])
        if isinstance(candidate, dict)
    ]
    rejection_reasons: dict[str, int] = defaultdict(int)
    for candidate in rejected_candidates:
        for reason in candidate.get("reasons", []):
            rejection_reasons[str(reason)] += 1
    repair_attempted_units = sum(
        1 for extraction in extractions.values() if extraction.get("repair_attempted")
    )
    repaired_relationships = sum(
        int(extraction.get("repair_recovered_relationships", 0) or 0)
        for extraction in extractions.values()
    )
    repaired_attribute_facts = sum(
        int(extraction.get("repair_recovered_attribute_facts", 0) or 0)
        for extraction in extractions.values()
    )
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
        "attribute_facts": attribute_facts,
        "text_units": text_units,
        "evidence": [*text_evidence, *circuit_evidence],
        "communities": communities,
        "community_reports": reports,
        "symbol_definitions": display_metadata.get("symbol_definitions", []),
        "stats": {
            "entities": len(nodes),
            "relationships": len(edges),
            "relationship_mentions": len(mentions),
            "attribute_facts": len(attribute_facts),
            "text_units": len(units),
            "communities": len(communities),
            "circuit_evidence": len(circuit_evidence),
            "extraction_method": "text_llm" if client is not None else "rule_fallback",
            "extraction_model": str(
                getattr(getattr(client, "config", None), "model", "rule")
            ),
            "localized_entity_names": int(
                display_metadata.get("stats", {}).get("localized_entity_names", 0)
            ),
            "repair_attempted_units": repair_attempted_units,
            "repair_recovered_relationships": repaired_relationships,
            "repair_recovered_attribute_facts": repaired_attribute_facts,
            "rejected_candidates": len(rejected_candidates),
            "rejection_reasons": dict(sorted(rejection_reasons.items())),
        },
    }


def is_semantic_graph(graph: dict[str, Any]) -> bool:
    return str(graph.get("schema_version", "")).startswith("3.")
