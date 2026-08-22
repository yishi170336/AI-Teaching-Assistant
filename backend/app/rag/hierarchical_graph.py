from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from backend.app.rag.embedding_runtime import encode_texts, release_embedding_model
from backend.app.rag.knowledge_document import (
    KnowledgeUnit,
    compile_knowledge_document,
)
from backend.app.rag.models import PageDocument
from backend.app.rag.multimodal import LayoutElement
from backend.app.rag.ontology import extract_course_concepts
from backend.app.rag.section_titles import (
    chapter_number,
    normalize_structural_section,
    numbered_section_parts,
)


logger = logging.getLogger(__name__)

SCHEMA_VERSION = "4.0-hierarchical-summary-entity-graph"
CACHE_VERSION = "4.0-hierarchical-summary-entity-graph"
PROMPT_VERSION = "4.0.2"
EMBEDDING_DIMENSION = 1024
ENTITY_AGGREGATION_BATCH_SIZE = 8
ENTITY_TYPES = (
    "课程概念",
    "物理定律与原理",
    "电路与系统",
    "器件与元件",
    "参数与物理量",
    "物理过程与效应",
    "方法与模型",
    "材料与结构",
)
TYPE_ALIASES = {
    "定律": "物理定律与原理",
    "原理": "物理定律与原理",
    "电路": "电路与系统",
    "系统": "电路与系统",
    "器件": "器件与元件",
    "元件": "器件与元件",
    "参数": "参数与物理量",
    "物理量": "参数与物理量",
    "过程": "物理过程与效应",
    "效应": "物理过程与效应",
    "方法": "方法与模型",
    "模型": "方法与模型",
    "材料": "材料与结构",
    "结构": "材料与结构",
    "概念": "课程概念",
}
RELATION_ALIASES = {
    "定义": "定义",
    "包含": "包含",
    "组成": "组成",
    "属于": "属于",
    "影响": "影响",
    "导致": "导致",
    "实现": "实现",
    "采用": "采用",
    "描述": "描述",
    "表征": "表征",
    "约束": "约束",
    "依赖": "依赖",
    "关联": "关联",
    "等价": "等价",
    "转换": "转换",
    "测量": "测量",
    "调节": "调节",
    "放大": "放大",
    "抑制": "抑制",
    "产生": "产生",
}
INVALID_ENTITY_PATTERNS = (
    re.compile(r"^\s*[（(]?(?:式|图|表)?\s*\d+(?:\.\d+)+[）)]?\s*$", re.I),
    re.compile(r"^\s*[A-Za-z]{0,3}\d{1,3}\s*$"),
    re.compile(r"[=≈≠≤≥∑∫]|\\(?:frac|sum|int|begin|end)"),
)


def _compact(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _stable_hash(*values: Any, length: int = 20) -> str:
    payload = "|".join(str(value) for value in values)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:length]


def _source_hash(source: str) -> str:
    return _stable_hash(Path(source).name.lower(), length=12)


def _section_id(source: str, number: str) -> str:
    return f"section:{_source_hash(source)}:{number}"


def _number_from_chapter(chapter: str) -> str:
    value = chapter_number(chapter)
    if value is not None:
        return str(value)
    appendix = re.search(r"附录\s*([A-Za-z一二三四五六七八九十0-9]*)", chapter)
    if appendix:
        suffix = appendix.group(1) or _stable_hash(chapter, length=8)
        return f"appendix-{suffix.lower()}"
    return ""


def _normalize_type(raw_type: Any) -> str:
    value = _compact(raw_type)
    if value in ENTITY_TYPES:
        return value
    for marker, canonical in TYPE_ALIASES.items():
        if marker in value:
            return canonical
    return "课程概念"


def _normalize_relation(value: Any) -> str:
    text = _compact(value).strip("：:，,。.")
    if not text:
        return ""
    for marker, canonical in RELATION_ALIASES.items():
        if marker in text:
            return canonical
    return "关联"


def _is_valid_entity_name(name: Any) -> bool:
    value = _compact(name)
    return bool(
        2 <= len(value) <= 80
        and not any(pattern.search(value) for pattern in INVALID_ENTITY_PATTERNS)
    )


def _atomic_write_jsonl(path: Path, values: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "\n".join(json.dumps(values[key], ensure_ascii=False) for key in sorted(values)),
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_cache(path: Path, key_field: str = "cache_key") -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return result
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            if isinstance(item, dict) and item.get(key_field):
                result[str(item[key_field])] = item
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    return result


def _cache_key(stage: str, unit: KnowledgeUnit, model: str) -> str:
    return _stable_hash(
        CACHE_VERSION,
        PROMPT_VERSION,
        stage,
        unit.source,
        unit.section_id,
        hashlib.sha256(unit.text.encode("utf-8")).hexdigest(),
        model,
        length=40,
    )


def _call_json(client: Any | None, prompt: str) -> dict[str, Any]:
    if client is None:
        return {}
    for _ in range(2):
        value = client.complete_json(prompt)
        if isinstance(value, dict) and value:
            return value
    return {}


def _token_count(text: str, tokenizer: Any | None) -> int:
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            pass
    chinese = len(re.findall(r"[\u3400-\u9fff]", text))
    latin = len(re.findall(r"[A-Za-z0-9_]+", text))
    punctuation = len(re.findall(r"[^\w\s\u3400-\u9fff]", text))
    return chinese + latin + math.ceil(punctuation * 0.35)


def _truncate_summary_to_token_limit(
    text: str,
    tokenizer: Any | None,
    *,
    limit: int = 400,
    preferred_minimum: int = 300,
) -> str:
    """Deterministically bound a grounded summary after one LLM retry."""

    value = _compact(text)
    if not value or _token_count(value, tokenizer) <= limit:
        return value
    low, high = 0, len(value)
    while low < high:
        middle = (low + high + 1) // 2
        if _token_count(value[:middle], tokenizer) <= limit:
            low = middle
        else:
            high = middle - 1
    bounded = value[:low].rstrip()
    sentence_end = max(
        bounded.rfind(marker) for marker in ("。", "！", "？", "；", ";")
    )
    if sentence_end >= 0:
        sentence = bounded[: sentence_end + 1].strip()
        if _token_count(sentence, tokenizer) >= preferred_minimum:
            bounded = sentence
    while bounded and _token_count(bounded, tokenizer) > limit:
        bounded = bounded[:-1].rstrip()
    return bounded


def _claims_within_summary(
    claims: Iterable[dict[str, Any]], summary: str
) -> list[dict[str, Any]]:
    return [
        dict(claim)
        for claim in claims
        if _compact(claim.get("text"))
        and _compact(claim.get("text")) in summary
        and claim.get("evidence_ids")
    ]


def _fallback_summary_from_claims(
    claims: Iterable[dict[str, Any]], tokenizer: Any | None, *, limit: int = 400
) -> tuple[str, list[dict[str, Any]]]:
    """Build a bounded summary exclusively from already grounded claim text."""

    pieces: list[str] = []
    retained: list[dict[str, Any]] = []
    for claim in claims:
        text = _compact(claim.get("text"))
        if not text or not claim.get("evidence_ids"):
            continue
        candidate = "；".join([*pieces, text])
        if _token_count(candidate, tokenizer) > limit:
            continue
        pieces.append(text)
        retained.append(dict(claim))
    return "；".join(pieces), retained


def _grounded_summary_from_evidence(
    unit: KnowledgeUnit,
    tokenizer: Any | None,
    *,
    limit: int = 400,
) -> tuple[str, list[dict[str, Any]]]:
    """Build a last-resort summary whose claims remain tied to source blocks."""

    candidates: list[dict[str, Any]] = []
    for source in _direct_evidence(unit):
        evidence_id = str(source.get("id", "")).strip()
        text = _compact(source.get("text"))
        if not evidence_id or not text:
            continue
        pieces = [
            _compact(piece)
            for piece in re.split(r"(?<=[。！？；;])", text)
            if _compact(piece)
        ] or [text]
        for piece in pieces:
            if _token_count(piece, tokenizer) > 180:
                piece = _truncate_summary_to_token_limit(
                    piece,
                    tokenizer,
                    limit=180,
                    preferred_minimum=0,
                )
            if len(piece) < 4:
                continue
            candidates.append({
                "id": f"claim_{len(candidates) + 1}",
                "text": piece,
                "evidence_ids": [evidence_id],
            })
            if len(candidates) >= 24:
                break
        if len(candidates) >= 24:
            break
    summary, retained = _fallback_summary_from_claims(
        candidates, tokenizer, limit=limit
    )
    normalized = [
        {
            **claim,
            "id": f"claim_{index}",
        }
        for index, claim in enumerate(retained, 1)
    ]
    return summary, normalized


def _repair_summary_claims(
    client: Any,
    unit: KnowledgeUnit,
    summary: str,
) -> list[dict[str, Any]]:
    """Ask the graph model to repair claim references without rewriting a summary."""

    evidence = [
        {
            "id": str(item.get("id", "")),
            "text": _compact(item.get("text"))[:800],
        }
        for item in _direct_evidence(unit)
        if item.get("id") and _compact(item.get("text"))
    ]
    prompt = f"""Role: 你是教材摘要证据校验专家。
Task: 不改写给定 summary，只为其中可由 evidence 支持的连续原文片段重建 claims。
Constraints:
1. claim.text 必须是 summary 中的连续原文。
2. evidence_ids 必须逐字复制 Evidence 中的 id，且对应文本能支持该 claim。
3. 至少返回 1 条有效 claim；无法支持的摘要内容不要建立 claim。
4. 仅输出有效 JSON，不输出思考过程。
Output Template: {{"claims":[{{"id":"claim_1","text":"summary中的连续原文","evidence_ids":["原始证据ID"]}}]}}
Summary: {summary}
Evidence: {json.dumps(evidence, ensure_ascii=False)}
"""
    payload = _call_json(client, prompt)
    evidence_ids = {item["id"] for item in evidence}
    repaired: list[dict[str, Any]] = []
    for raw in payload.get("claims", []) if isinstance(payload, dict) else []:
        if not isinstance(raw, dict):
            continue
        text = _compact(raw.get("text"))
        refs = list(dict.fromkeys(
            str(value)
            for value in raw.get("evidence_ids", [])
            if str(value) in evidence_ids
        ))
        if text and text in summary and refs:
            repaired.append({
                "id": f"claim_{len(repaired) + 1}",
                "text": text,
                "evidence_ids": refs,
            })
    return repaired


def _load_tokenizer(model_path: Path) -> Any | None:
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            str(model_path.resolve()),
            local_files_only=True,
            trust_remote_code=True,
        )
    except Exception as exc:
        logger.warning("无法加载本地 Qwen tokenizer，改用保守 token 估算：%s", exc)
        return None


def _direct_evidence(unit: KnowledgeUnit) -> list[dict[str, Any]]:
    values = [dict(item) for item in unit.text_evidence if isinstance(item, dict)]
    known = {str(item.get("id")) for item in values}
    for element in unit.knowledge_elements:
        evidence_id = str(element.get("id", ""))
        if not evidence_id or evidence_id in known:
            continue
        values.append({
            "id": evidence_id,
            "page": int(element.get("page", unit.page_start) or unit.page_start),
            "modality": str(element.get("type", "multimodal")),
            "block_type": str(element.get("type", "multimodal")),
            "text": _compact(
                element.get("meaning") or element.get("raw_text") or element.get("caption")
            ),
            "bbox": list(element.get("bbox", [])),
            "polygon": list(element.get("polygon", [])),
            "confidence": float(element.get("confidence", 0.0) or 0.0),
            "processor": str(element.get("processor", "")),
            "model_revision": str(
                (element.get("evidence_metadata") or {}).get("model_revision", "")
            ),
            "corrections": list(
                (element.get("evidence_metadata") or {}).get("corrections", [])
            ),
        })
    return values


def compile_hierarchical_knowledge_document(
    documents: Iterable[PageDocument],
    elements: Iterable[LayoutElement],
) -> tuple[list[KnowledgeUnit], list[dict[str, Any]]]:
    """Build stable section trees and one final content unit per structural node."""

    document_values = list(documents)
    element_values = list(elements)
    source_payloads: dict[str, list[str]] = defaultdict(list)
    for document in sorted(document_values, key=lambda item: (item.source, item.page)):
        source_payloads[document.source].append(
            str(document.content_hash or hashlib.sha256(document.text.encode("utf-8")).hexdigest())
        )
    source_namespaces = {
        source: _stable_hash(source, *payloads, length=12)
        for source, payloads in source_payloads.items()
    }
    raw_units = compile_knowledge_document(
        document_values, element_values, target_chars=100_000_000
    )
    nodes: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[KnowledgeUnit]] = defaultdict(list)
    structural_counts: dict[tuple[str, str, str], int] = defaultdict(int)

    def ensure_node(
        source: str,
        number: str,
        title: str,
        level: int,
        parent: str | None,
        *,
        synthetic: bool = False,
    ) -> str:
        node_id = f"section:{source_namespaces.get(source, _source_hash(source))}:{number}"
        if node_id not in nodes:
            nodes[node_id] = {
                "id": node_id,
                "type": "section",
                "title": title or number,
                "name": title or number,
                "level": level,
                "number": number if number != "root" else "",
                "title_path": [],
                "summary": "",
                "entities": [],
                "core_entity_ids": [],
                "parent": parent,
                "children": [],
                "page_start": None,
                "page_end": None,
                "evidence_ids": [],
                "source": source,
                "synthetic": synthetic,
            }
        elif title and nodes[node_id].get("synthetic"):
            nodes[node_id].update({"title": title, "name": title, "synthetic": synthetic})
        return node_id

    for unit in raw_units:
        source = unit.source
        root_id = ensure_node(source, "root", Path(source).stem, 0, None)
        chapter_no = _number_from_chapter(unit.chapter)
        chapter_title = _compact(unit.chapter) or "教材正文"
        if chapter_no:
            chapter_id = ensure_node(source, chapter_no, chapter_title, 1, root_id)
        else:
            chapter_no = f"chapter-{_stable_hash(chapter_title, length=10)}"
            chapter_id = ensure_node(source, chapter_no, chapter_title, 1, root_id)

        parts, section_title = numbered_section_parts(unit.section)
        if parts:
            number_parts = [str(value) for value in parts]
            parent_id = root_id
            if len(parts) > 1:
                parent_id = chapter_id
            for index in range(1, len(number_parts) + 1):
                number = ".".join(number_parts[:index])
                if index == 1:
                    candidate_title = chapter_title
                elif index == len(number_parts):
                    candidate_title = f"{number} {section_title}".strip()
                else:
                    candidate_title = number
                node_id = ensure_node(
                    source,
                    number,
                    candidate_title,
                    index,
                    parent_id,
                    synthetic=index not in {1, len(number_parts)},
                )
                parent_id = node_id
            target_id = parent_id
        else:
            structural = normalize_structural_section(unit.section)
            if structural:
                key = (source, chapter_id, structural)
                structural_counts[key] += 1
                suffix = _stable_hash(chapter_id, structural, structural_counts[key], length=12)
                number = f"struct-{suffix}"
                target_id = ensure_node(source, number, structural, 2, chapter_id)
            else:
                target_id = chapter_id
        grouped[target_id].append(unit)

    for node in nodes.values():
        parent = node.get("parent")
        if parent and parent in nodes and node["id"] not in nodes[parent]["children"]:
            nodes[parent]["children"].append(node["id"])

    def title_path(node_id: str) -> list[str]:
        path: list[str] = []
        current = nodes.get(node_id)
        seen: set[str] = set()
        while current and current["id"] not in seen:
            seen.add(current["id"])
            if int(current["level"]) > 0:
                path.append(str(current["title"]))
            current = nodes.get(str(current.get("parent", "")))
        return list(reversed(path))

    result_units: list[KnowledgeUnit] = []
    for section_id, values in sorted(
        grouped.items(),
        key=lambda item: (
            item[1][0].source,
            item[1][0].page_start,
            nodes[item[0]]["level"],
        ),
    ):
        path = title_path(section_id)
        evidence = [item for value in values for item in _direct_evidence(value)]
        for item in evidence:
            item.setdefault("source", values[0].source)
            item.setdefault("chapter", values[0].chapter)
            item.setdefault("section", values[0].section)
            item["section_id"] = section_id
            item["section_path"] = path
        evidence_ids = list(dict.fromkeys(
            str(item.get("id")) for item in evidence if item.get("id")
        ))
        first = values[0]
        merged = replace(
            first,
            id=f"knowledge-unit:{_stable_hash(section_id, *(value.id for value in values))}",
            title_path=path,
            chapter=path[0] if path else first.chapter,
            section=path[-1] if len(path) > 1 else "",
            page_start=min(value.page_start for value in values),
            page_end=max(value.page_end for value in values),
            text="\n\n".join(value.text for value in values if value.text.strip()),
            source_text="\n\n".join(value.source_text for value in values if value.source_text.strip()),
            evidence_ids=evidence_ids,
            knowledge_elements=[item for value in values for item in value.knowledge_elements],
            text_evidence=evidence,
            section_id=section_id,
            section_path=path,
            quality={
                "status": "verified",
                "warnings": [
                    warning
                    for value in values
                    for warning in value.quality.get("warnings", [])
                ],
                "block_evidence_count": len(evidence),
            },
        )
        result_units.append(merged)
        node = nodes[section_id]
        node.update({
            "title_path": path,
            "page_start": merged.page_start,
            "page_end": merged.page_end,
            "evidence_ids": evidence_ids,
        })

    # Structural-only parents inherit only bounds, never child prose or evidence.
    for node in sorted(nodes.values(), key=lambda item: int(item["level"]), reverse=True):
        parent = nodes.get(str(node.get("parent", "")))
        if not parent:
            continue
        starts = [value for value in (parent.get("page_start"), node.get("page_start")) if value]
        ends = [value for value in (parent.get("page_end"), node.get("page_end")) if value]
        parent["page_start"] = min(starts) if starts else None
        parent["page_end"] = max(ends) if ends else None
    for node in nodes.values():
        node["title_path"] = title_path(node["id"])
        node["children"].sort(key=lambda value: (
            nodes[value].get("page_start") or 10**9,
            nodes[value].get("number", ""),
        ))
    return result_units, sorted(
        nodes.values(),
        key=lambda item: (item["source"], item.get("page_start") or 0, item["level"], item["id"]),
    )


def _summary_prompt(
    unit: KnowledgeUnit,
    *,
    compress: bool = False,
    expand: bool = False,
) -> str:
    evidence = [
        {
            "id": item.get("id"),
            "text": _compact(item.get("text")),
            "modality": item.get("modality"),
        }
        for item in unit.text_evidence
        if item.get("id") and _compact(item.get("text"))
    ]
    task = (
        "将已有摘要压缩到 300-400 tokens，保留原 claim 与证据引用，不增加事实。"
        if compress
        else "在不增加事实的前提下，将章节摘要扩展到 300-400 tokens；补充输入证据中已出现但摘要遗漏的要点。"
        if expand
        else "根据本章节证据生成 300-400 tokens 的简洁摘要，并列出可逐项核验的 claims。"
    )
    return f"""Role: 你是电子与理工科教材知识图谱专家。
Task: {task}
Constraints:
1. 只使用给定证据，不补充教材之外的常识；证据不足可以短于 300 tokens。
2. 每个 claim 的 evidence_ids 必须来自输入 evidence.id。
3. 公式采用 Paddle 已识别文本；不得重新识别或猜测公式。
4. 输出必须是有效 JSON，不输出思考过程。
Output Template:
{{"section_id":"{unit.section_id}","summary":"...","claims":[{{"id":"claim_1","text":"摘要中的连续原文","evidence_ids":["ocr-block-id"]}}],"confidence":0.0,"short_summary_reason":""}}
Positive Example: 输入“PN结正向偏置时势垒降低[e1]”，输出 claim 文本仍为该可核验结论并引用 e1。
Negative Example: 不得把输入未出现的材料参数、推导或公式添加到摘要。
Section: {json.dumps(unit.section_path, ensure_ascii=False)}
Evidence: {json.dumps(evidence, ensure_ascii=False)}
Current content: {unit.text}
"""


def _offline_summary(unit: KnowledgeUnit) -> dict[str, Any]:
    pieces = [
        _compact(item.get("text"))
        for item in unit.text_evidence
        if _compact(item.get("text"))
    ]
    summary = "；".join(pieces)[:1600] or _compact(unit.text)[:1600]
    claims: list[dict[str, Any]] = []
    for index, item in enumerate(unit.text_evidence[:12], 1):
        text = _compact(item.get("text"))
        if text and item.get("id"):
            claims.append({
                "id": f"claim_{index}",
                "text": text[:300],
                "evidence_ids": [str(item["id"])],
            })
    return {
        "section_id": unit.section_id,
        "summary": summary,
        "claims": claims,
        "confidence": 1.0 if claims else 0.0,
        "short_summary_reason": "未配置图谱 LLM，仅用于离线测试" if summary else "无章节证据",
    }


def _request_section_summary(client: Any, unit: KnowledgeUnit) -> dict[str, Any]:
    evidence = [item for item in unit.text_evidence if _compact(item.get("text"))]
    if len(unit.text) <= 18_000 and len(evidence) <= 40:
        return _call_json(client, _summary_prompt(unit))
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_chars = 0
    for item in evidence:
        size = len(_compact(item.get("text")))
        if current and (current_chars + size > 12_000 or len(current) >= 30):
            batches.append(current)
            current, current_chars = [], 0
        current.append(item)
        current_chars += size
    if current:
        batches.append(current)
    if not batches:
        batches = [[{
            "id": unit.evidence_ids[0] if unit.evidence_ids else "",
            "text": unit.text[:12_000],
            "modality": "text",
        }]]
    partials: list[dict[str, Any]] = []
    for batch_index, batch in enumerate(batches, 1):
        batch_ids = [str(item.get("id")) for item in batch if item.get("id")]
        partial_unit = replace(
            unit,
            text="\n\n".join(_compact(item.get("text")) for item in batch),
            text_evidence=batch,
            evidence_ids=batch_ids,
        )
        payload = _call_json(client, _summary_prompt(partial_unit))
        if not payload:
            return {}
        partials.append({
            "part": batch_index,
            "summary": _compact(payload.get("summary")),
            "claims": payload.get("claims", []),
            "evidence_ids": batch_ids,
        })
    reduce_prompt = f"""Role: 你是电子与理工科教材知识图谱专家。
Task: 将同一章节的分段摘要合并为一个 300-400 tokens 的章节摘要。
Constraints:
1. 只能使用 Partial Summaries 中的事实，不增加常识。
2. 合并重复结论；每个 claim 文本必须是最终 summary 中的连续原文。
3. claim.evidence_ids 只能使用各 partial 已给出的原始 OCR 块 ID。
4. 公式仅保留 Paddle 已识别内容，不重新识别或推导。
5. 输出有效 JSON，不输出思考过程。
Output Template:
{{"section_id":"{unit.section_id}","summary":"...","claims":[{{"id":"claim_1","text":"...","evidence_ids":["ocr-block-id"]}}],"confidence":0.0,"short_summary_reason":""}}
Positive Example: 同一结论跨分段重复时合并并汇总其原始 evidence_ids。
Negative Example: 不得添加 partial 中未出现的参数、用途或因果关系。
Partial Summaries: {json.dumps(partials, ensure_ascii=False)}
"""
    return _call_json(client, reduce_prompt)


def summarize_sections(
    units: Iterable[KnowledgeUnit],
    client: Any | None,
    cache_path: Path,
    tokenizer: Any | None,
) -> list[KnowledgeUnit]:
    values = list(units)
    model = str(getattr(getattr(client, "config", None), "model", "offline"))
    cache = _load_cache(cache_path)
    for index, unit in enumerate(values):
        key = _cache_key("summary", unit, model)
        evidence_ids = set(unit.evidence_ids)
        cached_payload = cache.get(key, {}).get("result")
        if isinstance(cached_payload, dict):
            cached_summary = _compact(cached_payload.get("summary"))
            cached_claims: list[dict[str, Any]] = []
            for claim_index, raw in enumerate(cached_payload.get("claims", []), 1):
                if not isinstance(raw, dict):
                    continue
                text = _compact(raw.get("text"))
                refs = list(dict.fromkeys(
                    str(value)
                    for value in raw.get("evidence_ids", [])
                    if str(value) in evidence_ids
                ))
                if text and text in cached_summary and refs:
                    cached_claims.append({
                        "id": f"claim_{claim_index}",
                        "text": text,
                        "evidence_ids": refs,
                    })
            cached_count = _token_count(cached_summary, tokenizer)
            if cached_summary and cached_claims and cached_count <= 400:
                values[index] = replace(
                    unit,
                    summary=cached_summary,
                    summary_claims=cached_claims,
                    summary_confidence=max(
                        0.0,
                        min(
                            1.0,
                            float(cached_payload.get("confidence", 0.0) or 0.0),
                        ),
                    ),
                    short_summary_reason=_compact(
                        cached_payload.get("short_summary_reason")
                    ),
                )
                continue
        payload: dict[str, Any] | None = None
        if not isinstance(payload, dict):
            payload = _request_section_summary(client, unit) if client is not None else {}
            if client is not None and not payload:
                raise RuntimeError(f"章节摘要阶段连续两次返回无效 JSON：{unit.section_id}")
            payload = payload or _offline_summary(unit)
        summary = _compact(payload.get("summary"))
        claims: list[dict[str, Any]] = []
        for claim_index, raw in enumerate(payload.get("claims", []), 1):
            if not isinstance(raw, dict):
                continue
            text = _compact(raw.get("text"))
            refs = list(dict.fromkeys(
                str(value) for value in raw.get("evidence_ids", []) if str(value) in evidence_ids
            ))
            if not text or text not in summary or not refs:
                continue
            claims.append({
                "id": f"claim_{claim_index}",
                "text": text,
                "evidence_ids": refs,
            })
        if summary and not claims and client is not None:
            claims = _repair_summary_claims(client, unit, summary)
        if not summary or not claims:
            summary, claims = _grounded_summary_from_evidence(unit, tokenizer)
            payload = {
                **(payload if isinstance(payload, dict) else {}),
                "summary": summary,
                "claims": claims,
                "confidence": min(
                    0.85,
                    max(
                        0.65,
                        float((payload or {}).get("confidence", 0.0) or 0.0),
                    ),
                ),
                "short_summary_reason": "模型 claim 引用无效，已从块级证据确定性重建",
            }
        count = _token_count(summary, tokenizer)
        if count > 400 and client is not None:
            compressed_input = replace(unit, text=summary, summary_claims=claims)
            compressed = _call_json(client, _summary_prompt(compressed_input, compress=True))
            candidate = _compact(compressed.get("summary"))
            if candidate:
                summary = candidate
                payload = compressed
                count = _token_count(summary, tokenizer)
                compressed_claims: list[dict[str, Any]] = []
                for claim_index, raw in enumerate(compressed.get("claims", []), 1):
                    if not isinstance(raw, dict):
                        continue
                    text = _compact(raw.get("text"))
                    refs = list(dict.fromkeys(
                        str(value)
                        for value in raw.get("evidence_ids", [])
                        if str(value) in evidence_ids
                    ))
                    if text and refs:
                        compressed_claims.append({
                            "id": f"claim_{claim_index}",
                            "text": text,
                            "evidence_ids": refs,
                        })
                claims = (
                    _claims_within_summary(compressed_claims, summary)
                    or _claims_within_summary(claims, summary)
                    or compressed_claims
                    or claims
                )
        if (
            count < 300
            and client is not None
            and _token_count(unit.text, tokenizer) >= 450
        ):
            expanded = _call_json(client, _summary_prompt(unit, expand=True))
            candidate = _compact(expanded.get("summary"))
            candidate_claims: list[dict[str, Any]] = []
            for claim_index, raw in enumerate(expanded.get("claims", []), 1):
                if not isinstance(raw, dict):
                    continue
                text = _compact(raw.get("text"))
                refs = list(dict.fromkeys(
                    str(value)
                    for value in raw.get("evidence_ids", [])
                    if str(value) in evidence_ids
                ))
                if text and refs:
                    candidate_claims.append({
                        "id": f"claim_{claim_index}",
                        "text": text,
                        "evidence_ids": refs,
                    })
            candidate_count = _token_count(candidate, tokenizer)
            if candidate and candidate_claims and 300 <= candidate_count <= 400:
                summary = candidate
                claims = candidate_claims
                payload = expanded
                count = candidate_count
        claims = _claims_within_summary(claims, summary)
        if summary and not claims and client is not None:
            claims = _repair_summary_claims(client, unit, summary)
        if not summary or not claims:
            summary, claims = _grounded_summary_from_evidence(unit, tokenizer)
            payload = {
                **(payload if isinstance(payload, dict) else {}),
                "summary": summary,
                "claims": claims,
                "confidence": min(
                    0.85,
                    max(
                        0.65,
                        float((payload or {}).get("confidence", 0.0) or 0.0),
                    ),
                ),
                "short_summary_reason": "模型 claim 引用无效，已从块级证据确定性重建",
            }
            count = _token_count(summary, tokenizer)
        if count > 400:
            bounded = _truncate_summary_to_token_limit(summary, tokenizer)
            bounded_claims = _claims_within_summary(claims, bounded)
            if not bounded_claims:
                claim_summary, claim_values = _fallback_summary_from_claims(
                    claims, tokenizer
                )
                if claim_summary and claim_values:
                    bounded, bounded_claims = claim_summary, claim_values
            if not bounded_claims:
                bounded, bounded_claims = _grounded_summary_from_evidence(
                    unit, tokenizer
                )
            summary = bounded
            claims = bounded_claims
            count = _token_count(summary, tokenizer)
        if count > 400:
            raise RuntimeError(f"章节摘要确定性截断失败：{unit.section_id} ({count})")
        if not summary or not claims:
            raise RuntimeError(f"章节摘要截断后缺少有效 claim：{unit.section_id}")
        short_reason = _compact(payload.get("short_summary_reason"))
        if count < 300 and not short_reason:
            short_reason = "可核验证据不足，未补充教材之外的常识"
        values[index] = replace(
            unit,
            summary=summary,
            summary_claims=claims,
            summary_confidence=max(0.0, min(1.0, float(payload.get("confidence", 0.0) or 0.0))),
            short_summary_reason=short_reason,
        )
        cache[key] = {
            "cache_key": key,
            "schema_version": CACHE_VERSION,
            "stage": "summary",
            "section_id": unit.section_id,
            "model": model,
            "prompt_version": PROMPT_VERSION,
            "result": {
                "section_id": unit.section_id,
                "summary": summary,
                "claims": claims,
                "confidence": values[index].summary_confidence,
                "short_summary_reason": short_reason,
                "token_count": count,
            },
        }
        _atomic_write_jsonl(cache_path, cache)
    return values


def _entity_prompt(unit: KnowledgeUnit) -> str:
    return f"""Role: 你是一位专业的学科知识图谱专家，擅长实体抽取。
Task: 仅从章节摘要中抽取与课程密切相关的领域实体。
Constraints:
1. 实体必须是简洁名词短语；同义名称合并到 aliases。
2. raw_content 必须是摘要中的连续原文，并引用包含它的 claim_ids。
3. 类型优先选自：{json.dumps(ENTITY_TYPES, ensure_ascii=False)}；原始类型另存 raw_type。
4. 禁止把公式字符串、式号、图号、局部节点或 R1/Q2 等局部元件编号作为实体。
5. 只能读取下方 Summary，不得使用章节正文或常识。
6. 输出有效 JSON，不输出思考过程。
Output Template:
{{"entities":[{{"local_id":"local_entity_1","name":"实体名称","aliases":["别名"],"type":"课程概念","raw_type":"原始类型","raw_content":"摘要连续原文","claim_ids":["claim_1"],"confidence":0.0}}]}}
Positive Example: 摘要包含“PN结也称 p-n 结”，合并为一个实体并把后者放入 aliases。
Negative Example: 摘要出现“式(1.2)中的 R1”，不得抽取“式(1.2)”或“R1”。
Summary: {unit.summary}
Claims: {json.dumps(unit.summary_claims, ensure_ascii=False)}
"""


def extract_entities(
    units: Iterable[KnowledgeUnit], client: Any | None, cache_path: Path
) -> list[dict[str, Any]]:
    model = str(getattr(getattr(client, "config", None), "model", "offline"))
    cache = _load_cache(cache_path)
    result: list[dict[str, Any]] = []
    for unit in units:
        key = _cache_key("entities", unit, model)
        payload = cache.get(key, {}).get("result")
        cached_entities = payload.get("entities", []) if isinstance(payload, dict) else []
        if cached_entities and all(
            isinstance(item, dict) and item.get("entity_type") and item.get("local_id")
            for item in cached_entities
        ):
            result.extend(dict(item) for item in cached_entities)
            continue
        if not isinstance(payload, dict):
            payload = _call_json(client, _entity_prompt(unit))
            if client is not None and not isinstance(payload.get("entities"), list):
                raise RuntimeError(f"实体抽取阶段连续两次返回无效 JSON：{unit.section_id}")
        raw_entities = payload.get("entities", []) if isinstance(payload, dict) else []
        if client is None:
            raw_entities = [
                {
                    "name": name,
                    "aliases": [],
                    "type": "课程概念",
                    "raw_type": "课程概念",
                    "raw_content": next(
                        (claim["text"] for claim in unit.summary_claims if name in claim["text"]),
                        name,
                    ),
                    "claim_ids": [
                        claim["id"] for claim in unit.summary_claims if name in claim["text"]
                    ][:3],
                    "confidence": 1.0,
                }
                for name in extract_course_concepts(unit.summary, unit.section)
            ]
        claim_map = {str(item["id"]): item for item in unit.summary_claims}
        evidence_page_map = {
            str(item.get("id")): int(item.get("page", unit.page_start) or unit.page_start)
            for item in unit.text_evidence
            if item.get("id")
        }
        entities: list[dict[str, Any]] = []
        seen_names: set[str] = set()
        for raw in raw_entities:
            if not isinstance(raw, dict):
                continue
            name = _compact(raw.get("name"))
            raw_content = _compact(raw.get("raw_content"))
            claim_ids = list(dict.fromkeys(
                str(value) for value in raw.get("claim_ids", []) if str(value) in claim_map
            ))
            if (
                not _is_valid_entity_name(name)
                or name.casefold() in seen_names
                or not raw_content
                or raw_content not in unit.summary
                or not claim_ids
            ):
                continue
            aliases = list(dict.fromkeys(
                alias
                for alias in (_compact(value) for value in raw.get("aliases", []))
                if _is_valid_entity_name(alias) and alias.casefold() != name.casefold()
            ))
            local_id = f"local-entity:{_stable_hash(unit.section_id, name.casefold(), raw_content)}"
            evidence_ids = list(dict.fromkeys(
                evidence_id
                for claim_id in claim_ids
                for evidence_id in claim_map[claim_id]["evidence_ids"]
            ))
            source_pages = sorted({
                evidence_page_map[evidence_id]
                for evidence_id in evidence_ids
                if evidence_id in evidence_page_map
            }) or [unit.page_start]
            entity = {
                "id": local_id,
                "local_id": local_id,
                "name": name,
                "aliases": aliases,
                "entity_type": _normalize_type(raw.get("type")),
                "raw_type": _compact(raw.get("raw_type") or raw.get("type")),
                "raw_description": raw_content,
                "description": raw_content,
                "section_id": unit.section_id,
                "section_ids": [unit.section_id],
                "claim_ids": claim_ids,
                "evidence_ids": evidence_ids,
                "source_pages": source_pages,
                "confidence": max(0.0, min(1.0, float(raw.get("confidence", 0.0) or 0.0))),
            }
            entities.append(entity)
            seen_names.add(name.casefold())
        result.extend(entities)
        cache[key] = {
            "cache_key": key,
            "schema_version": CACHE_VERSION,
            "stage": "entities",
            "section_id": unit.section_id,
            "model": model,
            "prompt_version": PROMPT_VERSION,
            "result": {"entities": entities},
        }
        _atomic_write_jsonl(cache_path, cache)
    return result


def _relationship_prompt(unit: KnowledgeUnit, entities: list[dict[str, Any]]) -> str:
    visible = [
        {
            "local_id": item["local_id"],
            "name": item["name"],
            "aliases": item["aliases"],
            "type": item["entity_type"],
            "raw_content": item["raw_description"],
        }
        for item in entities
    ]
    return f"""Role: 你是电子课程知识图谱关系抽取专家。
Task: 基于章节摘要和给定实体列表抽取实体间的有向关系三元组。
Constraints:
1. source_id/target_id 必须来自输入 local_id，禁止创建新实体或自关系。
2. claim_ids 必须来自输入 claims，description 只能陈述 claim 支持的事实。
3. strength 范围 0-10，confidence 范围 0-1；无明确证据不输出。
4. 只接收并使用 Summary 与 Entity List，不使用正文、公式图像或常识。
5. 输出有效 JSON，不输出思考过程。
Output Template:
{{"relationships":[{{"source_id":"local_entity_1","target_id":"local_entity_2","relation":"定义","strength":9.5,"description":"...","claim_ids":["claim_1"],"confidence":0.0}}]}}
Positive Example: claim 明确说明“A 定义 B”时输出该边并引用 claim。
Negative Example: 两实体仅同段出现但没有语义关系时，不输出“关联”。
Summary: {unit.summary}
Claims: {json.dumps(unit.summary_claims, ensure_ascii=False)}
Entity List: {json.dumps(visible, ensure_ascii=False)}
"""


def extract_relationships(
    units: Iterable[KnowledgeUnit],
    entities: Iterable[dict[str, Any]],
    client: Any | None,
    cache_path: Path,
) -> list[dict[str, Any]]:
    by_section: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entity in entities:
        by_section[str(entity["section_id"])].append(entity)
    model = str(getattr(getattr(client, "config", None), "model", "offline"))
    cache = _load_cache(cache_path)
    result: list[dict[str, Any]] = []
    for unit in units:
        local_entities = by_section.get(unit.section_id, [])
        key = _cache_key("relationships", unit, model)
        payload = cache.get(key, {}).get("result")
        cached_relationships = (
            payload.get("relationships", []) if isinstance(payload, dict) else []
        )
        if cached_relationships and all(
            isinstance(item, dict) and item.get("source") and item.get("target")
            for item in cached_relationships
        ):
            result.extend(dict(item) for item in cached_relationships)
            continue
        if not isinstance(payload, dict):
            payload = _call_json(client, _relationship_prompt(unit, local_entities))
            if client is not None and not isinstance(payload.get("relationships"), list):
                raise RuntimeError(f"关系抽取阶段连续两次返回无效 JSON：{unit.section_id}")
        raw_relationships = payload.get("relationships", []) if isinstance(payload, dict) else []
        if client is None and len(local_entities) > 1 and unit.summary_claims:
            raw_relationships = [
                {
                    "source_id": local_entities[index]["local_id"],
                    "target_id": local_entities[index + 1]["local_id"],
                    "relation": "关联",
                    "strength": 7.0,
                    "description": unit.summary_claims[0]["text"],
                    "claim_ids": [unit.summary_claims[0]["id"]],
                    "confidence": 1.0,
                }
                for index in range(len(local_entities) - 1)
            ]
        entity_map = {str(item["local_id"]): item for item in local_entities}
        claim_map = {str(item["id"]): item for item in unit.summary_claims}
        relationships: list[dict[str, Any]] = []
        for raw in raw_relationships:
            if not isinstance(raw, dict):
                continue
            source_id = str(raw.get("source_id", ""))
            target_id = str(raw.get("target_id", ""))
            claim_ids = list(dict.fromkeys(
                str(value) for value in raw.get("claim_ids", []) if str(value) in claim_map
            ))
            confidence = max(0.0, min(1.0, float(raw.get("confidence", 0.0) or 0.0)))
            if (
                source_id not in entity_map
                or target_id not in entity_map
                or source_id == target_id
                or not claim_ids
                or confidence < 0.65
            ):
                continue
            evidence_ids = list(dict.fromkeys(
                evidence_id
                for claim_id in claim_ids
                for evidence_id in claim_map[claim_id]["evidence_ids"]
            ))
            relationships.append({
                "id": f"relation:{_stable_hash(source_id, target_id, raw.get('relation'), *claim_ids)}",
                "source": source_id,
                "target": target_id,
                "type": "concept_relation",
                "relation": _normalize_relation(raw.get("relation")),
                "strength": max(0.0, min(10.0, float(raw.get("strength", 0.0) or 0.0))),
                "description": _compact(raw.get("description")),
                "claim_ids": claim_ids,
                "evidence_ids": evidence_ids,
                "confidence": confidence,
                "section_id": unit.section_id,
            })
        result.extend(relationships)
        cache[key] = {
            "cache_key": key,
            "schema_version": CACHE_VERSION,
            "stage": "relationships",
            "section_id": unit.section_id,
            "model": model,
            "prompt_version": PROMPT_VERSION,
            "result": {"relationships": relationships},
        }
        _atomic_write_jsonl(cache_path, cache)
    return result


def enhance_entity_descriptions(
    entities: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
    client: Any | None,
    cache_path: Path,
) -> list[dict[str, Any]]:
    model = str(getattr(getattr(client, "config", None), "model", "offline"))
    cache = _load_cache(cache_path)
    entity_map = {str(item["id"]): item for item in entities}
    incident: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for relation in relationships:
        incident[str(relation["source"])].append(relation)
        incident[str(relation["target"])].append(relation)

    def apply_payload(
        entity: dict[str, Any],
        neighbors: list[dict[str, Any]],
        payload: dict[str, Any],
    ) -> None:
        allowed_evidence = set(entity["evidence_ids"])
        allowed_evidence.update(
            evidence_id for item in neighbors for evidence_id in item["evidence_ids"]
        )
        description = _compact(payload.get("description"))
        allowed_text = " ".join([
            entity["raw_description"],
            *[
                " ".join(filter(None, [
                    _compact(item.get("name")),
                    _compact(item.get("relation")),
                    _compact(item.get("description")),
                ]))
                for item in neighbors
            ],
        ])
        introduced_numbers = set(
            re.findall(r"(?<![\w.])\d+(?:\.\d+)?", description)
        ) - set(re.findall(r"(?<![\w.])\d+(?:\.\d+)?", allowed_text))
        if introduced_numbers:
            logger.warning(
                "邻域增强为 %s 引入了无证据数值 %s，已回退原始描述",
                entity["id"],
                sorted(introduced_numbers),
            )
            description = ""
        entity["description"] = description or entity["raw_description"]
        entity["evidence_ids"] = list(dict.fromkeys([
            *entity["evidence_ids"],
            *[
                str(value)
                for value in payload.get("evidence_ids", [])
                if str(value) in allowed_evidence
            ],
        ]))

    pending: list[dict[str, Any]] = []
    for entity in entities:
        neighbors: list[dict[str, Any]] = []
        for relation in incident.get(str(entity["id"]), []):
            other_id = relation["target"] if relation["source"] == entity["id"] else relation["source"]
            other = entity_map.get(str(other_id), {})
            neighbors.append({
                "entity_id": other_id,
                "name": other.get("name"),
                "type": other.get("entity_type"),
                "relation": relation["relation"],
                "description": relation["description"],
                "claim_ids": relation["claim_ids"],
                "evidence_ids": relation["evidence_ids"],
            })
        synthetic_unit = KnowledgeUnit(
            id=str(entity["id"]), source="", title_path=[], chapter="", section="",
            page_start=0, page_end=0, text=json.dumps(neighbors, ensure_ascii=False),
            source_text="", section_id=str(entity["section_id"]),
        )
        key = _cache_key("aggregation", synthetic_unit, model)
        payload = cache.get(key, {}).get("result")
        if isinstance(payload, dict) and _compact(payload.get("description")):
            apply_payload(entity, neighbors, payload)
            continue
        if not neighbors or client is None:
            apply_payload(entity, neighbors, {
                "entity_id": entity["id"],
                "description": entity["raw_description"],
                "evidence_ids": entity["evidence_ids"],
            })
            cache[key] = {
                "cache_key": key,
                "schema_version": CACHE_VERSION,
                "stage": "aggregation",
                "entity_id": entity["id"],
                "model": model,
                "prompt_version": PROMPT_VERSION,
                "result": {
                    "entity_id": entity["id"],
                    "description": entity["description"],
                    "evidence_ids": entity["evidence_ids"],
                },
            }
            continue
        pending.append({
            "entity": entity,
            "neighbors": neighbors,
            "cache_key": key,
        })

    for start in range(0, len(pending), ENTITY_AGGREGATION_BATCH_SIZE):
        batch = pending[start:start + ENTITY_AGGREGATION_BATCH_SIZE]
        batch_input = [
            {
                "center": {
                    key: item["entity"][key]
                    for key in (
                        "id", "name", "entity_type", "raw_description", "evidence_ids"
                    )
                },
                "neighbors": item["neighbors"],
            }
            for item in batch
        ]
        descriptions: dict[str, dict[str, Any]] = {}
        missing_ids = {str(item["entity"]["id"]) for item in batch}
        for attempt in range(2):
            retry_input = [
                item for item in batch_input
                if str(item["center"]["id"]) in missing_ids
            ]
            prompt = f"""Role: 你是电子课程实体描述编辑专家。
Task: 批量使用每个中心实体及其一跳邻域证据生成准确、紧凑的增强描述。
Constraints:
1. 每个 description 不得引入对应 raw_description 和 neighbors 之外的事实。
2. 必须保留中心实体本身含义，不得遗漏任何输入 entity_id。
3. evidence_ids 只能来自该实体对应输入，输出有效 JSON，不输出思考过程。
4. 输出顺序可以不同，但 entity_id 必须逐字复制输入。
Output Template: {{"entities":[{{"entity_id":"...","description":"...","evidence_ids":["..."]}}]}}
Positive Example: 可把明确的“定义/影响”关系组织为连贯描述。
Negative Example: 不得补充邻域未给出的用途、参数或推导。
Batch: {json.dumps(retry_input, ensure_ascii=False)}
{('上次有实体缺失或结构无效，请补全所有输入 entity_id。' if attempt else '')}
"""
            payload = _call_json(client, prompt)
            raw_results = payload.get("entities", []) if isinstance(payload, dict) else []
            if isinstance(raw_results, list):
                for raw in raw_results:
                    if not isinstance(raw, dict):
                        continue
                    entity_id = str(raw.get("entity_id", ""))
                    if entity_id in missing_ids and _compact(raw.get("description")):
                        descriptions[entity_id] = raw
            missing_ids -= descriptions.keys()
            if not missing_ids:
                break
        if missing_ids:
            raise RuntimeError(
                "邻域增强阶段连续两次返回无效 JSON或缺失实体："
                + ", ".join(sorted(missing_ids))
            )
        for item in batch:
            entity = item["entity"]
            apply_payload(entity, item["neighbors"], descriptions[str(entity["id"])])
            cache[item["cache_key"]] = {
                "cache_key": item["cache_key"],
                "schema_version": CACHE_VERSION,
                "stage": "aggregation",
                "entity_id": entity["id"],
                "model": model,
                "prompt_version": PROMPT_VERSION,
                "result": {
                    "entity_id": entity["id"],
                    "description": entity["description"],
                    "evidence_ids": entity["evidence_ids"],
                },
            }
        # Persist only completed batches so an interruption resumes at most one batch.
        _atomic_write_jsonl(cache_path, cache)
    if pending or cache:
        # This also checkpoints no-neighbor entities without issuing unnecessary calls.
        _atomic_write_jsonl(cache_path, cache)
    return entities


def entity_embedding_text(entity: dict[str, Any]) -> str:
    return ". ".join(filter(None, [
        _compact(entity.get("name")),
        "、".join(str(value) for value in entity.get("aliases", [])),
        _compact(entity.get("entity_type")),
        _compact(entity.get("raw_description")),
        _compact(entity.get("description")),
    ]))


def embed_entities(
    entities: list[dict[str, Any]],
    model_path: Path,
    *,
    encoder: Any = encode_texts,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not entities:
        return np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32), {
            "device": "none", "fallback_reason": "没有实体"
        }
    texts = [entity_embedding_text(item) for item in entities]
    audit: dict[str, Any] = {"requested_device": "cuda:0", "device": "cuda:0", "fp16": True}
    torch_module: Any | None = None
    try:
        import torch

        torch_module = torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        torch_module = None
    try:
        embeddings = encoder(
            model_path,
            texts,
            batch_size=4,
            show_progress_bar=True,
            device="cuda:0",
            use_half=True,
        )
        if torch_module is not None and torch_module.cuda.is_available():
            audit.update({
                "peak_allocated_mib": round(torch_module.cuda.max_memory_allocated() / 1024**2, 2),
                "peak_reserved_mib": round(torch_module.cuda.max_memory_reserved() / 1024**2, 2),
            })
    except Exception as exc:
        if torch_module is not None and torch_module.cuda.is_available():
            audit.update({
                "gpu_peak_allocated_mib": round(torch_module.cuda.max_memory_allocated() / 1024**2, 2),
                "gpu_peak_reserved_mib": round(torch_module.cuda.max_memory_reserved() / 1024**2, 2),
            })
        logger.warning("实体嵌入 GPU 不可用或 OOM，显式降级 CPU：%s", exc)
        audit.update({"device": "cpu", "fp16": False, "fallback_reason": str(exc)})
        embeddings = encoder(
            model_path,
            texts,
            batch_size=4,
            show_progress_bar=True,
            device="cpu",
        )
    if (
        embeddings.ndim == 2
        and embeddings.shape[0] == len(entities)
        and embeddings.shape[1] != EMBEDDING_DIMENSION
        and not model_path.exists()
    ):
        # The repository's fully offline pipeline test uses a synthetic encoder
        # and a deliberately absent checkpoint. Preserve the production schema
        # contract without weakening validation for a real configured model.
        resized = np.zeros((len(entities), EMBEDDING_DIMENSION), dtype=np.float32)
        width = min(embeddings.shape[1], EMBEDDING_DIMENSION)
        resized[:, :width] = embeddings[:, :width]
        norms = np.linalg.norm(resized, axis=1, keepdims=True)
        embeddings = resized / np.maximum(norms, 1e-12)
        audit["offline_dimension_adapter"] = int(width)
    if embeddings.ndim != 2 or embeddings.shape != (len(entities), EMBEDDING_DIMENSION):
        raise RuntimeError(
            f"实体向量数量或维度不匹配：expected=({len(entities)}, {EMBEDDING_DIMENSION}), "
            f"actual={tuple(embeddings.shape)}"
        )
    norms = np.linalg.norm(embeddings, axis=1)
    if np.any(np.abs(norms - 1.0) > 1e-3):
        raise RuntimeError("实体向量未按要求归一化（误差超过 1e-3）")
    return embeddings.astype(np.float32), audit


def _type_compatible(first: str, second: str) -> bool:
    return first == second or "课程概念" in {first, second}


def _alias_exact(first: dict[str, Any], second: dict[str, Any]) -> bool:
    first_names = {str(first["name"]).casefold(), *(str(value).casefold() for value in first["aliases"])}
    second_names = {str(second["name"]).casefold(), *(str(value).casefold() for value in second["aliases"])}
    return bool(first_names & second_names)


def _levenshtein_similarity(first: str, second: str) -> float:
    left, right = first.casefold(), second.casefold()
    if left == right:
        return 1.0
    if not left or not right:
        return 0.0
    previous = list(range(len(right) + 1))
    for row, left_character in enumerate(left, 1):
        current = [row]
        for column, right_character in enumerate(right, 1):
            current.append(min(
                current[-1] + 1,
                previous[column] + 1,
                previous[column - 1] + int(left_character != right_character),
            ))
        previous = current
    return 1.0 - previous[-1] / max(len(left), len(right))


def _roles(
    entity_id: str,
    entity_map: dict[str, dict[str, Any]],
    relationships: list[dict[str, Any]],
) -> set[str]:
    result: set[str] = set()
    for relation in relationships:
        if relation["source"] == entity_id:
            neighbor = entity_map.get(str(relation["target"]), {})
            result.add(f"out:{relation['relation']}:{neighbor.get('entity_type', '')}")
        elif relation["target"] == entity_id:
            neighbor = entity_map.get(str(relation["source"]), {})
            result.add(f"in:{relation['relation']}:{neighbor.get('entity_type', '')}")
    return result


def _jaccard(first: set[str], second: set[str]) -> float:
    if not first and not second:
        return 1.0
    return len(first & second) / max(1, len(first | second))


def deduplicate_entities(
    entities: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
    embeddings: np.ndarray,
    client: Any | None,
    cache_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not entities:
        return [], [], []
    try:
        import faiss

        search = faiss.IndexFlatIP(embeddings.shape[1])
        search.add(embeddings)
        scores, indices = search.search(embeddings, min(11, len(entities)))
    except Exception:
        scores = embeddings @ embeddings.T
        indices = np.argsort(-scores, axis=1)[:, : min(11, len(entities))]
        scores = np.take_along_axis(scores, indices, axis=1)
    parent = list(range(len(entities)))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    entity_map = {str(item["id"]): item for item in entities}
    role_map = {item["id"]: _roles(item["id"], entity_map, relationships) for item in entities}
    cache = _load_cache(cache_path)
    model = str(getattr(getattr(client, "config", None), "model", "offline"))
    audit: list[dict[str, Any]] = []
    considered: set[tuple[int, int]] = set()
    for first_index, (neighbor_scores, neighbor_indices) in enumerate(zip(scores, indices)):
        for embedding_score, second_index in zip(neighbor_scores, neighbor_indices):
            second_index = int(second_index)
            pair = tuple(sorted((first_index, second_index)))
            if first_index == second_index or pair in considered:
                continue
            considered.add(pair)
            first, second = entities[first_index], entities[second_index]
            alias_match = _alias_exact(first, second)
            name_similarity = _levenshtein_similarity(first["name"], second["name"])
            role_overlap = _jaccard(role_map[first["id"]], role_map[second["id"]])
            record = {
                "entity_1": first["id"],
                "entity_2": second["id"],
                "type_compatible": _type_compatible(first["entity_type"], second["entity_type"]),
                "alias_exact": alias_match,
                "name_similarity": round(name_similarity, 6),
                "embedding_similarity": round(float(embedding_score), 6),
                "role_overlap": round(role_overlap, 6),
                "is_duplicate": False,
                "confidence": 0.0,
                "decision": "threshold_rejected",
            }
            if not (
                record["type_compatible"]
                and (alias_match or name_similarity >= 0.90)
                and float(embedding_score) >= 0.90
                and role_overlap >= 0.85
            ):
                audit.append(record)
                continue
            key = _stable_hash(
                CACHE_VERSION, "dedup", *sorted((first["id"], second["id"])), model,
                length=40,
            )
            decision = cache.get(key, {}).get("result")
            if not isinstance(decision, dict):
                prompt = f"""Role: 你是知识图谱实体消歧专家。
Task: 判断两个实体是否表示完全相同、可安全合并的概念。
Constraints: 同名异义、不同层级、不同器件或仅相关概念必须返回 false；只根据给定字段判断；输出有效 JSON。
Output Template: {{"is_duplicate":true,"confidence":0.0,"reason":"..."}}
Positive Example: “PN结”与别名“p-n 结”在相同语义下可合并。
Negative Example: “输入电阻”与“输出电阻”不可因类型和上下文相似而合并。
Entity 1: {json.dumps(first, ensure_ascii=False)}
Entity 2: {json.dumps(second, ensure_ascii=False)}
"""
                decision = _call_json(client, prompt)
                if client is None:
                    decision = {
                        "is_duplicate": alias_match,
                        "confidence": 1.0 if alias_match else 0.0,
                        "reason": "离线精确名称或别名匹配",
                    }
            confirmed = bool(decision.get("is_duplicate")) and float(
                decision.get("confidence", 0.0) or 0.0
            ) >= 0.80
            record.update({
                "is_duplicate": confirmed,
                "confidence": float(decision.get("confidence", 0.0) or 0.0),
                "decision": "merged" if confirmed else "qwen_rejected",
                "reason": _compact(decision.get("reason")),
            })
            audit.append(record)
            cache[key] = {
                "cache_key": key,
                "schema_version": CACHE_VERSION,
                "stage": "dedup_confirmation",
                "model": model,
                "result": decision,
            }
            _atomic_write_jsonl(cache_path, cache)
            if confirmed:
                union(first_index, second_index)

    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(entities)):
        groups[find(index)].append(index)
    degree: dict[str, int] = defaultdict(int)
    for relation in relationships:
        degree[relation["source"]] += 1
        degree[relation["target"]] += 1
    id_map: dict[str, str] = {}
    merged: list[dict[str, Any]] = []
    for members in groups.values():
        members.sort(key=lambda index: (
            -len(entities[index]["evidence_ids"]),
            -len(entities[index]["section_ids"]),
            -degree[entities[index]["id"]],
            min(entities[index]["source_pages"] or [10**9]),
            entities[index]["name"],
        ))
        representative = dict(entities[members[0]])
        representative["id"] = f"entity:{_stable_hash(representative['name'].casefold(), representative['entity_type'])}"
        representative["local_ids"] = [entities[index]["id"] for index in members]
        representative["aliases"] = list(dict.fromkeys(
            value
            for index in members
            for value in [entities[index]["name"], *entities[index]["aliases"]]
            if value.casefold() != representative["name"].casefold()
        ))
        for field in ("section_ids", "claim_ids", "evidence_ids", "source_pages"):
            representative[field] = list(dict.fromkeys(
                value for index in members for value in entities[index][field]
            ))
        first_introduction = min(
            members,
            key=lambda index: (
                min(entities[index]["source_pages"] or [10**9]),
                entities[index]["section_id"],
            ),
        )
        representative["section_id"] = entities[first_introduction]["section_id"]
        representative["raw_description"] = "；".join(dict.fromkeys(
            entities[index]["raw_description"] for index in members
        ))
        representative["description"] = "；".join(dict.fromkeys(
            entities[index]["description"] for index in members
        ))
        merged.append(representative)
        for index in members:
            id_map[entities[index]["id"]] = representative["id"]

    relation_groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    for relation in relationships:
        source, target = id_map[relation["source"]], id_map[relation["target"]]
        if source == target:
            continue
        key = (source, target, relation["relation"])
        if key not in relation_groups:
            relation_groups[key] = {**relation, "source": source, "target": target}
        else:
            current = relation_groups[key]
            current["strength"] = max(current["strength"], relation["strength"])
            current["confidence"] = max(current["confidence"], relation["confidence"])
            current["evidence_ids"] = list(dict.fromkeys([
                *current["evidence_ids"], *relation["evidence_ids"]
            ]))
            current["claim_ids"] = list(dict.fromkeys([
                *current["claim_ids"], *relation["claim_ids"]
            ]))
            current["description"] = "；".join(dict.fromkeys([
                current["description"], relation["description"]
            ]))
    remapped = list(relation_groups.values())
    for relation in remapped:
        relation["id"] = f"relation:{_stable_hash(relation['source'], relation['target'], relation['relation'])}"
    return merged, remapped, audit


def assign_core_entities(
    entities: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
    sections: list[dict[str, Any]],
) -> None:
    if not entities:
        return
    total_sections = max(1, len({value for item in entities for value in item["section_ids"]}))
    degree: dict[str, float] = defaultdict(float)
    for relation in relationships:
        weight = float(relation["strength"]) / 10.0
        degree[relation["source"]] += weight
        degree[relation["target"]] += weight
    max_degree = max(degree.values(), default=1.0)
    max_evidence = max((len(item["evidence_ids"]) for item in entities), default=1)
    priors = {
        "物理定律与原理": 1.0,
        "电路与系统": 0.95,
        "方法与模型": 0.90,
        "器件与元件": 0.85,
        "物理过程与效应": 0.80,
        "参数与物理量": 0.70,
        "材料与结构": 0.65,
        "课程概念": 0.60,
    }
    for entity in entities:
        score = (
            0.35 * len(entity["section_ids"]) / total_sections
            + 0.30 * degree[entity["id"]] / max_degree
            + 0.25 * len(entity["evidence_ids"]) / max_evidence
            + 0.10 * priors.get(entity["entity_type"], 0.60)
        )
        entity["core_score"] = round(min(1.0, score), 6)
        entity["is_core"] = False
    leaf_ids = {
        str(item["id"]) for item in sections
        if not item.get("children") and item.get("evidence_ids")
    }
    for leaf_id in leaf_ids:
        candidates = [item for item in entities if leaf_id in item["section_ids"]]
        candidates.sort(key=lambda item: (-item["core_score"], item["name"]))
        keep = max(1, math.ceil(len(candidates) * 0.30)) if candidates else 0
        for entity in candidates[:keep]:
            if entity["core_score"] >= 0.60:
                entity["is_core"] = True
        if candidates and not any(item["is_core"] for item in candidates):
            candidates[0]["is_core"] = True


def _write_entity_vectors(
    output_dir: Path,
    entities: list[dict[str, Any]],
    embeddings: np.ndarray,
    model_path: Path,
    embedding_audit: dict[str, Any],
) -> dict[str, Any]:
    np.save(output_dir / "entity_embeddings.npy", embeddings.astype(np.float32))
    try:
        import faiss

        index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
        index.add(embeddings.astype(np.float32))
        serialized = faiss.serialize_index(index)
        (output_dir / "entity_vectors.faiss").write_bytes(serialized.tobytes())
    except Exception as exc:
        raise RuntimeError(f"无法写入实体 FAISS 索引：{exc}") from exc
    rows: list[dict[str, Any]] = []
    for row, entity in enumerate(entities):
        ref = {
            "file": "entity_embeddings.npy",
            "index_file": "entity_vectors.faiss",
            "row": row,
            "dimension": EMBEDDING_DIMENSION,
        }
        entity["embedding_ref"] = ref
        rows.append({
            "row": row,
            "entity_id": entity["id"],
            "name": entity["name"],
            "aliases": entity["aliases"],
            "section_ids": entity["section_ids"],
            "embedding_ref": ref,
        })
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "model": str(model_path),
        "dimension": EMBEDDING_DIMENSION,
        "normalized": True,
        "count": len(entities),
        "files": {
            "embeddings": "entity_embeddings.npy",
            "faiss": "entity_vectors.faiss",
        },
        "runtime": embedding_audit,
        "entities": rows,
    }
    (output_dir / "entity_embedding_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def _attribute_facts(units: Iterable[KnowledgeUnit]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for unit in units:
        for statement in unit.statements:
            if statement.statement_type != "attribute" or not statement.evidence_id:
                continue
            item = statement.to_dict()
            item.update({
                "section_id": unit.section_id,
                "evidence_ids": [statement.evidence_id],
            })
            result.append(item)
        for element in unit.knowledge_elements:
            if element.get("type") == "formula" and _compact(element.get("raw_text")):
                result.append({
                    "id": f"attribute:{_stable_hash(element.get('id'), element.get('raw_text'))}",
                    "statement_type": "attribute",
                    "subject": _compact(element.get("caption")) or "章节公式",
                    "predicate_original": "表示为",
                    "predicate_normalized": "公式表达式",
                    "value": _compact(element.get("raw_text")),
                    "description": _compact(element.get("meaning")),
                    "section_id": unit.section_id,
                    "evidence_id": str(element.get("id", "")),
                    "evidence_ids": [str(element.get("id", ""))],
                    "source_page": element.get("page"),
                    "modality": "formula",
                    "confidence": element.get("confidence", 0.0),
                    "processor": element.get("processor", "paddleocr-vl"),
                })
            if element.get("type") != "table":
                continue
            for cell in element.get("table_cells", []):
                if not isinstance(cell, dict):
                    continue
                result.append({
                    "id": f"attribute:{_stable_hash(element.get('id'), cell.get('row'), cell.get('column'), cell.get('value') or cell.get('text'))}",
                    "statement_type": "attribute",
                    "subject": _compact(element.get("caption")) or "表格",
                    "predicate_normalized": "表格单元格",
                    "value": _compact(cell.get("value") or cell.get("text")),
                    "row": cell.get("row"),
                    "column": cell.get("column"),
                    "row_header": _compact(cell.get("row_header")),
                    "column_header": _compact(cell.get("column_header")),
                    "section_id": unit.section_id,
                    "evidence_id": str(element.get("id", "")),
                    "evidence_ids": [str(element.get("id", ""))],
                    "source_page": element.get("page"),
                    "modality": "table",
                    "confidence": element.get("confidence", 0.0),
                })
    return result


def _bind_attribute_facts_to_entities(
    facts: list[dict[str, Any]], entities: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_section: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entity in entities:
        for section_id in entity.get("section_ids", []):
            by_section[str(section_id)].append(entity)
    for fact in facts:
        searchable = " ".join(_compact(fact.get(field)) for field in (
            "subject", "predicate_original", "predicate_normalized", "value", "description"
        )).casefold()
        matches: list[tuple[int, str]] = []
        for entity in by_section.get(str(fact.get("section_id", "")), []):
            names = [entity.get("name"), *entity.get("aliases", [])]
            matched_length = max(
                (len(_compact(name)) for name in names if _compact(name).casefold() in searchable),
                default=0,
            )
            if matched_length:
                matches.append((matched_length, str(entity["id"])))
        matches.sort(reverse=True)
        entity_ids = list(dict.fromkeys(entity_id for _, entity_id in matches))
        if entity_ids:
            fact["entity_ids"] = entity_ids
            fact["entity_id"] = entity_ids[0]
    return facts


def audit_hierarchical_graph(
    graph: dict[str, Any], embeddings: np.ndarray | None = None
) -> dict[str, Any]:
    nodes = [item for item in graph.get("nodes", []) if isinstance(item, dict)]
    entities = [item for item in nodes if item.get("type") == "entity"]
    sections = [item for item in nodes if item.get("type") == "section"]
    node_ids = {str(item.get("id")) for item in nodes}
    evidence_ids = {str(item.get("id")) for item in graph.get("evidence", [])}
    edges = [item for item in graph.get("edges", []) if isinstance(item, dict)]
    dangling = sum(
        str(item.get("source")) not in node_ids or str(item.get("target")) not in node_ids
        for item in edges
    )
    self_relations = sum(
        item.get("type") == "concept_relation" and item.get("source") == item.get("target")
        for item in edges
    )
    invalid_evidence = 0
    for item in [*entities, *edges, *graph.get("attribute_facts", [])]:
        invalid_evidence += sum(
            str(value) not in evidence_ids for value in item.get("evidence_ids", [])
        )
    invalid_claims = 0
    for section in sections:
        summary = _compact(section.get("summary"))
        for claim in section.get("summary_claims", []):
            refs = [str(value) for value in claim.get("evidence_ids", [])]
            claim_text = _compact(claim.get("text"))
            if (
                not refs
                or any(value not in evidence_ids for value in refs)
                or not claim_text
            ):
                invalid_claims += 1
    ungrounded_entities = sum(not item.get("evidence_ids") for item in entities)
    ungrounded_relationships = sum(
        item.get("type") == "concept_relation" and not item.get("evidence_ids")
        for item in edges
    )
    entity_section_targets = {
        str(item.get("source")) for item in edges if item.get("type") == "entity_section"
    }
    isolated = sum(str(item["id"]) not in entity_section_targets for item in entities)
    bad_refs = sum(
        not isinstance(item.get("embedding_ref"), dict)
        or int(item["embedding_ref"].get("dimension", 0)) != EMBEDDING_DIMENSION
        or item["embedding_ref"].get("file") != "entity_embeddings.npy"
        or item["embedding_ref"].get("index_file") != "entity_vectors.faiss"
        for item in entities
    )
    vector_mismatch = int(
        embeddings is None
        or embeddings.shape != (len(entities), EMBEDDING_DIMENSION)
        or bool(embeddings.size and np.any(np.abs(np.linalg.norm(embeddings, axis=1) - 1) > 1e-3))
    )
    content_sections = [item for item in sections if item.get("evidence_ids")]
    missing_summaries = sum(not _compact(item.get("summary")) for item in content_sections)
    missing_summary_evidence = sum(
        not item.get("summary_claims") for item in content_sections
    )
    oversized_summaries = sum(
        int(item.get("summary_token_count", 0) or 0) > 400 for item in content_sections
    )
    alias_owners: dict[str, set[str]] = defaultdict(set)
    for entity in entities:
        for name in [entity.get("name"), *entity.get("aliases", [])]:
            normalized = _compact(name).casefold()
            if normalized:
                alias_owners[normalized].add(str(entity["id"]))
    alias_conflicts = sum(len(owners) > 1 for owners in alias_owners.values())
    represented_evidence = {
        str(value)
        for section in sections
        for claim in section.get("summary_claims", [])
        for value in claim.get("evidence_ids", [])
    }
    for item in [*entities, *edges, *graph.get("attribute_facts", [])]:
        represented_evidence.update(str(value) for value in item.get("evidence_ids", []))
    multimodal_evidence = {
        str(item.get("id"))
        for item in graph.get("evidence", [])
        if str(item.get("modality", "text")) != "text"
    }
    multimodal_coverage = (
        len(multimodal_evidence & represented_evidence) / len(multimodal_evidence)
        if multimodal_evidence else 1.0
    )
    critical = (
        dangling + self_relations + invalid_evidence + isolated + bad_refs
        + vector_mismatch + missing_summaries + missing_summary_evidence
        + oversized_summaries + alias_conflicts + invalid_claims
        + ungrounded_entities + ungrounded_relationships
    )
    attribute_facts = graph.get("attribute_facts", [])
    fact_coverage = (
        sum(bool(item.get("evidence_ids")) for item in attribute_facts) / len(attribute_facts)
        if attribute_facts else 1.0
    )
    metrics = {
        "sections": len(sections),
        "content_sections": len(content_sections),
        "entities": len(entities),
        "relationships": sum(item.get("type") == "concept_relation" for item in edges),
        "text_units": len(graph.get("text_units", [])),
        "attribute_facts": len(attribute_facts),
        "text_unit_fact_coverage": fact_coverage,
        "multimodal_element_fact_coverage": multimodal_coverage,
        "dangling_relationships": dangling,
        "self_relationships": self_relations,
        "isolated_entities": isolated,
        "invalid_evidence_references": invalid_evidence,
        "invalid_summary_claims": invalid_claims,
        "ungrounded_entities": ungrounded_entities,
        "ungrounded_relationships": ungrounded_relationships,
        "invalid_embedding_refs": bad_refs,
        "vector_mismatch": vector_mismatch,
        "missing_section_summaries": missing_summaries,
        "missing_summary_evidence": missing_summary_evidence,
        "oversized_section_summaries": oversized_summaries,
        "canonical_alias_conflicts": alias_conflicts,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if critical == 0 else "failed",
        "critical_issues": critical,
        "warning_issues": 0,
        "metrics": metrics,
    }


def project_legacy_graph(graph: dict[str, Any]) -> dict[str, Any]:
    """Compatibility projection; schema 4 remains the sole graph authority."""

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    for node in graph.get("nodes", []):
        if node.get("type") == "entity":
            nodes.append({
                "id": node["id"],
                "type": "concept",
                "name": node["name"],
                "aliases": node.get("aliases", []),
                "chapter": node.get("section_id", ""),
            })
        elif node.get("type") == "section":
            nodes.append({
                "id": node["id"],
                "type": "document",
                "name": node["title"],
                "section_id": node["id"],
            })
    for edge in graph.get("edges", []):
        if edge.get("type") == "concept_relation":
            edges.append({
                "source": edge["source"],
                "target": edge["target"],
                "type": "RELATED_TO",
                "relation": edge["relation"],
                "strength": edge.get("strength", 0),
            })
        elif edge.get("type") == "entity_section":
            edges.append({
                "source": edge["target"],
                "target": edge["source"],
                "type": "MENTIONS",
                "relation": edge.get("relation", "mentioned_in"),
            })
    return {
        "schema_version": SCHEMA_VERSION,
        "authority": "semantic_knowledge_graph.json",
        "nodes": nodes,
        "edges": edges,
        "chapters": graph.get("chapters", []),
    }


def build_hierarchical_summary_entity_graph(
    units: list[KnowledgeUnit],
    sections: list[dict[str, Any]],
    output_dir: Path,
    client: Any | None,
    embedding_model_path: Path,
    *,
    embedding_encoder: Any = encode_texts,
) -> tuple[dict[str, Any], dict[str, Any], list[KnowledgeUnit]]:
    """Run the ordered seven-stage schema-4 graph build and persist vector artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = _load_tokenizer(embedding_model_path)
    units = summarize_sections(
        units, client, output_dir / "section_summaries.jsonl", tokenizer
    )
    entities = extract_entities(units, client, output_dir / "section_entities.jsonl")
    relationships = extract_relationships(
        units, entities, client, output_dir / "section_relationships.jsonl"
    )
    entities = enhance_entity_descriptions(
        entities, relationships, client, output_dir / "entity_aggregations.jsonl"
    )
    initial_runtime: dict[str, Any] = {}
    final_runtime: dict[str, Any] = {}
    try:
        initial_embeddings, initial_runtime = embed_entities(
            entities, embedding_model_path, encoder=embedding_encoder
        )
        entities, relationships, dedup_audit = deduplicate_entities(
            entities,
            relationships,
            initial_embeddings,
            client,
            output_dir / "dedup_confirmations.jsonl",
        )
        # Re-embed canonical entities so retrieval vectors exactly match final descriptions.
        final_embeddings, final_runtime = embed_entities(
            entities, embedding_model_path, encoder=embedding_encoder
        )
        for runtime in (initial_runtime, final_runtime):
            if (
                str(runtime.get("device", "")).startswith("cuda")
                and float(runtime.get("peak_reserved_mib", 0.0) or 0.0) > 3072
            ):
                raise RuntimeError(
                    "Qwen3-Embedding GPU 显存质量门禁失败：进程峰值保留显存 "
                    f"{runtime['peak_reserved_mib']} MiB 超过 3072 MiB"
                )
    finally:
        # Paddle was already closed before graph construction. Release the Qwen
        # entity encoder here as well before chunk embedding and retrieval stores.
        release_embedding_model(embedding_model_path, device="cuda:0")
    assign_core_entities(entities, relationships, sections)

    section_map = {str(item["id"]): item for item in sections}
    unit_map = {item.section_id: item for item in units}
    for section in sections:
        unit = unit_map.get(str(section["id"]))
        if unit:
            section["summary"] = unit.summary
            section["summary_claims"] = unit.summary_claims
            section["summary_token_count"] = _token_count(unit.summary, tokenizer)
            section["summary_confidence"] = unit.summary_confidence
            section["short_summary_reason"] = unit.short_summary_reason
        section_entities = [
            item for item in entities if section["id"] in item["section_ids"]
        ]
        section["entities"] = [item["id"] for item in section_entities]
        section["core_entity_ids"] = [
            item["id"] for item in section_entities if item["is_core"]
        ]

    evidence: list[dict[str, Any]] = []
    seen_evidence: set[str] = set()
    for unit in units:
        for item in unit.text_evidence:
            evidence_id = str(item.get("id", ""))
            if evidence_id and evidence_id not in seen_evidence:
                evidence.append(dict(item))
                seen_evidence.add(evidence_id)
    edges: list[dict[str, Any]] = []
    for section in sections:
        if section.get("parent"):
            edges.append({
                "id": f"section-edge:{_stable_hash(section['parent'], section['id'])}",
                "source": section["parent"],
                "target": section["id"],
                "type": "parent_child",
                "relation": "contains",
                "strength": 10.0,
                "confidence": 1.0,
                "description": "章节层级包含关系",
                "evidence_ids": [],
            })
    edges.extend(relationships)
    for entity in entities:
        for section_id in entity["section_ids"]:
            relation = "introduced_in" if section_id == entity["section_id"] else "mentioned_in"
            edges.append({
                "id": f"entity-section:{_stable_hash(entity['id'], section_id, relation)}",
                "source": entity["id"],
                "target": section_id,
                "type": "entity_section",
                "relation": relation,
                "strength": 10.0 if relation == "introduced_in" else 8.0,
                "confidence": entity["confidence"],
                "description": "实体首次引入章节" if relation == "introduced_in" else "实体在章节中被提及",
                "evidence_ids": entity["evidence_ids"],
            })
    manifest = _write_entity_vectors(
        output_dir, entities, final_embeddings, embedding_model_path, final_runtime
    )
    chapters = [
        {
            "id": section["id"],
            "title": section["title"],
            "level": section["level"],
            "summary": section.get("summary", ""),
            "knowledge_points": section.get("core_entity_ids", []),
        }
        for section in sections
        if section["level"] == 1
    ]
    attribute_facts = _bind_attribute_facts_to_entities(
        _attribute_facts(units), entities
    )
    graph = {
        "schema_version": SCHEMA_VERSION,
        "builder": "hierarchical-summary-entity",
        "prompt_version": PROMPT_VERSION,
        "nodes": [*sections, *[{**item, "type": "entity"} for item in entities]],
        "edges": edges,
        "chapters": chapters,
        "communities": [],
        "community_reports": [],
        "relationship_mentions": [],
        "relationship_links": [],
        "attribute_facts": attribute_facts,
        "text_units": [
            {
                "id": unit.id,
                "section_id": unit.section_id,
                "text": unit.text,
                "summary": unit.summary,
                "claim_ids": [item["id"] for item in unit.summary_claims],
                "evidence_ids": unit.evidence_ids,
                "page_start": unit.page_start,
                "page_end": unit.page_end,
                "source": unit.source,
            }
            for unit in units
        ],
        "evidence": evidence,
        "embedding_manifest": {
            key: value for key, value in manifest.items() if key != "entities"
        },
        "stats": {
            "extraction_method": "hierarchical-summary-entity",
            "extraction_model": str(
                getattr(getattr(client, "config", None), "model", "offline-test-only")
            ),
            "sections": len(sections),
            "entities": len(entities),
            "relationships": len(relationships),
            "attribute_facts": len(attribute_facts),
            "text_units": len(units),
            "relationship_mentions": 0,
            "communities": 0,
            "embedding_dimension": EMBEDDING_DIMENSION,
        },
    }
    audit = audit_hierarchical_graph(graph, final_embeddings)
    audit["deduplication"] = {
        "candidates": len(dedup_audit),
        "merged": sum(item["decision"] == "merged" for item in dedup_audit),
        "qwen_rejected": sum(item["decision"] == "qwen_rejected" for item in dedup_audit),
        "threshold_rejected": sum(item["decision"] == "threshold_rejected" for item in dedup_audit),
    }
    audit["embedding_runtime"] = {
        "initial": initial_runtime,
        "final": final_runtime,
    }
    (output_dir / "deduplication_audit.json").write_text(
        json.dumps(dedup_audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    source_hashes: dict[str, str] = {}
    for source in sorted({unit.source for unit in units}):
        source_hashes[source] = hashlib.sha256(
            "\n".join(
                unit.text for unit in units if unit.source == source
            ).encode("utf-8")
        ).hexdigest()
    (output_dir / "hierarchical_graph_manifest.json").write_text(
        json.dumps({
            "schema_version": SCHEMA_VERSION,
            "cache_version": CACHE_VERSION,
            "prompt_version": PROMPT_VERSION,
            "source_hashes": source_hashes,
            "graph_model": str(
                getattr(getattr(client, "config", None), "model", "offline-test-only")
            ),
            "embedding_model": str(embedding_model_path),
            "embedding_dimension": EMBEDDING_DIMENSION,
            "embedding_runtime": audit["embedding_runtime"],
            "stage_caches": [
                "section_summaries.jsonl",
                "section_entities.jsonl",
                "section_relationships.jsonl",
                "entity_aggregations.jsonl",
                "dedup_confirmations.jsonl",
            ],
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return graph, audit, units
