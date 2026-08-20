from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import unicodedata
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterable

import fitz
import numpy as np
from docx import Document
from openpyxl import load_workbook

from backend.app.config import settings
from backend.app.rag.models import PageDocument, TextChunk
from backend.app.rag.embedding_runtime import encode_texts
from backend.app.rag.graphrag_adapter import (
    audit_microsoft_graphrag,
    run_microsoft_graphrag,
)
from backend.app.rag.knowledge_document import (
    compile_knowledge_document,
    enrich_formula_knowledge,
    enrich_knowledge_statements,
    knowledge_units_to_chunks,
    write_knowledge_document,
)
from backend.app.rag.section_titles import (
    chapter_number,
    is_measurement_section,
    normalize_structural_section,
    numbered_section_parts,
    visible_structural_section,
)
from backend.app.rag.ontology import (
    COURSE_CONCEPTS,
    extract_course_concepts,
    is_course_concept,
    normalize_concept_name,
)
from backend.app.rag.multimodal import (
    BuildModelConfig,
    CompatibleMultimodalClient,
    LayoutElement,
    SCANNED_PAGE_PLACEHOLDER,
    audit_visual_semantics,
    build_chapter_knowledge_summaries,
    build_local_knowledge_graph,
    enhance_pdf,
    multimodal_chunks,
)
from backend.app.rag.semantic_graph import (
    audit_semantic_graph_quality,
    bind_chapter_knowledge_points,
    build_semantic_knowledge_graph,
    is_semantic_graph,
)
from backend.app.rag.stores import build_qdrant_indexes, sync_neo4j_graph


logger = logging.getLogger(__name__)


class KnowledgeBaseBuildCancelled(RuntimeError):
    """Raised when a cooperative knowledge-base build cancellation is requested."""


BuildProgressCallback = Callable[[int, str, str], None]

KNOWLEDGE_EXTENSIONS = {".pdf", ".md", ".txt", ".docx"}
QUESTION_BANK_EXTENSIONS = {".xlsx", ".json"}
SUPPORTED_EXTENSIONS = KNOWLEDGE_EXTENSIONS | QUESTION_BANK_EXTENSIONS
AD_NOISE = (
    "扫码关注",
    "微信公众号",
    "关注公众号",
    "购买正版",
    "资源下载",
    "广告",
)
TAG_KEYWORDS = COURSE_CONCEPTS


def _normalize_line(line: str) -> str:
    line = unicodedata.normalize("NFKC", line).replace("\u200b", "")
    line = re.sub(r"https?\s*:\s*[/\\]+\s*\S+", "", line, flags=re.I)
    line = re.sub(r"www\s*\.\s*\S+", "", line, flags=re.I)
    line = re.sub(r"(?<=[A-Za-z])\s+(?=[A-Za-z0-9])", "", line)
    line = re.sub(r"(?<=[0-9])\s+(?=[A-Za-z])", "", line)
    line = re.sub(r"(?<=[A-Za-z])\s+(?=[0-9])", "", line)
    line = re.sub(r"P\s*N\s*结", "PN结", line, flags=re.I)
    line = re.sub(r"([NP])\s*型", r"\1型", line, flags=re.I)
    line = re.sub(r"M\s*O\s*S", "MOS", line, flags=re.I)
    line = re.sub(r"[ \t]+", " ", line).strip()
    return line


def _edge_noise(raw_pages: list[str]) -> set[str]:
    candidates: Counter[str] = Counter()
    for text in raw_pages:
        lines = [_normalize_line(line) for line in text.splitlines() if _normalize_line(line)]
        for line in lines[:2] + lines[-2:]:
            if len(line) <= 80:
                candidates[line] += 1
    threshold = max(3, int(len(raw_pages) * 0.08))
    return {line for line, count in candidates.items() if count >= threshold}


def clean_page_text(text: str, repeated_noise: set[str] | None = None) -> str:
    repeated_noise = repeated_noise or set()
    cleaned_lines: list[str] = []
    for raw_line in text.splitlines():
        line = _normalize_line(raw_line)
        if not line or line in repeated_noise:
            continue
        if re.fullmatch(r"[-—·•\s]*\d{1,4}[-—·•\s]*", line):
            continue
        if any(noise in line for noise in AD_NOISE):
            continue
        if sum(char == "�" for char in line) > 1:
            continue
        cleaned_lines.append(line)

    paragraphs: list[str] = []
    buffer = ""
    heading_pattern = re.compile(
        r"^(?:第[一二三四五六七八九十百0-9]+章|\d+(?:\.\d+){0,3}\s+|本章小结|自测题|习题)"
    )
    for line in cleaned_lines:
        is_heading = bool(heading_pattern.match(line)) and len(line) < 70
        if is_heading:
            if buffer:
                paragraphs.append(buffer)
                buffer = ""
            paragraphs.append(line)
            continue
        buffer += line
        if line.endswith(("。", "！", "？", ":", "；")) or len(buffer) >= 260:
            paragraphs.append(buffer)
            buffer = ""
    if buffer:
        paragraphs.append(buffer)
    return "\n\n".join(paragraphs)


def _is_chapter_title(title: str) -> bool:
    return bool(re.match(r"^第[一二三四五六七八九十百0-9]+章", title.replace(" ", "")))


def extract_pdf(path: Path, chapter_limit: int | None = None) -> list[PageDocument]:
    """Create one OCR-required placeholder per PDF page.

    PyMuPDF is deliberately limited to page geometry/rendering in the PDF
    pipeline. It is no longer used for native text, block, outline, or formula
    extraction, so a stale or malformed PDF text layer cannot silently become
    the knowledge source. ``chapter_limit`` is applied after OCR has recovered
    real headings in ``_ocr_scanned_pages``.
    """

    document = fitz.open(path)
    page_docs: list[PageDocument] = []
    page_range_match = re.search(r"(?:pages?|页)[_-]?(\d+)[_-](\d+)", path.stem, re.I)
    source_page_offset = int(page_range_match.group(1)) - 1 if page_range_match else 0
    for page_number in range(1, document.page_count + 1):
        page_docs.append(
            PageDocument(
                text=SCANNED_PAGE_PLACEHOLDER,
                source=path.name,
                page=page_number,
                chapter=path.stem,
                section=path.stem,
                source_page=source_page_offset + page_number,
                extra={"ocr_required": True, "native_text_layer_ignored": True},
            )
        )
    document.close()
    return page_docs


def extract_markdown_or_text(path: Path) -> list[PageDocument]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    chapter = path.stem
    section = path.stem
    documents: list[PageDocument] = []
    page = 1
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer
        cleaned = clean_page_text("\n".join(buffer))
        if cleaned:
            documents.append(PageDocument(cleaned, path.name, page, chapter, section))
        buffer = []

    for line in text.splitlines():
        heading = re.match(r"^(#{1,3})\s+(.+)$", line.strip())
        if heading:
            flush()
            title = heading.group(2).strip()
            if len(heading.group(1)) == 1:
                chapter = title
            section = title
        elif line.strip() == "\f":
            flush()
            page += 1
        else:
            buffer.append(line)
    flush()
    return documents


def extract_docx(path: Path) -> list[PageDocument]:
    document = Document(path)
    chapter = path.stem
    section = path.stem
    blocks: list[PageDocument] = []
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer
        text = clean_page_text("\n".join(buffer))
        if text:
            blocks.append(PageDocument(text, path.name, 1, chapter, section))
        buffer = []

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        style = (paragraph.style.name or "").lower()
        if "heading" in style or "标题" in style:
            flush()
            section = text
            if style.endswith("1"):
                chapter = text
        else:
            buffer.append(text)
    flush()
    return blocks


def _split_tags(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in re.split(r"[,，、;；|]", str(value or "")) if item.strip()]


QUESTION_HEADER_ALIASES = {
    "question_id": ("题号", "question_id", "id"),
    "question_text": ("题目文本", "题目", "question_text"),
    "knowledge_tags": ("知识点标签", "知识点", "knowledge_tags"),
    "standard_answer": ("标准答案", "答案", "standard_answer"),
    "common_mistakes": ("易错点", "common_mistakes"),
    "difficulty": ("难度", "difficulty"),
    "question_type": ("题型", "question_type"),
    "solution_steps": ("解题步骤", "解析", "solution_steps"),
}


def extract_question_xlsx(path: Path) -> list[dict[str, Any]]:
    workbook = load_workbook(path, read_only=True, data_only=True)
    sheet = workbook.active
    rows = list(sheet.iter_rows(values_only=True))
    if not rows:
        return []
    header_index = next(
        (
            index
            for index, row in enumerate(rows[:12])
            if "题号" in [str(value or "").strip() for value in row]
            and "题目文本" in [str(value or "").strip() for value in row]
        ),
        0,
    )
    headers = [str(value or "").strip() for value in rows[header_index]]
    mapping: dict[str, int] = {}
    for field, aliases in QUESTION_HEADER_ALIASES.items():
        for alias in aliases:
            if alias in headers:
                mapping[field] = headers.index(alias)
                break
    required = {"question_id", "question_text", "knowledge_tags", "standard_answer", "common_mistakes"}
    if not required.issubset(mapping):
        workbook.close()
        return []
    questions: list[dict[str, Any]] = []
    for row in rows[header_index + 1 :]:
        if not row or not row[mapping["question_text"]]:
            continue
        item = {
            field: (row[index] if index < len(row) else "")
            for field, index in mapping.items()
        }
        item["question_id"] = str(item["question_id"])
        item["question_text"] = str(item["question_text"]).strip()
        item["knowledge_tags"] = _split_tags(item["knowledge_tags"])
        item["standard_answer"] = str(item["standard_answer"] or "").strip()
        item["common_mistakes"] = str(item["common_mistakes"] or "").strip()
        item["difficulty"] = str(item.get("difficulty") or "基础").strip()
        item["question_type"] = str(item.get("question_type") or "综合题").strip()
        item["solution_steps"] = str(item.get("solution_steps") or "").strip()
        item["source"] = path.name
        questions.append(item)
    workbook.close()
    return questions


def extract_question_json(path: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeError):
        return []
    if isinstance(data, dict):
        data = data.get("questions", [])
    if not isinstance(data, list):
        return []
    questions = []
    for index, item in enumerate(data, 1):
        if not isinstance(item, dict) or not item.get("question_text"):
            continue
        normalized = dict(item)
        normalized.setdefault("question_id", f"JSON-{index:03d}")
        normalized["knowledge_tags"] = _split_tags(normalized.get("knowledge_tags"))
        normalized.setdefault("standard_answer", "")
        normalized.setdefault("common_mistakes", "")
        normalized.setdefault("difficulty", "基础")
        normalized.setdefault("question_type", "综合题")
        normalized.setdefault("solution_steps", "")
        normalized["source"] = path.name
        questions.append(normalized)
    return questions


def _knowledge_tags(
    text: str,
    section: str,
    supplemental: Iterable[str] = (),
) -> list[str]:
    tags = extract_course_concepts(text, section)
    compact_text = re.sub(r"\s+", "", text).lower()
    for value in supplemental:
        concept = normalize_concept_name(str(value))
        if (
            concept
            and is_course_concept(concept)
            and re.sub(r"\s+", "", concept).lower() in compact_text
            and concept not in tags
        ):
            tags.append(concept)
    return tags[:12]


def _sentence_pieces(text: str, max_chars: int = 900) -> list[str]:
    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
    pieces: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= max_chars:
            pieces.append(paragraph)
            continue
        sentences = [piece for piece in re.split(r"(?<=[。！？；])", paragraph) if piece]
        current = ""
        for sentence in sentences:
            if current and len(current) + len(sentence) > max_chars:
                pieces.append(current)
                current = ""
            if len(sentence) > max_chars:
                pieces.extend(
                    sentence[index : index + max_chars]
                    for index in range(0, len(sentence), max_chars)
                )
                continue
            current += sentence
        if current:
            pieces.append(current)
    return pieces


def chunk_documents(documents: Iterable[PageDocument], max_chars: int = 900) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    for document in documents:
        pieces = _sentence_pieces(document.text, max_chars=max_chars)
        supplemental_concepts = (
            document.extra.get("ocr_concepts", [])
            if isinstance(document.extra, dict)
            and isinstance(document.extra.get("ocr_concepts"), list)
            else []
        )
        current: list[str] = []
        current_length = 0

        def flush() -> None:
            nonlocal current, current_length
            if not current:
                return
            text = "\n\n".join(current).strip()
            raw_id = f"{document.source}|{document.page}|{document.section}|{text[:120]}"
            chunk_id = hashlib.sha1(raw_id.encode("utf-8")).hexdigest()[:16]
            chunks.append(
                TextChunk(
                    id=chunk_id,
                    text=text,
                    source=document.source,
                    chapter=document.chapter,
                    section=document.section,
                    page_start=document.source_page or document.page,
                    page_end=document.source_page or document.page,
                    doc_type=document.doc_type,
                    knowledge_tags=_knowledge_tags(
                        text, document.section, supplemental_concepts
                    ),
                    element_type=document.element_type,
                    bbox=document.bbox,
                    parent_id=document.parent_id,
                    image_path=document.image_path,
                    content_hash=document.content_hash,
                    multimodal=document.extra,
                )
            )
            overlap = text[-140:] if len(text) > 140 else ""
            current = [overlap] if overlap else []
            current_length = len(overlap)

        for piece in pieces:
            if current and current_length + len(piece) > max_chars:
                flush()
            if current and current_length + len(piece) > max_chars:
                current = []
                current_length = 0
            current.append(piece)
            current_length += len(piece)
        flush()
    return chunks


def question_chunks(questions: Iterable[dict[str, Any]]) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    for item in questions:
        tags = _split_tags(item.get("knowledge_tags"))
        text = (
            f"题目：{item.get('question_text', '')}\n"
            f"标准答案：{item.get('standard_answer', '')}\n"
            f"解题步骤：{item.get('solution_steps', '')}\n"
            f"易错点：{item.get('common_mistakes', '')}"
        ).strip()
        question_id = str(item.get("question_id", "Q"))
        chunks.append(
            TextChunk(
                id=f"question-{question_id}",
                text=text,
                source=str(item.get("source", "question_bank.json")),
                chapter="示例题库",
                section="、".join(tags) or "综合",
                page_start=None,
                page_end=None,
                doc_type="question",
                knowledge_tags=tags,
            )
        )
    return chunks


def _write_clean_markdown(path: Path, documents: list[PageDocument], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"# {path.stem}", "", f"> 清洗来源：{path.name}", ""]
    last_chapter = last_section = ""
    for document in documents:
        if document.chapter and document.chapter != last_chapter:
            lines.extend([f"## {document.chapter}", ""])
            last_chapter = document.chapter
        if document.section and document.section != last_section and document.section != document.chapter:
            lines.extend([f"### {document.section}", ""])
            last_section = document.section
        lines.extend([
            f"<!-- source={document.source}; page={document.source_page or document.page} -->",
            document.text,
            "",
        ])
    (output_dir / f"{path.stem}.clean.md").write_text("\n".join(lines), encoding="utf-8")


def validate_extracted_content(chunks: list[TextChunk]) -> dict[str, int]:
    text_chunks = [
        chunk for chunk in chunks
        if chunk.doc_type == "textbook" and chunk.element_type == "text"
    ]
    placeholders = [
        chunk for chunk in text_chunks
        if chunk.text.strip() == SCANNED_PAGE_PLACEHOLDER
    ]
    if placeholders and len(placeholders) >= max(3, (len(text_chunks) + 3) // 4):
        raise RuntimeError(
            "扫描版 PDF 的正文 OCR 未成功："
            f"{len(placeholders)}/{len(text_chunks)} 个正文片段仍是图形占位符。"
            "请配置可用的 Qwen3-VL API 后重建，旧索引不会被替换。"
        )
    return {
        "text_chunks": len(text_chunks),
        "placeholder_text_chunks": len(placeholders),
    }


def _numbered_section_parts(value: str) -> tuple[tuple[int, ...], str]:
    return numbered_section_parts(value)


def _section_is_visible_on_page(section: str, text: str) -> bool:
    target = re.sub(r"\s+", "", section).replace("．", ".")
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", "", raw_line).replace("．", ".")
        if line.startswith(("图", "表", "式", "例", "【例")):
            continue
        if line.startswith(target):
            return True
    return False


def validate_section_semantics(
    documents: Iterable[PageDocument],
) -> dict[str, Any]:
    """Audit section provenance and sequence before titles enter chunks and graphs."""

    by_source: dict[str, list[PageDocument]] = {}
    for document in documents:
        by_source.setdefault(document.source, []).append(document)

    issues: list[dict[str, Any]] = []
    variants: dict[tuple[str, str, tuple[int, ...]], set[str]] = {}
    sectioned_pages = verified_pages = inherited_pages = structural_pages = 0
    toc_corrected_pages = heading_crop_pages = transition_count = 0
    critical_count = hard_invariant_count = 0

    for source, source_documents in by_source.items():
        previous_chapter = ""
        previous_number: tuple[int, ...] = ()
        previous_section = ""
        for document in sorted(source_documents, key=lambda item: item.page):
            if document.chapter.startswith(("部分习题", "参考文献", "附录", "目录")):
                previous_chapter = document.chapter
                previous_number = ()
                previous_section = ""
                continue
            extra = document.extra if isinstance(document.extra, dict) else {}
            section_source = str(extra.get("ocr_section_source", "native"))
            page_number = document.source_page or document.page
            visible_structural = visible_structural_section(document.text)
            structural = normalize_structural_section(document.section)

            if visible_structural and structural != visible_structural:
                critical_count += 1
                hard_invariant_count += 1
                issues.append({
                    "severity": "error",
                    "code": "missing_structural_heading",
                    "source": source,
                    "page": page_number,
                    "visible_section": visible_structural,
                    "section": document.section,
                })

            if is_measurement_section(document.section):
                critical_count += 1
                hard_invariant_count += 1
                issues.append({
                    "severity": "error",
                    "code": "measurement_unit_heading",
                    "source": source,
                    "page": page_number,
                    "section": document.section,
                })

            if structural:
                sectioned_pages += 1
                structural_pages += 1
                if section_source in {
                    "page-text", "heading-crop", "toc", "structural-heading",
                }:
                    verified_pages += 1
                if section_source == "inherited":
                    inherited_pages += 1
                if document.section != previous_section:
                    transition_count += 1
                    previous_section = document.section
                    previous_number = ()
                previous_chapter = document.chapter or previous_chapter
                continue

            number, title = _numbered_section_parts(document.section)
            if not number:
                previous_chapter = document.chapter or previous_chapter
                continue
            sectioned_pages += 1
            if section_source in {
                "page-text", "heading-crop", "toc", "structural-heading",
            }:
                verified_pages += 1
            if section_source == "inherited":
                inherited_pages += 1
            elif section_source == "toc":
                toc_corrected_pages += int(bool(extra.get("ocr_section_corrected")))
            elif section_source == "heading-crop":
                heading_crop_pages += 1

            key = (source, document.chapter, number)
            variants.setdefault(key, set()).add(document.section)

            expected_chapter = chapter_number(document.chapter)
            if expected_chapter is not None and number[0] != expected_chapter:
                critical_count += 1
                hard_invariant_count += 1
                issues.append({
                    "severity": "error",
                    "code": "chapter_section_mismatch",
                    "source": source,
                    "page": page_number,
                    "chapter": document.chapter,
                    "section": document.section,
                })

            if (
                section_source == "page-text"
                and not _section_is_visible_on_page(document.section, document.text)
            ):
                critical_count += 1
                issues.append({
                    "severity": "error",
                    "code": "ungrounded_page_heading",
                    "source": source,
                    "page": page_number,
                    "section": document.section,
                })

            try:
                confidence = float(extra.get("ocr_section_confidence", 1.0))
            except (TypeError, ValueError):
                confidence = 0.0
            if section_source == "heading-crop" and confidence < 0.65:
                critical_count += 1
                issues.append({
                    "severity": "error",
                    "code": "low_confidence_heading_crop",
                    "source": source,
                    "page": page_number,
                    "section": document.section,
                    "confidence": confidence,
                })

            if (
                len(title) > 60
                or re.search(r"[=；;！？!?]", title)
                or re.match(r"^(?:可得|已知|求|试|答|解|[（(]\d)", title)
            ):
                issues.append({
                    "severity": "warning",
                    "code": "suspicious_section_title",
                    "source": source,
                    "page": page_number,
                    "section": document.section,
                })

            chapter_changed = bool(
                previous_chapter
                and document.chapter
                and document.chapter != previous_chapter
            )
            if document.section != previous_section:
                transition_count += 1
                if (
                    previous_number
                    and not chapter_changed
                    and number < previous_number
                    and not (
                        len(number) <= len(previous_number)
                        and previous_number[: len(number)] == number
                    )
                ):
                    critical_count += 1
                    issues.append({
                        "severity": "error",
                        "code": "backward_section_transition",
                        "source": source,
                        "page": page_number,
                        "previous_section": previous_section,
                        "section": document.section,
                    })
                previous_number = number
                previous_section = document.section
            previous_chapter = document.chapter or previous_chapter

    inconsistent_numbers = 0
    for (source, chapter, _), titles in variants.items():
        if len(titles) <= 1:
            continue
        inconsistent_numbers += 1
        critical_count += 1
        issues.append({
            "severity": "error",
            "code": "inconsistent_section_titles",
            "source": source,
            "chapter": chapter,
            "variants": sorted(titles),
        })

    failure_threshold = max(3, int(max(1, transition_count) * 0.05 + 0.999))
    status = (
        "failed"
        if (
            hard_invariant_count > 0
            or critical_count >= failure_threshold
            or inconsistent_numbers >= 2
        )
        else "warning"
        if issues
        else "passed"
    )
    return {
        "status": status,
        "audited_pages": sum(len(items) for items in by_source.values()),
        "sectioned_pages": sectioned_pages,
        "structural_heading_pages": structural_pages,
        "section_transitions": transition_count,
        "verified_heading_pages": verified_pages,
        "inherited_heading_pages": inherited_pages,
        "heading_crop_pages": heading_crop_pages,
        "toc_corrected_pages": toc_corrected_pages,
        "inconsistent_section_numbers": inconsistent_numbers,
        "critical_issues": critical_count,
        "hard_invariant_issues": hard_invariant_count,
        "failure_threshold": failure_threshold,
        "issues": issues[:100],
    }


def validate_graph_semantics(
    chunks: list[TextChunk],
    graph: dict[str, Any],
    quality_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    textbook_chunks = [chunk for chunk in chunks if chunk.doc_type == "textbook"]
    if is_semantic_graph(graph):
        entities = {
            str(node.get("name", "")).strip()
            for node in graph.get("nodes", [])
            if node.get("type") == "entity" and str(node.get("name", "")).strip()
        }
        relationships = [
            edge for edge in graph.get("edges", [])
            if edge.get("source") and edge.get("target") and edge.get("relation")
        ]
        if len(textbook_chunks) >= 12 and not relationships:
            raise RuntimeError(
                "知识图谱语义校验失败：教材内容较多，但没有抽取到任何实体关系。"
                "请检查 TextUnit 切分、图谱抽取模型或关系证据，旧索引不会被替换。"
            )
        audit = quality_audit or audit_semantic_graph_quality(graph)
        if audit.get("status") != "passed":
            raise RuntimeError(
                "知识图谱质量门禁失败："
                f"发现 {int(audit.get('critical_issues', 0))} 个关键问题。"
                "请查看 semantic_quality_audit.json；旧索引不会被替换。"
            )
        return {
            "concept_nodes": len(entities),
            "entity_nodes": len(entities),
            "semantic_relationships": len(relationships),
            "semantic_quality_status": audit.get("status", "unknown"),
            "semantic_quality_warnings": int(audit.get("warning_issues", 0)),
        }
    concepts = {
        str(node.get("name", "")).strip()
        for node in graph.get("nodes", [])
        if node.get("type") == "concept" and str(node.get("name", "")).strip()
    }
    if len(textbook_chunks) >= 12 and len(concepts) <= 1:
        raise RuntimeError(
            "知识图谱语义校验失败：教材内容较多，但只识别出"
            f" {len(concepts)} 个知识点。请检查 OCR/章节识别结果，旧索引不会被替换。"
        )
    return {"concept_nodes": len(concepts)}


def validate_build_artifacts(
    chunks: list[TextChunk],
    embeddings: np.ndarray,
    graph: dict[str, Any],
) -> dict[str, Any]:
    if any(chunk.doc_type == "question" for chunk in chunks):
        raise RuntimeError("构建校验失败：题库 Chunk 不得进入课程知识库")
    if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
        raise RuntimeError("构建校验失败：向量数量与 Chunk 数量不一致")
    node_ids = {
        str(node.get("id", "")) for node in graph.get("nodes", []) if node.get("id")
    }
    dangling = [
        edge for edge in graph.get("edges", [])
        if str(edge.get("source", "")) not in node_ids
        or str(edge.get("target", "")) not in node_ids
    ]
    if dangling:
        raise RuntimeError(f"构建校验失败：知识图谱存在 {len(dangling)} 条悬空关系")
    invalid_bbox = [
        chunk.id for chunk in chunks
        if chunk.bbox is not None and len(chunk.bbox) != 4
    ]
    if invalid_bbox:
        raise RuntimeError(f"构建校验失败：{len(invalid_bbox)} 个多模态元素坐标不完整")
    return {
        "status": "passed",
        "chunks": len(chunks),
        "vectors": int(embeddings.shape[0]),
        "vector_dimension": int(embeddings.shape[1]),
        "graph_nodes": len(node_ids),
        "graph_edges": len(graph.get("edges", [])),
        "question_chunks": 0,
        "dangling_graph_edges": 0,
        "concept_nodes": sum(
            node.get("type") in {"concept", "entity"} for node in graph.get("nodes", [])
        ),
        "semantic_relationships": sum(
            bool(edge.get("relation")) for edge in graph.get("edges", [])
        ),
        "placeholder_text_chunks": sum(
            chunk.text.strip() == SCANNED_PAGE_PLACEHOLDER
            for chunk in chunks
            if chunk.doc_type == "textbook" and chunk.element_type == "text"
        ),
    }


def _formula_pipeline_stats(output_dir: Path, elements: list[LayoutElement]) -> dict[str, Any]:
    categories: Counter[str] = Counter()
    audit_counts: Counter[str] = Counter()
    for path in output_dir.glob("*.pdf_extract_kit.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for region in payload.get("regions", []):
            if not isinstance(region, dict):
                continue
            category = str(region.get("category", "")).lower()
            detector = str(region.get("detector", ""))
            if detector.endswith(":formula") or category in {
                "isolate_formula", "isolated", "isolated_formula"
            }:
                categories[category] += 1
    for path in output_dir.glob("*.formula_audit.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for name in ("detected", "recognized", "fallback", "uncertain"):
            try:
                audit_counts[name] += int(payload.get(name, 0))
            except (TypeError, ValueError):
                continue
    formulas = [element for element in elements if element.element_type == "formula"]
    display_candidates = audit_counts["detected"] or sum(
        categories.get(name, 0)
        for name in ("isolate_formula", "isolated", "isolated_formula")
    )
    return {
        "detected_regions": sum(categories.values()),
        "inline_regions_kept_in_text": categories.get("inline", 0),
        "display_candidates": display_candidates,
        "recognized_formulas": audit_counts["recognized"],
        "fallback_formulas": audit_counts["fallback"],
        "indexed_formulas": len(formulas),
        "rejected_or_merged_regions": max(0, display_candidates - len(formulas)),
        "uncertain_formulas": (
            audit_counts["uncertain"]
            if audit_counts["detected"]
            else sum(element.uncertain for element in formulas)
        ),
        "recognition": (
            "PDF-Extract-Kit localization + page OCR + "
            f"qwen/{settings.qwen_circuit_vision_model} formula recognition"
            if settings.qwen_api_key
            else "PDF-Extract-Kit localization + page OCR formula fallback"
        ),
    }


def build_knowledge_base(
    resources_dir: Path,
    output_dir: Path,
    embedding_model_path: Path,
    *,
    chapter_limit: int | None = None,
    model_config: BuildModelConfig | None = None,
    knowledge_base_id: str | None = None,
    sync_graph_store: bool = True,
    progress_callback: BuildProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> dict[str, Any]:
    def report(progress: int, stage: str, message: str) -> None:
        if progress_callback is not None:
            progress_callback(progress, stage, message)
        if cancel_event is not None and cancel_event.is_set():
            raise KnowledgeBaseBuildCancelled("用户已取消知识库构建")

    resources_dir = resources_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cleaned_dir = output_dir / "cleaned_documents"
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    report(5, "document_scanning", "正在扫描教材与讲义")

    documents: list[PageDocument] = []
    elements: list[LayoutElement] = []
    cleaning_audits: list[dict[str, Any]] = []
    candidate_files = [
        path for path in sorted(resources_dir.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    source_files = [
        path for path in candidate_files if path.suffix.lower() in KNOWLEDGE_EXTENSIONS
    ]
    excluded_sources = [
        {
            "source": path.name,
            "reason": "题库文件与课程知识库隔离，不参与分块、向量、图谱或检索",
        }
        for path in candidate_files
        if path.suffix.lower() in QUESTION_BANK_EXTENSIONS
    ]
    source_count = max(1, len(source_files))
    for source_index, path in enumerate(source_files):
        report(
            10 + int(source_index / source_count * 35),
            "document_parsing",
            f"正在解析 {path.name}（{source_index + 1}/{len(source_files)}）",
        )
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            extracted = extract_pdf(path, chapter_limit)
            extracted, pdf_elements, audit = enhance_pdf(
                path,
                extracted,
                output_dir,
                model_config=model_config,
                chapter_limit=chapter_limit,
            )
            repeated_noise = _edge_noise([item.text for item in extracted])
            extracted = [
                replace(
                    item,
                    text=clean_page_text(item.text, repeated_noise) or item.text,
                )
                for item in extracted
            ]
            documents.extend(extracted)
            elements.extend(pdf_elements)
            cleaning_audits.extend(
                {**item, "source": path.name} for item in audit
            )
            _write_clean_markdown(path, extracted, cleaned_dir)
        elif suffix in {".md", ".txt"}:
            extracted = extract_markdown_or_text(path)
            documents.extend(extracted)
            _write_clean_markdown(path, extracted, cleaned_dir)
        elif suffix == ".docx":
            extracted = extract_docx(path)
            documents.extend(extracted)
            _write_clean_markdown(path, extracted, cleaned_dir)
        report(
            10 + int((source_index + 1) / source_count * 35),
            "document_cleaning",
            f"已完成 {path.name} 的解析与清洗",
        )
    report(47, "section_validation", "正在校验章节标题、目录映射与继承顺序")
    section_quality = validate_section_semantics(documents)
    (output_dir / "section_quality_audit.json").write_text(
        json.dumps(section_quality, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if section_quality["status"] == "failed":
        raise RuntimeError(
            "章节语义质检失败：检测到 "
            f"{section_quality['critical_issues']} 个高风险标题问题，"
            "请检查 section_quality_audit.json；旧索引不会被替换。"
        )
    structured_questions = {
        "schema_version": "2.0",
        "questions": [],
        "excluded_sources": excluded_sources,
        "message": "题库与课程知识库隔离；本文件仅保留兼容占位。",
    }
    (output_dir / "question_bank.json").write_text(
        json.dumps(structured_questions, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    visual_quality = audit_visual_semantics(elements)
    (output_dir / "visual_semantic_quality_audit.json").write_text(
        json.dumps(visual_quality, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if visual_quality["status"] == "failed":
        raise RuntimeError(
            "图像语义质量门禁失败："
            f"{visual_quality['critical_issues']} 个电路类型与拓扑冲突"
        )

    report(49, "knowledge_document", "正在融合正文、公式、电路图和表格知识")
    knowledge_units = compile_knowledge_document(documents, elements)
    if not knowledge_units:
        raise RuntimeError(f"在 {resources_dir} 中没有编译出可用的教材知识单元")
    document_client = (
        CompatibleMultimodalClient(model_config)
        if model_config and model_config.enabled
        else None
    )
    knowledge_units = enrich_formula_knowledge(
        knowledge_units,
        document_client,
        cache_path=output_dir / "knowledge_document_enrichment.jsonl",
    )
    knowledge_units = enrich_knowledge_statements(
        knowledge_units,
        document_client,
        cache_path=output_dir / "knowledge_statements.jsonl",
    )
    write_knowledge_document(knowledge_units, output_dir)

    report(52, "chunking", "正在按教材语义单元切分文本并整理多模态证据")
    chunks = knowledge_units_to_chunks(knowledge_units) + multimodal_chunks(elements)
    if not chunks:
        raise RuntimeError(f"在 {resources_dir} 中没有提取到可索引内容")
    extraction_quality = validate_extracted_content(chunks)

    chunk_path = output_dir / "chunks.jsonl"
    chunk_path.write_text(
        "\n".join(json.dumps(chunk.to_dict(), ensure_ascii=False) for chunk in chunks),
        encoding="utf-8",
    )
    (output_dir / "multimodal_elements.jsonl").write_text(
        "\n".join(json.dumps(element.to_dict(), ensure_ascii=False) for element in elements),
        encoding="utf-8",
    )
    (output_dir / "cleaning_audit.json").write_text(
        json.dumps(cleaning_audits, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report(55, "semantic_text_units", "正在生成教材语义单元与 GraphRAG 输入")
    graph_model_config = model_config
    if (
        model_config
        and model_config.provider == "qwen"
        and settings.qwen_graph_model
    ):
        graph_model_config = replace(
            model_config,
            model="qwen3.7-flash",
            enable_thinking=False,
        )
    report(60, "knowledge_graph", "正在使用 Microsoft GraphRAG 构建实体关系与社区")
    if graph_model_config and graph_model_config.enabled:
        semantic_graph, semantic_audit = run_microsoft_graphrag(
            knowledge_units,
            output_dir,
            graph_model_config,
            embedding_model_path,
        )
    else:
        # Unit tests and explicitly offline deployments retain a deterministic
        # fallback; production Qwen builds always take the Microsoft path.
        semantic_graph = build_semantic_knowledge_graph(
            documents,
            elements,
            client=None,
            extraction_cache_path=output_dir / "semantic_extractions.jsonl",
        )
        semantic_audit = audit_semantic_graph_quality(semantic_graph)
    chapter_summaries = build_chapter_knowledge_summaries(chunks)
    semantic_chapters, chapter_alignment = bind_chapter_knowledge_points(
        chapter_summaries, semantic_graph.get("nodes", [])
    )
    semantic_graph["chapters"] = semantic_chapters
    semantic_graph.setdefault("stats", {})["chapter_alignment"] = chapter_alignment
    if str(semantic_graph.get("schema_version", "")).startswith("3.4-microsoft"):
        semantic_audit = audit_microsoft_graphrag(semantic_graph)
    else:
        semantic_audit = audit_semantic_graph_quality(semantic_graph)
    semantic_audit["chapter_alignment"] = chapter_alignment
    (output_dir / "semantic_quality_audit.json").write_text(
        json.dumps(semantic_audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    semantic_quality = validate_graph_semantics(chunks, semantic_graph, semantic_audit)
    legacy_graph = build_local_knowledge_graph(chunks)
    legacy_graph["chapters"] = chapter_summaries
    (output_dir / "knowledge_graph.json").write_text(
        json.dumps(legacy_graph, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "semantic_knowledge_graph.json").write_text(
        json.dumps(semantic_graph, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "chapter_knowledge_points.json").write_text(
        json.dumps(
            {
                "schema_version": "2.0-semantic-entities",
                "chapters": semantic_chapters,
                "alignment": chapter_alignment,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    for artifact_name, values in (
        ("text_units.jsonl", semantic_graph.get("text_units", [])),
        ("evidence_store.jsonl", semantic_graph.get("evidence", [])),
        ("relationship_mentions.jsonl", semantic_graph.get("relationship_mentions", [])),
        ("attribute_facts.jsonl", semantic_graph.get("attribute_facts", [])),
    ):
        (output_dir / artifact_name).write_text(
            "\n".join(json.dumps(item, ensure_ascii=False) for item in values),
            encoding="utf-8",
        )
    for artifact_name, values in (
        ("relationship_links.json", semantic_graph.get("relationship_links", [])),
        ("communities.json", semantic_graph.get("communities", [])),
        ("community_reports.json", semantic_graph.get("community_reports", [])),
    ):
        (output_dir / artifact_name).write_text(
            json.dumps(values, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    if settings.qdrant_url:
        # Establish native-module import order before Torch on Windows.
        import qdrant_client  # noqa: F401

    embedding_texts = [
        "\n".join(
            filter(
                None,
                [chunk.doc_type, chunk.chapter, chunk.section, " ".join(chunk.knowledge_tags), chunk.text],
            )
        )
        for chunk in chunks
    ]
    report(68, "embedding", f"正在生成 {len(chunks)} 个内容向量")
    embeddings = encode_texts(
        embedding_model_path,
        embedding_texts,
        batch_size=32,
        show_progress_bar=True,
    )
    report(82, "validation", "正在校验向量与图谱完整性")
    validation = validate_build_artifacts(chunks, embeddings, legacy_graph)
    validation["semantic_graph"] = validate_build_artifacts(
        chunks, embeddings, semantic_graph
    )
    validation["section_semantics"] = section_quality
    import faiss

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    # FAISS' Windows file writer cannot open paths containing Chinese characters.
    # Serialize in memory and let Python handle the Unicode path instead.
    serialized_index = faiss.serialize_index(index)
    (output_dir / "vectors.faiss").write_bytes(serialized_index.tobytes())

    report(85, "indexing", "正在写入向量索引")
    qdrant_status = build_qdrant_indexes(output_dir, chunks, embeddings)
    neo4j_status = (
        sync_neo4j_graph(knowledge_base_id or output_dir.name, legacy_graph)
        if sync_graph_store
        else {"enabled": False, "reason": "deferred until atomic index activation"}
    )

    formula_processing = _formula_pipeline_stats(output_dir, elements)
    metadata = {
        "state": "populated",
        "schema_version": "2.3-circuit-image-retrieval",
        "resource_dir": str(resources_dir),
        "embedding_model": str(embedding_model_path),
        "dimension": int(embeddings.shape[1]),
        "documents": len(source_files),
        "text_pages": len(documents),
        "knowledge_units": len(knowledge_units),
        "ocr_pages": sum(
            isinstance(item.extra, dict) and bool(item.extra.get("ocr_processor"))
            for item in documents
        ),
        "questions": 0,
        "excluded_sources": excluded_sources,
        "chunks": len(chunks),
        "layout_elements": len(elements),
        "circuit_diagrams": sum(item.element_type == "circuit" for item in elements),
        "formula_elements": sum(item.element_type == "formula" for item in elements),
        "formula_processing": formula_processing,
        "table_elements": sum(item.element_type == "table" for item in elements),
        "discarded_pages": sum(not item.get("keep", True) for item in cleaning_audits),
        "knowledge_graph": {
            "nodes": len(legacy_graph["nodes"]),
            "edges": len(legacy_graph["edges"]),
            "chapters": len(chapter_summaries),
            "neo4j": neo4j_status,
        },
        "semantic_knowledge_graph": {
            "display_only": False,
            "provider": semantic_graph.get("stats", {}).get("extraction_method", "fallback"),
            "nodes": len(semantic_graph["nodes"]),
            "edges": len(semantic_graph["edges"]),
            "chapters": len(semantic_chapters),
            "entities": semantic_graph.get("stats", {}).get("entities", len(semantic_graph["nodes"])),
            "semantic_relationships": semantic_graph.get("stats", {}).get("relationships", len(semantic_graph["edges"])),
            "relationship_mentions": semantic_graph.get("stats", {}).get("relationship_mentions", 0),
            "text_units": semantic_graph.get("stats", {}).get("text_units", 0),
            "communities": semantic_graph.get("stats", {}).get("communities", 0),
        },
        "qdrant": qdrant_status,
        "vision_model": (
            f"qwen/{settings.qwen_circuit_vision_model}"
            if settings.qwen_api_key
            else "not-configured (safe fallback)"
        ),
        "pdf_extract_kit": (
            json.loads((output_dir / "pdf_extract_kit_manifest.json").read_text(encoding="utf-8"))
            if (output_dir / "pdf_extract_kit_manifest.json").exists()
            else {"enabled": False}
        ),
        "chapter_limit": chapter_limit,
        "sources": [path.name for path in source_files],
        "validation": validation,
        "extraction_quality": {
            **extraction_quality,
            **semantic_quality,
            "section_semantics": section_quality,
        },
    }
    metadata["pipeline_layers"] = {
        "document_cleaning": {
            "status": "ready",
            "pages_reviewed": len(cleaning_audits),
            "pages_discarded": metadata["discarded_pages"],
            "partial_characters_removed": sum(
                int(item.get("removed_characters", 0)) for item in cleaning_audits
            ),
            "question_banks_excluded": len(excluded_sources),
        },
        "document_parsing": {
            "status": "ready",
            "engine": "Qwen3-VL all-page OCR (cache-first) + PDF-Extract-Kit layout; PyMuPDF rendering only",
            "ocr_pages": metadata["ocr_pages"],
            "placeholder_text_chunks": extraction_quality["placeholder_text_chunks"],
            "layout_elements": len(elements),
            "preserves_page_bbox": True,
            "section_semantics": section_quality,
        },
        "modality_processing": {
            "status": "ready",
            "text_chunks": sum(chunk.element_type == "text" for chunk in chunks),
            "circuit_diagrams": metadata["circuit_diagrams"],
            "formula_elements": metadata["formula_elements"],
            "formula_processing": formula_processing,
            "table_elements": metadata["table_elements"],
            "circuit_vision_model": metadata["vision_model"],
        },
        "knowledge_fusion": {
            "status": "ready",
            "knowledge_document": "book_knowledge_document.json",
            "knowledge_units": len(knowledge_units),
            "graphrag_provider": semantic_graph.get("stats", {}).get("extraction_method", "fallback"),
            "graphrag_model": semantic_graph.get("stats", {}).get("extraction_model", "rule"),
            "embedding_model": Path(embedding_model_path).name,
            "vector_store": qdrant_status.get("mode", "faiss") if qdrant_status.get("enabled") else "faiss",
            "vector_points": len(chunks),
            "circuit_vector_points": qdrant_status.get("circuit_points", 0),
            "circuit_vector_store": (
                "qdrant+faiss"
                if qdrant_status.get("multimodal_qdrant_enabled")
                else "faiss"
                if qdrant_status.get("local_faiss_enabled")
                else "disabled"
            ),
            "graph_nodes": len(legacy_graph["nodes"]),
            "graph_edges": len(legacy_graph["edges"]),
            "semantic_preview_nodes": len(semantic_graph["nodes"]),
            "semantic_preview_edges": len(semantic_graph["edges"]),
            "chapter_summaries": len(chapter_summaries),
            "graph_store": "neo4j" if neo4j_status.get("enabled") else "local-json",
        },
        "retrieval_service": {
            "status": "ready",
            "strategies": [
                "vector", "BM25", "knowledge-graph", "rerank", "circuit-image"
            ],
            "question_bank_search": False,
            "circuit_image_min_score": settings.circuit_image_retrieval_min_score,
            "circuit_image_max_references": settings.circuit_image_retrieval_max_references,
        },
        "application": {
            "status": "ready",
            "context_modalities": ["text", "formula", "table", "circuit-description", "netlist", "image"],
        },
    }
    (output_dir / "pipeline_audit.json").write_text(
        json.dumps(metadata["pipeline_layers"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "index_meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report(88, "artifacts_ready", "构建产物已生成，等待原子切换")
    logger.info("Knowledge base populated: %s", metadata)
    return metadata
