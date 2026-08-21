from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import math
import mimetypes
import os
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable
from uuid import NAMESPACE_URL, uuid5

import fitz
import httpx

from backend.app.config import settings
from backend.app.rag.pdf_extract_kit import DetectedRegion, PDFExtractKitAdapter
from backend.app.rag.models import PageDocument, TextChunk
from backend.app.rag.section_titles import (
    normalize_numbered_section,
    normalize_structural_section,
    numbered_section_parts,
    section_matches_chapter,
    visible_numbered_sections,
    visible_structural_section,
)
from backend.app.rag.ontology import (
    COMPONENT_CONCEPTS,
    component_display_name,
    component_role,
    extract_course_concepts,
    extract_formula_concepts,
    is_course_concept,
    meaningful_section,
    normalize_concept_name,
)
from backend.app.rag.paddleocr_vl import (
    PADDLEOCR_VL_API_MODEL,
    PADDLEOCR_VL_GIT_REVISION,
    PADDLEOCR_VL_MODEL_ID,
    PADDLEOCR_VL_PIPELINE_VERSION,
    PADDLEOCR_VL_SCHEMA_VERSION,
    PaddleOCRVLClient,
    PaddleOCRVLInferenceError,
    markdown_table_cells,
    stable_block_content_hash,
)
from backend.app.services.qwen_multimodal_client import QwenMultimodalAPIError, QwenVisionClient


logger = logging.getLogger(__name__)

PARTIAL_NOISE_MARKERS = (
    "版权所有", "版权", "ISBN", "责任编辑", "封面设计", "版次", "印次", "出版社",
    "扫码", "公众号", "购买正版", "资源下载", "广告", "网址", "http://", "https://",
)

PAGE_CLEANING_POLICY_VERSION = "disabled-pass-through-v1"

SCANNED_PAGE_PLACEHOLDER = "[本页主要包含电路图、公式或其他图形内容]"
PAGE_OCR_SCHEMA_VERSION = PADDLEOCR_VL_SCHEMA_VERSION
LEGACY_PAGE_OCR_SCHEMA_VERSIONS: set[str] = set()
CIRCUIT_ANALYSIS_SCHEMA_VERSION = "2.1-grounded-circuit-family"
VISUAL_SUMMARY_SCHEMA_VERSION = "1.1-qwen3.7-json-repair"

CHAPTER_MARKER_PATTERN = r"第[零〇一二三四五六七八九十百两0-9]+章"
CHAPTER_SENTENCE_PREFIXES = (
    "中", "里", "内", "的", "介绍", "已经", "我们", "可以", "给出", "所述",
)
CHAPTER_SENTENCE_FRAGMENTS = (
    "如图", "所示", "我们已经", "可以看成", "介绍了", "给出了", "已经知道",
)
SECTION_SENTENCE_FRAGMENTS = (
    "如图", "所示", "试求", "试画", "试分析", "已知", "求出", "判断",
    "可得", "因此", "所以", "其中", "这时", "由此",
)
CHAPTER_EXERCISE_CONCEPT_PATTERN = re.compile(
    r"[？?]|图题\s*\d|判断下列|回答下列|试证明|哪些能够|怎样用|电路.*所示"
)

OCR_NON_CONCEPTS = {
    "模拟电子技术", "模拟电子技术基础", "常用半导体器件", "基本放大电路",
    "本章讨论的问题", "本章小结", "问题", "公式", "图形", "图示", "教材",
    "材料", "物质", "元件", "器件", "电路", "电流", "电压", "电子",
}


def _safe_partial_noise_fragment(fragment: str) -> bool:
    lowered = fragment.lower()
    return any(marker.lower() in lowered for marker in PARTIAL_NOISE_MARKERS)


@dataclass(frozen=True)
class BuildModelConfig:
    """Model profile used only for one background knowledge-base build."""

    provider: str = "deepseek"
    model: str = ""
    api_key: str = field(default="", repr=False)
    base_url: str = ""
    enable_thinking: bool | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.model and self.base_url and (self.api_key or self.provider == "ollama"))


@dataclass
class LayoutElement:
    id: str
    source: str
    page: int
    element_type: str
    bbox: list[float]
    text: str = ""
    image_path: str | None = None
    parent_id: str | None = None
    reading_order: int = 0
    chapter: str = ""
    section: str = ""
    caption: str = ""
    nearby_text: str = ""
    content_hash: str = ""
    components: list[dict[str, Any]] = field(default_factory=list)
    nets: list[dict[str, Any]] = field(default_factory=list)
    netlist: str = ""
    description: str = ""
    confidence: float = 0.0
    processor: str = "ocr-layout"
    uncertain: bool = False
    source_page: int | None = None
    polygon: list[list[float]] = field(default_factory=list)
    ocr_block_id: str | None = None
    evidence_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_visual_element_cache(
    paths: Iterable[Path], source: str
) -> dict[str, dict[str, Any]]:
    cached: dict[str, dict[str, Any]] = {}
    for cache_path in paths:
        if not cache_path.exists():
            continue
        try:
            lines = cache_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                item = json.loads(line)
            except (TypeError, ValueError, json.JSONDecodeError):
                # A process can stop during the final append. Earlier complete
                # page records remain usable and must not be discarded.
                continue
            if (
                isinstance(item, dict)
                and item.get("source") == source
                and item.get("image_path")
                and item.get("content_hash")
            ):
                cached[str(item["content_hash"])] = item
    return cached


def _append_visual_element_checkpoint(
    path: Path,
    elements: Iterable[LayoutElement],
    *,
    source: str,
    page: int,
) -> int:
    records = [
        element.to_dict()
        for element in elements
        if element.source == source
        and element.page == page
        and element.element_type in {"image", "circuit", "table"}
        and element.image_path
        and element.content_hash
    ]
    if not records:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(item, ensure_ascii=False) + "\n" for item in records
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return len(records)


def _json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        try:
            from json_repair import repair_json

            value = json.loads(repair_json(text))
        except Exception:
            match = re.search(r"\{.*\}", text, flags=re.S)
            if not match:
                return {}
            try:
                value = json.loads(match.group(0))
            except json.JSONDecodeError:
                return {}
    return value if isinstance(value, dict) else {}


class CompatibleMultimodalClient:
    """Small synchronous OpenAI-compatible client for offline ingestion workers."""

    def __init__(self, config: BuildModelConfig) -> None:
        self.config = config
        base_url = config.base_url.rstrip("/")
        if config.provider == "ollama" and not base_url.endswith("/v1"):
            base_url += "/v1"
        self.endpoint = f"{base_url}/chat/completions"

    def complete_json(
        self,
        prompt: str,
        *,
        image_bytes: bytes | None = None,
        image_mime: str = "image/png",
    ) -> dict[str, Any]:
        if not self.config.enabled:
            return {}
        content: str | list[dict[str, Any]] = prompt
        if image_bytes:
            content = [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{image_mime};base64,{base64.b64encode(image_bytes).decode('ascii')}"
                    },
                },
            ]
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": 4096,
            "response_format": {"type": "json_object"},
        }
        if self.config.enable_thinking is not None:
            payload["enable_thinking"] = self.config.enable_thinking
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        try:
            with httpx.Client(timeout=httpx.Timeout(180, connect=15)) as client:
                response = client.post(self.endpoint, headers=headers, json=payload)
                if response.status_code == 400:
                    payload.pop("response_format", None)
                    response = client.post(self.endpoint, headers=headers, json=payload)
                response.raise_for_status()
            choices = response.json().get("choices") or []
            raw = choices[0].get("message", {}).get("content", "") if choices else ""
            return _json_object(str(raw))
        except Exception as exc:
            logger.warning("Multimodal model call failed and will degrade safely: %s", exc)
            return {}


def _page_cleaning_decisions(
    pages: list[PageDocument], client: CompatibleMultimodalClient | None
) -> dict[int, dict[str, Any]]:
    decisions: dict[int, dict[str, Any]] = {}
    inside_exercises = False
    exercise_heading = re.compile(
        r"^(?:习题|复习题|思考题|自测题|练习题|部分习题参考答案)(?:\s|$)"
    )
    for page in sorted(pages, key=lambda item: item.page):
        lines = [
            re.sub(r"\s+", " ", line).strip()
            for line in page.text.splitlines()
            if line.strip()
        ]
        if lines and (
            _normalize_chapter_heading(lines[0])
            or re.match(r"^附录\s+[^0-9]", lines[0])
        ):
            inside_exercises = False
        if any(exercise_heading.match(line) for line in lines[:8]):
            inside_exercises = True
        decisions[page.page] = {
            "page": page.page,
            "source_page": page.source_page or page.page,
            "keep": not inside_exercises,
            "page_type": "exercise" if inside_exercises else "course_content",
            "reason": (
                "规则识别为连续课后习题区间"
                if inside_exercises else "默认保留课程内容"
            ),
            "method": "rule",
            "cleaning_policy_version": PAGE_CLEANING_POLICY_VERSION,
            "remove_fragments": [],
        }
    if not client or not client.config.enabled:
        return decisions
    for start in range(0, len(pages), 12):
        batch = pages[start : start + 12]
        samples = "\n\n".join(
            f"<PAGE number=\"{item.page}\">\n{item.text[:900]}\n</PAGE>" for item in batch
        )
        result = client.complete_json(
            """你是电路教材清洗器。判断每页是否属于可用于教学问答的有效课程内容。
将页面分为 course_content、exercise、noise 三类。正文、带完整讲解或解答过程的例题、公式、表格、电路图、目录和章节导读属于 course_content。
独立的课后习题、复习题、思考题、自测题、练习题以及仅提供这些题目答案的页面属于 exercise；即使包含课程概念、公式或电路图，也必须将 keep 设为 false。不要把带讲解或解答过程的例题误判为 exercise。
封面、版权/出版信息、空白页、广告、二维码和下载说明、与课程无关的目录、序言或噪声文本属于 noise，并将 keep 设为 false。
若页面包含少量与课程无关的版本说明、广告或页眉噪音但同时有技术正文，必须保留页面，
并在 remove_fragments 中逐字列出需要删除的短片段；不得删除公式、图题、例题或技术段落。
返回 JSON：{"decisions":[{"page":1,"page_type":"course_content|exercise|noise","keep":true,"reason":"...","remove_fragments":["原文片段"]}]}，不得改写页码。\n"""
            + samples
        )
        for item in result.get("decisions", []):
            if not isinstance(item, dict):
                continue
            try:
                page_no = int(item.get("page"))
            except (TypeError, ValueError):
                continue
            if page_no in decisions:
                fragments = item.get("remove_fragments", [])
                if not isinstance(fragments, list):
                    fragments = []
                keep = item.get("keep", True)
                if not isinstance(keep, bool):
                    keep = str(keep).strip().lower() not in {"false", "0", "no"}
                page_type = str(item.get("page_type", "")).strip().lower()
                if page_type not in {"course_content", "exercise", "noise"}:
                    page_type = "course_content" if keep else "noise"
                if page_type == "exercise":
                    keep = False
                decisions[page_no] = {
                    "page": page_no,
                    "source_page": next(
                        (page.source_page or page.page for page in pages if page.page == page_no),
                        page_no,
                    ),
                    "keep": keep,
                    "page_type": page_type,
                    "reason": str(item.get("reason", "模型语义清洗"))[:240],
                    "method": f"llm:{client.config.provider}/{client.config.model}",
                    "cleaning_policy_version": PAGE_CLEANING_POLICY_VERSION,
                    "requested_remove_fragments": [
                        str(fragment).strip()
                        for fragment in fragments[:12]
                        if 4 <= len(str(fragment).strip()) <= 500
                    ],
                }
                decisions[page_no]["remove_fragments"] = [
                    fragment
                    for fragment in decisions[page_no]["requested_remove_fragments"]
                    if _safe_partial_noise_fragment(fragment)
                ]
                page_text = next(
                    (page.text for page in pages if page.page == page_no), ""
                )
                if (
                    not decisions[page_no]["keep"]
                    and decisions[page_no]["page_type"] != "exercise"
                    and (
                        extract_course_concepts(page_text)
                        or re.search(r"[=+−±√∫ΣΩπ^_]", page_text)
                    )
                ):
                    decisions[page_no]["keep"] = True
                    decisions[page_no]["reason"] = (
                        "模型建议丢弃，但检测到课程概念或公式，安全策略强制保留"
                    )
    return decisions


def _element_id(source: str, page: int, order: int, content: bytes | str) -> tuple[str, str]:
    raw = content if isinstance(content, bytes) else content.encode("utf-8", errors="ignore")
    digest = hashlib.sha256(raw).hexdigest()
    return uuid5(NAMESPACE_URL, f"{source}|{page}|{order}|{digest}").hex, digest


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _ocr_text(value: Any) -> str:
    if isinstance(value, list):
        lines = [
            str(item.get("text", "")).strip() if isinstance(item, dict) else str(item).strip()
            for item in value
            if (
                str(item.get("text", "")).strip()
                if isinstance(item, dict)
                else str(item).strip()
            )
        ]
        return "\n".join(lines)
    return str(value or "").strip()


OCR_BLOCK_TYPES = {
    "chapter_heading", "section_heading", "paragraph", "list_item", "formula",
    "figure_caption", "table", "image", "exercise", "page_header", "page_footer", "noise",
}


def _ocr_blocks(value: Any) -> list[dict[str, Any]]:
    """Normalize OCR layout blocks while preserving the model's reading order."""

    if not isinstance(value, list):
        return []
    blocks: list[dict[str, Any]] = []
    for index, raw in enumerate(value, 1):
        if isinstance(raw, dict):
            text = str(raw.get("text", "")).strip()
            block_type = str(raw.get("type", "paragraph")).strip().lower()
            bbox = _normalized_heading_bbox(raw.get("bbox"))
            try:
                reading_order = max(1, int(raw.get("reading_order", index)))
            except (TypeError, ValueError):
                reading_order = index
        else:
            text = str(raw).strip()
            block_type = "paragraph"
            bbox = []
            reading_order = index
        if not text and block_type != "image":
            continue
        if block_type not in OCR_BLOCK_TYPES:
            block_type = "paragraph"
        block = {
            "id": str(raw.get("id", f"ocr-block-{index}")) if isinstance(raw, dict) else f"ocr-block-{index}",
            "type": block_type,
            "text": text,
            "bbox": bbox,
            "reading_order": reading_order,
        }
        if isinstance(raw, dict):
            for field_name in (
                "polygon",
                "confidence",
                "model_reading_order",
                "raw_label",
                "source_engine",
                "model_revision",
                "corrections",
                "uncertain",
                "content_hash",
            ):
                if field_name in raw:
                    block[field_name] = raw[field_name]
        block.setdefault("content_hash", stable_block_content_hash(block))
        blocks.append(block)
    return sorted(blocks, key=lambda item: int(item["reading_order"]))


def _legacy_ocr_blocks(text: str) -> list[dict[str, Any]]:
    """Promote legacy page text to layout blocks without inventing geometry."""

    blocks: list[dict[str, Any]] = []
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer
        paragraph = "".join(buffer).strip()
        if paragraph:
            blocks.append({
                "type": "paragraph",
                "text": paragraph,
                "bbox": [],
                "reading_order": len(blocks) + 1,
            })
        buffer = []

    for raw_line in str(text).splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            flush()
            continue
        if re.match(r"^第[零〇一二三四五六七八九十百两0-9]+章", line):
            flush()
            block_type = "chapter_heading"
        elif re.match(r"^\*?\d+(?:\.\d+){1,3}\s*\S+", line) and len(line) < 90:
            flush()
            block_type = "section_heading"
        elif re.match(r"^(?:图|表)(?:题)?\s*\d", line):
            flush()
            block_type = "figure_caption"
        elif len(line) < 180 and re.search(r"[=≈≠≤≥∑∫√]", line):
            flush()
            block_type = "formula"
        else:
            buffer.append(line)
            if line.endswith(("。", "！", "？", "；", ":", "：")):
                flush()
            continue
        blocks.append({
            "type": block_type,
            "text": line,
            "bbox": [],
            "reading_order": len(blocks) + 1,
        })
    flush()
    return _ocr_blocks(blocks)


def _normalize_chapter_heading(value: Any) -> str:
    """Return a real chapter heading, rejecting prose and answer-index labels."""

    heading = re.sub(r"\s+", " ", str(value or "")).strip()
    heading = re.sub(r"\s*[.．·…]{3,}.*$", "", heading).strip()
    heading = re.sub(r"\s*[（(]\s*\d+\s*[）)]\s*$", "", heading).strip()
    if re.fullmatch(CHAPTER_MARKER_PATTERN, heading):
        return heading
    match = re.fullmatch(
        rf"(?P<marker>{CHAPTER_MARKER_PATTERN})\s*(?P<title>.+)",
        heading,
    )
    if not match:
        return ""
    title = match.group("title").strip(" -—_:：、")
    compact_title = re.sub(r"\s+", "", title)
    if not 2 <= len(compact_title) <= 32:
        return ""
    if compact_title.startswith(CHAPTER_SENTENCE_PREFIXES):
        return ""
    if any(fragment in compact_title for fragment in CHAPTER_SENTENCE_FRAGMENTS):
        return ""
    if re.search(r"[，,。；;！？!?]", title):
        return ""
    return f"{match.group('marker')} {title}"


def _chapter_marker(value: str) -> str:
    match = re.match(CHAPTER_MARKER_PATTERN, value)
    return match.group(0) if match else ""


def _special_chapter_heading(lines: list[str], previous_chapter: str) -> str:
    compact_lead = re.sub(r"\s+", "", "".join(lines[:8]))
    if "目录" in compact_lead:
        return previous_chapter or "目录"
    for line in lines[:8]:
        if re.fullmatch(r"部分习题参考答案", line):
            return "部分习题参考答案"
        if re.fullmatch(r"参考文献", line):
            return "参考文献"
        if re.match(r"^附录(?:\s|\d|[一二三四五六七八九十])", line):
            if previous_chapter.startswith("附录"):
                return previous_chapter
            if re.match(r"^附录\s+[^0-9]", line):
                return line[:48].strip()
            return "附录"
    return ""


def _normalize_section_heading(value: Any) -> str:
    return normalize_numbered_section(value)


def _section_number(value: str) -> tuple[int, ...]:
    return numbered_section_parts(value)[0]


def _visible_section_headings(text: str) -> list[str]:
    return visible_numbered_sections(text)


def _section_transition_allowed(
    candidate: str,
    previous_section: str,
    *,
    chapter_changed: bool,
) -> bool:
    """Reject page-header regressions while allowing real forward section changes."""

    if not candidate or not previous_section or chapter_changed:
        return bool(candidate)
    candidate_structural = normalize_structural_section(candidate)
    previous_structural = normalize_structural_section(previous_section)
    if candidate_structural:
        return candidate_structural != previous_structural
    if previous_structural:
        return False
    candidate_number = _section_number(candidate)
    previous_number = _section_number(previous_section)
    if not candidate_number or not previous_number:
        return True
    if candidate_number == previous_number:
        return candidate == previous_section
    if (
        len(candidate_number) <= len(previous_number)
        and previous_number[: len(candidate_number)] == candidate_number
    ):
        return False
    return candidate_number > previous_number


def _ocr_heading_context_details(
    value: dict[str, Any],
    text: str,
    previous_chapter: str,
    previous_section: str,
    *,
    verified_section: str = "",
    heading_verification_attempted: bool = False,
) -> tuple[str, str, str]:
    """Resolve chapter/section using only page-grounded or independently verified text."""

    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines() if line.strip()]
    special_chapter = _special_chapter_heading(lines, previous_chapter)
    if special_chapter:
        return (
            special_chapter,
            previous_section if special_chapter == previous_chapter else "",
            "inherited" if special_chapter == previous_chapter else "special-section",
        )

    visible_chapter_lines = [
        (line_index, chapter)
        for line_index, line in enumerate(lines)
        if (chapter := _normalize_chapter_heading(line))
    ]
    visible_chapters = [chapter for _, chapter in visible_chapter_lines]
    compact_lead = re.sub(r"\s+", "", "".join(lines[:6]))
    contents_entry_count = sum(
        len(re.findall(r"(?:[.．·…]{3,}|[（(]\s*\d+\s*[）)])", line))
        for line in lines
    )
    numbered_entry_count = sum(
        bool(re.match(r"^\*?\d+(?:\.\d+)+\s*\S", line)) for line in lines
    )
    structural_section = visible_structural_section(text)
    is_contents_page = not structural_section and (
        "目录" in compact_lead
        or len({_chapter_marker(chapter) for chapter in visible_chapters}) >= 2
        or contents_entry_count >= 5
        or numbered_entry_count >= 8
    )
    if is_contents_page:
        return previous_chapter, previous_section, "inherited"
    if (
        previous_chapter == "目录"
        and (not visible_chapter_lines or visible_chapter_lines[0][0] > 1)
    ):
        return previous_chapter, previous_section, "inherited"

    raw_chapter = _normalize_chapter_heading(value.get("chapter", ""))
    candidate = visible_chapters[0] if visible_chapters else raw_chapter
    if candidate and _chapter_marker(candidate) == _chapter_marker(previous_chapter):
        chapter = previous_chapter or candidate
    elif visible_chapters:
        chapter = candidate
    elif raw_chapter and re.sub(r"\s+", "", raw_chapter) in re.sub(r"\s+", "", "".join(lines[:4])):
        chapter = raw_chapter
    else:
        chapter = previous_chapter
    chapter_changed = bool(chapter and chapter != previous_chapter)
    if chapter_changed:
        previous_section = ""

    if structural_section:
        return chapter, structural_section, "structural-heading"

    visible_sections = _visible_section_headings(text)
    raw_section = _normalize_section_heading(value.get("section", ""))
    verified = _normalize_section_heading(verified_section)
    page_candidate = next(
        (
            candidate
            for candidate in reversed(visible_sections)
            if section_matches_chapter(candidate, chapter)
        ),
        "",
    )
    if verified:
        verified_number = _section_number(verified)
        candidate_number = _section_number(page_candidate or raw_section)
        if candidate_number and candidate_number == verified_number:
            page_candidate = verified
            source = "heading-crop"
        else:
            source = "page-text" if page_candidate else "inherited"
    elif heading_verification_attempted:
        page_candidate = ""
        source = "inherited"
    else:
        source = "page-text" if page_candidate else "inherited"

    if page_candidate and not _section_transition_allowed(
        page_candidate,
        previous_section,
        chapter_changed=chapter_changed,
    ):
        page_candidate = ""
        source = "inherited"
    section = page_candidate or previous_section
    return chapter, section, source


def _ocr_heading_context(
    value: dict[str, Any],
    text: str,
    previous_chapter: str,
    previous_section: str,
    *,
    verified_section: str = "",
    heading_verification_attempted: bool = False,
) -> tuple[str, str]:
    chapter, section, _ = _ocr_heading_context_details(
        value,
        text,
        previous_chapter,
        previous_section,
        verified_section=verified_section,
        heading_verification_attempted=heading_verification_attempted,
    )
    return chapter, section


def _ocr_concepts(value: Any, text: str) -> list[str]:
    if not isinstance(value, list):
        return []
    compact_text = re.sub(r"\s+", "", text).lower()
    concepts: list[str] = []
    for item in value:
        concept = normalize_concept_name(str(item))
        compact = re.sub(r"\s+", "", concept).lower()
        if (
            not (2 <= len(concept) <= 24)
            or concept in OCR_NON_CONCEPTS
            or not re.search(r"[\u4e00-\u9fffA-Za-z]", concept)
            or compact not in compact_text
        ):
            continue
        if concept not in concepts:
            concepts.append(concept)
    return concepts[:18]


def _write_page_ocr_cache(path: Path, entries: dict[int, dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "\n".join(
            json.dumps(entries[page], ensure_ascii=False)
            for page in sorted(entries)
        ),
        encoding="utf-8",
    )
    for attempt in range(6):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            # Windows readers and antivirus scanners can briefly hold the old
            # JSONL file open. Keep the fully written temporary file and retry
            # the atomic swap instead of losing the just-completed OCR page.
            if attempt == 5:
                raise
            time.sleep(0.1 * (attempt + 1))


def _page_ocr_cache_identity_is_compatible(
    item: dict[str, Any],
    expected_identity: dict[str, Any],
) -> bool:
    """Reuse pinned PaddleOCR-VL 1.6 pages across local/API providers.

    Runtime fields remain attached to every page as audit evidence. They are not
    content invalidators when both sides are the pinned local revision or the
    official PaddleOCR-VL 1.6 API, which lets an interrupted local build finish
    through the API without discarding already completed OCR pages.
    """

    if not expected_identity:
        return True
    identity_fields = ("model_revision", "engine", "dtype", "pipeline_version")
    if all(
        str(item.get(field_name, "")) == str(expected_identity.get(field_name, ""))
        for field_name in identity_fields
    ):
        return True

    compatible_revisions = {
        PADDLEOCR_VL_GIT_REVISION,
        f"aistudio-api:{PADDLEOCR_VL_API_MODEL}",
    }
    return (
        str(item.get("model_revision", "")) in compatible_revisions
        and str(expected_identity.get("model_revision", "")) in compatible_revisions
        and str(item.get("pipeline_version", "")) == PADDLEOCR_VL_PIPELINE_VERSION
        and str(expected_identity.get("pipeline_version", ""))
        == PADDLEOCR_VL_PIPELINE_VERSION
    )


def _reuse_near_complete_page_ocr_cache(
    pages: Iterable[PageDocument],
    entries: dict[int, dict[str, Any]],
) -> bool:
    """Avoid paid retries for a few pages missing from an otherwise complete cache."""

    scanned_pages = {
        page.page
        for page in pages
        if page.text.strip() == SCANNED_PAGE_PLACEHOLDER
    }
    if not scanned_pages:
        return False
    cached_pages = scanned_pages.intersection(entries)
    missing_pages = scanned_pages.difference(cached_pages)
    return (
        len(cached_pages) / len(scanned_pages) >= 0.99
        and len(missing_pages) <= 3
    )


def _normalized_heading_bbox(value: Any) -> list[float]:
    if not isinstance(value, list) or len(value) != 4:
        return []
    try:
        bbox = [float(item) for item in value]
    except (TypeError, ValueError):
        return []
    left, top, right, bottom = bbox
    if not (
        0 <= left < right <= 1000
        and 0 <= top < bottom <= 1000
        and right - left >= 20
        and bottom - top >= 8
    ):
        return []
    return bbox


def _retry_low_confidence_key_blocks(
    page: fitz.Page,
    blocks: list[dict[str, Any]],
    client: PaddleOCRVLClient | Any,
    *,
    source: str,
    page_number: int,
) -> list[dict[str, Any]]:
    """Retry only uncertain titles, formulas and tables with a sharper crop."""

    key_types = {"chapter_heading", "section_heading", "formula", "table"}
    page_width = max(1.0, float(page.rect.width))
    page_height = max(1.0, float(page.rect.height))
    retried: list[dict[str, Any]] = []
    for block in blocks:
        updated = {**block, "corrections": list(block.get("corrections", []))}
        block_type = str(block.get("type", ""))
        try:
            confidence = float(block.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        bbox = _normalized_heading_bbox(block.get("bbox"))
        if (
            block_type not in key_types
            or confidence >= settings.paddleocr_key_block_confidence
            or not bbox
        ):
            retried.append(updated)
            continue

        left, top, right, bottom = bbox
        horizontal_margin = max(8.0, (right - left) / 1000 * page_width * 0.06)
        vertical_margin = max(6.0, (bottom - top) / 1000 * page_height * 0.45)
        clip = fitz.Rect(
            max(0.0, left / 1000 * page_width - horizontal_margin),
            max(0.0, top / 1000 * page_height - vertical_margin),
            min(page_width, right / 1000 * page_width + horizontal_margin),
            min(page_height, bottom / 1000 * page_height + vertical_margin),
        )
        pixmap = page.get_pixmap(
            matrix=fitz.Matrix(settings.paddleocr_retry_scale, settings.paddleocr_retry_scale),
            clip=clip,
            alpha=False,
        )
        try:
            result = client.predict_page(
                pixmap.tobytes("png"),
                source=f"{source}#retry-{block.get('id', '')}",
                page=page_number,
            )
            candidates = _ocr_blocks(result.get("blocks", []))
        except PaddleOCRVLInferenceError as exc:
            if block_type in {"formula", "table"} and str(updated.get("text", "")).strip():
                updated["uncertain"] = True
                updated["corrections"].append({
                    "type": "paddle-high-resolution-retry-failed",
                    "previous_confidence": confidence,
                    "reason": str(exc),
                    "fallback": "original-paddle-block",
                })
                updated["content_hash"] = stable_block_content_hash(updated)
                retried.append(updated)
                continue
            raise PaddleOCRVLInferenceError(
                f"{source} 第 {page_number} 页的低置信度 {block_type} "
                f"高清重试失败：{exc}"
            ) from exc

        compatible_types = (
            {"chapter_heading", "section_heading"}
            if block_type in {"chapter_heading", "section_heading"}
            else {block_type}
        )
        compatible = [
            item
            for item in candidates
            if item.get("type") in compatible_types and str(item.get("text", "")).strip()
        ]
        if not compatible:
            compatible = [
                item
                for item in candidates
                if item.get("type") not in {"image", "page_header", "page_footer", "noise"}
                and str(item.get("text", "")).strip()
            ]
        candidate = max(
            compatible,
            key=lambda item: (
                float(item.get("confidence", 0.0) or 0.0),
                len(str(item.get("text", ""))),
            ),
            default=None,
        )
        if candidate is None:
            if block_type in {"formula", "table"} and str(updated.get("text", "")).strip():
                updated["uncertain"] = True
                updated["corrections"].append({
                    "type": "paddle-high-resolution-retry-failed",
                    "previous_confidence": confidence,
                    "reason": "retry-returned-no-compatible-content",
                    "fallback": "original-paddle-block",
                })
                updated["content_hash"] = stable_block_content_hash(updated)
                retried.append(updated)
                continue
            raise PaddleOCRVLInferenceError(
                f"{source} 第 {page_number} 页的低置信度 {block_type} "
                "高清重试未返回可用内容"
            )

        replacement_text = str(candidate.get("text", "")).strip()
        replacement_confidence = float(candidate.get("confidence", 0.0) or 0.0)
        if replacement_text:
            updated["text"] = replacement_text
        updated["confidence"] = max(confidence, replacement_confidence)
        updated["uncertain"] = updated["confidence"] < settings.paddleocr_key_block_confidence
        updated["corrections"].append({
            "type": "paddle-high-resolution-retry",
            "previous_confidence": confidence,
            "retry_confidence": replacement_confidence,
        })
        updated["content_hash"] = stable_block_content_hash(updated)
        retried.append(updated)
    return retried


def _normalize_toc_section_heading(value: str) -> str:
    heading = re.sub(r"\s+", " ", str(value or "")).strip()
    match = re.match(
        r"^(?P<number>\d{1,2}(?:\s*[.．]\s*\d{1,2}){1,3})\s+"
        r"(?P<title>.+?)\s*$",
        heading,
    )
    if not match:
        return ""
    title = re.sub(r"\s*[.．·…]{2,}\s*(?:[（(]?\d+[）)]?)?\s*$", "", match.group("title"))
    title = re.sub(r"\s+[（(]\s*\d+\s*[）)]\s*$", "", title).strip(" .．、:：-")
    compact_title = re.sub(r"\s+", "", title)
    if (
        not 2 <= len(compact_title) <= 60
        or not re.search(r"[\u4e00-\u9fff]", title)
        or re.search(r"[=；;！？!?]", title)
        or any(fragment in title for fragment in SECTION_SENTENCE_FRAGMENTS)
    ):
        return ""
    number = re.sub(r"\s*[.．]\s*", ".", match.group("number"))
    return f"{number} {title}"


def _toc_section_catalog(
    documents: Iterable[PageDocument],
) -> dict[str, str]:
    candidates: dict[str, Counter[str]] = {}
    inside_contents = False
    for document in sorted(documents, key=lambda item: item.page):
        lines = [re.sub(r"\s+", " ", line).strip() for line in document.text.splitlines() if line.strip()]
        compact_lead = re.sub(r"\s+", "", "".join(lines[:6]))
        if "目录" in compact_lead:
            inside_contents = True
        if inside_contents and lines:
            opening_chapter = _normalize_chapter_heading(lines[0])
            if (
                opening_chapter
                and not re.search(r"[.．·…]{2,}|[（(]\s*\d+\s*[）)]\s*$", lines[0])
                and any(len(line) >= 24 and not _normalize_toc_section_heading(line) for line in lines[1:5])
            ):
                inside_contents = False
        if not inside_contents:
            continue
        for line in lines:
            section = _normalize_toc_section_heading(line)
            if not section:
                continue
            number = section.split(" ", 1)[0]
            candidates.setdefault(number, Counter())[section] += 1

    catalog: dict[str, str] = {}
    for number, variants in candidates.items():
        ranked = variants.most_common()
        if len(ranked) == 1 or ranked[0][1] > ranked[1][1]:
            catalog[number] = ranked[0][0]
        elif len({variant for variant, _ in ranked}) == 1:
            catalog[number] = ranked[0][0]
    return catalog


def _replace_page_section_heading(text: str, canonical: str) -> str:
    canonical_number = _section_number(canonical)
    lines = text.splitlines()
    for index, line in enumerate(lines):
        visible = _normalize_section_heading(line)
        if visible and _section_number(visible) == canonical_number:
            lines[index] = canonical
            break
    return "\n".join(lines)


def _apply_toc_section_catalog(
    documents: list[PageDocument],
) -> list[PageDocument]:
    catalog = _toc_section_catalog(documents)
    if not catalog:
        return documents
    corrected: list[PageDocument] = []
    for document in documents:
        section = _normalize_section_heading(document.section)
        number = section.split(" ", 1)[0] if section else ""
        canonical = catalog.get(number, "")
        if (
            not canonical
            or canonical == section
            or document.chapter.startswith(("部分习题", "参考文献", "附录"))
        ):
            corrected.append(document)
            continue
        extra = {
            **(document.extra or {}),
            "ocr_section_raw": section,
            "ocr_section_source": "toc",
            "ocr_section_confidence": 1.0,
            "ocr_section_corrected": True,
            "ocr_toc_catalog_size": len(catalog),
        }
        corrected.append(replace(
            document,
            text=_replace_page_section_heading(document.text, canonical),
            section=canonical,
            extra=extra,
        ))
    return corrected


def _annotate_ocr_block_contexts(
    documents: list[PageDocument],
) -> list[PageDocument]:
    """Attach grounded chapter/section context to every Paddle layout block."""

    annotated: list[PageDocument] = []
    current_chapter = ""
    current_section = ""
    current_source = ""
    for document in sorted(documents, key=lambda item: (item.source, item.page)):
        if document.source != current_source:
            current_source = document.source
            current_chapter = ""
            current_section = ""
        extra = dict(document.extra or {})
        blocks = _ocr_blocks(extra.get("text_blocks", []))
        if not blocks:
            annotated.append(document)
            current_chapter = document.chapter or current_chapter
            current_section = document.section or current_section
            continue
        lines = [str(block.get("text", "")) for block in blocks if block.get("text")]
        compact_lead = re.sub(r"\s+", "", "".join(lines[:8]))
        contents_entries = sum(
            bool(_normalize_toc_section_heading(line)) for line in lines
        )
        is_contents = "目录" in compact_lead or contents_entries >= 5
        if not current_chapter:
            current_chapter = document.chapter
        if not current_section:
            current_section = document.section
        contextualized: list[dict[str, Any]] = []
        for block in blocks:
            value = {**block}
            text = str(value.get("text", "")).strip()
            if not is_contents and value.get("type") == "chapter_heading":
                candidate = _normalize_chapter_heading(text)
                if candidate:
                    current_chapter = candidate
                    current_section = ""
            elif not is_contents and value.get("type") == "section_heading":
                candidate = (
                    _normalize_section_heading(text)
                    or normalize_structural_section(text)
                )
                if candidate and (
                    normalize_structural_section(candidate)
                    or section_matches_chapter(candidate, current_chapter or document.chapter)
                ):
                    current_section = candidate
            value["chapter"] = current_chapter or document.chapter
            value["section"] = current_section or document.section
            contextualized.append(value)
        extra["text_blocks"] = contextualized
        annotated.append(replace(document, extra=extra))
        current_chapter = document.chapter or current_chapter
        current_section = document.section or current_section
    return annotated


def _is_full_page_scan(
    bbox: list[float],
    page_width: float,
    page_height: float,
    page_document: PageDocument,
) -> bool:
    if not (
        isinstance(page_document.extra, dict)
        and page_document.extra.get("ocr_processor")
        and len(bbox) == 4
    ):
        return False
    left, top, right, bottom = bbox
    image_area = max(0.0, right - left) * max(0.0, bottom - top)
    page_area = max(1.0, page_width * page_height)
    return image_area / page_area >= 0.8


def _ocr_scanned_pages(
    path: Path,
    pages: list[PageDocument],
    output_dir: Path,
    client: PaddleOCRVLClient | Any | None,
    document_hash: str,
    chapter_limit: int | None = None,
) -> list[PageDocument]:
    """Recover every PDF page with PaddleOCR-VL and durable block evidence."""

    def page_chapter(document: PageDocument) -> str:
        lines = [line.strip() for line in document.text.splitlines() if line.strip()]
        compact_lead = re.sub(r"\s+", "", "".join(lines[:6]))
        visible = [
            heading for line in lines[:8]
            if (heading := _normalize_chapter_heading(line))
        ]
        contents_entry_count = sum(
            len(re.findall(r"(?:[.．·…]{3,}|[（(]\s*\d+\s*[）)])", line))
            for line in lines
        )
        numbered_entry_count = sum(
            bool(re.match(r"^\*?\d+(?:\.\d+)+\s*\S", line)) for line in lines
        )
        if (
            document.chapter == "目录"
            or document.section == "目录"
            or "目录" in compact_lead
            or len({_chapter_marker(item) for item in visible}) >= 2
            or contents_entry_count >= 5
            or numbered_entry_count >= 8
        ):
            return ""
        normalized = _normalize_chapter_heading(document.chapter)
        if normalized:
            return normalized
        return visible[0] if visible else ""

    def limited_chapters(values: list[PageDocument]) -> list[PageDocument]:
        if not chapter_limit:
            return values
        starts: list[tuple[str, int]] = []
        for index, document in enumerate(values):
            chapter = page_chapter(document)
            marker = _chapter_marker(chapter)
            if marker and (not starts or starts[-1][0] != marker):
                starts.append((marker, index))
        if not starts:
            return values
        start = starts[0][1]
        end = starts[chapter_limit][1] if len(starts) > chapter_limit else len(values)
        return values[start:end]

    if not any(page.text.strip() == SCANNED_PAGE_PLACEHOLDER for page in pages):
        return limited_chapters(pages)
    cache_path = output_dir / f"{path.stem}.page_ocr.jsonl"
    cache_entries: dict[int, dict[str, Any]] = {}
    expected_identity = (
        dict(getattr(client, "cache_identity", {}) or {}) if client is not None else {}
    )
    if cache_path.exists():
        try:
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                item = json.loads(line)
                schema_version = str(item.get("schema_version", "")) if isinstance(item, dict) else ""
                if not (
                    isinstance(item, dict)
                    and schema_version == PAGE_OCR_SCHEMA_VERSION
                    and item.get("document_hash") == document_hash
                    and float(item.get("render_scale", 0) or 0)
                    == settings.paddleocr_render_scale
                    and isinstance(item.get("blocks"), list)
                    and item.get("blocks")
                ):
                    continue
                if not _page_ocr_cache_identity_is_compatible(item, expected_identity):
                    continue
                cache_entries[int(item["page"])] = item
        except (OSError, ValueError, json.JSONDecodeError):
            cache_entries = {}

    recovered: list[PageDocument] = []
    seen_chapter_markers: list[str] = []
    first_chapter_index: int | None = None
    previous_chapter = ""
    previous_section = ""
    document = fitz.open(path)
    try:
        for page_document in sorted(pages, key=lambda item: item.page):
            recovered_document: PageDocument | None = None
            if page_document.text.strip() != SCANNED_PAGE_PLACEHOLDER:
                previous_chapter = page_document.chapter or previous_chapter
                previous_section = page_document.section or previous_section
                recovered_document = page_document
            else:
                cached = cache_entries.get(page_document.page)
            if recovered_document is None and cached:
                cached_text = str(cached["text"]).strip()
                cached_blocks = _ocr_blocks(cached.get("blocks"))
                chapter, section, section_source = _ocr_heading_context_details(
                    cached,
                    cached_text,
                    previous_chapter,
                    previous_section,
                    verified_section=str(cached.get("section_verified", "")),
                    heading_verification_attempted=bool(
                        cached.get("heading_verification_attempted")
                    ),
                )
                concepts = [str(item) for item in cached.get("concepts", []) if str(item).strip()]
                try:
                    section_confidence = float(cached.get("section_confidence", 0.0))
                except (TypeError, ValueError):
                    section_confidence = 0.0
                previous_chapter, previous_section = chapter, section
                recovered_document = replace(
                    page_document,
                    text=cached_text,
                    chapter=chapter or page_document.chapter,
                    section=section or chapter or page_document.section,
                    extra={
                        **(page_document.extra or {}),
                        "ocr_concepts": concepts,
                        "ocr_processor": f"paddleocr-vl:{cached.get('model', PADDLEOCR_VL_MODEL_ID)}",
                        "ocr_schema_version": PAGE_OCR_SCHEMA_VERSION,
                        "ocr_runtime": {
                            field_name: cached.get(field_name)
                            for field_name in (
                                "model", "model_revision", "engine", "device", "dtype",
                                "pipeline_version", "render_scale",
                            )
                        },
                        "ocr_section_raw": str(cached.get("section_raw", "")),
                        "ocr_section_source": section_source,
                        "ocr_section_confidence": section_confidence,
                        "ocr_section_bbox": cached.get("section_bbox", []),
                        "ocr_heading_verification_attempted": bool(
                            cached.get("heading_verification_attempted")
                        ),
                        "text_blocks": cached_blocks,
                    },
                )
            elif recovered_document is None and client is None:
                raise RuntimeError(
                    f"{path.name} 第 {page_document.page} 页没有可用的 PaddleOCR-VL 缓存，"
                    "且当前构建未提供 PaddleOCR-VL 客户端"
                )

            if recovered_document is None:
                page = document[page_document.page - 1]
                width, height = max(1.0, float(page.rect.width)), max(1.0, float(page.rect.height))
                scale = min(
                    settings.paddleocr_render_scale,
                    2800 / max(width, height),
                    math.sqrt(7_000_000 / (width * height)),
                )
                pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
                image_bytes = pixmap.tobytes("png")
                try:
                    result = client.predict_page(
                        image_bytes,
                        source=path.name,
                        page=page_document.page,
                    )
                except PaddleOCRVLInferenceError:
                    # The build manager stages output separately, so a hard failure
                    # preserves the currently active index instead of degrading it.
                    raise
                blocks = _ocr_blocks(result.get("blocks"))
                blocks = _retry_low_confidence_key_blocks(
                    page,
                    blocks,
                    client,
                    source=path.name,
                    page_number=page_document.page,
                )
                visible_blocks = [
                    block
                    for block in blocks
                    if block.get("type") not in {"image", "page_header", "page_footer", "noise"}
                    and str(block.get("text", "")).strip()
                ]
                text = _ocr_text(visible_blocks)
                if not text:
                    text = "[PaddleOCR-VL未识别到正文，保留本页视觉版面证据]"
                chapter_blocks = [
                    block for block in blocks if block.get("type") == "chapter_heading"
                ]
                section_blocks = [
                    block for block in blocks if block.get("type") == "section_heading"
                ]
                raw_chapter = str(chapter_blocks[0].get("text", "")) if chapter_blocks else ""
                section_block = section_blocks[-1] if section_blocks else {}
                raw_section = str(section_block.get("text", ""))
                heading_verification_attempted = any(
                    any(
                        correction.get("type") == "paddle-high-resolution-retry"
                        for correction in block.get("corrections", [])
                        if isinstance(correction, dict)
                    )
                    for block in section_blocks
                )
                verified_section = (
                    _normalize_section_heading(raw_section)
                    if heading_verification_attempted
                    else ""
                )
                value = {
                    "chapter": raw_chapter,
                    "section": raw_section,
                    "section_bbox": section_block.get("bbox", []),
                }
                chapter, section, section_source = _ocr_heading_context_details(
                    value,
                    text,
                    previous_chapter,
                    previous_section,
                    verified_section=verified_section,
                    heading_verification_attempted=heading_verification_attempted,
                )
                concepts = _ocr_concepts(extract_course_concepts(text), text)
                try:
                    detected_section_confidence = float(
                        section_block.get("confidence", 0.0) or 0.0
                    )
                except (TypeError, ValueError):
                    detected_section_confidence = 0.0
                section_confidence = (
                    detected_section_confidence
                    if section_source in {"heading-crop", "page-text"}
                    else 0.7 if section else 0.0
                )
                previous_chapter, previous_section = chapter, section
                identity = {
                    "model": getattr(client, "model", PADDLEOCR_VL_MODEL_ID),
                    "model_revision": PADDLEOCR_VL_GIT_REVISION,
                    "engine": settings.paddleocr_engine,
                    "device": settings.paddleocr_device,
                    "dtype": settings.paddleocr_dtype,
                    "pipeline_version": settings.paddleocr_pipeline_version,
                    **expected_identity,
                }
                cache_entries[page_document.page] = {
                    "schema_version": PAGE_OCR_SCHEMA_VERSION,
                    "document_hash": document_hash,
                    **identity,
                    "render_scale": settings.paddleocr_render_scale,
                    "page": page_document.page,
                    "source_page": page_document.source_page or page_document.page,
                    "text": text,
                    "blocks": blocks,
                    "raw_result": result.get("raw", {}),
                    "chapter": chapter,
                    "section": section,
                    "section_raw": _normalize_section_heading(raw_section),
                    "section_verified": verified_section,
                    "heading_verification_attempted": heading_verification_attempted,
                    "section_source": section_source,
                    "section_confidence": section_confidence,
                    "section_bbox": _normalized_heading_bbox(section_block.get("bbox")),
                    "concepts": concepts,
                }
                _write_page_ocr_cache(cache_path, cache_entries)
                recovered_document = replace(
                    page_document,
                    text=text,
                    chapter=chapter or page_document.chapter,
                    section=section or chapter or page_document.section,
                    extra={
                        **(page_document.extra or {}),
                        "ocr_concepts": concepts,
                        "ocr_processor": f"paddleocr-vl:{identity['model']}",
                        "ocr_schema_version": PAGE_OCR_SCHEMA_VERSION,
                        "ocr_runtime": {**identity, "render_scale": settings.paddleocr_render_scale},
                        "ocr_section_raw": _normalize_section_heading(raw_section),
                        "ocr_section_source": section_source,
                        "ocr_section_confidence": section_confidence,
                        "ocr_section_bbox": _normalized_heading_bbox(section_block.get("bbox")),
                        "ocr_heading_verification_attempted": heading_verification_attempted,
                        "text_blocks": blocks,
                    },
                )

            assert recovered_document is not None
            chapter = page_chapter(recovered_document)
            marker = _chapter_marker(chapter)
            if marker and (not seen_chapter_markers or seen_chapter_markers[-1] != marker):
                if chapter_limit and len(seen_chapter_markers) >= chapter_limit:
                    break
                seen_chapter_markers.append(marker)
                if first_chapter_index is None:
                    first_chapter_index = len(recovered)
            recovered.append(recovered_document)
    finally:
        document.close()
    if chapter_limit and first_chapter_index is not None:
        recovered = recovered[first_chapter_index:]
    return _annotate_ocr_block_contexts(_apply_toc_section_catalog(recovered))


def _looks_like_formula(text: str) -> bool:
    if len(text) > 500 or not text.strip():
        return False
    math_chars = sum(char in "=+-±×÷√∫ΣΩμφλπ^_<>" for char in text)
    return math_chars >= 2 and bool(re.search(r"[A-Za-z0-9]", text))


def _looks_like_table(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    return len(lines) >= 3 and sum(bool(re.search(r"\s{2,}|\t", line)) for line in lines) >= 2


def _overlapping_text(
    target_bbox: list[float], text_blocks: list[tuple[list[float], str]]
) -> str:
    left, top, right, bottom = target_bbox
    matches: list[str] = []
    for bbox, text in text_blocks:
        block_left, block_top, block_right, block_bottom = bbox
        horizontal = min(right, block_right) - max(left, block_left)
        vertical = min(bottom, block_bottom) - max(top, block_top)
        if horizontal > 0 and vertical > 0:
            matches.append(text)
    return "\n".join(matches).strip()


def _bbox_iou_points(first: list[float], second: list[float]) -> float:
    if len(first) != 4 or len(second) != 4:
        return 0.0
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    if intersection <= 0:
        return 0.0
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    return intersection / max(1.0, first_area + second_area - intersection)


def _matching_layout_element(
    elements: Iterable[LayoutElement],
    *,
    source: str,
    page: int,
    element_type: str,
    bbox: list[float],
) -> LayoutElement | None:
    candidates = [
        element
        for element in elements
        if element.source == source
        and element.page == page
        and element.element_type == element_type
        and _bbox_iou_points(element.bbox, bbox) >= 0.25
    ]
    return max(
        candidates,
        key=lambda element: _bbox_iou_points(element.bbox, bbox),
        default=None,
    )


def _table_fact_description(markdown: str, block_id: str) -> tuple[str, list[dict[str, Any]]]:
    cells = markdown_table_cells(markdown, block_id=block_id)
    if not cells:
        return "", []
    facts: list[str] = []
    for cell in cells:
        row_header = str(cell.get("row_header", "")).strip()
        column_header = str(cell.get("column_header", "")).strip()
        value = str(cell.get("value", "")).strip()
        if not value or (value == row_header and int(cell.get("column", 0)) == 0):
            continue
        label = " / ".join(part for part in (row_header, column_header) if part)
        fact = f"{label}：{value}" if label else value
        if fact not in facts:
            facts.append(fact)
    return "；".join(facts[:80]), cells


def _summary_numbers_are_grounded(summary: str, evidence: str) -> bool:
    """Reject new Arabic numeric claims that are absent from Paddle evidence."""

    number_pattern = r"(?<![A-Za-z0-9_])[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:\s*[%℃℉])?"
    summary_numbers = {
        re.sub(r"\s+", "", item) for item in re.findall(number_pattern, summary)
    }
    evidence_numbers = {
        re.sub(r"\s+", "", item) for item in re.findall(number_pattern, evidence)
    }
    return summary_numbers.issubset(evidence_numbers)


def _summarize_table(
    element: LayoutElement,
    image_bytes: bytes,
    client: QwenVisionClient | None,
) -> None:
    """Summarize table meaning without allowing the VLM to transcribe cells."""

    paddle_text = str(element.text).strip()
    if client is None or not paddle_text or not _image_is_safe(image_bytes):
        return
    prompt = f"""你只负责总结教材表格表达的知识，不负责 OCR、单元格转写或公式识别。
下方 PaddleOCR-VL Markdown 是表格文字和数值的唯一事实来源；图片只用于理解表头层级、行列分组和视觉强调。
不得增加 Markdown 中没有的数值、单位、公式或因果关系；公式单元格只可逐字引用，不得推导、改写。
若表格与当前课程无关，令 is_course_relevant=false 并留空 summary。

章节：{element.chapter} / {element.section}
邻近正文：{element.nearby_text[:1200]}
PaddleOCR-VL 表格 Markdown：
{paddle_text[:12000]}

返回 JSON：{{"is_course_relevant":true,"summary":"只总结表格的比较维度、变化趋势或关键结论","knowledge_points":["表格直接支持的知识点"],"confidence":0.0}}。
""".strip()
    summary = ""
    summary_confidence = 0.0
    rejection_reason = ""
    for attempt in range(2):
        attempt_prompt = prompt
        if attempt:
            attempt_prompt += (
                f"\n\n上一次结构化总结未通过证据校验（{rejection_reason}）。"
                "请重新输出完整 JSON；summary 和 knowledge_points 不得写表号、图号、式号，"
                "所有数值必须逐字存在于 PaddleOCR-VL Markdown 中。"
            )
        try:
            raw = client.complete_json(
                attempt_prompt,
                image_bytes=image_bytes,
                image_mime=mimetypes.guess_type(element.image_path or "table.png")[0]
                or "image/png",
            )
        except QwenMultimodalAPIError as exc:
            logger.warning(
                "Qwen visual table summary failed; retaining Paddle evidence: %s",
                exc,
            )
            element.evidence_metadata["visual_summary_fallback_reason"] = str(exc)
            return
        relevant_value = raw.get("is_course_relevant", raw.get("summary"))
        relevant = (
            relevant_value.strip().lower() in {"true", "1", "yes", "是"}
            if isinstance(relevant_value, str)
            else bool(relevant_value)
        )
        if not relevant:
            element.evidence_metadata["is_course_relevant"] = False
            return
        summary = str(raw.get("summary", "")).strip()[:4000]
        points = [
            str(item).strip() for item in raw.get("knowledge_points", [])
            if str(item).strip()
        ][:12] if isinstance(raw.get("knowledge_points"), list) else []
        if points:
            point_text = "；".join(points)
            summary = (
                f"{summary}\n表格知识点：{point_text}"
                if summary else f"表格知识点：{point_text}"
            )
        try:
            summary_confidence = max(
                0.0, min(1.0, float(raw.get("confidence", 0)))
            )
        except (TypeError, ValueError):
            summary_confidence = 0.0
        if not summary:
            rejection_reason = "summary 为空"
        elif summary_confidence < 0.45:
            rejection_reason = f"confidence={summary_confidence:.2f}"
        elif not _summary_numbers_are_grounded(summary, paddle_text):
            rejection_reason = "总结含 Paddle 表格证据之外的数值或编号"
        else:
            rejection_reason = ""
            break
    if rejection_reason:
        element.evidence_metadata.update({
            "visual_summary_fallback_reason": rejection_reason,
            "visual_summary_schema": VISUAL_SUMMARY_SCHEMA_VERSION,
            "visual_summary_model": client.model,
        })
        logger.warning(
            "Qwen visual table summary rejected after retry; retaining Paddle evidence: %s",
            rejection_reason,
        )
        return
    # Rebuild facts from Paddle Markdown so a refreshed summary can never carry
    # an older vision-generated description forward as if it were cell evidence.
    paddle_facts, table_cells = _table_fact_description(paddle_text, element.id)
    element.evidence_metadata["table_cells"] = table_cells
    element.description = (
        f"视觉总结：{summary}\n可核验单元格事实：{paddle_facts}"
        if paddle_facts
        else f"视觉总结：{summary}"
    )
    processor = _visual_processor(client, "table")
    if processor and processor not in element.processor:
        element.processor += f"+{processor}"
    element.evidence_metadata.update({
        "visual_summary_schema": VISUAL_SUMMARY_SCHEMA_VERSION,
        "visual_summary": summary,
        "visual_summary_confidence": summary_confidence,
        "summary_grounded_in": "paddleocr-vl-table-markdown",
    })


def _localized_nearby_text(
    target_bbox: list[float],
    text_blocks: list[tuple[list[float], str]],
    *,
    vertical_margin: float = 72.0,
) -> str:
    """Return nearby prose without attaching the whole page to every element."""

    left, top, right, bottom = target_bbox
    candidates: list[tuple[float, str]] = []
    for bbox, text in text_blocks:
        block_left, block_top, block_right, block_bottom = bbox
        horizontal_overlap = min(right, block_right) - max(left, block_left)
        same_column = horizontal_overlap > 0 or not (
            block_right < left - 48 or block_left > right + 48
        )
        vertical_gap = max(0.0, top - block_bottom, block_top - bottom)
        if same_column and vertical_gap <= vertical_margin:
            candidates.append((vertical_gap, text))
    return "\n".join(text for _, text in sorted(candidates, key=lambda item: item[0])[:3])[:1800]


def _nearby_text_block_ids(
    target_bbox: list[float],
    text_blocks: list[tuple[list[float], str]],
    block_records: list[dict[str, Any]],
    *,
    vertical_margin: float = 72.0,
) -> list[str]:
    if len(target_bbox) != 4:
        return []
    left, top, right, bottom = target_bbox
    candidates: list[tuple[float, str]] = []
    for index, (bbox, _text) in enumerate(text_blocks):
        if index >= len(block_records) or len(bbox) != 4:
            continue
        block_left, block_top, block_right, block_bottom = bbox
        horizontal_overlap = min(right, block_right) - max(left, block_left)
        same_column = horizontal_overlap > 0 or not (
            block_right < left - 48 or block_left > right + 48
        )
        vertical_gap = max(0.0, top - block_bottom, block_top - bottom)
        block_id = str(block_records[index].get("id", ""))
        if same_column and vertical_gap <= vertical_margin and block_id:
            candidates.append((vertical_gap, block_id))
    return [
        block_id
        for _, block_id in sorted(candidates, key=lambda item: item[0])[:8]
    ]


_FORMULA_NUMBER_PATTERN = re.compile(
    r"[（(]\s*((?:\d+\.)+\d+[A-Za-z]?)\s*[)）]\s*$"
)


def _formula_latex_from_ocr_text(text: str) -> str:
    """Convert a conservative OCR equation subset into searchable LaTeX."""

    value = _FORMULA_NUMBER_PATTERN.sub("", text).strip().strip("$；;，,")
    if "=" not in value:
        return ""
    value = value.replace("−", "-").replace("×", r"\cdot ").replace("*", " ")
    value = value.replace("β", r"\beta ").replace("α", r"\alpha ")
    value = value.replace("γ", r"\gamma ").replace("Δ", r"\Delta ")
    value = re.sub(r"\b([A-Za-z])_([A-Za-z0-9]+)\b", r"\1_{\2}", value)
    left, right = (part.strip() for part in value.split("=", 1))
    fraction = re.fullmatch(r"\(?\s*(.+?)\s*\)?\s*/\s*([^/]+)", right)
    if fraction:
        numerator = fraction.group(1).strip()
        denominator = fraction.group(2).strip()
        right = rf"\frac{{{numerator}}}{{{denominator}}}"
    latex = re.sub(r"\s+", " ", f"{left} = {right}").strip()
    return latex if re.search(r"[A-Za-z0-9]", latex) else ""


def _formula_candidates_from_page_text(text: str) -> list[dict[str, str]]:
    """Recover display equations and their printed numbers from page OCR order."""

    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines() if line.strip()]
    candidates: list[dict[str, str]] = []
    for index, line in enumerate(lines):
        if "=" not in line or len(line) > 240:
            continue
        # Prose can contain assignments such as “令 ui = 0”; display equations on
        # these textbooks are overwhelmingly Latin/Greek mathematical lines.
        if len(re.findall(r"[\u4e00-\u9fff]", line)) > 3:
            continue
        number_match = _FORMULA_NUMBER_PATTERN.search(line)
        if number_match is None and index + 1 < len(lines):
            number_match = _FORMULA_NUMBER_PATTERN.fullmatch(lines[index + 1])
        caption = f"({number_match.group(1)})" if number_match else ""
        plain_text = _FORMULA_NUMBER_PATTERN.sub("", line).strip()
        latex = _formula_latex_from_ocr_text(plain_text)
        if latex:
            candidates.append(
                {"plain_text": plain_text, "latex": latex, "caption": caption}
            )
    return candidates


def _normalize_formula_result(value: dict[str, Any], fallback_text: str) -> dict[str, Any]:
    raw_is_formula = value.get("is_formula", bool(value.get("latex") or fallback_text))
    is_formula = (
        raw_is_formula.strip().lower() in {"true", "1", "yes", "是"}
        if isinstance(raw_is_formula, str)
        else bool(raw_is_formula)
    )
    latex = str(value.get("latex", "")).strip().strip("$")
    plain_text = str(value.get("plain_text", "")).strip()
    if not latex:
        plain_text = plain_text or re.sub(r"\s+", "", fallback_text)
    try:
        confidence = max(0.0, min(1.0, float(value.get("confidence", 0))))
    except (TypeError, ValueError):
        confidence = 0.0
    valid_content = latex or plain_text
    if len(valid_content) > 600 or not re.search(r"[A-Za-z0-9α-ωΑ-Ω=]", valid_content):
        is_formula = False
    return {
        "is_formula": is_formula,
        "latex": latex,
        "plain_text": plain_text,
        "variables": value.get("variables", []) if isinstance(value.get("variables"), list) else [],
        "knowledge": str(value.get("knowledge", "")).strip(),
        "confidence": confidence,
    }


def _formula_symbol_skeleton(latex: str) -> str:
    """Return a conservative symbol-order signature for OCR/VL reconciliation."""

    value = re.sub(r"\\(?:bar|overline)\s*", "", latex)
    value = re.sub(r"\\(?:frac|left|right|cdot|times)\b", "", value)
    value = value.replace(r"\beta", "beta")
    value = value.replace(r"\alpha", "alpha").replace(r"\gamma", "gamma")
    return re.sub(r"[^A-Za-z0-9]", "", value).casefold()


def _reconcile_formula_with_page_ocr(
    result: dict[str, Any], ocr_formula: dict[str, str]
) -> tuple[dict[str, Any], str]:
    """Correct VL typography when page OCR independently confirms the symbols."""

    vl_latex = str(result.get("latex", "")).strip()
    ocr_latex = str(ocr_formula.get("latex", "")).strip()
    if not vl_latex or not ocr_latex or vl_latex == ocr_latex:
        return result, ""
    if _formula_symbol_skeleton(vl_latex) != _formula_symbol_skeleton(ocr_latex):
        return result, ""
    reconciled = dict(result)
    reconciled["latex"] = ocr_latex
    if ocr_formula.get("plain_text"):
        reconciled["plain_text"] = ocr_formula["plain_text"]
    return reconciled, "matching-symbol-skeleton"


def _nearest_formula_caption_region(
    formula_region: DetectedRegion,
    caption_regions: list[DetectedRegion],
) -> DetectedRegion | None:
    if not caption_regions:
        return None
    left, top, right, bottom = formula_region.bbox_pixels
    center_y = (top + bottom) / 2
    height = max(1.0, bottom - top)
    eligible: list[tuple[float, DetectedRegion]] = []
    for caption in caption_regions:
        cap_left, cap_top, _cap_right, cap_bottom = caption.bbox_pixels
        cap_center_y = (cap_top + cap_bottom) / 2
        cap_height = max(1.0, cap_bottom - cap_top)
        vertical_gap = abs(center_y - cap_center_y)
        if vertical_gap > max(24.0, 1.5 * max(height, cap_height)):
            continue
        horizontal_gap = max(0.0, cap_left - right, left - cap_left)
        eligible.append((vertical_gap + 0.05 * horizontal_gap, caption))
    return min(eligible, key=lambda item: item[0])[1] if eligible else None


def _indexable_pdfkit_regions(regions: list[DetectedRegion]) -> list[DetectedRegion]:
    """Keep structural regions and display formulas, never one node per inline symbol."""

    layout_formulas = [
        region
        for region in regions
        if region.category.lower() == "isolate_formula"
        and region.detector.endswith(":layout")
    ]
    selected: list[DetectedRegion] = []
    for region in regions:
        category = region.category.lower()
        if category in {"figure", "table"}:
            selected.append(region)
        elif category == "isolate_formula":
            selected.append(region)
        elif category in {"isolated", "isolated_formula"} and not layout_formulas:
            selected.append(region)
    return selected


def _circuit_image_heuristic(image_bytes: bytes) -> tuple[bool, float]:
    try:
        import cv2
        import numpy as np

        image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape[0] * image.shape[1] < settings.multimodal_min_image_area:
            return False, 0.0
        edges = cv2.Canny(image, 60, 160)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 35, minLineLength=18, maxLineGap=7)
        line_count = 0 if lines is None else len(lines)
        density = float((edges > 0).mean())
        score = min(0.85, 0.15 + line_count / 80 + min(density, 0.12) * 2)
        return line_count >= 6 and 0.008 <= density <= 0.35, score
    except Exception:
        return False, 0.0


def _image_is_safe(image_bytes: bytes) -> bool:
    if not image_bytes or len(image_bytes) > 25 * 1024 * 1024:
        return False
    try:
        from PIL import Image

        with Image.open(io.BytesIO(image_bytes)) as image:
            width, height = image.size
        return 0 < width <= 12000 and 0 < height <= 12000 and width * height <= 40_000_000
    except Exception:
        return False


def _crop_png(image_bgr: Any, bbox: list[float]) -> bytes:
    import cv2

    height, width = image_bgr.shape[:2]
    left = max(0, min(width - 1, int(math.floor(bbox[0]))))
    top = max(0, min(height - 1, int(math.floor(bbox[1]))))
    right = max(left + 1, min(width, int(math.ceil(bbox[2]))))
    bottom = max(top + 1, min(height, int(math.ceil(bbox[3]))))
    crop = image_bgr[top:bottom, left:right]
    ok, encoded = cv2.imencode(".png", crop)
    if not ok:
        return b""
    return encoded.tobytes()


def _normalize_circuit_result(value: dict[str, Any]) -> dict[str, Any]:
    components = [
        item for item in value.get("components", [])
        if isinstance(item, dict)
    ] if isinstance(value.get("components"), list) else []
    nets = [
        item for item in value.get("nets", [])
        if isinstance(item, dict)
    ] if isinstance(value.get("nets"), list) else []
    try:
        confidence = max(0.0, min(1.0, float(value.get("confidence", 0))))
    except (TypeError, ValueError):
        confidence = 0.0
    raw_is_circuit = value.get("is_circuit", components or nets or value.get("netlist"))
    is_circuit = (
        raw_is_circuit.strip().lower() in {"true", "1", "yes", "是"}
        if isinstance(raw_is_circuit, str)
        else bool(raw_is_circuit)
    )
    raw_course_relevant = value.get(
        "is_course_relevant",
        is_circuit
        or value.get("summary")
        or value.get("description")
        or value.get("knowledge_points"),
    )
    is_course_relevant = is_circuit or (
        raw_course_relevant.strip().lower() in {"true", "1", "yes", "是"}
        if isinstance(raw_course_relevant, str)
        else bool(raw_course_relevant)
    )
    visual_type = str(value.get("visual_type", "circuit" if is_circuit else "other"))
    visual_type = visual_type.strip().lower()[:80] or "other"
    summary = str(value.get("summary", value.get("description", ""))).strip()[:5000]
    knowledge_points = [
        str(item).strip() for item in value.get("knowledge_points", [])
        if str(item).strip()
    ][:12] if isinstance(value.get("knowledge_points"), list) else []
    circuit_type = str(value.get("circuit_type", "")).strip()[:200]
    grounding_quotes = [
        str(item).strip() for item in value.get("grounding_quotes", [])
        if str(item).strip()
    ][:8] if isinstance(value.get("grounding_quotes"), list) else []
    contradictions = [
        str(item).strip() for item in value.get("contradictions", [])
        if str(item).strip()
    ][:8] if isinstance(value.get("contradictions"), list) else []
    description = str(value.get("description") or summary).strip()[:8000]
    if not is_circuit and knowledge_points:
        point_text = "；".join(knowledge_points)
        description = (
            f"{description}\n图示知识点：{point_text}" if description else f"图示知识点：{point_text}"
        )
    if circuit_type and circuit_type not in description:
        description = f"电路类型：{circuit_type}。{description}"
    if grounding_quotes:
        description = (
            f"{description}\n类型判定依据：" + "；".join(grounding_quotes)
        ).strip()
    if not is_course_relevant:
        description = ""
    if contradictions:
        confidence = min(confidence, 0.69)
    # Always serialize from the structured component list. This prevents a VLM
    # from silently inserting numeric values in an otherwise correct raw netlist.
    netlist = _synthesize_netlist(components) if components else ""
    return {
        "is_circuit": is_circuit,
        "is_course_relevant": is_course_relevant,
        "visual_type": visual_type,
        "knowledge_points": knowledge_points,
        "components": components,
        "nets": nets,
        "netlist": netlist,
        "description": description,
        "caption": str(value.get("caption", ""))[:1000],
        "circuit_type": circuit_type,
        "grounding_quotes": grounding_quotes,
        "contradictions": contradictions,
        "confidence": confidence,
    }


def _circuit_processor(client: QwenVisionClient | None) -> str:
    return (
        f"{_visual_processor(client, 'image')}+circuit-{CIRCUIT_ANALYSIS_SCHEMA_VERSION}"
        if client else ""
    )


def _visual_processor(client: QwenVisionClient | None, kind: str) -> str:
    return (
        f"qwen-vl:{client.model}:{kind}-summary-{VISUAL_SUMMARY_SCHEMA_VERSION}"
        if client else ""
    )


def _visual_cache_compatible(
    cached: dict[str, Any] | None,
    client: QwenVisionClient | None,
    kind: str,
) -> bool:
    if not cached:
        return False
    expected = _visual_processor(client, kind)
    processor = str(cached.get("processor", ""))
    evidence_metadata = cached.get("evidence_metadata", {})
    if not isinstance(evidence_metadata, dict):
        evidence_metadata = {}
    if (
        kind == "table"
        and evidence_metadata.get("visual_summary_fallback_reason")
        and evidence_metadata.get("visual_summary_schema")
        == VISUAL_SUMMARY_SCHEMA_VERSION
        and evidence_metadata.get("visual_summary_model")
        == str(getattr(client, "model", ""))
    ):
        return True
    if expected and expected not in processor:
        return False
    if kind == "image" and str(cached.get("element_type", "")) == "circuit":
        return f"circuit-{CIRCUIT_ANALYSIS_SCHEMA_VERSION}" in processor
    return True


def _ground_circuit_family(
    value: dict[str, Any],
    *,
    caption: str,
    nearby_text: str,
) -> dict[str, Any]:
    """Apply deterministic topology guards to high-risk textbook circuit families."""

    result = dict(value)
    components = [
        item for item in result.get("components", []) if isinstance(item, dict)
    ]
    component_types = [str(item.get("type", "")).lower() for item in components]
    bjt_count = sum(
        item in {"bjt", "npn", "pnp", "bipolar_junction_transistor"}
        for item in component_types
    )
    resistor_ids = {
        str(item.get("id", "")).replace("_", "").upper()
        for item in components
        if str(item.get("type", "")).lower() == "resistor"
    }
    context = f"{caption}\n{nearby_text}"
    description = str(result.get("description", ""))
    circuit_type = str(result.get("circuit_type", ""))
    classification = f"{circuit_type}\n{description}".lower()

    if (
        ("威尔逊" in classification or "wilson" in classification)
        and bjt_count < 3
    ):
        microcurrent_grounded = (
            bjt_count == 2
            and any(identifier.startswith("RE") for identifier in resistor_ids)
            and "微电流源" in context
            and re.search(r"微电流源.{0,30}图\s*2\.6\.8", context, re.S)
        )
        replacement = "微电流源" if microcurrent_grounded else "两晶体管电流源（具体类型待核验）"
        if microcurrent_grounded:
            description = (
                "电路类型：微电流源。图内确认两只晶体管构成镜像支路，"
                "输出支路的发射极经 R_E 接地，参考支路发射极直接接地。"
                "正文明确说明将带射极电阻的镜像电流源中 R_E1 短路便构成"
                "图 2.6.8 所示的微电流源，与图内拓扑一致。"
            )
        else:
            description = re.sub(
                r"[^。；\n]*(?:威尔逊|Wilson)[^。；\n]*[。；]?",
                "",
                description,
                flags=re.I,
            ).strip()
            description = (
                f"电路类型校验：{replacement}。图内仅确认 {bjt_count} 只晶体管，"
                "不满足三管反馈拓扑，因此不保留原类型名称。" + description
            )
        result["circuit_type"] = replacement
        result["description"] = description
        result["contradictions"] = [
            *result.get("contradictions", []),
            "Wilson 名称与图内晶体管数量冲突，已按拓扑降级",
        ]

    has_current_source = any(item in {"current_source", "isource"} for item in component_types)
    has_voltage_source = any(item in {"voltage_source", "vsource"} for item in component_types)
    if "戴维南" in description and has_current_source and not has_voltage_source:
        description = description.replace("戴维南等效模型", "诺顿型电流源等效模型")
        description = description.replace("戴维南模型", "诺顿型电流源模型")
        result["circuit_type"] = "诺顿型电流源等效模型"
        result["description"] = description
    return result


def audit_visual_semantics(elements: Iterable[LayoutElement]) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    circuits = 0
    course_images = 0
    table_summaries = 0
    formula_vision_violations = 0
    for element in elements:
        if element.element_type == "formula" and "qwen-vl:" in element.processor:
            formula_vision_violations += 1
            issues.append({
                "severity": "critical",
                "code": "formula_processed_by_vision_model",
                "element_id": element.id,
                "page": int(element.source_page or element.page),
            })
        if (
            element.element_type == "table"
            and element.evidence_metadata.get("visual_summary")
        ):
            table_summaries += 1
        if (
            element.element_type in {"image", "circuit"}
            and element.evidence_metadata.get("is_course_relevant")
            and element.description
        ):
            course_images += 1
        if element.element_type != "circuit":
            continue
        circuits += 1
        component_types = [
            str(item.get("type", "")).lower()
            for item in element.components if isinstance(item, dict)
        ]
        bjt_count = sum(
            item in {"bjt", "npn", "pnp", "bipolar_junction_transistor"}
            for item in component_types
        )
        description = str(element.description)
        if (
            ("威尔逊" in description or "wilson" in description.lower())
            and bjt_count < 3
        ):
            issues.append({
                "severity": "critical",
                "code": "wilson_without_three_transistors",
                "element_id": element.id,
                "page": int(element.source_page or element.page),
            })
        if (
            "戴维南" in description
            and any(item in {"current_source", "isource"} for item in component_types)
            and not any(item in {"voltage_source", "vsource"} for item in component_types)
        ):
            issues.append({
                "severity": "critical",
                "code": "current_source_mislabeled_thevenin",
                "element_id": element.id,
                "page": int(element.source_page or element.page),
            })
        if element.uncertain:
            issues.append({
                "severity": "warning",
                "code": "uncertain_circuit_semantics",
                "element_id": element.id,
                "page": int(element.source_page or element.page),
            })
    critical = sum(issue["severity"] == "critical" for issue in issues)
    warnings = sum(issue["severity"] == "warning" for issue in issues)
    return {
        "schema_version": VISUAL_SUMMARY_SCHEMA_VERSION,
        "status": "failed" if critical else "passed",
        "circuits": circuits,
        "course_image_summaries": course_images,
        "table_summaries": table_summaries,
        "formula_vision_violations": formula_vision_violations,
        "critical_issues": critical,
        "warning_issues": warnings,
        "issues": issues,
    }


def _is_verified_circuit_result(value: dict[str, Any]) -> bool:
    """Require auditable topology before promoting a figure to a circuit node."""

    components = value.get("components") if isinstance(value.get("components"), list) else []
    components = [item for item in components if isinstance(item, dict)]
    if len(components) < 2:
        return False

    try:
        if float(value.get("confidence", 0)) < 0.7:
            return False
    except (TypeError, ValueError):
        return False

    component_types = {
        str(item.get("type", "")).strip().lower() for item in components
    }
    # These describe a system/block diagram, not an electrical schematic whose
    # component connectivity can be audited.
    if component_types & {"black_box", "microphone", "speaker"}:
        return False

    description = str(value.get("description", "")).lower()
    explicit_non_circuit = (
        "各元件独立，无连接点",
        "未绘制实际电路元件",
        "无spice可建模",
        "未绘制具体电路元件",
        "仅展示其特性曲线",
        "示意框图",
    )
    if any(marker in description for marker in explicit_non_circuit):
        return False

    terminal_counts: dict[str, int] = {}
    for component in components:
        terminals = component.get("terminals", [])
        if not isinstance(terminals, list):
            continue
        for terminal in terminals:
            node = str(terminal).strip()
            if node:
                terminal_counts[node] = terminal_counts.get(node, 0) + 1
    has_shared_node = any(count >= 2 for count in terminal_counts.values())

    # A few valid composite textbook figures are described as a circuit even
    # when the VLM omits a shared node from its structured output. Keep those
    # only when the short caption/description explicitly says circuit/model.
    summary = (
        str(value.get("caption", "")) + "\n" + str(value.get("description", ""))[:180]
    ).lower()
    explicitly_circuit = "电路" in summary or "等效模型" in summary or "通路" in summary
    return has_shared_node or explicitly_circuit


def _enforce_verified_circuit(element: LayoutElement) -> None:
    if element.element_type != "circuit":
        return
    if _is_verified_circuit_result({
        "components": element.components,
        "nets": element.nets,
        "description": element.description,
        "caption": element.caption,
        "confidence": element.confidence,
    }):
        return
    element.element_type = "image"
    element.components = []
    element.nets = []
    element.netlist = ""
    element.uncertain = False


def _synthesize_netlist(components: list[dict[str, Any]]) -> str:
    """Create an auditable SPICE-like fallback without inventing values."""

    prefixes = {
        "resistor": "R",
        "capacitor": "C",
        "inductor": "L",
        "diode": "D",
        "voltage_source": "V",
        "current_source": "I",
        "bipolar_junction_transistor": "Q",
        "bjt": "Q",
        "npn": "Q",
        "pnp": "Q",
        "mosfet": "M",
        "vsource": "V",
        "isource": "I",
    }
    lines = ["* Generated from Qwen3-VL structured detection; UNKNOWN means unreadable."]
    for position, component in enumerate(components, 1):
        component_type = str(component.get("type", "component")).strip().lower()
        raw_id = str(component.get("id", "")).strip()
        terminals = component.get("terminals", [])
        nodes = [str(item).strip() for item in terminals if str(item).strip()]
        if component_type == "port":
            lines.append(" ".join(["* PORT", raw_id or str(position), *nodes]))
            continue
        prefix = prefixes.get(component_type, "X")
        identifier = raw_id or f"{prefix}{position}"
        if not identifier.upper().startswith(prefix):
            identifier = f"{prefix}{identifier}"
        if not nodes:
            nodes = ["UNKNOWN_NODE"]
        value = str(component.get("value") or "UNKNOWN").strip()
        if value.lower() in {"null", "none", "n/a", "unknown"}:
            value = "UNKNOWN"
        lines.append(" ".join([identifier, *nodes, value]))
    return "\n".join(lines)


def _analyze_image(
    element: LayoutElement,
    image_bytes: bytes,
    client: QwenVisionClient | None,
) -> None:
    likely, heuristic_score = _circuit_image_heuristic(image_bytes)
    prompt = f"""你负责总结教材中与课程内容相关的图片，包括电路图、半导体结构图、器件实物或外形、特性曲线、波形图、系统框图和其他教学插图。
你不负责页面文字或公式转写；公式由 PaddleOCR-VL 独占识别，不得根据图片改写、推导或补全公式。
只描述图中可见信息与邻近正文直接支持的课程知识；出版社标识、二维码、装饰图和与课程无关的照片令 is_course_relevant=false。

只有包含至少两个电气元件且存在可核验导线连接的原理图、等效电路或小信号模型才可令 is_circuit=true。器件实物/外形、单个器件符号、物理结构、特性曲线、波形图和系统框图必须令 is_circuit=false，但与课程相关时仍需总结其含义。
电路类型必须有“图内拓扑 + 图题/邻近正文”双重证据，不得从其他图补入元件。电流源与电阻并联是诺顿/电流源模型，不是戴维南模型；戴维南模型必须是电压源与电阻串联。Wilson 电流源必须确认第三只晶体管的反馈拓扑。若含 (a)/(b) 等子图，必须分别描述，不得合并拓扑。
若是电路图，识别可核验的元件、端口、节点和导线连接；跨线但无连接点不得当作连接，看不清的值写 null。

章节：{element.chapter} / {element.section}
邻近正文：{element.nearby_text[:1800]}
返回 JSON：{{"is_course_relevant":true,"visual_type":"circuit|characteristic_curve|waveform|physical_structure|device_photo|system_block_diagram|illustration|other","caption":"","summary":"图片所表达的课程知识总结","knowledge_points":["可核验知识点"],"grounding_quotes":["图内/图题可核验的短证据"],"is_circuit":false,"circuit_type":"","contradictions":[],"components":[],"nets":[],"description":"与 summary 一致；电路图可补充拓扑与功能","confidence":0.0}}。"""
    try:
        raw_vlm_result = (
            client.complete_json(
                prompt,
                image_bytes=image_bytes,
                image_mime=mimetypes.guess_type(element.image_path or "figure.png")[0] or "image/png",
            )
            if client
            else {}
        )
    except QwenMultimodalAPIError as exc:
        logger.warning("Qwen visual image summary failed; retaining localized evidence: %s", exc)
        raw_vlm_result = {}
    vlm_result = _ground_circuit_family(
        _normalize_circuit_result(raw_vlm_result),
        caption=element.caption,
        nearby_text=element.nearby_text,
    )
    result = vlm_result
    nearby_lower = f"{element.caption}\n{element.nearby_text}".lower()
    chart_markers = ("波形", "曲线", "坐标", "频谱", "特性图")
    heuristic_circuit = likely and not any(marker in nearby_lower for marker in chart_markers)
    # The local edge/line heuristic is deliberately not authoritative. Crystal
    # lattices, device cross-sections and characteristic plots are visually
    # similar to schematics and previously became false circuit nodes whenever
    # the vision endpoint timed out.
    is_circuit = bool(result.get("is_circuit"))
    course_relevant = bool(result.get("is_course_relevant"))
    if is_circuit:
        element.element_type = "circuit"
        element.components = result.get("components", [])
        element.nets = result.get("nets", [])
        element.netlist = result.get("netlist", "")
        element.description = result.get("description") or (
            "检测到疑似电路原理图；专用识别服务未返回可验证的元件与连接关系。"
        )
        element.caption = result.get("caption", "")
        element.confidence = float(result.get("confidence") or heuristic_score)
        qwen_processor = (
            _circuit_processor(client)
            if vlm_result.get("is_circuit") and client
            else "opencv-heuristic"
        )
        element.processor = (
            f"{element.processor}+{qwen_processor}"
            if element.processor.startswith("pdf-extract-kit")
            else qwen_processor
        )
        element.uncertain = not bool(element.components and (element.nets or element.netlist))
    else:
        element.element_type = "image"
        element.components = []
        element.nets = []
        element.netlist = ""
        element.description = result.get("description", "") if course_relevant else ""
        element.caption = result.get("caption", "")
        element.confidence = float(result.get("confidence") or heuristic_score)
        visual_processor = _visual_processor(client, "image") if raw_vlm_result else ""
        if visual_processor:
            element.processor = (
                f"{element.processor}+{visual_processor}"
                if element.processor and element.processor != "ocr-layout"
                else visual_processor
            )
        if heuristic_circuit and not raw_vlm_result:
            heuristic_processor = "opencv-heuristic-unconfirmed"
            element.processor = (
                f"{element.processor}+{heuristic_processor}"
                if element.processor.startswith("pdf-extract-kit")
                else heuristic_processor
            )
            element.uncertain = True
    element.evidence_metadata.update({
        "visual_summary_schema": VISUAL_SUMMARY_SCHEMA_VERSION,
        "is_course_relevant": course_relevant,
        "visual_type": str(result.get("visual_type", "")),
        "visual_knowledge_points": list(result.get("knowledge_points", [])),
        "visual_grounding_quotes": list(result.get("grounding_quotes", [])),
    })
    _enforce_verified_circuit(element)


def _external_pdf_extract_elements(path: Path) -> list[dict[str, Any]]:
    """Read normalized PDF-Extract-Kit/MinerU JSON when a worker exported it.

    Keeping parsing out-of-process avoids forcing its large GPU dependency set
    into the FastAPI environment. Accepted files are ``<stem>.json`` under
    ``PDF_EXTRACT_KIT_OUTPUT_DIR`` and contain either a list or ``elements``.
    """

    if not settings.pdf_extract_kit_output_dir:
        return []
    candidate = Path(settings.pdf_extract_kit_output_dir) / f"{path.stem}.json"
    if not candidate.exists():
        return []
    try:
        value = json.loads(candidate.read_text(encoding="utf-8"))
        items = value.get("elements", value.get("content_list", [])) if isinstance(value, dict) else value
        return items if isinstance(items, list) else []
    except Exception as exc:
        logger.warning("Cannot read PDF-Extract-Kit output %s: %s", candidate, exc)
        return []


def enhance_pdf(
    path: Path,
    page_documents: list[PageDocument],
    output_dir: Path,
    *,
    model_config: BuildModelConfig | None = None,
    chapter_limit: int | None = None,
    ocr_client: PaddleOCRVLClient | Any | None = None,
) -> tuple[list[PageDocument], list[LayoutElement], list[dict[str, Any]]]:
    """Add layout, visual and circuit semantics while preserving original pages."""

    artifacts_dir = output_dir / "artifacts" / path.stem
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    vision_client = (
        QwenVisionClient(
            api_key=settings.qwen_api_key,
            model=settings.qwen_visual_summary_model,
            base_url=settings.qwen_base_url,
        )
        if settings.qwen_api_key
        else None
    )
    owns_ocr_client = False
    if (
        ocr_client is None
        and any(item.text.strip() == SCANNED_PAGE_PLACEHOLDER for item in page_documents)
    ):
        ocr_client = PaddleOCRVLClient(
            device=settings.paddleocr_device,
            engine=settings.paddleocr_engine,
            dtype=settings.paddleocr_dtype,
            pipeline_version=settings.paddleocr_pipeline_version,
            model_source=settings.paddleocr_model_source,
        )
        owns_ocr_client = True
    document_hash = _file_sha256(path)
    try:
        page_documents = _ocr_scanned_pages(
            path, page_documents, output_dir, ocr_client, document_hash, chapter_limit
        )
    except Exception:
        if vision_client is not None:
            vision_client.close()
        if owns_ocr_client and ocr_client is not None:
            ocr_client.close()
        raise
    page_text_hashes = {
        item.page: hashlib.sha256(item.text.encode("utf-8")).hexdigest()
        for item in page_documents
    }
    audit_path = output_dir / f"{path.stem}.cleaning_audit.json"
    # Page cleaning is intentionally disabled. Preserve every OCR page and every
    # character so exercises, appendices and publication matter remain available
    # to the later section/evidence pipeline. The pass-through audit is written
    # immediately, before visual calls, so an interrupted build is still auditable.
    decisions: dict[int, dict[str, Any]] = {
        item.page: {
            "page": item.page,
            "source_page": item.source_page or item.page,
            "keep": True,
            "page_type": "unfiltered",
            "reason": "页面清洗已禁用，原页完整保留",
            "method": "disabled",
            "cleaning_policy_version": PAGE_CLEANING_POLICY_VERSION,
            "requested_remove_fragments": [],
            "remove_fragments": [],
            "removed_characters": 0,
            "document_hash": document_hash,
            "page_text_hash": page_text_hashes[item.page],
        }
        for item in page_documents
    }
    audit_path.write_text(
        json.dumps(
            [decisions[number] for number in sorted(decisions)],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    kept_docs = list(page_documents)
    page_meta = {item.page: item for item in kept_docs}
    allowed_pages = set(page_meta)
    external = _external_pdf_extract_elements(path)
    elements: list[LayoutElement] = []
    image_counter = 0
    pdf_extract_kit = PDFExtractKitAdapter()
    pdf_extract_kit.write_manifest(output_dir)
    layout_records: list[dict[str, Any]] = []
    formula_audit_records: list[dict[str, Any]] = []
    analysis_pages = sorted(allowed_pages)
    if settings.pdf_extract_kit_page_limit > 0:
        analysis_pages = analysis_pages[: settings.pdf_extract_kit_page_limit]
    analysis_page_set = set(analysis_pages)
    element_cache = output_dir / "multimodal_elements.jsonl"
    visual_checkpoint = output_dir / f"{path.stem}.visual_elements.checkpoint.jsonl"
    cached_images = _load_visual_element_cache(
        (element_cache, visual_checkpoint), path.name
    )

    document = fitz.open(path)
    try:
        for page_no in sorted(allowed_pages):
            page = document[page_no - 1]
            meta = page_meta[page_no]
            text_blocks: list[tuple[list[float], str]] = []
            text_block_types: list[str] = []
            text_block_records: list[dict[str, Any]] = []
            paddle_image_blocks: list[tuple[list[float], dict[str, Any]]] = []
            ocr_blocks = (
                meta.extra.get("text_blocks", [])
                if isinstance(meta.extra, dict)
                else []
            )
            for block in ocr_blocks:
                if not isinstance(block, dict):
                    continue
                block_type = str(block.get("type", "paragraph")).strip().lower()
                if block_type in {"page_header", "page_footer", "noise"}:
                    continue
                normalized_bbox = block.get("bbox", [])
                if not isinstance(normalized_bbox, list) or len(normalized_bbox) != 4:
                    continue
                try:
                    bbox = [
                        round(float(normalized_bbox[0]) / 1000 * float(page.rect.width), 2),
                        round(float(normalized_bbox[1]) / 1000 * float(page.rect.height), 2),
                        round(float(normalized_bbox[2]) / 1000 * float(page.rect.width), 2),
                        round(float(normalized_bbox[3]) / 1000 * float(page.rect.height), 2),
                    ]
                except (TypeError, ValueError):
                    continue
                if block_type == "image":
                    paddle_image_blocks.append((bbox, block))
                    continue
                text = str(block.get("text", "")).strip()
                if not text:
                    continue
                text_blocks.append((bbox, text))
                text_block_types.append(block_type)
                text_block_records.append(block)

            order = 0
            page_image_count = 0
            for block_index, (bbox, text) in enumerate(text_blocks):
                block_type = text_block_types[block_index]
                block = text_block_records[block_index]
                element_type = (
                    "table"
                    if block_type == "table" or _looks_like_table(text)
                    else "formula"
                    if block_type == "formula"
                    else "text"
                )
                generated_id, digest = _element_id(path.name, page_no, order, text)
                element_id = str(block.get("id", generated_id))
                meta = page_meta[page_no]
                description = ""
                table_cells: list[dict[str, Any]] = []
                if element_type == "table":
                    description, table_cells = _table_fact_description(text, element_id)
                element = LayoutElement(
                    id=element_id, source=path.name, page=page_no, element_type=element_type,
                    bbox=bbox, text=text, reading_order=order, chapter=meta.chapter,
                    section=meta.section, content_hash=digest, description=description,
                    nearby_text=_localized_nearby_text(bbox, text_blocks),
                    confidence=float(block.get("confidence", 0.0) or 0.0),
                    processor="paddleocr-vl-layout",
                    uncertain=bool(block.get("uncertain", False)),
                    source_page=meta.source_page or page_no,
                    polygon=[
                        [float(value) for value in point[:2]]
                        for point in block.get("polygon", [])
                        if isinstance(point, list) and len(point) >= 2
                    ],
                    ocr_block_id=element_id,
                    evidence_metadata={
                        "schema_version": PAGE_OCR_SCHEMA_VERSION,
                        "normalized_bbox": list(block.get("bbox", [])),
                        "raw_label": str(block.get("raw_label", "")),
                        "source_engine": str(block.get("source_engine", "paddleocr-vl")),
                        "model_revision": str(
                            block.get("model_revision", PADDLEOCR_VL_GIT_REVISION)
                        ),
                        "model_reading_order": block.get("model_reading_order"),
                        "corrections": list(block.get("corrections", [])),
                        "table_cells": table_cells,
                        "context_block_ids": _nearby_text_block_ids(
                            bbox, text_blocks, text_block_records
                        ),
                    },
                )
                if element_type == "table":
                    clip = fitz.Rect(*bbox) & page.rect
                    table_image_bytes = b""
                    if not clip.is_empty and clip.width >= 2 and clip.height >= 2:
                        table_image_bytes = page.get_pixmap(
                            matrix=fitz.Matrix(2.0, 2.0), clip=clip, alpha=False
                        ).tobytes("png")
                    if _image_is_safe(table_image_bytes):
                        table_image_path = artifacts_dir / (
                            f"p{page_no:04d}-{digest[:10]}-paddle-table.png"
                        )
                        table_image_path.write_bytes(table_image_bytes)
                        element.image_path = str(
                            table_image_path.relative_to(output_dir)
                        ).replace("\\", "/")
                        cached_table = cached_images.get(digest)
                        if _visual_cache_compatible(
                            cached_table, vision_client, "table"
                        ):
                            element.description = str(
                                cached_table.get("description", element.description)
                            )
                            cached_processor = str(cached_table.get("processor", ""))
                            expected_processor = _visual_processor(vision_client, "table")
                            if expected_processor and expected_processor not in element.processor:
                                element.processor += f"+{expected_processor}"
                            cached_evidence = cached_table.get("evidence_metadata", {})
                            if isinstance(cached_evidence, dict):
                                element.evidence_metadata.update({
                                    key: value for key, value in cached_evidence.items()
                                    if key.startswith("visual_") or key == "summary_grounded_in"
                                })
                            if cached_processor and expected_processor not in cached_processor:
                                element.evidence_metadata["previous_processor"] = cached_processor
                        else:
                            _summarize_table(element, table_image_bytes, vision_client)
                if element_type in {"formula", "table"}:
                    element.uncertain = element.uncertain or not bool(text.strip())
                elements.append(element)
                order += 1

            pdfkit_figure_count = 0
            if page_no in analysis_page_set and pdf_extract_kit.available:
                import cv2
                import numpy as np

                render_scale = 2.0  # PDF-Extract-Kit convention: 144 DPI from 72-DPI PDF points.
                pixmap = page.get_pixmap(matrix=fitz.Matrix(render_scale, render_scale), alpha=False)
                rgb = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
                    pixmap.height, pixmap.width, pixmap.n
                )
                image_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                detected_regions = pdf_extract_kit.detect(image_bgr)
                for region in detected_regions:
                    bbox_points = [round(value / render_scale, 2) for value in region.bbox_pixels]
                    layout_records.append({
                        "source": path.name,
                        "page": page_no,
                        "category": region.category,
                        "bbox_pixels": region.bbox_pixels,
                        "bbox": bbox_points,
                        "confidence": region.confidence,
                        "detector": region.detector,
                    })
                selected_regions = _indexable_pdfkit_regions(detected_regions)
                formula_ocr_candidates = _formula_candidates_from_page_text(meta.text)
                formula_caption_regions = [
                    region for region in detected_regions
                    if region.category.lower() == "formula_caption"
                ]
                formula_candidate_index = 0
                for region in selected_regions:
                    bbox_points = [round(value / render_scale, 2) for value in region.bbox_pixels]
                    category = region.category.lower()
                    is_formula_region = category in {
                        "isolate_formula", "isolated", "isolated_formula"
                    }
                    ocr_formula: dict[str, str] = {}
                    caption_region: DetectedRegion | None = None
                    if is_formula_region:
                        if formula_candidate_index < len(formula_ocr_candidates):
                            ocr_formula = formula_ocr_candidates[formula_candidate_index]
                        formula_candidate_index += 1
                        caption_region = _nearest_formula_caption_region(
                            region, formula_caption_regions
                        )
                    crop_bytes = _crop_png(image_bgr, region.bbox_pixels)
                    crop_usable = len(crop_bytes) >= 300 and _image_is_safe(crop_bytes)
                    if not crop_usable and not is_formula_region:
                        continue
                    if category == "figure":
                        image_counter += 1
                        if (
                            settings.multimodal_image_limit
                            and image_counter > settings.multimodal_image_limit
                        ):
                            continue
                    id_content: bytes | str = crop_bytes or json.dumps(
                        {"category": category, "bbox": bbox_points}, sort_keys=True
                    )
                    element_id, digest = _element_id(path.name, page_no, order, id_content)
                    image_path = artifacts_dir / (
                        f"p{page_no:04d}-{element_id[:10]}-pdfkit-{category}.png"
                    )
                    if crop_bytes:
                        image_path.write_bytes(crop_bytes)
                    meta = page_meta[page_no]
                    nearby = _localized_nearby_text(bbox_points, text_blocks) or meta.text[-3000:]
                    element = LayoutElement(
                        id=element_id,
                        source=path.name,
                        page=page_no,
                        element_type=(
                            "image" if category == "figure"
                            else "table" if category == "table"
                            else "formula"
                        ),
                        bbox=bbox_points,
                        image_path=(
                            str(image_path.relative_to(output_dir)).replace("\\", "/")
                            if crop_bytes else ""
                        ),
                        reading_order=order,
                        chapter=meta.chapter,
                        section=meta.section,
                        caption=ocr_formula.get("caption", ""),
                        nearby_text=nearby,
                        content_hash=digest,
                        confidence=region.confidence,
                        processor=region.detector,
                        source_page=meta.source_page or page_no,
                        evidence_metadata={
                            "context_block_ids": _nearby_text_block_ids(
                                bbox_points, text_blocks, text_block_records
                            ),
                        },
                    )
                    cached = cached_images.get(digest)
                    append_element = True
                    if category == "figure":
                        cache_compatible = _visual_cache_compatible(
                            cached, vision_client, "image"
                        )
                        if cached and cache_compatible:
                            for field_name in (
                                "element_type", "caption", "components", "nets", "netlist",
                                "description", "confidence", "processor", "uncertain",
                            ):
                                if field_name in cached:
                                    setattr(element, field_name, cached[field_name])
                            if isinstance(cached.get("evidence_metadata"), dict):
                                element.evidence_metadata.update(
                                    cached["evidence_metadata"]
                                )
                            if element.components:
                                element.netlist = _synthesize_netlist(element.components)
                        else:
                            _analyze_image(element, crop_bytes, vision_client)
                        _enforce_verified_circuit(element)
                        pdfkit_figure_count += 1
                        page_image_count += 1
                    elif category == "table":
                        matched = _matching_layout_element(
                            elements,
                            source=path.name,
                            page=page_no,
                            element_type="table",
                            bbox=bbox_points,
                        )
                        if matched is not None:
                            append_element = False
                            element = matched
                            element.confidence = max(element.confidence, region.confidence)
                            if region.detector not in element.processor:
                                element.processor += f"+{region.detector}-localized"
                            if crop_bytes and not element.image_path:
                                element.image_path = str(
                                    image_path.relative_to(output_dir)
                                ).replace("\\", "/")
                            element.evidence_metadata["pdf_extract_kit_bbox"] = bbox_points
                            if (
                                crop_bytes
                                and _visual_processor(vision_client, "table")
                                not in element.processor
                            ):
                                _summarize_table(element, crop_bytes, vision_client)
                        else:
                            element.text = _overlapping_text(bbox_points, text_blocks)
                            description, cells = _table_fact_description(
                                element.text, element.id
                            )
                            element.description = description
                            element.processor += "+paddleocr-vl-page-evidence"
                            element.uncertain = not bool(element.text)
                            element.evidence_metadata = {
                                "schema_version": PAGE_OCR_SCHEMA_VERSION,
                                "source_engine": "paddleocr-vl",
                                "model_revision": PADDLEOCR_VL_GIT_REVISION,
                                "table_cells": cells,
                                "pdf_extract_kit_bbox": bbox_points,
                                "context_block_ids": _nearby_text_block_ids(
                                    bbox_points, text_blocks, text_block_records
                                ),
                            }
                            _summarize_table(element, crop_bytes, vision_client)
                    else:
                        matched = _matching_layout_element(
                            elements,
                            source=path.name,
                            page=page_no,
                            element_type="formula",
                            bbox=bbox_points,
                        )
                        formula_audit: dict[str, Any] = {
                            "source": path.name,
                            "page": page_no,
                            "source_page": meta.source_page or page_no,
                            "bbox": bbox_points,
                            "detector": region.detector,
                            "detector_confidence": region.confidence,
                            "image_path": element.image_path,
                            "caption": element.caption,
                            "caption_bbox": (
                                [round(value / render_scale, 2) for value in caption_region.bbox_pixels]
                                if caption_region else None
                            ),
                            "ocr_candidate": ocr_formula,
                            "attempts": [],
                            "fallback_source": "",
                        }
                        if matched is not None:
                            append_element = False
                            element = matched
                            element.confidence = max(element.confidence, region.confidence)
                            if region.detector not in element.processor:
                                element.processor += f"+{region.detector}-localized"
                            if crop_bytes and not element.image_path:
                                element.image_path = str(
                                    image_path.relative_to(output_dir)
                                ).replace("\\", "/")
                            if not element.caption:
                                element.caption = ocr_formula.get("caption", "")
                            element.evidence_metadata["pdf_extract_kit_bbox"] = bbox_points
                            formula_audit["status"] = "recognized"
                            formula_audit["attempts"].append({
                                "stage": "paddle-page-ocr",
                                "accepted": bool(element.text),
                                "ocr_block_id": element.ocr_block_id,
                            })
                        else:
                            fallback_text = ocr_formula.get("plain_text", "")
                            fallback_latex = (
                                ocr_formula.get("latex", "")
                                or _formula_latex_from_ocr_text(fallback_text)
                            )
                            if fallback_latex or fallback_text:
                                result = _normalize_formula_result(
                                    {
                                        "is_formula": True,
                                        "latex": fallback_latex,
                                        "plain_text": fallback_text,
                                        "confidence": max(0.55, region.confidence),
                                    },
                                    fallback_text,
                                )
                                formula_audit["status"] = "recognized"
                                formula_audit["fallback_source"] = "paddle-page-ocr"
                            else:
                                result = {
                                    "is_formula": True,
                                    "latex": "",
                                    "plain_text": "",
                                    "variables": [],
                                    "confidence": 0.0,
                                }
                                formula_audit["status"] = "uncertain"
                            latex = str(result.get("latex", "")).strip()
                            plain_text = str(result.get("plain_text", "")).strip()
                            element.text = (
                                f"LaTeX: ${latex}$\n检索文本: {plain_text}"
                                if latex and plain_text
                                else f"LaTeX: ${latex}$" if latex else plain_text
                            )
                            if not element.text:
                                element.text = "独立公式（未能可靠转写，详见公式审计）"
                            variables = result.get("variables", [])
                            formula_knowledge = str(result.get("knowledge", "")).strip()
                            if formula_knowledge:
                                element.description = formula_knowledge
                            elif variables:
                                element.description = "变量：" + json.dumps(
                                    variables, ensure_ascii=False
                                )
                            if formula_audit["status"] == "uncertain":
                                element.processor += "+formula-unresolved"
                            else:
                                element.processor += "+paddleocr-vl-page-evidence"
                            element.confidence = max(
                                element.confidence, float(result.get("confidence", 0))
                            )
                            recognition_confidence = float(result.get("confidence", 0))
                            element.uncertain = (
                                formula_audit["status"] == "uncertain"
                                or not bool(latex)
                                or recognition_confidence < 0.6
                            )
                            element.evidence_metadata = {
                                "schema_version": PAGE_OCR_SCHEMA_VERSION,
                                "source_engine": "paddleocr-vl",
                                "model_revision": PADDLEOCR_VL_GIT_REVISION,
                                "pdf_extract_kit_bbox": bbox_points,
                                "context_block_ids": _nearby_text_block_ids(
                                    bbox_points, text_blocks, text_block_records
                                ),
                            }
                        formula_audit["final"] = {
                            "status": formula_audit.get("status", "uncertain"),
                            "caption": element.caption,
                            "text": element.text,
                            "processor": element.processor,
                            "confidence": element.confidence,
                            "uncertain": element.uncertain,
                        }
                        formula_audit_records.append(formula_audit)
                    if append_element:
                        elements.append(element)
                    order += 1

            for bbox, block in paddle_image_blocks:
                duplicate_visual = any(
                    element.source == path.name
                    and element.page == page_no
                    and element.element_type in {"image", "circuit", "figure"}
                    and _bbox_iou_points(element.bbox, bbox) >= 0.25
                    for element in elements
                )
                if duplicate_visual:
                    continue
                image_counter += 1
                if (
                    settings.multimodal_image_limit
                    and image_counter > settings.multimodal_image_limit
                ):
                    break
                clip = fitz.Rect(*bbox) & page.rect
                if clip.is_empty or clip.width < 2 or clip.height < 2:
                    continue
                image_bytes = page.get_pixmap(
                    matrix=fitz.Matrix(2.0, 2.0), clip=clip, alpha=False
                ).tobytes("png")
                if not _image_is_safe(image_bytes):
                    continue
                element_id, digest = _element_id(
                    path.name, page_no, order, image_bytes
                )
                image_path = artifacts_dir / (
                    f"p{page_no:04d}-{element_id[:10]}-paddle-image.png"
                )
                image_path.write_bytes(image_bytes)
                element = LayoutElement(
                    id=element_id,
                    source=path.name,
                    page=page_no,
                    element_type="image",
                    bbox=bbox,
                    image_path=str(image_path.relative_to(output_dir)).replace("\\", "/"),
                    reading_order=order,
                    chapter=meta.chapter,
                    section=meta.section,
                    nearby_text=_localized_nearby_text(bbox, text_blocks) or meta.text[-3000:],
                    content_hash=digest,
                    confidence=float(block.get("confidence", 0.0) or 0.0),
                    processor="paddleocr-vl-layout",
                    source_page=meta.source_page or page_no,
                    polygon=list(block.get("polygon", [])),
                    ocr_block_id=str(block.get("id", "")) or None,
                    evidence_metadata={
                        "schema_version": PAGE_OCR_SCHEMA_VERSION,
                        "normalized_bbox": list(block.get("bbox", [])),
                        "source_engine": "paddleocr-vl",
                        "model_revision": PADDLEOCR_VL_GIT_REVISION,
                        "corrections": list(block.get("corrections", [])),
                        "context_block_ids": _nearby_text_block_ids(
                            bbox, text_blocks, text_block_records
                        ),
                    },
                )
                cached = cached_images.get(digest)
                cache_compatible = _visual_cache_compatible(
                    cached, vision_client, "image"
                )
                if cached and cache_compatible:
                    for field_name in (
                        "element_type", "caption", "components", "nets", "netlist",
                        "description", "confidence", "processor", "uncertain",
                    ):
                        if field_name in cached:
                            setattr(element, field_name, cached[field_name])
                    if isinstance(cached.get("evidence_metadata"), dict):
                        element.evidence_metadata.update(cached["evidence_metadata"])
                    if element.components:
                        element.netlist = _synthesize_netlist(element.components)
                else:
                    _analyze_image(element, image_bytes, vision_client)
                _enforce_verified_circuit(element)
                elements.append(element)
                page_image_count += 1
                order += 1

            if page_image_count == 0:
                seen_xrefs: set[int] = set()
                for image_info in page.get_images(full=True):
                    xref = int(image_info[0])
                    if xref <= 0 or xref in seen_xrefs:
                        continue
                    seen_xrefs.add(xref)
                    extracted = document.extract_image(xref)
                    image_bytes = bytes(extracted.get("image", b""))
                    if len(image_bytes) < 700 or not _image_is_safe(image_bytes):
                        continue
                    rects = page.get_image_rects(xref)
                    bbox = (
                        [round(float(value), 2) for value in tuple(rects[0])]
                        if rects
                        else [0.0, 0.0, float(page.rect.width), float(page.rect.height)]
                    )
                    if _is_full_page_scan(
                        bbox, float(page.rect.width), float(page.rect.height), meta
                    ):
                        continue
                    image_counter += 1
                    if (
                        settings.multimodal_image_limit
                        and image_counter > settings.multimodal_image_limit
                    ):
                        break
                    ext = str(extracted.get("ext", "png")).lower()
                    if ext not in {"png", "jpg", "jpeg", "webp", "bmp"}:
                        ext = "png"
                    element_id, digest = _element_id(
                        path.name, page_no, order, image_bytes
                    )
                    image_path = artifacts_dir / (
                        f"p{page_no:04d}-{element_id[:10]}.{ext}"
                    )
                    image_path.write_bytes(image_bytes)
                    nearby = (
                        "\n".join(text for _, text in text_blocks)[-3000:]
                        or meta.text[-3000:]
                    )
                    element = LayoutElement(
                        id=element_id,
                        source=path.name,
                        page=page_no,
                        element_type="image",
                        bbox=bbox,
                        image_path=str(image_path.relative_to(output_dir)).replace(
                            "\\", "/"
                        ),
                        reading_order=order,
                        chapter=meta.chapter,
                        section=meta.section,
                        nearby_text=nearby,
                        content_hash=digest,
                        processor="pdf-image-object",
                        source_page=meta.source_page or page_no,
                    )
                    cached = cached_images.get(digest)
                    cache_compatible = _visual_cache_compatible(
                        cached, vision_client, "image"
                    )
                    if cached and cache_compatible:
                        for field_name in (
                            "element_type",
                            "caption",
                            "components",
                            "nets",
                            "netlist",
                            "description",
                            "confidence",
                            "processor",
                            "uncertain",
                        ):
                            if field_name in cached:
                                setattr(element, field_name, cached[field_name])
                        if isinstance(cached.get("evidence_metadata"), dict):
                            element.evidence_metadata.update(cached["evidence_metadata"])
                        if element.components:
                            element.netlist = _synthesize_netlist(element.components)
                    else:
                        _analyze_image(element, image_bytes, vision_client)
                    _enforce_verified_circuit(element)
                    elements.append(element)
                    page_image_count += 1
                    order += 1

            # Many electronic textbooks store schematics as PDF vector paths,
            # not raster images. Render such a page so Qwen3-VL can still see it.
            if (
                page_image_count == 0
                and len(page.get_drawings()) >= 3
                and (not settings.multimodal_image_limit or image_counter < settings.multimodal_image_limit)
            ):
                image_counter += 1
                width, height = max(1.0, float(page.rect.width)), max(1.0, float(page.rect.height))
                scale = min(1.5, 2200 / max(width, height), math.sqrt(4_000_000 / (width * height)))
                if scale <= 0.05:
                    continue
                pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
                image_bytes = pixmap.tobytes("png")
                if not _image_is_safe(image_bytes):
                    continue
                element_id, digest = _element_id(path.name, page_no, order, image_bytes)
                image_path = artifacts_dir / f"p{page_no:04d}-{element_id[:10]}-vector.png"
                image_path.write_bytes(image_bytes)
                meta = page_meta[page_no]
                element = LayoutElement(
                    id=element_id,
                    source=path.name,
                    page=page_no,
                    element_type="image",
                    bbox=[0.0, 0.0, float(page.rect.width), float(page.rect.height)],
                    image_path=str(image_path.relative_to(output_dir)).replace("\\", "/"),
                    reading_order=order,
                    chapter=meta.chapter,
                    section=meta.section,
                    nearby_text=(
                        "\n".join(text for _, text in text_blocks)[-3000:]
                        or meta.text[-3000:]
                    ),
                    content_hash=digest,
                    processor="pymupdf-vector-render",
                    source_page=meta.source_page or page_no,
                )
                cached = cached_images.get(digest)
                cache_compatible = _visual_cache_compatible(
                    cached, vision_client, "image"
                )
                if cached and cache_compatible:
                    for field_name in (
                        "element_type", "caption", "components", "nets", "netlist",
                        "description", "confidence", "processor", "uncertain",
                    ):
                        if field_name in cached:
                            setattr(element, field_name, cached[field_name])
                    if isinstance(cached.get("evidence_metadata"), dict):
                        element.evidence_metadata.update(cached["evidence_metadata"])
                    if element.components:
                        element.netlist = _synthesize_netlist(element.components)
                else:
                    _analyze_image(element, image_bytes, vision_client)
                _enforce_verified_circuit(element)
                elements.append(element)
            _append_visual_element_checkpoint(
                visual_checkpoint,
                elements,
                source=path.name,
                page=page_no,
            )
    finally:
        document.close()
        if vision_client is not None:
            vision_client.close()
        if owns_ocr_client and ocr_client is not None:
            ocr_client.close()

    # External parser output enriches, but never erases, the auditable fallback extraction.
    for index, item in enumerate(external):
        if not isinstance(item, dict):
            continue
        try:
            page_no = int(item.get("page", item.get("page_idx", 0)))
            if page_no == 0 and "page_idx" in item:
                page_no = int(item["page_idx"]) + 1
        except (TypeError, ValueError):
            continue
        if page_no not in allowed_pages:
            continue
        text = str(item.get("text", item.get("content", ""))).strip()
        if not text:
            continue
        element_id, digest = _element_id(path.name, page_no, 100000 + index, text)
        bbox = item.get("bbox") if isinstance(item.get("bbox"), list) else [0, 0, 0, 0]
        kind = str(item.get("type", item.get("category", "text"))).lower()
        if "formula" in kind or "equation" in kind:
            kind = "formula"
        elif "table" in kind:
            kind = "table"
        else:
            kind = "text"
        meta = page_meta[page_no]
        bbox_values = [float(v) for v in bbox[:4]]
        matched = (
            _matching_layout_element(
                elements,
                source=path.name,
                page=page_no,
                element_type=kind,
                bbox=bbox_values,
            )
            if kind in {"formula", "table"}
            else None
        )
        if matched is not None:
            if "pdf-extract-kit-external" not in matched.processor:
                matched.processor += "+pdf-extract-kit-external"
            matched.evidence_metadata["external_parser_bbox"] = bbox_values
            continue
        description = ""
        table_cells: list[dict[str, Any]] = []
        if kind == "table":
            description, table_cells = _table_fact_description(text, element_id)
        elements.append(LayoutElement(
            id=element_id, source=path.name, page=page_no, element_type=kind,
            bbox=bbox_values, text=text, reading_order=100000 + index,
            chapter=meta.chapter, section=meta.section, content_hash=digest,
            description=description,
            processor="pdf-extract-kit",
            source_page=meta.source_page or page_no,
            evidence_metadata={"table_cells": table_cells} if table_cells else {},
        ))

    audit = [decisions[number] for number in sorted(decisions)]
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / f"{path.stem}.formula_audit.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "source": path.name,
                "detected": len(formula_audit_records),
                "recognized": sum(
                    item.get("status") in {"recognized", "cached"}
                    for item in formula_audit_records
                ),
                "fallback": sum(
                    item.get("status") == "fallback" for item in formula_audit_records
                ),
                "uncertain": sum(
                    bool((item.get("final") or {}).get("uncertain"))
                    for item in formula_audit_records
                ),
                "formulas": formula_audit_records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / f"{path.stem}.pdf_extract_kit.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "source": path.name,
                "analyzed_pages": analysis_pages,
                "regions": layout_records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return kept_docs, elements, audit


def multimodal_chunks(elements: Iterable[LayoutElement]) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    seen: set[tuple[str, int, str, str]] = set()
    seen_semantic: set[tuple[str, int, str, str]] = set()
    for element in elements:
        if element.element_type == "text":
            continue  # page-level text chunks already provide coherent overlap.
        dedup_key = (element.source, element.page, element.element_type, element.content_hash)
        if element.content_hash and dedup_key in seen:
            continue
        seen.add(dedup_key)
        body_parts = [element.caption, element.text, element.description]
        if element.components:
            body_parts.append("元件：" + json.dumps(element.components, ensure_ascii=False))
        if element.nets:
            body_parts.append("连接网络：" + json.dumps(element.nets, ensure_ascii=False))
        if element.netlist:
            body_parts.append("Netlist：\n" + element.netlist)
        text = "\n".join(part for part in body_parts if part).strip()
        if not text:
            continue
        semantic_key = (
            element.source,
            element.source_page or element.page,
            element.element_type,
            re.sub(r"\s+", "", text).lower()[:500],
        )
        if semantic_key in seen_semantic:
            continue
        seen_semantic.add(semantic_key)
        semantic_parts = [element.caption, element.text, element.description]
        if element.element_type != "formula":
            semantic_parts.append(element.nearby_text)
        semantic_text = "\n".join(part for part in semantic_parts if part)
        tags = extract_course_concepts(semantic_text, element.section)
        if element.element_type == "formula":
            for concept in extract_formula_concepts(element.text):
                if concept not in tags:
                    tags.append(concept)
        for component in element.components:
            if not isinstance(component, dict):
                continue
            component_type = str(component.get("type", "")).strip()
            component_concept = COMPONENT_CONCEPTS.get(component_type.lower())
            if component_concept and component_concept not in tags:
                tags.append(component_concept)
        chunks.append(TextChunk(
            id=f"element-{element.id}", text=text, source=element.source,
            chapter=element.chapter, section=element.section,
            page_start=element.source_page or element.page,
            page_end=element.source_page or element.page,
            doc_type="multimodal",
            knowledge_tags=tags[:12], element_type=element.element_type,
            bbox=element.bbox, parent_id=element.id, image_path=element.image_path,
            content_hash=element.content_hash,
            multimodal={
                "components": element.components,
                "nets": element.nets,
                "netlist": element.netlist,
                "confidence": element.confidence,
                "processor": element.processor,
                "uncertain": element.uncertain,
            },
        ))
    return chunks


def build_chapter_knowledge_summaries(
    chunks: Iterable[TextChunk],
) -> list[dict[str, Any]]:
    """Group indexed course concepts into chapter-level browse summaries."""

    grouped: dict[str, dict[str, Any]] = {}
    for chunk_index, chunk in enumerate(chunks):
        if chunk.doc_type == "question":
            continue
        chapter = _normalize_chapter_heading(chunk.chapter)
        if not chapter:
            continue
        # A bare marker is useful when real content inherits only "第一章", but an
        # isolated marker copied from a contents/answer page is not a chapter
        # summary by itself. Require some evidence beyond the repeated heading.
        if (
            re.fullmatch(CHAPTER_MARKER_PATTERN, chapter)
            and re.sub(r"\s+", "", chunk.text) == re.sub(r"\s+", "", chapter)
            and meaningful_section(chunk.section) == chapter
        ):
            continue
        summary = grouped.setdefault(
            chapter,
            {
                "id": "chapter:" + hashlib.sha1(chapter.encode("utf-8")).hexdigest()[:16],
                "name": chapter,
                "order": len(grouped) + 1,
                "pages": set(),
                "sources": set(),
                "sections": set(),
                "concepts": {},
            },
        )
        summary["sources"].add(chunk.source)
        section = meaningful_section(chunk.section)
        if section and section != chapter:
            summary["sections"].add(section)
        chunk_pages = {
            page
            for page in (chunk.page_start, chunk.page_end)
            if isinstance(page, int) and page > 0
        }
        summary["pages"].update(chunk_pages)
        if chunk.doc_type == "exercise":
            continue
        for concept in dict.fromkeys(chunk.knowledge_tags):
            concept_name = normalize_concept_name(concept)
            if (
                not concept_name
                or not is_course_concept(concept_name)
                or CHAPTER_EXERCISE_CONCEPT_PATTERN.search(concept_name)
            ):
                continue
            concept_summary = summary["concepts"].setdefault(
                concept_name,
                {
                    "id": "concept:" + hashlib.sha1(
                        concept_name.encode("utf-8")
                    ).hexdigest()[:16],
                    "name": concept_name,
                    "evidence_count": 0,
                    "pages": set(),
                    "first_seen": chunk_index,
                },
            )
            concept_summary["evidence_count"] += 1
            concept_summary["pages"].update(chunk_pages)

    result: list[dict[str, Any]] = []
    for summary in grouped.values():
        pages = sorted(summary["pages"])
        concepts = sorted(
            summary["concepts"].values(),
            key=lambda item: (
                -int(item["evidence_count"]),
                int(item["first_seen"]),
                str(item["name"]),
            ),
        )
        result.append({
            "id": summary["id"],
            "name": summary["name"],
            "order": summary["order"],
            "page_start": pages[0] if pages else None,
            "page_end": pages[-1] if pages else None,
            "pages": pages,
            "sources": sorted(summary["sources"]),
            "section_count": len(summary["sections"]),
            "concept_count": len(concepts),
            "concepts": [
                {
                    "id": item["id"],
                    "name": item["name"],
                    "evidence_count": item["evidence_count"],
                    "pages": sorted(item["pages"]),
                }
                for item in concepts
            ],
        })
    return result


def build_local_knowledge_graph(chunks: Iterable[TextChunk]) -> dict[str, Any]:
    chunk_items = list(chunks)
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, str]] = []
    seen_edges: set[tuple[str, str, str]] = set()

    def add_edge(source: str, relation: str, target: str) -> None:
        edge = (source, relation, target)
        if edge not in seen_edges:
            edges.append({"source": source, "type": relation, "target": target})
            seen_edges.add(edge)

    for chunk in chunk_items:
        if chunk.doc_type in {"question", "exercise"}:
            continue
        document_id = "document:" + hashlib.sha1(chunk.source.encode("utf-8")).hexdigest()[:16]
        source_stem = Path(chunk.source).stem
        source_range = re.search(r"(?:pages?|页)[_-]?(\d+)[_-](\d+)", source_stem, re.I)
        document_name = (
            f"第 {source_range.group(1)}–{source_range.group(2)} 页教材节选"
            if source_range else source_stem
        )
        nodes.setdefault(document_id, {
            "id": document_id,
            "type": "document",
            "name": document_name,
            "source": chunk.source,
        })
        page_number = chunk.page_start or chunk.page_end
        page_id = f"page:{document_id}:{page_number or 'unknown'}"
        nodes.setdefault(page_id, {
            "id": page_id,
            "type": "page",
            "name": f"第 {page_number} 页" if page_number else "无页码片段",
            "source": chunk.source,
            "page": page_number,
        })
        add_edge(document_id, "HAS_PAGE", page_id)
        chunk_node = f"chunk:{chunk.id}"
        element_labels = {
            "circuit": "电路图", "formula": "公式", "table": "表格",
            "image": "图片", "text": "正文",
        }
        section_name = meaningful_section(chunk.section)
        snippet = re.sub(r"\s+", " ", chunk.text).strip()[:24]
        chunk_label = section_name or f"{element_labels.get(chunk.element_type, '资料')} · {snippet}"
        nodes[chunk_node] = {
            "id": chunk_node,
            "type": "chunk",
            "name": chunk_label,
            "chunk_id": chunk.id,
            "source": chunk.source,
            "page": page_number,
            "element_type": chunk.element_type,
        }
        add_edge(page_id, "HAS_CHUNK", chunk_node)
        concepts = list(dict.fromkeys(
            normalized
            for concept in chunk.knowledge_tags
            if (normalized := normalize_concept_name(concept))
            and is_course_concept(normalized)
        ))
        for concept in concepts:
            concept_id = "concept:" + hashlib.sha1(concept.encode("utf-8")).hexdigest()[:16]
            nodes.setdefault(concept_id, {"id": concept_id, "type": "concept", "name": concept})
            add_edge(chunk_node, "MENTIONS", concept_id)
        if chunk.element_type == "circuit" and chunk.multimodal:
            component_nodes: dict[str, str] = {}
            components = [
                item for item in chunk.multimodal.get("components", [])
                if isinstance(item, dict)
            ]
            nets = [
                item for item in chunk.multimodal.get("nets", [])
                if isinstance(item, dict)
            ]
            for component in components:
                ref = str(component.get("id") or component.get("ref") or "component")
                component_type = str(component.get("type", "unknown"))
                role = component_role(
                    ref,
                    component_type,
                    explicit_role=str(component.get("role") or component.get("function") or ""),
                    context=chunk.text,
                )
                component_id = f"component:{chunk.id}:{ref}"
                component_nodes[ref] = component_id
                nodes[component_id] = {
                    "id": component_id,
                    "type": "component",
                    "name": component_display_name(
                        ref,
                        component_type,
                        explicit_role=role,
                    ),
                    "symbol": ref,
                    "component_role": role,
                    "component_type": component_type,
                    "chunk_id": chunk.id,
                }
                add_edge(chunk_node, "CONTAINS", component_id)
                component_concept = COMPONENT_CONCEPTS.get(
                    str(component.get("type", "")).strip().lower()
                )
                if component_concept:
                    concept_id = "concept:" + hashlib.sha1(
                        component_concept.encode("utf-8")
                    ).hexdigest()[:16]
                    nodes.setdefault(concept_id, {
                        "id": concept_id,
                        "type": "concept",
                        "name": component_concept,
                    })
                    add_edge(component_id, "INSTANCE_OF", concept_id)
            for position, net in enumerate(nets, 1):
                net_ref = str(net.get("id") or net.get("name") or f"n{position}")
                net_id = f"net:{chunk.id}:{net_ref}"
                nodes[net_id] = {
                    "id": net_id,
                    "type": "net",
                    "name": net_ref,
                    "chunk_id": chunk.id,
                }
                add_edge(chunk_node, "CONTAINS", net_id)
                terminals = net.get("terminals", net.get("connections", []))
                if not isinstance(terminals, list):
                    terminals = []
                for terminal in terminals:
                    component_ref = re.split(r"[.:/]", str(terminal), maxsplit=1)[0]
                    if component_ref in component_nodes:
                        add_edge(component_nodes[component_ref], "CONNECTED_TO", net_id)
                for component in components:
                    component_ref = str(component.get("id") or component.get("ref") or "component")
                    terminal_nets = component.get("terminals", [])
                    if (
                        component_ref in component_nodes
                        and isinstance(terminal_nets, list)
                        and net_ref in map(str, terminal_nets)
                    ):
                        add_edge(component_nodes[component_ref], "CONNECTED_TO", net_id)
    return {
        "schema_version": "2.3-semantic-components",
        "nodes": list(nodes.values()),
        "edges": edges,
        "chapters": build_chapter_knowledge_summaries(chunk_items),
    }


def project_student_knowledge_graph(graph: dict[str, Any]) -> dict[str, Any]:
    """Collapse the provenance graph into a readable student-facing semantic map.

    Chunk, formula and net nodes remain in the persisted graph for retrieval and
    auditing. The UI receives a compact projection where those records become
    evidence metadata instead of dozens of visible nodes and edges.
    """

    raw_nodes = {
        str(node.get("id")): node
        for node in graph.get("nodes", [])
        if isinstance(node, dict) and node.get("id")
    }
    raw_edges = [edge for edge in graph.get("edges", []) if isinstance(edge, dict)]
    outgoing: dict[str, list[dict[str, Any]]] = {}
    incoming: dict[str, list[dict[str, Any]]] = {}
    for edge in raw_edges:
        source, target = str(edge.get("source", "")), str(edge.get("target", ""))
        outgoing.setdefault(source, []).append(edge)
        incoming.setdefault(target, []).append(edge)

    visible: dict[str, dict[str, Any]] = {}
    concept_aliases: dict[str, str] = {}
    for node_id, node in raw_nodes.items():
        if node.get("type") in {"document", "page"}:
            visible[node_id] = dict(node)
        elif node.get("type") == "concept":
            concept_name = normalize_concept_name(str(node.get("name", "")))
            if not concept_name:
                continue
            canonical_id = "concept:" + hashlib.sha1(
                concept_name.encode("utf-8")
            ).hexdigest()[:16]
            concept_aliases[node_id] = canonical_id
            visible.setdefault(canonical_id, {
                **node,
                "id": canonical_id,
                "name": concept_name,
            })

    projected_edges: list[dict[str, Any]] = []
    seen_edges: set[tuple[str, str, str]] = set()

    def add_edge(source: str, relation: str, target: str, evidence_count: int = 1) -> None:
        key = (source, relation, target)
        if source not in visible or target not in visible:
            return
        if key in seen_edges:
            for edge in projected_edges:
                if (edge["source"], edge["type"], edge["target"]) == key:
                    edge["evidence_count"] = int(edge.get("evidence_count", 1)) + evidence_count
                    return
        seen_edges.add(key)
        projected_edges.append({
            "source": source,
            "type": relation,
            "target": target,
            "evidence_count": evidence_count,
        })

    for edge in raw_edges:
        if edge.get("type") == "HAS_PAGE":
            add_edge(str(edge.get("source")), "HAS_PAGE", str(edge.get("target")))

    chunk_to_page: dict[str, str] = {}
    for edge in raw_edges:
        if edge.get("type") == "HAS_CHUNK":
            chunk_to_page[str(edge.get("target"))] = str(edge.get("source"))

    concept_evidence: dict[str, set[str]] = {}
    concept_pages: dict[str, set[int]] = {}
    for edge in raw_edges:
        if edge.get("type") != "MENTIONS":
            continue
        chunk_id = str(edge.get("source"))
        concept_id = concept_aliases.get(str(edge.get("target")), "")
        page_id = chunk_to_page.get(chunk_id)
        if not page_id or concept_id not in visible:
            continue
        concept_evidence.setdefault(concept_id, set()).add(chunk_id)
        page_number = raw_nodes.get(page_id, {}).get("page")
        if isinstance(page_number, int):
            concept_pages.setdefault(concept_id, set()).add(page_number)
        add_edge(page_id, "COVERS", concept_id)

    original_to_visible_component: dict[str, str] = {}
    component_pages: dict[str, set[int]] = {}
    for node_id, node in raw_nodes.items():
        if node.get("type") != "component":
            continue
        raw_name = str(node.get("name", "")).strip()
        symbol = str(node.get("symbol") or raw_name).strip()
        if not symbol or symbol.lower() in {"component", "unknown", "?"}:
            continue
        component_type = str(node.get("component_type", "unknown"))
        role = component_role(
            symbol,
            component_type,
            explicit_role=str(node.get("component_role") or node.get("role") or ""),
        )
        name = component_display_name(symbol, component_type, explicit_role=role)
        merged_id = "component:" + hashlib.sha1(
            f"{symbol.lower()}|{component_type.lower()}|{role}".encode("utf-8")
        ).hexdigest()[:16]
        original_to_visible_component[node_id] = merged_id
        visible.setdefault(merged_id, {
            "id": merged_id,
            "type": "component",
            "name": name,
            "symbol": symbol,
            "component_role": role,
            "component_type": component_type,
            "pages": [],
            "evidence_count": 0,
        })
        chunk_edges = [
            edge for edge in incoming.get(node_id, []) if edge.get("type") == "CONTAINS"
        ]
        for chunk_edge in chunk_edges:
            chunk_id = str(chunk_edge.get("source"))
            page_id = chunk_to_page.get(chunk_id)
            page_number = raw_nodes.get(page_id or "", {}).get("page")
            if isinstance(page_number, int):
                component_pages.setdefault(merged_id, set()).add(page_number)
            visible[merged_id]["evidence_count"] += 1
        for relation in outgoing.get(node_id, []):
            if relation.get("type") == "INSTANCE_OF":
                concept_id = concept_aliases.get(str(relation.get("target")), "")
                if concept_id:
                    add_edge(merged_id, "INSTANCE_OF", concept_id)

    for chunk_id, page_id in chunk_to_page.items():
        chunk = raw_nodes.get(chunk_id, {})
        if chunk.get("element_type") != "circuit":
            continue
        circuit_id = "circuit:" + chunk_id.removeprefix("chunk:")
        raw_name = str(chunk.get("name") or "")
        figure_number = re.search(r"图\s*\d+(?:\.\d+)+", raw_name)
        page_number = raw_nodes.get(page_id, {}).get("page")
        visible[circuit_id] = {
            "id": circuit_id,
            "type": "circuit",
            "name": (
                f"{figure_number.group(0).replace(' ', '')} 电路图"
                if figure_number
                else f"第 {page_number} 页电路图" if page_number else "电路图"
            ),
            "page": page_number,
            "chunk_id": chunk.get("chunk_id"),
        }
        add_edge(page_id, "HAS_CIRCUIT", circuit_id)
        for edge in outgoing.get(chunk_id, []):
            component_id = original_to_visible_component.get(str(edge.get("target")))
            if edge.get("type") == "CONTAINS" and component_id:
                add_edge(circuit_id, "CONTAINS", component_id)

    for concept_id, evidence in concept_evidence.items():
        visible[concept_id]["evidence_count"] = len(evidence)
        visible[concept_id]["pages"] = sorted(concept_pages.get(concept_id, set()))
    for component_id, pages in component_pages.items():
        visible[component_id]["pages"] = sorted(pages)

    return {
        "schema_version": "2.4-semantic-component-labels",
        "nodes": list(visible.values()),
        "edges": projected_edges,
        "chapters": graph.get("chapters", []),
    }
