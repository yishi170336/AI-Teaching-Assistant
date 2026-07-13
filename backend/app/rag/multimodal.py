from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import math
import mimetypes
import re
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable
from uuid import NAMESPACE_URL, uuid5

import fitz
import httpx

from backend.app.config import settings
from backend.app.rag.pdf_extract_kit import DetectedRegion, PDFExtractKitAdapter
from backend.app.rag.models import PageDocument, TextChunk
from backend.app.rag.ontology import (
    COMPONENT_CONCEPTS,
    extract_course_concepts,
    extract_formula_concepts,
    is_course_concept,
    meaningful_section,
    normalize_concept_name,
)
from backend.app.services.qwen_multimodal_client import QwenMultimodalAPIError, QwenVisionClient


logger = logging.getLogger(__name__)

PARTIAL_NOISE_MARKERS = (
    "版权所有", "版权", "ISBN", "责任编辑", "封面设计", "版次", "印次", "出版社",
    "扫码", "公众号", "购买正版", "资源下载", "广告", "网址", "http://", "https://",
)

SCANNED_PAGE_PLACEHOLDER = "[本页主要包含电路图、公式或其他图形内容]"
PAGE_OCR_SCHEMA_VERSION = "1.0-qwen-page-ocr"
PAGE_OCR_PROMPT = """你是模拟电子技术教材的高保真 OCR 与结构识别器。请完整转写本页，严格保持阅读顺序、标题层级、图题、表题、公式、变量、上下标和单位；不得概括、改写或补写看不清的内容。省略页码和重复的页眉。
text 必须是按阅读顺序排列的字符串数组，每个元素是一行或一个自然段；chapter 填本页可见的章标题，否则为空；section 填本页最后出现、层级最深的编号教学小节（例如“1.1.3 PN结”），否则为空；concepts 只列正文中明确出现的 2-18 个具体模拟电子技术知识点，不得列书名、章名、泛化词或举例材料。
仅返回 JSON：{"text":["..."],"chapter":"","section":"","concepts":["..."]}。"""

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
    processor: str = "pymupdf-fallback"
    uncertain: bool = False
    source_page: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
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
    decisions = {
        page.page: {
            "page": page.page,
            "source_page": page.source_page or page.page,
            "keep": True,
            "reason": "默认保留课程内容",
            "method": "rule",
            "remove_fragments": [],
        }
        for page in pages
    }
    if not client or not client.config.enabled:
        return decisions
    for start in range(0, len(pages), 12):
        batch = pages[start : start + 12]
        samples = "\n\n".join(
            f"<PAGE number=\"{item.page}\">\n{item.text[:900]}\n</PAGE>" for item in batch
        )
        result = client.complete_json(
            """你是电路教材清洗器。判断每页是否包含可用于教学问答的正文、例题、公式、表格或电路图。
仅当整页只是封面、版权/版本说明、空白页、纯广告、与课程无关的序言时才丢弃；目录、章节导读和任何技术内容必须保留。
若页面包含少量与课程无关的版本说明、广告或页眉噪音但同时有技术正文，必须保留页面，
并在 remove_fragments 中逐字列出需要删除的短片段；不得删除公式、图题、例题或技术段落。
返回 JSON：{"decisions":[{"page":1,"keep":true,"reason":"...","remove_fragments":["原文片段"]}]}，不得改写页码。\n"""
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
                decisions[page_no] = {
                    "page": page_no,
                    "source_page": next(
                        (page.source_page or page.page for page in pages if page.page == page_no),
                        page_no,
                    ),
                    "keep": bool(item.get("keep", True)),
                    "reason": str(item.get("reason", "模型语义清洗"))[:240],
                    "method": f"llm:{client.config.provider}/{client.config.model}",
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
                if not decisions[page_no]["keep"] and (
                    extract_course_concepts(page_text)
                    or re.search(r"[=+−±√∫ΣΩπ^_]", page_text)
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
        lines = [str(item).strip() for item in value if str(item).strip()]
        return "\n".join(lines)
    return str(value or "").strip()


def _ocr_heading_context(
    value: dict[str, Any],
    text: str,
    previous_chapter: str,
    previous_section: str,
) -> tuple[str, str]:
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines() if line.strip()]
    chapter_pattern = re.compile(r"^第[一二三四五六七八九十百0-9]+章\s*[^。；]{0,40}")
    visible_chapters = [match.group(0).strip() for line in lines if (match := chapter_pattern.match(line))]
    raw_chapter = re.sub(r"\s+", " ", str(value.get("chapter", ""))).strip()
    chapter = visible_chapters[-1] if visible_chapters else (
        raw_chapter if chapter_pattern.match(raw_chapter) else previous_chapter
    )
    if chapter and chapter != previous_chapter:
        previous_section = ""

    section_pattern = re.compile(
        r"^\s*(\d{1,2}(?:\s*[.．]\s*\d{1,2}){1,3})\s+([^=。；]{2,42})\s*$"
    )
    visible_sections: list[str] = []
    for line in lines:
        match = section_pattern.match(line)
        if not match:
            continue
        number = re.sub(r"\s*[.．]\s*", ".", match.group(1))
        title = match.group(2).strip(" .．、:：-")
        visible_sections.append(f"{number} {title}")
    raw_section = re.sub(r"\s+", " ", str(value.get("section", ""))).strip()
    if visible_sections:
        section = visible_sections[-1]
    elif section_pattern.match(raw_section):
        match = section_pattern.match(raw_section)
        assert match is not None
        number = re.sub(r"\s*[.．]\s*", ".", match.group(1))
        section = f"{number} {match.group(2).strip(' .．、:：-')}"
    else:
        section = previous_section
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
    temporary.replace(path)


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
    client: QwenVisionClient | None,
    document_hash: str,
) -> list[PageDocument]:
    """Recover the text layer of image-only textbook pages with a durable cache."""

    if not any(page.text.strip() == SCANNED_PAGE_PLACEHOLDER for page in pages):
        return pages
    cache_path = output_dir / f"{path.stem}.page_ocr.jsonl"
    cache_entries: dict[int, dict[str, Any]] = {}
    if cache_path.exists():
        try:
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                item = json.loads(line)
                if (
                    isinstance(item, dict)
                    and item.get("schema_version") == PAGE_OCR_SCHEMA_VERSION
                    and item.get("document_hash") == document_hash
                    and (client is None or item.get("model") == client.model)
                    and str(item.get("text", "")).strip()
                ):
                    cache_entries[int(item["page"])] = item
        except (OSError, ValueError, json.JSONDecodeError):
            cache_entries = {}

    recovered: list[PageDocument] = []
    previous_chapter = ""
    previous_section = ""
    document = fitz.open(path)
    try:
        for page_document in sorted(pages, key=lambda item: item.page):
            if page_document.text.strip() != SCANNED_PAGE_PLACEHOLDER:
                previous_chapter = page_document.chapter or previous_chapter
                previous_section = page_document.section or previous_section
                recovered.append(page_document)
                continue

            cached = cache_entries.get(page_document.page)
            if cached:
                chapter = str(cached.get("chapter", "")).strip() or previous_chapter
                section = str(cached.get("section", "")).strip() or previous_section
                concepts = [str(item) for item in cached.get("concepts", []) if str(item).strip()]
                previous_chapter, previous_section = chapter, section
                recovered.append(replace(
                    page_document,
                    text=str(cached["text"]).strip(),
                    chapter=chapter or page_document.chapter,
                    section=section or chapter or page_document.section,
                    extra={
                        **(page_document.extra or {}),
                        "ocr_concepts": concepts,
                        "ocr_processor": f"qwen-vl:{cached.get('model', '')}",
                    },
                ))
                continue
            if client is None:
                recovered.append(page_document)
                continue

            page = document[page_document.page - 1]
            width, height = max(1.0, float(page.rect.width)), max(1.0, float(page.rect.height))
            scale = min(1.7, 2200 / max(width, height), math.sqrt(4_500_000 / (width * height)))
            pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
            image_bytes = pixmap.tobytes("png")
            try:
                value = client.complete_json(
                    PAGE_OCR_PROMPT,
                    image_bytes=image_bytes,
                    image_mime="image/png",
                )
            except QwenMultimodalAPIError as exc:
                logger.warning("Qwen page OCR failed for %s page %s: %s", path.name, page_document.page, exc)
                recovered.append(page_document)
                continue
            text = _ocr_text(value.get("text"))
            if len(re.sub(r"\s+", "", text)) < 30:
                logger.warning("Qwen page OCR returned too little text for %s page %s", path.name, page_document.page)
                recovered.append(page_document)
                continue
            chapter, section = _ocr_heading_context(
                value, text, previous_chapter, previous_section
            )
            concepts = _ocr_concepts(value.get("concepts"), text)
            previous_chapter, previous_section = chapter, section
            cache_entries[page_document.page] = {
                "schema_version": PAGE_OCR_SCHEMA_VERSION,
                "document_hash": document_hash,
                "model": client.model,
                "page": page_document.page,
                "source_page": page_document.source_page or page_document.page,
                "text": text,
                "chapter": chapter,
                "section": section,
                "concepts": concepts,
            }
            _write_page_ocr_cache(cache_path, cache_entries)
            recovered.append(replace(
                page_document,
                text=text,
                chapter=chapter or page_document.chapter,
                section=section or chapter or page_document.section,
                extra={
                    **(page_document.extra or {}),
                    "ocr_concepts": concepts,
                    "ocr_processor": f"qwen-vl:{client.model}",
                },
            ))
    finally:
        document.close()
    return recovered


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


def _formula_text_from_words(page: fitz.Page, bbox: list[float]) -> str:
    """Extract only glyphs inside a formula box instead of its containing paragraph."""

    words: list[tuple[float, float, float, float, str, int, int, int]] = []
    target = fitz.Rect(*bbox)
    for raw in page.get_text("words"):
        word_rect = fitz.Rect(raw[:4])
        intersection = target & word_rect
        if intersection.is_empty:
            continue
        overlap = intersection.get_area() / max(word_rect.get_area(), 1e-6)
        if overlap >= 0.45:
            words.append(raw)
    words.sort(key=lambda item: (item[5], item[6], item[7], item[0]))
    grouped: dict[tuple[int, int], list[str]] = {}
    for word in words:
        grouped.setdefault((int(word[5]), int(word[6])), []).append(str(word[4]))
    lines = ["".join(parts) for _, parts in sorted(grouped.items())]
    return "\n".join(line for line in lines if line).strip()


def _formula_latex_from_pdf_geometry(page: fitz.Page, bbox: list[float]) -> str:
    """Recover display-math structure from native PDF spans and coordinates."""

    blocks = page.get_text("dict", clip=fitz.Rect(*bbox)).get("blocks", [])
    lines: list[dict[str, Any]] = []
    max_size = 0.0
    for block in blocks:
        if int(block.get("type", -1)) != 0:
            continue
        for line in block.get("lines", []):
            spans = [span for span in line.get("spans", []) if str(span.get("text", "")).strip()]
            if not spans:
                continue
            max_size = max(max_size, *(float(span.get("size", 0)) for span in spans))
            lines.append({"bbox": list(line.get("bbox", [0, 0, 0, 0])), "spans": spans})
    if not lines or max_size <= 0:
        return ""

    def format_span(span: dict[str, Any]) -> str:
        text = re.sub(r"\s+", "", str(span.get("text", "")))
        text = text.replace("β", r"\beta ").replace("α", r"\alpha ")
        text = text.replace("γ", r"\gamma ").replace("Δ", r"\Delta ")
        if not text:
            return ""
        if float(span.get("size", 0)) <= max_size * 0.72:
            return "_{" + text + "}"
        return text

    for line in lines:
        line["latex"] = "".join(format_span(span) for span in line["spans"])
        main_origins = [
            float(span.get("origin", [0, 0])[1])
            for span in line["spans"]
            if float(span.get("size", 0)) > max_size * 0.72
        ]
        line["baseline"] = sum(main_origins) / len(main_origins) if main_origins else float(line["bbox"][3])

    equality_lines = [line for line in lines if "=" in str(line["latex"])]
    if equality_lines:
        equality = min(equality_lines, key=lambda item: float(item["bbox"][0]))
        baseline = float(equality["baseline"])
        rhs_lines = [
            line for line in lines
            if line is not equality and float(line["bbox"][0]) >= float(equality["bbox"][2]) - 1
        ]
        numerator = [line for line in rhs_lines if float(line["baseline"]) < baseline - 4]
        denominator = [line for line in rhs_lines if float(line["baseline"]) > baseline + 4]
        if numerator and denominator:
            top = "".join(str(line["latex"]) for line in sorted(numerator, key=lambda item: item["bbox"][0]))
            bottom = "".join(str(line["latex"]) for line in sorted(denominator, key=lambda item: item["bbox"][0]))
            latex = str(equality["latex"]) + rf"\frac{{{top}}}{{{bottom}}}"
        else:
            same_baseline = [
                line for line in rhs_lines if abs(float(line["baseline"]) - baseline) <= 4
            ]
            latex = str(equality["latex"]) + "".join(
                str(line["latex"]) for line in sorted(same_baseline, key=lambda item: item["bbox"][0])
            )
    else:
        latex = "".join(str(line["latex"]) for line in sorted(lines, key=lambda item: (item["baseline"], item["bbox"][0])))

    latex = re.sub(r"\s+", " ", latex).strip()
    return latex if re.search(r"[=+\-\\]", latex) else ""


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
        "confidence": confidence,
    }


def _recognize_formula(
    client: QwenVisionClient | None,
    image_bytes: bytes,
    fallback_text: str,
    nearby_text: str,
) -> dict[str, Any]:
    """Recognize one display formula; inline math stays embedded in prose."""

    if client is None:
        return _normalize_formula_result({}, fallback_text)
    prompt = f"""你是电子电路教材公式识别器。图片只包含一个独立公式。
输出严格 JSON：{{"is_formula":true,"latex":"不含外层美元符号的 LaTeX","plain_text":"便于全文检索的线性文本","variables":[{{"symbol":"I_BQ","meaning":"静态基极电流"}}],"confidence":0.0}}。
要求：准确恢复上下标、希腊字母、分数、绝对值、单位与公式编号；不得把邻近正文补进公式，不清楚的字符使用 ?，不得猜造数值。
邻近正文仅用于消歧：{nearby_text[:800]}"""
    try:
        result = client.complete_json(prompt, image_bytes=image_bytes, image_mime="image/png")
    except QwenMultimodalAPIError as exc:
        logger.warning("Qwen3-VL formula recognition failed; using PDF text fallback: %s", exc)
        result = {}
    return _normalize_formula_result(result, fallback_text)


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
    # Always serialize from the structured component list. This prevents a VLM
    # from silently inserting numeric values in an otherwise correct raw netlist.
    netlist = _synthesize_netlist(components) if components else ""
    return {
        "is_circuit": is_circuit,
        "components": components,
        "nets": nets,
        "netlist": netlist,
        "description": str(value.get("description", ""))[:8000],
        "caption": str(value.get("caption", ""))[:1000],
        "confidence": confidence,
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
    prompt = """只有包含至少两个电气元件且存在可核验导线连接的原理图、等效电路或小信号模型才可令 is_circuit=true。器件实物/外形、单个器件符号、半导体物理结构、特性曲线、波形图和系统框图必须令 is_circuit=false，即使它们包含端子、箭头或直线。\n""" + f"""你是电路图结构化识别器。判断图片是否为电路图；若是，结合邻近教材正文识别所有元件、端口、节点和导线连接，输出可复核的 SPICE 风格 Netlist 和中文结构/功能描述。跨线但无连接点时不得当作连接。看不清的值写 null，不得猜测。components.terminals 必须直接填写网络 ID；BJT 顺序为 collector/base/emitter，MOS 顺序为 drain/gate/source/bulk，其它二端元件按图中方向列出。
附近正文：{element.nearby_text[:1800]}
返回 JSON：{{"is_circuit":true,"caption":"","components":[{{"id":"R1","type":"resistor","value":"4 ohm","terminals":["n1","n2"],"bbox":[]}}],"nets":[{{"id":"n1","terminals":["R1.1"]}}],"netlist":"R1 n1 n2 4","description":"...","confidence":0.0}}。"""
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
        logger.warning("Qwen3-VL circuit analysis failed; using local uncertain fallback: %s", exc)
        raw_vlm_result = {}
    vlm_result = _normalize_circuit_result(raw_vlm_result)
    result = vlm_result
    nearby_lower = f"{element.caption}\n{element.nearby_text}".lower()
    chart_markers = ("波形", "曲线", "坐标", "频谱", "特性图")
    heuristic_circuit = likely and not any(marker in nearby_lower for marker in chart_markers)
    # The local edge/line heuristic is deliberately not authoritative. Crystal
    # lattices, device cross-sections and characteristic plots are visually
    # similar to schematics and previously became false circuit nodes whenever
    # the vision endpoint timed out.
    is_circuit = bool(result.get("is_circuit"))
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
            f"qwen-vl:{client.model}"
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
        element.description = result.get("description") or element.nearby_text[:1200]
        element.caption = result.get("caption", "")
        element.confidence = float(result.get("confidence") or heuristic_score)
        if heuristic_circuit and not raw_vlm_result:
            heuristic_processor = "opencv-heuristic-unconfirmed"
            element.processor = (
                f"{element.processor}+{heuristic_processor}"
                if element.processor.startswith("pdf-extract-kit")
                else heuristic_processor
            )
            element.uncertain = True
    _enforce_verified_circuit(element)


def _qwen_table(
    client: QwenVisionClient | None,
    image_bytes: bytes,
    nearby_text: str,
) -> dict[str, Any]:
    if client is None:
        return {}
    prompt = f"""识别图片中的电路课程表格，只返回 JSON：
{{"markdown":"完整Markdown表格","columns":["列名"],"rows":[["单元格"]],"description":"表格含义","confidence":0.0}}
保留单位、公式和空单元格，不得编造被遮挡内容。
附近教材文字：{nearby_text[:1200]}"""
    try:
        return client.complete_json(prompt, image_bytes=image_bytes, image_mime="image/png")
    except QwenMultimodalAPIError as exc:
        logger.warning("Qwen3-VL table recognition failed: %s", exc)
        return {}


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
) -> tuple[list[PageDocument], list[LayoutElement], list[dict[str, Any]]]:
    """Add layout, visual and circuit semantics while preserving original pages."""

    artifacts_dir = output_dir / "artifacts" / path.stem
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    vision_client = (
        QwenVisionClient(
            api_key=settings.qwen_api_key,
            model=settings.qwen_circuit_vision_model,
            base_url=settings.qwen_base_url,
        )
        if settings.qwen_api_key
        else None
    )
    cleaning_client = (
        CompatibleMultimodalClient(model_config)
        if model_config and model_config.enabled
        else None
    )
    document_hash = _file_sha256(path)
    page_documents = _ocr_scanned_pages(
        path, page_documents, output_dir, vision_client, document_hash
    )
    page_text_hashes = {
        item.page: hashlib.sha256(item.text.encode("utf-8")).hexdigest()
        for item in page_documents
    }
    audit_path = output_dir / f"{path.stem}.cleaning_audit.json"
    decisions: dict[int, dict[str, Any]] = {}
    if audit_path.exists():
        try:
            cached_audit = json.loads(audit_path.read_text(encoding="utf-8"))
            expected_method = (
                f"llm:{cleaning_client.config.provider}/{cleaning_client.config.model}"
                if cleaning_client
                else "rule"
            )
            decisions = {
                int(item["page"]): item
                for item in cached_audit
                if isinstance(item, dict)
                and item.get("method") == expected_method
                and item.get("document_hash") == document_hash
                and item.get("page_text_hash") == page_text_hashes.get(int(item["page"]))
            }
        except Exception:
            decisions = {}
    if not all(item.page in decisions for item in page_documents):
        decisions = _page_cleaning_decisions(page_documents, cleaning_client)
    # Re-apply safety policy to cached audits as the policy may become stricter
    # between builds even when the source document and model are unchanged.
    for page_document in page_documents:
        decision = decisions.setdefault(
            page_document.page,
            {
                "page": page_document.page,
                "source_page": page_document.source_page or page_document.page,
                "keep": True,
                "reason": "默认保留课程内容",
                "method": "rule",
                "remove_fragments": [],
            },
        )
        requested = decision.get(
            "requested_remove_fragments", decision.get("remove_fragments", [])
        )
        decision["requested_remove_fragments"] = requested
        decision["remove_fragments"] = [
            fragment for fragment in requested
            if _safe_partial_noise_fragment(str(fragment))
        ]
        if not decision.get("keep", True) and (
            extract_course_concepts(page_document.text)
            or re.search(r"[=+−±√∫ΣΩπ^_]", page_document.text)
        ):
            decision["keep"] = True
            decision["reason"] = "检测到课程概念或公式，安全策略强制保留"
    for decision in decisions.values():
        decision["document_hash"] = document_hash
        try:
            decision["page_text_hash"] = page_text_hashes.get(int(decision["page"]), "")
        except (TypeError, ValueError):
            decision["page_text_hash"] = ""
    kept_docs: list[PageDocument] = []
    for item in page_documents:
        decision = decisions.get(item.page, {})
        if not decision.get("keep", True):
            continue
        cleaned_text = item.text
        removed_characters = 0
        for fragment in decision.get("remove_fragments", []):
            if fragment in cleaned_text:
                cleaned_text = cleaned_text.replace(fragment, "")
                removed_characters += len(fragment)
        decision["removed_characters"] = removed_characters
        kept_docs.append(replace(item, text=cleaned_text.strip()))
    page_meta = {item.page: item for item in kept_docs}
    allowed_pages = set(page_meta)
    external = _external_pdf_extract_elements(path)
    elements: list[LayoutElement] = []
    image_counter = 0
    pdf_extract_kit = PDFExtractKitAdapter()
    pdf_extract_kit.write_manifest(output_dir)
    layout_records: list[dict[str, Any]] = []
    analysis_pages = sorted(allowed_pages)
    if settings.pdf_extract_kit_page_limit > 0:
        analysis_pages = analysis_pages[: settings.pdf_extract_kit_page_limit]
    analysis_page_set = set(analysis_pages)
    cached_images: dict[str, dict[str, Any]] = {}
    element_cache = output_dir / "multimodal_elements.jsonl"
    if element_cache.exists():
        try:
            for line in element_cache.read_text(encoding="utf-8").splitlines():
                item = json.loads(line)
                if item.get("source") == path.name and item.get("image_path") and item.get("content_hash"):
                    cached_images[str(item["content_hash"])] = item
        except Exception:
            cached_images = {}

    document = fitz.open(path)
    try:
        for page_no in sorted(allowed_pages):
            page = document[page_no - 1]
            meta = page_meta[page_no]
            blocks = page.get_text(
                "dict",
                flags=fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_PRESERVE_IMAGES,
            ).get("blocks", [])
            text_blocks: list[tuple[list[float], str]] = []
            for block in blocks:
                if int(block.get("type", -1)) != 0:
                    continue
                text = "\n".join(
                    "".join(str(span.get("text", "")) for span in line.get("spans", []))
                    for line in block.get("lines", [])
                ).strip()
                if text:
                    text_blocks.append(([round(float(v), 2) for v in block.get("bbox", [0, 0, 0, 0])], text))

            order = 0
            page_image_count = 0
            for bbox, text in text_blocks:
                element_type = (
                    "table"
                    if _looks_like_table(text)
                    else "formula"
                    if not pdf_extract_kit.available and _looks_like_formula(text)
                    else "text"
                )
                element_id, digest = _element_id(path.name, page_no, order, text)
                meta = page_meta[page_no]
                elements.append(LayoutElement(
                    id=element_id, source=path.name, page=page_no, element_type=element_type,
                    bbox=bbox, text=text, reading_order=order, chapter=meta.chapter,
                    section=meta.section, content_hash=digest,
                    source_page=meta.source_page or page_no,
                ))
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
                for region in _indexable_pdfkit_regions(detected_regions):
                    bbox_points = [round(value / render_scale, 2) for value in region.bbox_pixels]
                    category = region.category.lower()
                    crop_bytes = _crop_png(image_bgr, region.bbox_pixels)
                    if len(crop_bytes) < 300 or not _image_is_safe(crop_bytes):
                        continue
                    if category == "figure":
                        image_counter += 1
                        if (
                            settings.multimodal_image_limit
                            and image_counter > settings.multimodal_image_limit
                        ):
                            continue
                    element_id, digest = _element_id(path.name, page_no, order, crop_bytes)
                    image_path = artifacts_dir / (
                        f"p{page_no:04d}-{element_id[:10]}-pdfkit-{category}.png"
                    )
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
                        image_path=str(image_path.relative_to(output_dir)).replace("\\", "/"),
                        reading_order=order,
                        chapter=meta.chapter,
                        section=meta.section,
                        nearby_text=nearby,
                        content_hash=digest,
                        confidence=region.confidence,
                        processor=region.detector,
                        source_page=meta.source_page or page_no,
                    )
                    cached = cached_images.get(digest)
                    if category == "figure":
                        expected_vlm = f"qwen-vl:{vision_client.model}" if vision_client else ""
                        cache_compatible = bool(cached) and (
                            (bool(expected_vlm) and str(cached.get("processor", "")).endswith(expected_vlm))
                            or not expected_vlm
                        )
                        if cached and cache_compatible:
                            for field_name in (
                                "element_type", "caption", "components", "nets", "netlist",
                                "description", "confidence", "processor", "uncertain",
                            ):
                                if field_name in cached:
                                    setattr(element, field_name, cached[field_name])
                            if element.components:
                                element.netlist = _synthesize_netlist(element.components)
                        else:
                            _analyze_image(element, crop_bytes, vision_client)
                        _enforce_verified_circuit(element)
                        pdfkit_figure_count += 1
                        page_image_count += 1
                    elif category == "table":
                        expected_vlm = f"qwen-vl:{vision_client.model}" if vision_client else ""
                        if cached and expected_vlm and str(cached.get("processor", "")).endswith(expected_vlm):
                            for field_name in ("text", "description", "confidence", "processor", "uncertain"):
                                if field_name in cached:
                                    setattr(element, field_name, cached[field_name])
                        else:
                            result = _qwen_table(vision_client, crop_bytes, nearby)
                            element.text = str(result.get("markdown", "")).strip()
                            element.description = str(result.get("description", "")).strip()
                            element.processor += (
                                f"+{expected_vlm}" if result and expected_vlm else "+unrecognized"
                            )
                            try:
                                element.confidence = max(
                                    element.confidence, float(result.get("confidence", 0))
                                )
                            except (TypeError, ValueError):
                                pass
                            element.uncertain = not bool(element.text)
                    else:
                        fallback_text = _formula_text_from_words(page, bbox_points)
                        native_latex = _formula_latex_from_pdf_geometry(page, bbox_points)
                        expected_processor = (
                            "pymupdf-geometry-latex"
                            if native_latex
                            else f"formula-vl:{vision_client.model}"
                            if vision_client
                            else "pymupdf-formula"
                        )
                        cache_compatible = bool(cached) and (
                            str(cached.get("processor", "")).endswith(expected_processor)
                        )
                        if cache_compatible:
                            for field_name in (
                                "text", "description", "confidence", "processor", "uncertain"
                            ):
                                if field_name in cached:
                                    setattr(element, field_name, cached[field_name])
                        else:
                            result = (
                                _normalize_formula_result(
                                    {
                                        "is_formula": True,
                                        "latex": native_latex,
                                        "plain_text": re.sub(r"\s+", "", fallback_text),
                                        "confidence": 0.98,
                                    },
                                    fallback_text,
                                )
                                if native_latex
                                else _recognize_formula(
                                    vision_client, crop_bytes, fallback_text, nearby
                                )
                            )
                            if not result["is_formula"]:
                                continue
                            latex = str(result.get("latex", "")).strip()
                            plain_text = str(result.get("plain_text", "")).strip()
                            element.text = (
                                f"LaTeX: ${latex}$\n检索文本: {plain_text}"
                                if latex and plain_text
                                else f"LaTeX: ${latex}$" if latex else plain_text
                            )
                            variables = result.get("variables", [])
                            if variables:
                                element.description = "变量：" + json.dumps(
                                    variables, ensure_ascii=False
                                )
                            element.processor += f"+{expected_processor}"
                            element.confidence = max(
                                element.confidence, float(result.get("confidence", 0))
                            )
                            element.uncertain = not bool(latex) or element.confidence < 0.6
                    elements.append(element)
                    order += 1

            for block in blocks:
                if pdfkit_figure_count:
                    break
                if int(block.get("type", -1)) != 1 or not block.get("image"):
                    continue
                bbox = [round(float(v), 2) for v in block.get("bbox", [0, 0, 0, 0])]
                if _is_full_page_scan(
                    bbox, float(page.rect.width), float(page.rect.height), meta
                ):
                    continue
                image_counter += 1
                if settings.multimodal_image_limit and image_counter > settings.multimodal_image_limit:
                    break
                image_bytes = bytes(block["image"])
                if len(image_bytes) < 700 or not _image_is_safe(image_bytes):
                    continue
                ext = str(block.get("ext", "png")).lower()
                if ext not in {"png", "jpg", "jpeg", "webp", "bmp"}:
                    ext = "png"
                element_id, digest = _element_id(path.name, page_no, order, image_bytes)
                image_path = artifacts_dir / f"p{page_no:04d}-{element_id[:10]}.{ext}"
                image_path.write_bytes(image_bytes)
                nearby = "\n".join(text for _, text in text_blocks)[-3000:] or meta.text[-3000:]
                element = LayoutElement(
                    id=element_id, source=path.name, page=page_no, element_type="image",
                    bbox=bbox, image_path=str(image_path.relative_to(output_dir)).replace("\\", "/"),
                    reading_order=order, chapter=meta.chapter, section=meta.section,
                    nearby_text=nearby, content_hash=digest,
                    source_page=meta.source_page or page_no,
                )
                cached = cached_images.get(digest)
                expected_vlm = f"qwen-vl:{vision_client.model}" if vision_client else ""
                cache_compatible = bool(cached) and (
                    (bool(expected_vlm) and cached.get("processor") == expected_vlm)
                    or not expected_vlm
                )
                if cached and cache_compatible:
                    for field_name in (
                        "element_type", "caption", "components", "nets", "netlist",
                        "description", "confidence", "processor", "uncertain",
                    ):
                        if field_name in cached:
                            setattr(element, field_name, cached[field_name])
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
                expected_vlm = f"qwen-vl:{vision_client.model}" if vision_client else ""
                cache_compatible = bool(cached) and (
                    (bool(expected_vlm) and cached.get("processor") == expected_vlm)
                    or not expected_vlm
                )
                if cached and cache_compatible:
                    for field_name in (
                        "element_type", "caption", "components", "nets", "netlist",
                        "description", "confidence", "processor", "uncertain",
                    ):
                        if field_name in cached:
                            setattr(element, field_name, cached[field_name])
                    if element.components:
                        element.netlist = _synthesize_netlist(element.components)
                else:
                    _analyze_image(element, image_bytes, vision_client)
                _enforce_verified_circuit(element)
                elements.append(element)
    finally:
        document.close()
        if vision_client is not None:
            vision_client.close()

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
        elements.append(LayoutElement(
            id=element_id, source=path.name, page=page_no, element_type=kind,
            bbox=[float(v) for v in bbox[:4]], text=text, reading_order=100000 + index,
            chapter=meta.chapter, section=meta.section, content_hash=digest,
            processor="pdf-extract-kit",
            source_page=meta.source_page or page_no,
        ))

    audit = [decisions[number] for number in sorted(decisions)]
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
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


def build_local_knowledge_graph(chunks: Iterable[TextChunk]) -> dict[str, Any]:
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, str]] = []
    seen_edges: set[tuple[str, str, str]] = set()

    def add_edge(source: str, relation: str, target: str) -> None:
        edge = (source, relation, target)
        if edge not in seen_edges:
            edges.append({"source": source, "type": relation, "target": target})
            seen_edges.add(edge)

    for chunk in chunks:
        if chunk.doc_type == "question":
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
                component_id = f"component:{chunk.id}:{ref}"
                component_nodes[ref] = component_id
                nodes[component_id] = {
                    "id": component_id, "type": "component", "name": ref,
                    "component_type": str(component.get("type", "unknown")), "chunk_id": chunk.id,
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
    return {"schema_version": "2.1", "nodes": list(nodes.values()), "edges": edges}


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
        name = str(node.get("name", "")).strip()
        if not name or name.lower() in {"component", "unknown", "?"}:
            continue
        component_type = str(node.get("component_type", "unknown"))
        merged_id = "component:" + hashlib.sha1(
            f"{name.lower()}|{component_type.lower()}".encode("utf-8")
        ).hexdigest()[:16]
        original_to_visible_component[node_id] = merged_id
        visible.setdefault(merged_id, {
            "id": merged_id,
            "type": "component",
            "name": name,
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
        "schema_version": "2.2-student-projection",
        "nodes": list(visible.values()),
        "edges": projected_edges,
    }
