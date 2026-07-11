from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import math
import mimetypes
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable
from uuid import NAMESPACE_URL, uuid5

import fitz
import httpx

from backend.app.config import settings
from backend.app.rag.models import PageDocument, TextChunk


logger = logging.getLogger(__name__)


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
        page.page: {"page": page.page, "keep": True, "reason": "默认保留课程内容", "method": "rule"}
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
返回 JSON：{"decisions":[{"page":1,"keep":true,"reason":"..."}]}，不得改写页码。\n"""
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
                decisions[page_no] = {
                    "page": page_no,
                    "keep": bool(item.get("keep", True)),
                    "reason": str(item.get("reason", "模型语义清洗"))[:240],
                    "method": f"llm:{client.config.provider}/{client.config.model}",
                }
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


def _looks_like_formula(text: str) -> bool:
    if len(text) > 500 or not text.strip():
        return False
    math_chars = sum(char in "=+-±×÷√∫ΣΩμφλπ^_<>" for char in text)
    return math_chars >= 2 and bool(re.search(r"[A-Za-z0-9]", text))


def _looks_like_table(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    return len(lines) >= 3 and sum(bool(re.search(r"\s{2,}|\t", line)) for line in lines) >= 2


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


class SINAAdapter:
    """HTTP adapter around a separately deployed SINA GPU worker.

    The worker contract is deliberately small: multipart field ``image`` in,
    JSON with ``components``, ``nets``, ``netlist`` and optional ``confidence`` out.
    """

    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint.strip()

    def analyze(self, image_bytes: bytes, filename: str) -> dict[str, Any]:
        if not self.endpoint:
            return {}
        try:
            with httpx.Client(timeout=httpx.Timeout(300, connect=15)) as client:
                response = client.post(
                    self.endpoint,
                    files={"image": (filename, image_bytes, "application/octet-stream")},
                )
                response.raise_for_status()
            value = response.json()
            return value if isinstance(value, dict) else {}
        except Exception as exc:
            logger.warning("SINA worker unavailable; using VLM/heuristic fallback: %s", exc)
            return {}


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
    return {
        "is_circuit": is_circuit,
        "components": components,
        "nets": nets,
        "netlist": str(value.get("netlist", ""))[:16000],
        "description": str(value.get("description", ""))[:8000],
        "caption": str(value.get("caption", ""))[:1000],
        "confidence": confidence,
    }


def _analyze_image(
    element: LayoutElement,
    image_bytes: bytes,
    client: CompatibleMultimodalClient | None,
    sina: SINAAdapter,
) -> None:
    likely, heuristic_score = _circuit_image_heuristic(image_bytes)
    raw_sina_result = sina.analyze(image_bytes, Path(element.image_path or "figure.png").name)
    sina_result = _normalize_circuit_result(raw_sina_result)
    prompt = f"""你是电路图结构化识别器。判断图片是否为电路图；若是，结合邻近教材正文识别所有元件、端口、节点和导线连接，输出可复核的 SPICE 风格 Netlist 和中文结构/功能描述。跨线但无连接点时不得当作连接。看不清的值写 null，不得猜测。
附近正文：{element.nearby_text[:1800]}
返回 JSON：{{"is_circuit":true,"caption":"","components":[{{"id":"R1","type":"resistor","value":"4 ohm","terminals":["n1","n2"],"bbox":[]}}],"nets":[{{"id":"n1","terminals":["R1.1"]}}],"netlist":"R1 n1 n2 4","description":"...","confidence":0.0}}。"""
    raw_vlm_result = (
        client.complete_json(
            prompt,
            image_bytes=image_bytes,
            image_mime=mimetypes.guess_type(element.image_path or "figure.png")[0] or "image/png",
        )
        if client and client.config.enabled
        else {}
    )
    vlm_result = _normalize_circuit_result(raw_vlm_result)
    result = sina_result if sina_result.get("is_circuit") else vlm_result
    is_circuit = bool(result.get("is_circuit")) or (
        likely and not raw_sina_result and not raw_vlm_result
    )
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
        element.processor = "sina" if sina_result.get("is_circuit") else (
            f"vlm:{client.config.provider}/{client.config.model}" if vlm_result.get("is_circuit") and client else "opencv-heuristic"
        )
        element.uncertain = not bool(element.components and (element.nets or element.netlist))
    else:
        element.description = result.get("description") or element.nearby_text[:1200]
        element.caption = result.get("caption", "")
        element.confidence = float(result.get("confidence") or heuristic_score)


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
    client = CompatibleMultimodalClient(model_config) if model_config and model_config.enabled else None
    document_hash = _file_sha256(path)
    audit_path = output_dir / f"{path.stem}.cleaning_audit.json"
    decisions: dict[int, dict[str, Any]] = {}
    if audit_path.exists():
        try:
            cached_audit = json.loads(audit_path.read_text(encoding="utf-8"))
            expected_method = (
                f"llm:{client.config.provider}/{client.config.model}" if client else "rule"
            )
            decisions = {
                int(item["page"]): item
                for item in cached_audit
                if isinstance(item, dict)
                and item.get("method") == expected_method
                and item.get("document_hash") == document_hash
            }
        except Exception:
            decisions = {}
    if not all(item.page in decisions for item in page_documents):
        decisions = _page_cleaning_decisions(page_documents, client)
    for decision in decisions.values():
        decision["document_hash"] = document_hash
    kept_docs = [item for item in page_documents if decisions.get(item.page, {}).get("keep", True)]
    page_meta = {item.page: item for item in kept_docs}
    allowed_pages = set(page_meta)
    external = _external_pdf_extract_elements(path)
    elements: list[LayoutElement] = []
    image_counter = 0
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
                element_type = "table" if _looks_like_table(text) else "formula" if _looks_like_formula(text) else "text"
                element_id, digest = _element_id(path.name, page_no, order, text)
                meta = page_meta[page_no]
                elements.append(LayoutElement(
                    id=element_id, source=path.name, page=page_no, element_type=element_type,
                    bbox=bbox, text=text, reading_order=order, chapter=meta.chapter,
                    section=meta.section, content_hash=digest,
                ))
                order += 1

            for block in blocks:
                if int(block.get("type", -1)) != 1 or not block.get("image"):
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
                bbox = [round(float(v), 2) for v in block.get("bbox", [0, 0, 0, 0])]
                nearby = "\n".join(text for _, text in text_blocks)[-3000:]
                meta = page_meta[page_no]
                element = LayoutElement(
                    id=element_id, source=path.name, page=page_no, element_type="image",
                    bbox=bbox, image_path=str(image_path.relative_to(output_dir)).replace("\\", "/"),
                    reading_order=order, chapter=meta.chapter, section=meta.section,
                    nearby_text=nearby, content_hash=digest,
                )
                cached = cached_images.get(digest)
                expected_vlm = (
                    f"vlm:{client.config.provider}/{client.config.model}" if client else ""
                )
                cache_compatible = bool(cached) and (
                    (bool(settings.sina_endpoint) and cached.get("processor") == "sina")
                    or (bool(expected_vlm) and cached.get("processor") == expected_vlm)
                    or (not settings.sina_endpoint and not expected_vlm)
                )
                if cached and cache_compatible:
                    for field_name in (
                        "element_type", "caption", "components", "nets", "netlist",
                        "description", "confidence", "processor", "uncertain",
                    ):
                        if field_name in cached:
                            setattr(element, field_name, cached[field_name])
                else:
                    _analyze_image(element, image_bytes, client, SINAAdapter(settings.sina_endpoint))
                elements.append(element)
                page_image_count += 1
                order += 1

            # Many electronic textbooks store schematics as PDF vector paths,
            # not raster images. Render such a page so SINA/VLM can still see it.
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
                    nearby_text="\n".join(text for _, text in text_blocks)[-3000:],
                    content_hash=digest,
                    processor="pymupdf-vector-render",
                )
                cached = cached_images.get(digest)
                expected_vlm = (
                    f"vlm:{client.config.provider}/{client.config.model}" if client else ""
                )
                cache_compatible = bool(cached) and (
                    (bool(settings.sina_endpoint) and cached.get("processor") == "sina")
                    or (bool(expected_vlm) and cached.get("processor") == expected_vlm)
                    or (not settings.sina_endpoint and not expected_vlm)
                )
                if cached and cache_compatible:
                    for field_name in (
                        "element_type", "caption", "components", "nets", "netlist",
                        "description", "confidence", "processor", "uncertain",
                    ):
                        if field_name in cached:
                            setattr(element, field_name, cached[field_name])
                else:
                    _analyze_image(element, image_bytes, client, SINAAdapter(settings.sina_endpoint))
                elements.append(element)
    finally:
        document.close()

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
        ))

    audit = [decisions[number] for number in sorted(decisions)]
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return kept_docs, elements, audit


def multimodal_chunks(elements: Iterable[LayoutElement]) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    seen: set[tuple[str, int, str, str]] = set()
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
        tags = [element.element_type]
        for component in element.components:
            if not isinstance(component, dict):
                continue
            component_type = str(component.get("type", "")).strip()
            if component_type and component_type not in tags:
                tags.append(component_type)
        chunks.append(TextChunk(
            id=f"element-{element.id}", text=text, source=element.source,
            chapter=element.chapter, section=element.section,
            page_start=element.page, page_end=element.page, doc_type="multimodal",
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
        chunk_node = f"chunk:{chunk.id}"
        nodes[chunk_node] = {"id": chunk_node, "type": "chunk", "name": chunk.section, "chunk_id": chunk.id}
        concepts = list(dict.fromkeys(chunk.knowledge_tags))
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
    return {"schema_version": "2.0", "nodes": list(nodes.values()), "edges": edges}
