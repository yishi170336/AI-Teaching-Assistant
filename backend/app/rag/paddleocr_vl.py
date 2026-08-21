from __future__ import annotations

import gc
import hashlib
import logging
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


logger = logging.getLogger(__name__)

PADDLEOCR_VL_SCHEMA_VERSION = "3.0-paddleocr-vl-layout-evidence"
PADDLEOCR_VL_GIT_REVISION = "2661c7c0ef5c613e8f93c6e93b2e052399f0f854"
PADDLEOCR_VL_PIPELINE_VERSION = "v1.6"
PADDLEOCR_VL_MODEL_ID = f"PaddleOCR-VL@{PADDLEOCR_VL_GIT_REVISION[:12]}"


class PaddleOCRVLUnavailableError(RuntimeError):
    """Raised when the configured local PaddleOCR-VL runtime cannot be loaded."""


class PaddleOCRVLInferenceError(RuntimeError):
    """Raised when a page cannot be parsed by PaddleOCR-VL."""


@dataclass(frozen=True)
class PaddleOCRVLRuntime:
    requested_device: str
    device: str
    engine: str
    dtype: str
    pipeline_version: str
    model_revision: str = PADDLEOCR_VL_GIT_REVISION

    def to_dict(self) -> dict[str, str]:
        return {
            "requested_device": self.requested_device,
            "device": self.device,
            "engine": self.engine,
            "dtype": self.dtype,
            "pipeline_version": self.pipeline_version,
            "model_revision": self.model_revision,
            "model": PADDLEOCR_VL_MODEL_ID,
        }


_CHAPTER_PATTERN = re.compile(r"^第[零〇一二三四五六七八九十百两0-9]+章(?:\s|\S)")
_SECTION_PATTERN = re.compile(r"^\*?\d+(?:[.．]\d+){1,3}\s*\S+")
_LIST_PATTERN = re.compile(
    r"^(?:[-—–•·●○▪■◆◇※]|\(?\d{1,3}[)）.、]|[（(][一二三四五六七八九十]+[）)]|[①-⑳])\s*"
)
_EXERCISE_PATTERN = re.compile(
    r"^(?:例\s*\d|例题|习题|练习|思考题|自测题|复习题|\d+[.、]\s*(?:试|求|证明|画|分析))"
)

_HEADER_LABELS = {"header", "header_image", "page_header"}
_FOOTER_LABELS = {"footer", "footer_image", "page_footer", "number", "footnote"}
_FORMULA_LABELS = {
    "formula",
    "isolate_formula",
    "isolated_formula",
    "display_formula",
    "inline_formula",
    "equation",
}
_TABLE_LABELS = {"table", "table_body"}
_IMAGE_LABELS = {"image", "figure", "chart", "diagram"}
_CAPTION_LABELS = {
    "vision_footnote",
    "figure_caption",
    "image_caption",
    "chart_title",
    "table_caption",
    "formula_caption",
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def paddle_result_to_dict(result: Any) -> dict[str, Any]:
    """Convert PaddleX's dict-like result to durable, JSON-safe data."""

    json_value = getattr(result, "json", None)
    if isinstance(json_value, dict):
        value = json_value
    elif isinstance(result, dict):
        value = dict(result)
    else:
        try:
            value = dict(result)
        except (TypeError, ValueError) as exc:
            raise PaddleOCRVLInferenceError(
                f"PaddleOCR-VL returned an unsupported result: {type(result)!r}"
            ) from exc
    # Live PaddleX results expose ``json`` as {"res": payload}, while
    # save_to_json writes the payload directly. Normalize both contracts.
    while (
        isinstance(value, dict)
        and set(value) == {"res"}
        and isinstance(value.get("res"), dict)
    ):
        value = value["res"]
    normalized = _json_safe(value)
    return normalized if isinstance(normalized, dict) else {}


def compact_paddle_result(result: dict[str, Any]) -> dict[str, Any]:
    """Drop rendered image arrays while retaining replayable OCR/layout evidence."""

    layout = result.get("layout_det_res", {})
    layout_boxes = layout.get("boxes", []) if isinstance(layout, dict) else []
    return {
        "input_path": result.get("input_path"),
        "page_index": result.get("page_index"),
        "page_count": result.get("page_count"),
        "width": result.get("width"),
        "height": result.get("height"),
        "model_settings": result.get("model_settings", {}),
        "parsing_res_list": result.get("parsing_res_list", []),
        "layout_det_res": {"boxes": layout_boxes},
    }


def _valid_bbox(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return []
    bbox = [_safe_float(item, -1.0) for item in value]
    left, top, right, bottom = bbox
    if left < 0 or top < 0 or right <= left or bottom <= top:
        return []
    return bbox


def _normalized_bbox(value: Any, width: float, height: float) -> list[float]:
    bbox = _valid_bbox(value)
    if not bbox or width <= 0 or height <= 0:
        return []
    left, top, right, bottom = bbox
    return [
        round(max(0.0, min(1000.0, left / width * 1000)), 3),
        round(max(0.0, min(1000.0, top / height * 1000)), 3),
        round(max(0.0, min(1000.0, right / width * 1000)), 3),
        round(max(0.0, min(1000.0, bottom / height * 1000)), 3),
    ]


def _normalized_polygon(value: Any, width: float, height: float) -> list[list[float]]:
    if not isinstance(value, list) or width <= 0 or height <= 0:
        return []
    polygon: list[list[float]] = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            return []
        polygon.append([
            round(max(0.0, min(1000.0, _safe_float(point[0]) / width * 1000)), 3),
            round(max(0.0, min(1000.0, _safe_float(point[1]) / height * 1000)), 3),
        ])
    return polygon


def _bbox_iou(first: Iterable[float], second: Iterable[float]) -> float:
    a = list(first)
    b = list(second)
    if len(a) != 4 or len(b) != 4:
        return 0.0
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    if not intersection:
        return 0.0
    first_area = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    second_area = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / max(1.0, first_area + second_area - intersection)


def _block_type(label: str, text: str) -> str:
    normalized_label = label.strip().lower()
    compact = re.sub(r"\s+", " ", text).strip()
    if normalized_label in _HEADER_LABELS:
        return "page_header"
    if normalized_label in _FOOTER_LABELS:
        return "page_footer"
    if normalized_label in _FORMULA_LABELS:
        return "formula"
    if normalized_label in _TABLE_LABELS:
        return "table"
    if normalized_label in _IMAGE_LABELS:
        return "image"
    if normalized_label in _CAPTION_LABELS:
        return "figure_caption"
    if normalized_label in {"doc_title", "document_title"}:
        return "chapter_heading" if _CHAPTER_PATTERN.match(compact) else "section_heading"
    if normalized_label in {"paragraph_title", "section_title", "title"}:
        if _CHAPTER_PATTERN.match(compact):
            return "chapter_heading"
        return "section_heading"
    if _EXERCISE_PATTERN.match(compact):
        return "exercise"
    if normalized_label in {"list", "list_item", "reference"} or _LIST_PATTERN.match(compact):
        return "list_item"
    if _SECTION_PATTERN.match(compact) and len(compact) <= 90:
        return "section_heading"
    return "paragraph"


def _layout_confidence(
    label: str,
    bbox: list[float],
    layout_boxes: list[dict[str, Any]],
) -> float:
    candidates: list[tuple[float, float]] = []
    for box in layout_boxes:
        if not isinstance(box, dict):
            continue
        detected_bbox = _valid_bbox(box.get("coordinate"))
        if not detected_bbox:
            continue
        label_bonus = 1.0 if str(box.get("label", "")).lower() == label.lower() else 0.0
        overlap = _bbox_iou(bbox, detected_bbox)
        if overlap >= 0.65 or (label_bonus and overlap >= 0.4):
            candidates.append((label_bonus + overlap, _safe_float(box.get("score"))))
    return max(candidates, default=(0.0, 0.0))[1]


def repair_reading_order(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Repair local vertical inversions without destroying Paddle's column order."""

    if len(blocks) < 2:
        return blocks
    model_order = {
        index: int(block.get("model_reading_order", block.get("reading_order", index + 1)))
        for index, block in enumerate(blocks)
    }
    outgoing: dict[int, set[int]] = {index: set() for index in range(len(blocks))}
    incoming: dict[int, int] = {index: 0 for index in range(len(blocks))}
    for first_index, first in enumerate(blocks):
        first_bbox = first.get("bbox", [])
        if not isinstance(first_bbox, list) or len(first_bbox) != 4:
            continue
        first_width = max(1.0, first_bbox[2] - first_bbox[0])
        for second_index, second in enumerate(blocks):
            if first_index == second_index:
                continue
            second_bbox = second.get("bbox", [])
            if not isinstance(second_bbox, list) or len(second_bbox) != 4:
                continue
            horizontal = min(first_bbox[2], second_bbox[2]) - max(first_bbox[0], second_bbox[0])
            second_width = max(1.0, second_bbox[2] - second_bbox[0])
            overlap_ratio = horizontal / min(first_width, second_width)
            # Only constrain blocks in the same visual column. Paddle's ordering
            # remains authoritative between columns and overlapping regions.
            if overlap_ratio < 0.45 or first_bbox[3] > second_bbox[1] + 3:
                continue
            if second_index not in outgoing[first_index]:
                outgoing[first_index].add(second_index)
                incoming[second_index] += 1

    ready = [index for index, count in incoming.items() if count == 0]
    ordered_indices: list[int] = []
    while ready:
        ready.sort(key=lambda index: (model_order[index], index))
        current = ready.pop(0)
        ordered_indices.append(current)
        for target in sorted(outgoing[current]):
            incoming[target] -= 1
            if incoming[target] == 0:
                ready.append(target)
    if len(ordered_indices) != len(blocks):
        return blocks

    repaired: list[dict[str, Any]] = []
    for reading_order, original_index in enumerate(ordered_indices, 1):
        block = {**blocks[original_index]}
        old_order = int(block.get("model_reading_order", reading_order))
        corrections = list(block.get("corrections", []))
        if old_order != reading_order:
            corrections.append({
                "type": "geometry-reading-order",
                "from": old_order,
                "to": reading_order,
            })
        block["reading_order"] = reading_order
        block["corrections"] = corrections
        repaired.append(block)
    return repaired


def _deduplicate_blocks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for block in sorted(blocks, key=lambda item: float(item.get("confidence", 0)), reverse=True):
        duplicate = next(
            (
                existing
                for existing in kept
                if block.get("type") == existing.get("type")
                and re.sub(r"\s+", "", str(block.get("text", "")))
                == re.sub(r"\s+", "", str(existing.get("text", "")))
                and _bbox_iou(block.get("bbox", []), existing.get("bbox", [])) >= 0.65
            ),
            None,
        )
        if duplicate is None:
            kept.append(block)
    return kept


def _recover_list_items(blocks: list[dict[str, Any]]) -> None:
    for index, block in enumerate(blocks):
        if block.get("type") != "paragraph":
            continue
        bbox = block.get("bbox", [])
        text = str(block.get("text", ""))
        if len(text) > 180 or not isinstance(bbox, list) or len(bbox) != 4:
            continue
        neighbours = blocks[max(0, index - 2) : index] + blocks[index + 1 : index + 3]
        aligned_lists = [
            item
            for item in neighbours
            if item.get("type") == "list_item"
            and isinstance(item.get("bbox"), list)
            and len(item["bbox"]) == 4
            and abs(float(item["bbox"][0]) - float(bbox[0])) <= 35
        ]
        if aligned_lists:
            block["type"] = "list_item"
            block.setdefault("corrections", []).append({
                "type": "geometry-list-recovery",
                "reason": "aligned-with-neighbouring-list-items",
            })


def normalize_paddle_result(
    result: dict[str, Any],
    *,
    source: str,
    page: int,
) -> list[dict[str, Any]]:
    """Normalize one PaddleOCR-VL page to the ingestion layout-block contract."""

    width = max(1.0, _safe_float(result.get("width"), 1.0))
    height = max(1.0, _safe_float(result.get("height"), 1.0))
    layout = result.get("layout_det_res", {})
    layout_boxes = layout.get("boxes", []) if isinstance(layout, dict) else []
    raw_blocks = result.get("parsing_res_list", [])
    if not isinstance(raw_blocks, list):
        return []

    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_blocks, 1):
        if not isinstance(raw, dict):
            continue
        label = str(raw.get("block_label", "text")).strip().lower()
        text = str(raw.get("block_content", "")).strip()
        block_type = _block_type(label, text)
        if not text and block_type != "image":
            continue
        pixel_bbox = _valid_bbox(raw.get("block_bbox"))
        bbox = _normalized_bbox(pixel_bbox, width, height)
        if not bbox:
            continue
        raw_id = raw.get("block_id", index)
        try:
            model_order = int(raw.get("block_order") or index)
        except (TypeError, ValueError):
            model_order = index
        block_id = f"ocr:{source}:p{page}:b{raw_id}"
        normalized.append({
            "id": block_id,
            "type": block_type,
            "text": text,
            "bbox": bbox,
            "polygon": _normalized_polygon(
                raw.get("block_polygon_points", []), width, height
            ),
            "confidence": round(
                _layout_confidence(label, pixel_bbox, layout_boxes), 6
            ),
            "reading_order": model_order,
            "model_reading_order": model_order,
            "raw_label": label,
            "source_engine": "paddleocr-vl",
            "model_revision": PADDLEOCR_VL_GIT_REVISION,
            "corrections": [],
        })

    normalized = _deduplicate_blocks(normalized)
    normalized.sort(key=lambda item: (int(item["model_reading_order"]), item["bbox"][1]))
    _recover_list_items(normalized)
    return repair_reading_order(normalized)


def markdown_table_cells(markdown: str, *, block_id: str) -> list[dict[str, Any]]:
    """Return header-aware table-cell evidence from Paddle's Markdown output."""

    rows: list[list[str]] = []
    for raw_line in str(markdown).splitlines():
        line = raw_line.strip()
        if not (line.startswith("|") and line.endswith("|")):
            continue
        values = [item.strip() for item in line[1:-1].split("|")]
        if values and all(re.fullmatch(r":?-{2,}:?", item) for item in values):
            continue
        if any(values):
            rows.append(values)
    if len(rows) < 2:
        return []
    headers = rows[0]
    cells: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows[1:], 1):
        row_header = row[0] if row else ""
        for column_index, value in enumerate(row):
            if not value:
                continue
            column_header = headers[column_index] if column_index < len(headers) else ""
            cells.append({
                "id": f"{block_id}:r{row_index}:c{column_index}",
                "row": row_index,
                "column": column_index,
                "row_header": row_header,
                "column_header": column_header,
                "value": value,
            })
    return cells


class PaddleOCRVLClient:
    """Lazy local PaddleOCR-VL client intended to live for one build worker."""

    def __init__(
        self,
        *,
        device: str = "gpu:0",
        engine: str = "transformers",
        dtype: str = "float16",
        pipeline_version: str = PADDLEOCR_VL_PIPELINE_VERSION,
        model_source: str = "BOS",
    ) -> None:
        os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", model_source)
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        try:
            import torch
            import transformers
            from paddleocr import PaddleOCRVL
        except ImportError as exc:
            raise PaddleOCRVLUnavailableError(
                "PaddleOCR-VL 未安装；请在 llm 环境安装 requirements-paddleocr-vl-gpu.txt"
            ) from exc

        version_parts = transformers.__version__.split(".")[:2]
        try:
            transformers_version = tuple(int(item) for item in version_parts)
        except ValueError as exc:
            raise PaddleOCRVLUnavailableError(
                f"无法识别 transformers 版本：{transformers.__version__}"
            ) from exc
        if engine == "transformers" and transformers_version < (5, 8):
            raise PaddleOCRVLUnavailableError(
                "PaddleOCR-VL transformers GPU 后端要求 transformers>=5.8"
            )

        requested_device = device
        if device.startswith("gpu") and not torch.cuda.is_available():
            logger.warning("CUDA unavailable; PaddleOCR-VL explicitly falls back to CPU")
            device = "cpu"
            dtype = "float32"
        engine_config = {"dtype": dtype} if engine == "transformers" else None
        try:
            self._pipeline = PaddleOCRVL(
                device=device,
                engine=engine,
                engine_config=engine_config,
                pipeline_version=pipeline_version,
            )
        except Exception as exc:
            raise PaddleOCRVLUnavailableError(
                f"PaddleOCR-VL 初始化失败（device={device}, engine={engine}）：{exc}"
            ) from exc
        self._torch = torch
        self.runtime = PaddleOCRVLRuntime(
            requested_device=requested_device,
            device=device,
            engine=engine,
            dtype=dtype,
            pipeline_version=pipeline_version,
        )
        self.model = PADDLEOCR_VL_MODEL_ID

    @property
    def cache_identity(self) -> dict[str, str]:
        return self.runtime.to_dict()

    def predict_page(
        self,
        image_bytes: bytes,
        *,
        source: str,
        page: int,
    ) -> dict[str, Any]:
        temporary_path: Path | None = None
        try:
            handle, path = tempfile.mkstemp(prefix="paddleocr-vl-", suffix=".png")
            os.close(handle)
            temporary_path = Path(path)
            temporary_path.write_bytes(image_bytes)
            # PaddleOCR-VL's public path input preserves PaddleOCRVLBlock objects
            # through result serialization; ndarray input currently loses their
            # structured fields in some PaddleX releases.
            results = list(self._pipeline.predict(str(temporary_path)))
        except Exception as exc:
            raise PaddleOCRVLInferenceError(
                f"PaddleOCR-VL 解析 {source} 第 {page} 页失败：{exc}"
            ) from exc
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        if not results:
            raise PaddleOCRVLInferenceError(
                f"PaddleOCR-VL 未返回 {source} 第 {page} 页结果"
            )
        raw = paddle_result_to_dict(results[0])
        # Do not persist the deleted temporary filename as evidence provenance.
        raw["input_path"] = source
        raw.setdefault("page_index", max(0, page - 1))
        blocks = normalize_paddle_result(raw, source=source, page=page)
        if not blocks:
            parsing_values = raw.get("parsing_res_list", [])
            raise PaddleOCRVLInferenceError(
                f"PaddleOCR-VL 未识别出 {source} 第 {page} 页的版面块"
                f"（raw_keys={sorted(raw)}, parsing_count="
                f"{len(parsing_values) if isinstance(parsing_values, list) else 'invalid'}）"
            )
        return {
            "raw": compact_paddle_result(raw),
            "blocks": blocks,
            "width": int(_safe_float(raw.get("width"))),
            "height": int(_safe_float(raw.get("height"))),
        }

    def close(self) -> None:
        self._pipeline = None
        gc.collect()
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    def __enter__(self) -> PaddleOCRVLClient:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def stable_block_content_hash(block: dict[str, Any]) -> str:
    payload = "|".join(
        (
            str(block.get("type", "")),
            str(block.get("text", "")),
            ",".join(str(item) for item in block.get("bbox", [])),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
