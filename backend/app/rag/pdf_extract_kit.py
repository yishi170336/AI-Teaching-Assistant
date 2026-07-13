from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from backend.app.config import settings


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DetectedRegion:
    category: str
    bbox_pixels: list[float]
    confidence: float
    detector: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PDFExtractKitAdapter:
    """Use PDF-Extract-Kit's official Layout/MFD model wrappers in-process.

    OCR and UniMERNet are intentionally not imported because their pinned
    dependencies downgrade the platform's existing Torch/Transformers/OpenCV
    stack. Native PDF text plus Qwen3-VL handles their respective roles.
    """

    def __init__(self) -> None:
        self.root = (settings.root_dir / settings.pdf_extract_kit_dir).resolve()
        self.layout_weights = self.root / "models" / "Layout" / "YOLO" / "yolov10l_ft.pt"
        self.formula_weights = self.root / "models" / "MFD" / "YOLO" / "yolo_v8_ft.pt"
        self.layout = None
        self.formula = None
        self.device = 0 if _cuda_available() else "cpu"

    @property
    def available(self) -> bool:
        return (
            (self.root / "pdf_extract_kit").is_dir()
            and self.layout_weights.is_file()
            and self.formula_weights.is_file()
        )

    def load(self) -> bool:
        if self.layout is not None and self.formula is not None:
            return True
        if not self.available:
            logger.warning(
                "PDF-Extract-Kit is incomplete; expected %s and %s",
                self.layout_weights,
                self.formula_weights,
            )
            return False
        root_text = str(self.root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
        try:
            LayoutDetectionYOLO = _load_official_class(
                self.root / "pdf_extract_kit" / "tasks" / "layout_detection" / "models" / "yolo.py",
                "LayoutDetectionYOLO",
                "circuitmind_pdfkit_layout_yolo",
            )
            FormulaDetectionYOLO = _load_official_class(
                self.root / "pdf_extract_kit" / "tasks" / "formula_detection" / "models" / "yolo.py",
                "FormulaDetectionYOLO",
                "circuitmind_pdfkit_formula_yolo",
            )

            self.layout = LayoutDetectionYOLO({
                "model_path": str(self.layout_weights),
                "img_size": 1280,
                "conf_thres": 0.25,
                "iou_thres": 0.45,
                "batch_size": 1,
                "visualize": False,
                "device": self.device,
            })
            self.formula = FormulaDetectionYOLO({
                "model_path": str(self.formula_weights),
                "img_size": 1280,
                "conf_thres": 0.25,
                "iou_thres": 0.45,
                "batch_size": 1,
                "visualize": False,
                "device": self.device,
            })
            return True
        except Exception as exc:
            logger.exception("PDF-Extract-Kit model loading failed: %s", exc)
            self.layout = None
            self.formula = None
            return False

    def detect(self, image_bgr: np.ndarray) -> list[DetectedRegion]:
        if not self.load() or self.layout is None or self.formula is None:
            return []
        regions: list[DetectedRegion] = []
        regions.extend(self._predict(self.layout, image_bgr, "layout"))
        regions.extend(self._predict(self.formula, image_bgr, "formula"))
        return _deduplicate_regions(regions)

    def _predict(self, wrapper: Any, image_bgr: np.ndarray, detector: str) -> list[DetectedRegion]:
        result = wrapper.model.predict(
            image_bgr,
            imgsz=wrapper.img_size,
            conf=wrapper.conf_thres,
            iou=wrapper.iou_thres,
            device=self.device,
            verbose=False,
        )[0]
        boxes = result.boxes
        if boxes is None:
            return []
        names = wrapper.id_to_names
        values: list[DetectedRegion] = []
        for bbox, class_id, score in zip(
            boxes.xyxy.detach().cpu().tolist(),
            boxes.cls.detach().cpu().tolist(),
            boxes.conf.detach().cpu().tolist(),
        ):
            values.append(DetectedRegion(
                category=str(names.get(int(class_id), f"class_{int(class_id)}")),
                bbox_pixels=[round(float(value), 2) for value in bbox],
                confidence=round(float(score), 6),
                detector=f"pdf-extract-kit:{detector}",
            ))
        return values

    def manifest(self) -> dict[str, Any]:
        return {
            "enabled": self.available,
            "root": str(self.root),
            "device": str(self.device),
            "source_revision": "PDF-Extract-Kit-1.0.0-released/da99314",
            "layout_weights": _file_manifest(self.layout_weights),
            "formula_weights": _file_manifest(self.formula_weights),
        }

    def write_manifest(self, output_dir: Path) -> None:
        (output_dir / "pdf_extract_kit_manifest.json").write_text(
            json.dumps(self.manifest(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _load_official_class(path: Path, class_name: str, module_name: str) -> Any:
    """Load one official task module without importing optional PDF-Kit tasks."""

    if module_name in sys.modules:
        return getattr(sys.modules[module_name], class_name)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 PDF-Extract-Kit 模块：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return getattr(module, class_name)


def _file_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "exists": False}
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return {
        "path": str(path),
        "exists": True,
        "size": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _deduplicate_regions(regions: list[DetectedRegion]) -> list[DetectedRegion]:
    ordered = sorted(regions, key=lambda item: item.confidence, reverse=True)
    kept: list[DetectedRegion] = []
    for region in ordered:
        if any(
            _normalized_category(region.category) == _normalized_category(current.category)
            and _intersection_over_union(region.bbox_pixels, current.bbox_pixels) >= 0.75
            for current in kept
        ):
            continue
        kept.append(region)
    return sorted(kept, key=lambda item: (item.bbox_pixels[1], item.bbox_pixels[0]))


def _normalized_category(category: str) -> str:
    return "isolated_formula" if category in {"isolate_formula", "isolated"} else category


def _intersection_over_union(first: list[float], second: list[float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    return intersection / max(first_area + second_area - intersection, 1e-9)
