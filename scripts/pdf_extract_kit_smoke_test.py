from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import fitz
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.rag.pdf_extract_kit import PDFExtractKitAdapter


COLORS = {
    "figure": (0, 160, 255),
    "table": (255, 140, 0),
    "isolate_formula": (0, 210, 0),
    "inline": (180, 0, 180),
    "isolated": (0, 210, 0),
}


def _render_page(pdf_path: Path, page_number: int, dpi: int) -> np.ndarray:
    document = fitz.open(pdf_path)
    try:
        if not 1 <= page_number <= len(document):
            raise ValueError(f"页码必须在 1 到 {len(document)} 之间")
        scale = dpi / 72.0
        pixmap = document[page_number - 1].get_pixmap(
            matrix=fitz.Matrix(scale, scale), alpha=False
        )
        rgb = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
            pixmap.height, pixmap.width, pixmap.n
        )
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    finally:
        document.close()


def _draw_regions(image: np.ndarray, regions: list[dict[str, object]]) -> np.ndarray:
    canvas = image.copy()
    for item in regions:
        bbox = [int(round(float(value))) for value in item["bbox_pixels"]]
        category = str(item["category"])
        confidence = float(item["confidence"])
        color = COLORS.get(category, (50, 90, 220))
        left, top, right, bottom = bbox
        cv2.rectangle(canvas, (left, top), (right, bottom), color, 3)
        label = f"{category} {confidence:.2f}"
        label_width = max(120, len(label) * 11)
        label_top = max(0, top - 28)
        cv2.rectangle(
            canvas,
            (left, label_top),
            (min(canvas.shape[1] - 1, left + label_width), top),
            color,
            -1,
        )
        cv2.putText(
            canvas,
            label,
            (left + 4, max(18, top - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(
        description="对一个 PDF 页面执行 PDF-Extract-Kit Layout/MFD GPU 冒烟测试"
    )
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--page", type=int, default=101, help="从 1 开始的 PDF 页码")
    parser.add_argument("--dpi", type=int, default=144)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "tmp" / "pdfs" / "pdf_extract_kit_smoke"
    )
    args = parser.parse_args()

    pdf_path = args.pdf.resolve()
    if not pdf_path.is_file():
        raise FileNotFoundError(pdf_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    image = _render_page(pdf_path, args.page, args.dpi)
    rendered_path = args.output_dir / f"page-{args.page:04d}-rendered.png"
    if not cv2.imwrite(str(rendered_path), image):
        raise OSError(f"无法写入 {rendered_path}")

    adapter = PDFExtractKitAdapter()
    started = time.perf_counter()
    detected = adapter.detect(image)
    elapsed = time.perf_counter() - started
    regions = [item.to_dict() for item in detected]

    overlay = _draw_regions(image, regions)
    overlay_path = args.output_dir / f"page-{args.page:04d}-detected.png"
    if not cv2.imwrite(str(overlay_path), overlay):
        raise OSError(f"无法写入 {overlay_path}")

    cuda = {
        "available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "allocated_mib": (
            round(torch.cuda.memory_allocated(0) / 1024**2, 2)
            if torch.cuda.is_available()
            else 0
        ),
        "reserved_mib": (
            round(torch.cuda.memory_reserved(0) / 1024**2, 2)
            if torch.cuda.is_available()
            else 0
        ),
    }
    report = {
        "source": str(pdf_path),
        "page": args.page,
        "dpi": args.dpi,
        "image_size": {"width": int(image.shape[1]), "height": int(image.shape[0])},
        "elapsed_seconds": round(elapsed, 3),
        "cuda": cuda,
        "model_manifest": adapter.manifest(),
        "region_count": len(regions),
        "regions": regions,
        "rendered_image": str(rendered_path.resolve()),
        "detected_image": str(overlay_path.resolve()),
    }
    report_path = args.output_dir / f"page-{args.page:04d}-result.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
