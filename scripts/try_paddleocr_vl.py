"""Run an isolated PaddleOCR-VL document parsing experiment.

The production OCR pipeline still uses Qwen.  This script exists so a local
PaddleOCR-VL checkout or installation can be evaluated without importing it
into the FastAPI process.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib.parse import urlparse


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Image/PDF path or an http(s) URL")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tmp/paddleocr-vl"),
        help="Directory for Markdown, JSON, extracted images, and benchmark.json",
    )
    parser.add_argument(
        "--page",
        type=int,
        help="One-based PDF page to render before OCR; omit to parse the whole PDF",
    )
    parser.add_argument(
        "--engine",
        choices=("paddle", "transformers"),
        default="transformers",
        help="Use transformers for NVIDIA GPU inference",
    )
    parser.add_argument("--device", default="gpu:0")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--pipeline-version", default="v1.6")
    parser.add_argument(
        "--model-source",
        choices=("BOS", "HUGGINGFACE"),
        default="BOS",
    )
    return parser


def _is_url(value: str) -> bool:
    return urlparse(value).scheme.lower() in {"http", "https"}


def _prepare_input(value: str, page_number: int | None, output_dir: Path) -> str:
    if _is_url(value):
        if page_number is not None:
            raise ValueError("--page can only be used with a local PDF")
        return value

    source = Path(value).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if page_number is None:
        return str(source)
    if source.suffix.lower() != ".pdf":
        raise ValueError("--page requires a PDF input")
    if page_number < 1:
        raise ValueError("--page must be at least 1")

    import fitz

    with fitz.open(source) as document:
        if page_number > document.page_count:
            raise ValueError(
                f"--page {page_number} exceeds the PDF page count {document.page_count}"
            )
        pdf_page = document[page_number - 1]
        pixmap = pdf_page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        rendered = output_dir / f"{source.stem}-page-{page_number:04d}.png"
        pixmap.save(rendered)
    return str(rendered.resolve())


def _version(module: Any) -> str:
    return str(getattr(module, "__version__", "unknown"))


def main() -> int:
    args = _parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", args.model_source)
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

    prepared_input = _prepare_input(args.input, args.page, args.output_dir)

    try:
        import paddleocr
        from paddleocr import PaddleOCRVL
    except ImportError as exc:
        raise SystemExit(
            "PaddleOCR-VL is not installed. Follow the optional GPU setup in README.md."
        ) from exc

    engine_config: dict[str, Any] | None = None
    torch_module: Any | None = None
    if args.engine == "transformers":
        import torch
        import transformers

        torch_module = torch
        major, minor = (int(item) for item in transformers.__version__.split(".")[:2])
        if (major, minor) < (5, 8):
            raise SystemExit(
                "The PaddleOCR GitHub transformers engine requires transformers>=5.8. "
                "Use the isolated GPU environment described in README.md."
            )
        if args.device.startswith("gpu") and not torch.cuda.is_available():
            raise SystemExit("CUDA is not available to PyTorch in this environment.")
        engine_config = {"dtype": args.dtype}

    started = perf_counter()
    pipeline = PaddleOCRVL(
        device=args.device,
        engine=args.engine,
        engine_config=engine_config,
        pipeline_version=args.pipeline_version,
    )
    initialization_seconds = perf_counter() - started

    started = perf_counter()
    results = list(pipeline.predict(prepared_input))
    inference_seconds = perf_counter() - started

    for result in results:
        result.save_to_json(save_path=args.output_dir)
        result.save_to_markdown(save_path=args.output_dir)

    peak_vram_gb = None
    torch_version = None
    if torch_module is not None:
        torch_version = _version(torch_module)
        if torch_module.cuda.is_available():
            peak_vram_gb = round(
                torch_module.cuda.max_memory_allocated() / 1024**3,
                3,
            )

    benchmark = {
        "source_input": args.input,
        "prepared_input": prepared_input,
        "page": args.page,
        "engine": args.engine,
        "device": args.device,
        "dtype": args.dtype if args.engine == "transformers" else None,
        "pipeline_version": args.pipeline_version,
        "paddleocr_version": _version(paddleocr),
        "torch_version": torch_version,
        "initialization_seconds": round(initialization_seconds, 3),
        "inference_seconds": round(inference_seconds, 3),
        "peak_vram_gb": peak_vram_gb,
        "result_count": len(results),
    }
    benchmark_path = args.output_dir / "benchmark.json"
    benchmark_path.write_text(
        json.dumps(benchmark, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(benchmark, ensure_ascii=False, indent=2))
    print(f"Saved outputs to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
