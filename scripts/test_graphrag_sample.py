from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, fields
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.config import settings  # noqa: E402
from backend.app.rag.graphrag_adapter import run_microsoft_graphrag  # noqa: E402
from backend.app.rag.knowledge_document import (  # noqa: E402
    compile_knowledge_document,
    enrich_formula_knowledge,
    enrich_knowledge_statements,
    write_knowledge_document,
)
from backend.app.rag.models import PageDocument  # noqa: E402
from backend.app.rag.multimodal import (  # noqa: E402
    BuildModelConfig,
    CompatibleMultimodalClient,
    LayoutElement,
    audit_visual_semantics,
    _apply_toc_section_catalog,
    _legacy_ocr_blocks,
    _normalize_section_heading,
    _ocr_blocks,
    _analyze_image,
)
from backend.app.services.qwen_multimodal_client import QwenVisionClient  # noqa: E402


def _page_range(value: str) -> tuple[int, int]:
    parts = value.split("-", 1)
    start = int(parts[0])
    end = int(parts[1]) if len(parts) == 2 else start
    if start < 1 or end < start:
        raise argparse.ArgumentTypeError("页码范围应为 START-END")
    return start, end


def _load_documents(cache_path: Path) -> list[PageDocument]:
    documents: list[PageDocument] = []
    previous_chapter = ""
    previous_section = ""
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        page = int(item["source_page"] or item["page"])
        text = str(item.get("text", "")).strip()
        chapter = str(item.get("chapter", "")).strip() or previous_chapter
        section = _normalize_section_heading(item.get("section", "")) or previous_section or chapter
        previous_chapter = chapter or previous_chapter
        previous_section = section or previous_section
        blocks = _ocr_blocks(item.get("blocks")) or _legacy_ocr_blocks(text)
        documents.append(PageDocument(
            text=text,
            source=cache_path.name.removesuffix(".page_ocr.jsonl") + ".pdf",
            page=int(item["page"]),
            source_page=page,
            chapter=chapter,
            section=section,
            extra={
                "ocr_processor": f"qwen-vl:{item.get('model', '')}",
                "ocr_concepts": item.get("concepts", []),
                "text_blocks": blocks,
            },
        ))
    return _apply_toc_section_catalog(documents)


def _load_elements(path: Path) -> list[LayoutElement]:
    allowed = {item.name for item in fields(LayoutElement)}
    values: list[LayoutElement] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        values.append(LayoutElement(**{key: value for key, value in item.items() if key in allowed}))
    return values


def _refresh_visual_semantics(
    index_dir: Path,
    elements: list[LayoutElement],
) -> int:
    """Re-evaluate only figure crops; page OCR and formula/table caches stay reusable."""

    refreshed = 0
    with QwenVisionClient(
        api_key=settings.qwen_api_key,
        model=settings.qwen_circuit_vision_model,
        base_url=settings.qwen_base_url,
    ) as client:
        for element in elements:
            if element.element_type not in {"image", "circuit"} or not element.image_path:
                continue
            image_path = (index_dir / element.image_path).resolve()
            if not image_path.exists() or index_dir.resolve() not in image_path.parents:
                continue
            _analyze_image(element, image_path.read_bytes(), client)
            refreshed += 1
    return refreshed


def main() -> int:
    parser = argparse.ArgumentParser(description="连续教材页面 GraphRAG 质量测试")
    parser.add_argument("index_dir", type=Path)
    parser.add_argument("--pages", type=_page_range, default=(95, 99))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--with-graphrag", action="store_true")
    parser.add_argument(
        "--visual-cache",
        type=Path,
        help="复用已用当前电路 schema 复核过的样本图件 JSONL",
    )
    args = parser.parse_args()

    cache_path = next(args.index_dir.glob("*.page_ocr.jsonl"))
    element_path = args.index_dir / "multimodal_elements.jsonl"
    start, end = args.pages
    documents = [
        item for item in _load_documents(cache_path)
        if start <= int(item.source_page or item.page) <= end
    ]
    elements = [
        item for item in _load_elements(element_path)
        if start <= int(item.source_page or item.page) <= end
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    config = None
    if args.with_graphrag:
        config = BuildModelConfig(
            provider="qwen",
            model="qwen3.7-flash",
            api_key=settings.qwen_api_key,
            base_url=settings.qwen_base_url,
            enable_thinking=False,
        )
        if not config.enabled:
            raise RuntimeError("未配置 QWEN_API_KEY，无法执行 GraphRAG 样本测试")
        if args.visual_cache:
            if not args.visual_cache.exists():
                raise RuntimeError(f"图件复核缓存不存在：{args.visual_cache}")
            elements = [
                item for item in _load_elements(args.visual_cache)
                if start <= int(item.source_page or item.page) <= end
            ]
            refreshed = 0
            reused_visuals = len(elements)
        else:
            refreshed = _refresh_visual_semantics(args.index_dir, elements)
            reused_visuals = 0
        (args.output / "refreshed_multimodal_elements.jsonl").write_text(
            "\n".join(
                json.dumps(asdict(element), ensure_ascii=False) for element in elements
            ) + "\n",
            encoding="utf-8",
        )
        visual_audit = audit_visual_semantics(elements)
        (args.output / "visual_semantic_quality_audit.json").write_text(
            json.dumps(visual_audit, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if visual_audit["status"] == "failed":
            raise RuntimeError(
                "图像语义质量门禁失败："
                f"{visual_audit['critical_issues']} 个类型与拓扑冲突"
            )
    else:
        refreshed = 0
        reused_visuals = 0
        visual_audit = audit_visual_semantics(elements)
    units = compile_knowledge_document(documents, elements)
    if config is not None:
        units = enrich_formula_knowledge(
            units,
            CompatibleMultimodalClient(config),
            cache_path=args.output / "knowledge_document_enrichment.jsonl",
        )
        units = enrich_knowledge_statements(
            units,
            CompatibleMultimodalClient(config),
            cache_path=args.output / "knowledge_statements.jsonl",
        )
    write_knowledge_document(units, args.output)
    summary = {
        "pages": [start, end],
        "documents": len(documents),
        "elements": len(elements),
        "knowledge_units": len(units),
        "knowledge_statements": sum(len(item.statements) for item in units),
        "refreshed_visual_elements": refreshed,
        "reused_visual_elements": reused_visuals,
        "visual_quality": visual_audit,
        "unit_ranges": [[item.page_start, item.page_end] for item in units],
        "warnings": sum(len(item.quality.get("warnings", [])) for item in units),
    }
    if args.with_graphrag:
        assert config is not None
        graph, audit = run_microsoft_graphrag(
            units,
            args.output,
            config,
            settings.embedding_model_path,
        )
        (args.output / "semantic_knowledge_graph.json").write_text(
            json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (args.output / "semantic_quality_audit.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summary["graph"] = graph["stats"]
        summary["graph_quality"] = audit
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
