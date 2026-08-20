from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.config import settings  # noqa: E402
from backend.app.rag.graphrag_adapter import run_microsoft_graphrag  # noqa: E402
from backend.app.rag.knowledge_document import (  # noqa: E402
    compile_knowledge_document,
    enrich_formula_knowledge,
    write_knowledge_document,
)
from backend.app.rag.models import PageDocument  # noqa: E402
from backend.app.rag.multimodal import (  # noqa: E402
    BuildModelConfig,
    CompatibleMultimodalClient,
    LayoutElement,
    _apply_toc_section_catalog,
    _legacy_ocr_blocks,
    _normalize_section_heading,
    _ocr_blocks,
)


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


def main() -> int:
    parser = argparse.ArgumentParser(description="连续教材页面 GraphRAG 质量测试")
    parser.add_argument("index_dir", type=Path)
    parser.add_argument("--pages", type=_page_range, default=(95, 99))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--with-graphrag", action="store_true")
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
    units = compile_knowledge_document(documents, elements)
    if config is not None:
        units = enrich_formula_knowledge(
            units,
            CompatibleMultimodalClient(config),
            cache_path=args.output / "knowledge_document_enrichment.jsonl",
        )
    write_knowledge_document(units, args.output)
    summary = {
        "pages": [start, end],
        "documents": len(documents),
        "elements": len(elements),
        "knowledge_units": len(units),
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
