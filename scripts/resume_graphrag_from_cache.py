from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.config import settings  # noqa: E402
from backend.app.rag.graphrag_adapter import (  # noqa: E402
    audit_microsoft_graphrag,
    run_microsoft_graphrag,
)
from backend.app.rag.knowledge_document import (  # noqa: E402
    compile_knowledge_document,
    enrich_formula_knowledge,
    enrich_knowledge_statements,
    knowledge_units_to_chunks,
    write_knowledge_document,
)
from backend.app.rag.models import PageDocument  # noqa: E402
from backend.app.rag.multimodal import (  # noqa: E402
    BuildModelConfig,
    CompatibleMultimodalClient,
    LayoutElement,
    _file_sha256,
    _apply_toc_section_catalog,
    _legacy_ocr_blocks,
    _normalize_section_heading,
    _ocr_blocks,
    _ocr_scanned_pages,
    audit_visual_semantics,
    build_chapter_knowledge_summaries,
    multimodal_chunks,
)
from backend.app.rag.pipeline import (  # noqa: E402
    _edge_noise,
    clean_page_text,
    extract_pdf,
    validate_graph_semantics,
    validate_section_semantics,
)
from backend.app.rag.semantic_graph import bind_chapter_knowledge_points  # noqa: E402
from backend.app.rag.stores import sync_neo4j_graph  # noqa: E402


STATUS_FILE = "graphrag_resume_status.json"


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _write_status(index_dir: Path, stage: str, **values: Any) -> None:
    payload = {"state": "running", "stage": stage, "updated_at": _now(), **values}
    (index_dir / STATUS_FILE).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_stem(cache_path: Path) -> str:
    suffix = ".page_ocr.jsonl"
    return cache_path.name[: -len(suffix)] if cache_path.name.endswith(suffix) else cache_path.stem


def load_page_documents(cache_path: Path) -> list[PageDocument]:
    documents: list[PageDocument] = []
    seen_pages: set[int] = set()
    previous_chapter = ""
    previous_section = ""
    for raw_line in cache_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        item = json.loads(raw_line)
        page = int(item["page"])
        if page in seen_pages:
            raise RuntimeError(f"OCR 缓存存在重复页码：{page}")
        seen_pages.add(page)
        source_page = int(item.get("source_page") or page)
        text = str(item.get("text", "")).strip()
        if not text:
            raise RuntimeError(f"OCR 缓存第 {page} 页为空")
        chapter = str(item.get("chapter", "")).strip() or previous_chapter
        section = (
            _normalize_section_heading(item.get("section", ""))
            or previous_section
            or chapter
        )
        previous_chapter = chapter or previous_chapter
        previous_section = section or previous_section
        blocks = _ocr_blocks(item.get("blocks")) or _legacy_ocr_blocks(text)
        documents.append(PageDocument(
            text=text,
            source=f"{_source_stem(cache_path)}.pdf",
            page=page,
            source_page=source_page,
            chapter=chapter,
            section=section,
            extra={
                "ocr_processor": f"qwen-vl:{item.get('model', '')}",
                "ocr_concepts": item.get("concepts", []),
                "ocr_section_source": str(item.get("section_source", "cache")),
                "ocr_section_confidence": float(item.get("section_confidence", 0.0) or 0.0),
                "ocr_heading_verification_attempted": bool(
                    item.get("heading_verification_attempted")
                ),
                "text_blocks": blocks,
            },
        ))
    documents.sort(key=lambda item: item.page)
    if not documents:
        raise RuntimeError(f"OCR 缓存为空：{cache_path}")
    expected_pages = set(range(documents[0].page, documents[-1].page + 1))
    missing = sorted(expected_pages - seen_pages)
    if missing:
        raise RuntimeError(f"OCR 缓存缺页：{missing[:20]}")
    return _apply_toc_section_catalog(documents)


def load_canonical_page_documents(
    source_pdf: Path,
    index_dir: Path,
) -> list[PageDocument]:
    """Replay the production OCR-cache loader without invoking native PDF text."""

    source_pdf = source_pdf.resolve()
    if not source_pdf.is_file():
        raise FileNotFoundError(f"教材 PDF 不存在：{source_pdf}")
    return _ocr_scanned_pages(
        source_pdf,
        extract_pdf(source_pdf, None),
        index_dir,
        None,
        _file_sha256(source_pdf),
        None,
    )


def load_layout_elements(path: Path) -> list[LayoutElement]:
    allowed = {item.name for item in fields(LayoutElement)}
    elements: list[LayoutElement] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        item = json.loads(raw_line)
        elements.append(LayoutElement(**{
            key: value for key, value in item.items() if key in allowed
        }))
    if not elements:
        raise RuntimeError(f"多模态缓存为空：{path}")
    return elements


def apply_cleaning_audit(
    documents: list[PageDocument],
    audit_path: Path,
) -> list[PageDocument]:
    decisions = {
        int(item["page"]): item
        for item in json.loads(audit_path.read_text(encoding="utf-8"))
        if isinstance(item, dict) and item.get("page") is not None
    }
    missing = [item.page for item in documents if item.page not in decisions]
    if missing:
        raise RuntimeError(f"清洗审计缺少页码：{missing[:20]}")
    kept: list[PageDocument] = []
    for document in documents:
        decision = decisions[document.page]
        if not decision.get("keep", True):
            continue
        text = document.text
        for fragment in decision.get("remove_fragments", []):
            value = str(fragment).strip()
            if value:
                text = text.replace(value, "")
        kept.append(replace(document, text=text.strip()))
    repeated_noise = _edge_noise([item.text for item in kept])
    return [
        replace(item, text=clean_page_text(item.text, repeated_noise) or item.text)
        for item in kept
    ]


def _find_inputs(index_dir: Path) -> tuple[Path, Path, Path]:
    ocr_paths = sorted(index_dir.glob("*.page_ocr.jsonl"))
    if len(ocr_paths) != 1:
        raise RuntimeError(
            f"需要且只能有一个 page OCR 缓存，当前找到 {len(ocr_paths)} 个"
        )
    ocr_path = ocr_paths[0]
    audit_path = index_dir / f"{_source_stem(ocr_path)}.cleaning_audit.json"
    element_path = index_dir / "multimodal_elements.jsonl"
    for path in (audit_path, element_path):
        if not path.is_file():
            raise FileNotFoundError(f"续跑检查点不存在：{path}")
    return ocr_path, audit_path, element_path


def _write_graph_artifacts(
    index_dir: Path,
    graph: dict[str, Any],
    audit: dict[str, Any],
) -> None:
    (index_dir / "semantic_knowledge_graph.json").write_text(
        json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (index_dir / "semantic_quality_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for artifact_name, values in (
        ("text_units.jsonl", graph.get("text_units", [])),
        ("evidence_store.jsonl", graph.get("evidence", [])),
        ("relationship_mentions.jsonl", graph.get("relationship_mentions", [])),
        ("attribute_facts.jsonl", graph.get("attribute_facts", [])),
    ):
        (index_dir / artifact_name).write_text(
            "\n".join(json.dumps(item, ensure_ascii=False) for item in values),
            encoding="utf-8",
        )
    for artifact_name, values in (
        ("relationship_links.json", graph.get("relationship_links", [])),
        ("communities.json", graph.get("communities", [])),
        ("community_reports.json", graph.get("community_reports", [])),
    ):
        (index_dir / artifact_name).write_text(
            json.dumps(values, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def run(args: argparse.Namespace) -> dict[str, Any]:
    index_dir = args.index_dir.resolve()
    if not index_dir.is_dir():
        raise FileNotFoundError(f"索引目录不存在：{index_dir}")
    ocr_path, cleaning_path, element_path = _find_inputs(index_dir)
    _write_status(index_dir, "loading_validated_caches")
    documents = apply_cleaning_audit(
        load_canonical_page_documents(args.source_pdf, index_dir),
        cleaning_path,
    )
    allowed_pages = {int(item.source_page or item.page) for item in documents}
    elements = [
        item for item in load_layout_elements(element_path)
        if int(item.source_page or item.page) in allowed_pages
    ]
    section_audit = validate_section_semantics(documents)
    visual_audit = audit_visual_semantics(elements)
    if section_audit["status"] == "failed":
        raise RuntimeError(
            f"章节语义门禁失败：{section_audit['critical_issues']} 个关键问题"
        )
    if visual_audit["status"] == "failed":
        raise RuntimeError(
            f"图像语义门禁失败：{visual_audit['critical_issues']} 个关键问题"
        )
    manifest = {
        "schema_version": "1.0-graphrag-cache-resume",
        "created_at": _now(),
        "ocr_cache": {"path": str(ocr_path), "sha256": _sha256(ocr_path)},
        "cleaning_audit": {"path": str(cleaning_path), "sha256": _sha256(cleaning_path)},
        "multimodal_cache": {"path": str(element_path), "sha256": _sha256(element_path)},
        "documents": len(documents),
        "elements": len(elements),
        "section_audit": section_audit,
        "visual_audit": visual_audit,
    }
    (index_dir / "graphrag_resume_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.preflight_only:
        result = {**manifest, "state": "preflight_passed"}
        (index_dir / STATUS_FILE).write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return result

    if not settings.qwen_api_key:
        raise RuntimeError("未配置 QWEN_API_KEY，无法从缓存续跑 Microsoft GraphRAG")
    if not settings.embedding_model_path.is_dir():
        raise FileNotFoundError(f"本地嵌入模型不存在：{settings.embedding_model_path}")
    model_config = BuildModelConfig(
        provider="qwen",
        model="qwen3.7-flash",
        api_key=settings.qwen_api_key,
        base_url=settings.qwen_base_url,
        enable_thinking=False,
    )
    client = CompatibleMultimodalClient(model_config)

    _write_status(
        index_dir,
        "compiling_knowledge_document",
        documents=len(documents),
        elements=len(elements),
    )
    units = compile_knowledge_document(documents, elements)
    if not units:
        raise RuntimeError("从缓存没有编译出知识单元")
    _write_status(index_dir, "enriching_formula_knowledge", knowledge_units=len(units))
    units = enrich_formula_knowledge(
        units,
        client,
        cache_path=index_dir / "knowledge_document_enrichment.jsonl",
    )
    _write_status(index_dir, "enriching_atomic_statements", knowledge_units=len(units))
    units = enrich_knowledge_statements(
        units,
        client,
        cache_path=index_dir / "knowledge_statements.jsonl",
    )
    write_knowledge_document(units, index_dir)

    _write_status(
        index_dir,
        "extracting_entities_relationships",
        knowledge_units=len(units),
        model="qwen3.7-flash",
        embedding_model=str(settings.embedding_model_path),
    )
    graph, audit = run_microsoft_graphrag(
        units,
        index_dir,
        model_config,
        settings.embedding_model_path,
    )
    chunks = knowledge_units_to_chunks(units) + multimodal_chunks(elements)
    chapter_summaries = build_chapter_knowledge_summaries(chunks)
    semantic_chapters, chapter_alignment = bind_chapter_knowledge_points(
        chapter_summaries, graph.get("nodes", [])
    )
    graph["chapters"] = semantic_chapters
    graph.setdefault("stats", {})["chapter_alignment"] = chapter_alignment
    audit = audit_microsoft_graphrag(graph)
    audit["chapter_alignment"] = chapter_alignment
    graph_validation = validate_graph_semantics(chunks, graph, audit)
    _write_graph_artifacts(index_dir, graph, audit)
    (index_dir / "chapter_knowledge_points.json").write_text(
        json.dumps({
            "schema_version": "2.0-semantic-entities",
            "chapters": semantic_chapters,
            "alignment": chapter_alignment,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    neo4j_status: dict[str, Any] = {"enabled": False, "reason": "not requested"}
    if args.sync_neo4j:
        _write_status(
            index_dir,
            "syncing_semantic_graph_to_neo4j",
            entities=len(graph.get("nodes", [])),
            relationships=len(graph.get("edges", [])),
        )
        neo4j_status = sync_neo4j_graph(args.knowledge_base, graph)
        if not neo4j_status.get("enabled"):
            raise RuntimeError(f"Neo4j 同步失败：{neo4j_status.get('reason', 'unknown')}")

    result = {
        "state": "completed",
        "stage": "completed",
        "completed_at": _now(),
        "knowledge_base": args.knowledge_base,
        "index_dir": str(index_dir),
        "documents": len(documents),
        "elements": len(elements),
        "knowledge_units": len(units),
        "entities": len(graph.get("nodes", [])),
        "relationships": len(graph.get("edges", [])),
        "communities": len(graph.get("communities", [])),
        "graph_validation": graph_validation,
        "neo4j": neo4j_status,
    }
    (index_dir / STATUS_FILE).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="从已验证 OCR、清洗审计和多模态缓存直接续跑 Microsoft GraphRAG"
    )
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--source-pdf", type=Path, required=True)
    parser.add_argument("--knowledge-base", required=True)
    parser.add_argument("--sync-neo4j", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        index_dir = args.index_dir.resolve()
        if index_dir.is_dir():
            (index_dir / STATUS_FILE).write_text(
                json.dumps({
                    "state": "failed",
                    "stage": "failed",
                    "updated_at": _now(),
                    "error": str(exc),
                }, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
