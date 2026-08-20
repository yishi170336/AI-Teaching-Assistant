from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.visualize_graphrag_graph import generate_visualization, load_graphrag_graph


def _write_sample_outputs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True)
    pd.DataFrame([
        {
            "id": "entity-current-source",
            "human_readable_id": 0,
            "title": "镜像电流源",
            "type": "电路",
            "description": "复制参考电流的电流源电路。",
            "text_unit_ids": ["tu-1"],
            "frequency": 1,
            "degree": 1,
            "x": 0.0,
            "y": 0.0,
        },
        {
            "id": "entity-output-resistance",
            "human_readable_id": 1,
            "title": "输出电阻",
            "type": "电路参数",
            "description": "衡量电流源恒流特性的参数。",
            "text_unit_ids": ["tu-1"],
            "frequency": 1,
            "degree": 1,
            "x": 0.0,
            "y": 0.0,
        },
        {
            "id": "entity-attribute-only",
            "human_readable_id": 2,
            "title": "属性事实实体",
            "type": "课程概念",
            "description": "只用于属性事实。",
            "text_unit_ids": ["tu-2"],
            "frequency": 1,
            "degree": 0,
            "x": None,
            "y": None,
        },
    ]).to_parquet(output_dir / "entities.parquet")
    relation_description = json.dumps({
        "relation_original": "具有",
        "relation_normalized": "HAS_PROPERTY",
        "evidence_texts": ["镜像电流源具有较高输出电阻。"],
        "qualifiers": ["晶体管参数匹配时"],
        "source_pages": [95],
        "modalities": ["text"],
        "confidence": 0.96,
    }, ensure_ascii=False)
    self_description = json.dumps({
        "relation_original": "等于",
        "source_pages": [96],
        "modalities": ["formula"],
        "confidence": 1.0,
    }, ensure_ascii=False)
    pd.DataFrame([
        {
            "id": "relationship-1",
            "human_readable_id": 0,
            "source": "镜像电流源",
            "target": "输出电阻",
            "description": relation_description,
            "weight": 1.0,
            "combined_degree": 2,
            "text_unit_ids": ["tu-1"],
        },
        {
            "id": "relationship-self",
            "human_readable_id": 1,
            "source": "输出电阻",
            "target": "输出电阻",
            "description": self_description,
            "weight": 1.0,
            "combined_degree": 2,
            "text_unit_ids": ["tu-2"],
        },
    ]).to_parquet(output_dir / "relationships.parquet")
    pd.DataFrame([{
        "id": "community-0",
        "human_readable_id": 0,
        "community": 0,
        "level": 0,
        "parent": -1,
        "children": [],
        "title": "镜像电流源社区",
        "entity_ids": ["entity-current-source", "entity-output-resistance"],
        "relationship_ids": ["relationship-1"],
        "text_unit_ids": ["tu-1"],
        "period": "2026-08-20",
        "size": 2,
    }]).to_parquet(output_dir / "communities.parquet")


def test_loads_graphrag_parquet_and_filters_nonsemantic_self_loop(tmp_path: Path) -> None:
    output_dir = tmp_path / "graphrag" / "output"
    _write_sample_outputs(output_dir)

    graph, metadata = load_graphrag_graph(output_dir)

    assert set(graph.nodes) == {"镜像电流源", "输出电阻"}
    assert graph["镜像电流源"]["输出电阻"]["relation"] == "具有"
    assert graph["镜像电流源"]["输出电阻"]["pages"] == [95]
    assert graph.nodes["镜像电流源"]["community"] == 0
    assert metadata["self_relationships_excluded"] == 1
    assert metadata["isolated_entities_excluded"] == 1


def test_generates_png_html_graphml_and_summary(tmp_path: Path) -> None:
    output_dir = tmp_path / "graphrag" / "output"
    destination = tmp_path / "visualization"
    _write_sample_outputs(output_dir)

    summary = generate_visualization(
        output_dir,
        destination,
        pages="95–99",
        title="五页 GraphRAG 测试图谱",
    )

    assert summary["entities_drawn"] == 2
    assert summary["relationships_drawn"] == 1
    assert (destination / "graphrag-knowledge-graph.png").stat().st_size > 1_000
    html_text = (destination / "graphrag-knowledge-graph.html").read_text(encoding="utf-8")
    assert "镜像电流源" in html_text
    assert '"pages": [95]' in html_text
    assert (destination / "graphrag-knowledge-graph.graphml").stat().st_size > 100
    assert (destination / "graphrag-knowledge-graph-summary.json").exists()
