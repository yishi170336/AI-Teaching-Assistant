from __future__ import annotations

from pathlib import Path

import networkx as nx

from scripts.visualize_neo4j_knowledge_base import (
    _html_data,
    render_fragment,
    render_html,
)


def _sample_graph() -> tuple[nx.DiGraph, dict[str, tuple[float, float]]]:
    graph = nx.DiGraph()
    graph.add_node(
        "entity:mirror",
        display_name="镜像电流源",
        entity_type="电路",
        description="输出电流近似等于参考电流。",
        source_pages=[96],
    )
    graph.add_node(
        "entity:reference",
        display_name="参考电流",
        entity_type="电路参数",
        description="镜像支路的参考电流。",
        source_pages=[96],
    )
    graph.add_edge(
        "entity:mirror",
        "entity:reference",
        relationship_type="近似等于",
        description="当晶体管电流增益足够大时成立。",
        source_pages=[96],
        modalities=["text", "formula"],
        confidence=0.98,
    )
    return graph, {"entity:mirror": (-1.0, 0.0), "entity:reference": (1.0, 0.0)}


def test_html_data_preserves_semantics_and_evidence() -> None:
    graph, positions = _sample_graph()

    data = _html_data(graph, positions)

    assert data["typeCounts"] == {"电路": 1, "电路参数": 1}
    assert data["relationCounts"] == {"近似等于": 1}
    assert data["edges"][0]["pages"] == [96]
    assert data["edges"][0]["modalities"] == ["text", "formula"]
    assert data["edges"][0]["confidence"] == 0.98


def test_renderers_create_standalone_and_scoped_fragment(tmp_path: Path) -> None:
    graph, positions = _sample_graph()
    standalone = tmp_path / "graph.html"
    fragment = tmp_path / "fragment.html"

    render_html(graph, positions, standalone, "sample-95-99")
    render_fragment(graph, positions, fragment, "sample-95-99")

    standalone_text = standalone.read_text(encoding="utf-8")
    fragment_text = fragment.read_text(encoding="utf-8")
    assert standalone_text.startswith("<!doctype html>")
    assert "镜像电流源" in standalone_text
    assert '<div id="neo4j-five-page-graph">' in fragment_text
    assert "#neo4j-five-page-graph .node" in fragment_text
    assert "height:100vh" not in fragment_text
    assert "<!doctype" not in fragment_text
    assert "<html" not in fragment_text
    assert "<body" not in fragment_text

