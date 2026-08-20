from __future__ import annotations

import argparse
import html
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch

try:
    # Use the same dataframe-to-NetworkX operation as Microsoft GraphRAG 2.7.2.
    from graphrag.index.operations.create_graph import create_graph
except ImportError as exc:  # pragma: no cover - exercised by CLI environment checks
    raise RuntimeError(
        "未安装 Microsoft GraphRAG；请在 llm 环境中安装 requirements.txt 后重试"
    ) from exc


TYPE_COLORS = {
    "电路": "#38BDF8",
    "电路参数": "#FBBF24",
    "器件与元件": "#FB7185",
    "物理过程与效应": "#A78BFA",
    "方法与模型": "#34D399",
    "课程概念": "#F472B6",
}
DEFAULT_TYPE_COLOR = "#94A3B8"
MODALITY_COLORS = {
    "text": "#60A5FA",
    "circuit": "#FB923C",
    "formula": "#C084FC",
    "table": "#4ADE80",
    "image": "#F472B6",
}
DEFAULT_EDGE_COLOR = "#64748B"
COMMUNITY_COLORS = {0: "#FDE047", 1: "#2DD4BF"}


def _required_columns(frame: pd.DataFrame, columns: set[str], filename: str) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{filename} 缺少 GraphRAG 字段：{', '.join(missing)}")


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if hasattr(value, "tolist"):
        converted = value.tolist()
        return converted if isinstance(converted, list) else [converted]
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _description_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(str(value or ""))
    except json.JSONDecodeError:
        return {"relation_original": str(value or "关系")}
    return parsed if isinstance(parsed, dict) else {"relation_original": str(value or "关系")}


def _font_properties() -> font_manager.FontProperties:
    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/msyhbd.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/System/Library/Fonts/PingFang.ttc"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return font_manager.FontProperties(fname=str(candidate))
    return font_manager.FontProperties(family="sans-serif")


def _display_label(value: str, width: int = 10, max_chars: int = 28) -> str:
    text = str(value).strip()
    if len(text) > max_chars:
        text = text[: max_chars - 1] + "…"
    if not text:
        return "未命名实体"
    return "\n".join(text[index : index + width] for index in range(0, len(text), width))


def _community_memberships(
    entities: pd.DataFrame,
    communities: pd.DataFrame,
) -> dict[str, int]:
    id_to_title = {
        str(row["id"]): str(row["title"])
        for _, row in entities.iterrows()
    }
    memberships: dict[str, int] = {}
    if communities.empty:
        return memberships
    minimum_level = int(communities["level"].min()) if "level" in communities else 0
    for _, row in communities.iterrows():
        if int(row.get("level", minimum_level)) != minimum_level:
            continue
        community = int(row.get("community", -1))
        for entity_id in _as_list(row.get("entity_ids")):
            title = id_to_title.get(str(entity_id))
            if title:
                memberships[title] = community
    return memberships


def load_graphrag_graph(
    output_dir: Path,
    *,
    include_isolated: bool = False,
) -> tuple[nx.DiGraph, dict[str, Any]]:
    """Load GraphRAG parquet outputs through GraphRAG's own graph constructor."""

    output_dir = output_dir.resolve()
    entities_path = output_dir / "entities.parquet"
    relationships_path = output_dir / "relationships.parquet"
    communities_path = output_dir / "communities.parquet"
    for path in (entities_path, relationships_path):
        if not path.exists():
            raise FileNotFoundError(f"缺少 GraphRAG 构建产物：{path}")

    entities = pd.read_parquet(entities_path)
    relationships = pd.read_parquet(relationships_path)
    communities = (
        pd.read_parquet(communities_path)
        if communities_path.exists()
        else pd.DataFrame()
    )
    _required_columns(
        entities,
        {"id", "title", "type", "description", "degree"},
        entities_path.name,
    )
    _required_columns(
        relationships,
        {"id", "source", "target", "description", "weight"},
        relationships_path.name,
    )

    entities = entities.copy()
    relationships = relationships.copy()
    entities["title"] = entities["title"].astype(str).str.strip()
    relationships["source"] = relationships["source"].astype(str).str.strip()
    relationships["target"] = relationships["target"].astype(str).str.strip()
    self_loops = relationships[relationships["source"] == relationships["target"]].copy()
    valid_relationships = relationships[
        (relationships["source"] != relationships["target"])
        & relationships["source"].ne("")
        & relationships["target"].ne("")
    ].copy()

    edge_attributes = [
        column
        for column in ("id", "description", "weight", "text_unit_ids")
        if column in valid_relationships.columns
    ]
    # This is Microsoft GraphRAG's canonical dataframe -> nx.Graph operation.
    graphrag_graph = create_graph(
        valid_relationships,
        edge_attr=edge_attributes,
        nodes=entities.copy(),
        node_id="title",
    )
    memberships = _community_memberships(entities, communities)

    graph = nx.DiGraph()
    for title, attributes in graphrag_graph.nodes(data=True):
        attrs = dict(attributes)
        attrs["community"] = memberships.get(str(title), -1)
        graph.add_node(str(title), **attrs)
    for _, row in valid_relationships.iterrows():
        payload = _description_payload(row.get("description"))
        modalities = [str(item) for item in _as_list(payload.get("modalities")) if str(item)]
        pages = sorted({
            int(item)
            for item in _as_list(payload.get("source_pages"))
            if str(item).isdigit()
        })
        relation = str(payload.get("relation_original") or "关系").strip()
        graph.add_edge(
            str(row["source"]),
            str(row["target"]),
            id=str(row.get("id", "")),
            relation=relation,
            relation_normalized=str(payload.get("relation_normalized") or relation),
            evidence_texts=[str(item) for item in _as_list(payload.get("evidence_texts"))],
            qualifiers=[str(item) for item in _as_list(payload.get("qualifiers"))],
            pages=pages,
            modalities=modalities,
            confidence=float(payload.get("confidence", 0.0) or 0.0),
            weight=float(row.get("weight", 1.0) or 1.0),
        )

    if not include_isolated:
        graph.remove_nodes_from([node for node, degree in graph.degree() if degree == 0])

    metadata = {
        "graphrag_output": str(output_dir),
        "entities_total": int(len(entities)),
        "entities_drawn": int(graph.number_of_nodes()),
        "relationships_total": int(len(relationships)),
        "relationships_drawn": int(graph.number_of_edges()),
        "self_relationships_excluded": int(len(self_loops)),
        "isolated_entities_excluded": int(len(entities) - graph.number_of_nodes()),
        "communities": int(len(communities)),
        "self_relationships": [
            {
                "source": str(row["source"]),
                "relation": str(
                    _description_payload(row.get("description")).get("relation_original", "关系")
                ),
            }
            for _, row in self_loops.iterrows()
        ],
    }
    return graph, metadata


def _graph_positions(graph: nx.DiGraph, seed: int) -> dict[str, tuple[float, float]]:
    if not graph:
        return {}
    undirected = graph.to_undirected()
    components = sorted(
        nx.connected_components(undirected),
        key=lambda item: (-len(item), sorted(str(node) for node in item)),
    )
    columns = min(5, max(1, math.ceil(math.sqrt(len(components) * 1.55))))
    cell_width = 5.2
    cell_height = 4.1
    positions: dict[str, tuple[float, float]] = {}
    for component_index, component_nodes in enumerate(components):
        subgraph = undirected.subgraph(component_nodes)
        node_count = subgraph.number_of_nodes()
        ordered_nodes = sorted(subgraph.nodes(), key=str)
        if node_count == 1:
            local = {ordered_nodes[0]: (0.0, 0.0)}
        elif node_count == 2:
            local = {
                ordered_nodes[0]: (-0.9, 0.0),
                ordered_nodes[1]: (0.9, 0.0),
            }
        elif node_count <= 4:
            local = nx.circular_layout(subgraph, scale=1.0)
        else:
            local = nx.spring_layout(
                subgraph,
                seed=seed + component_index,
                weight="weight",
                k=1.1,
                iterations=500,
                scale=1.25,
            )
        column = component_index % columns
        row = component_index // columns
        center_x = column * cell_width
        center_y = -row * cell_height
        for node, point in local.items():
            positions[str(node)] = (
                center_x + float(point[0]),
                center_y + float(point[1]),
            )
    return positions


def _edge_modality(attributes: dict[str, Any]) -> str:
    modalities = [str(item) for item in attributes.get("modalities", [])]
    return modalities[0] if modalities else "unknown"


def render_static_graph(
    graph: nx.DiGraph,
    positions: dict[str, tuple[float, float]],
    output_path: Path,
    *,
    title: str,
    pages: str,
) -> None:
    font = _font_properties()
    figure, axis = plt.subplots(figsize=(30, 19), facecolor="#07111F")
    axis.set_facecolor("#07111F")

    components = sorted(
        nx.connected_components(graph.to_undirected()),
        key=lambda item: (-len(item), sorted(str(node) for node in item)),
    )
    for component_index, component in enumerate(components, 1):
        xs = [positions[node][0] for node in component]
        ys = [positions[node][1] for node in component]
        left, right = min(xs) - 1.25, max(xs) + 1.25
        bottom, top = min(ys) - 1.05, max(ys) + 1.05
        axis.add_patch(FancyBboxPatch(
            (left, bottom),
            right - left,
            top - bottom,
            boxstyle="round,pad=0.08,rounding_size=0.12",
            facecolor="#0B1728",
            edgecolor="#24364F",
            linewidth=1.0,
            alpha=0.78,
            zorder=0,
        ))
        relation_count = graph.subgraph(component).number_of_edges()
        axis.text(
            (left + right) / 2,
            top - 0.15,
            f"连通子图 {component_index} · {len(component)} 实体 / {relation_count} 关系",
            color="#64748B",
            fontsize=6.8,
            fontproperties=font,
            ha="center",
            va="top",
            zorder=1,
        )

    nodes = list(graph.nodes())
    node_colors = [
        TYPE_COLORS.get(str(graph.nodes[node].get("type", "")), DEFAULT_TYPE_COLOR)
        for node in nodes
    ]
    node_borders = [
        COMMUNITY_COLORS.get(int(graph.nodes[node].get("community", -1)), "#CBD5E1")
        for node in nodes
    ]
    node_sizes = [
        650 + min(1200, 160 * int(graph.degree(node)))
        for node in nodes
    ]
    nx.draw_networkx_nodes(
        graph,
        positions,
        ax=axis,
        nodelist=nodes,
        node_color=node_colors,
        edgecolors=node_borders,
        linewidths=2.2,
        node_size=node_sizes,
        alpha=0.96,
    )

    for modality in sorted({_edge_modality(data) for _, _, data in graph.edges(data=True)}):
        edges = [
            (source, target)
            for source, target, data in graph.edges(data=True)
            if _edge_modality(data) == modality
        ]
        nx.draw_networkx_edges(
            graph,
            positions,
            ax=axis,
            edgelist=edges,
            edge_color=MODALITY_COLORS.get(modality, DEFAULT_EDGE_COLOR),
            width=1.8,
            alpha=0.82,
            arrows=True,
            arrowsize=18,
            arrowstyle="-|>",
            node_size=node_sizes,
            connectionstyle="arc3,rad=0.045",
            min_source_margin=7,
            min_target_margin=7,
        )

    for node_index, (node, (x, y)) in enumerate(positions.items()):
        offset = 15 if node_index % 2 == 0 else -15
        axis.annotate(
            _display_label(node),
            (x, y),
            xytext=(0, offset),
            textcoords="offset points",
            color="#07111F",
            ha="center",
            va="bottom" if offset > 0 else "top",
            fontsize=6.7,
            fontproperties=font,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "#E2E8F0",
                "edgecolor": "#64748B",
                "alpha": 0.9,
            },
            zorder=5,
        )
    for source, target, attributes in graph.edges(data=True):
        x1, y1 = positions[source]
        x2, y2 = positions[target]
        relation = str(attributes.get("relation", "关系"))
        relation = relation if len(relation) <= 12 else relation[:11] + "…"
        page_text = ",".join(f"p{page}" for page in attributes.get("pages", []))
        label = relation + (f" · {page_text}" if page_text else "")
        axis.text(
            (x1 + x2) / 2,
            (y1 + y2) / 2,
            label,
            color="#E2E8F0",
            fontsize=5.7,
            fontproperties=font,
            ha="center",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "#0F172A",
                "edgecolor": "#334155",
                "alpha": 0.9,
            },
            zorder=4,
        )

    figure.suptitle(
        title,
        color="#F8FAFC",
        fontsize=23,
        fontproperties=font,
        fontweight="bold",
        y=0.975,
    )
    axis.set_title(
        f"教材页码 {pages}｜{graph.number_of_nodes()} 个关系实体｜{graph.number_of_edges()} 条有效关系｜{len(components)} 个连通子图｜箭头表示 GraphRAG source → target",
        color="#94A3B8",
        fontsize=12,
        fontproperties=font,
        pad=16,
    )
    type_handles = [
        Patch(facecolor=color, edgecolor="#CBD5E1", label=entity_type)
        for entity_type, color in TYPE_COLORS.items()
        if any(str(data.get("type")) == entity_type for _, data in graph.nodes(data=True))
    ]
    modality_handles = [
        Line2D([0], [0], color=color, lw=2.4, label=modality)
        for modality, color in MODALITY_COLORS.items()
        if any(_edge_modality(data) == modality for _, _, data in graph.edges(data=True))
    ]
    legend = axis.legend(
        handles=type_handles + modality_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.055),
        ncol=max(1, min(10, len(type_handles) + len(modality_handles))),
        facecolor="#0F172A",
        edgecolor="#334155",
        labelcolor="#E2E8F0",
        prop=font,
        framealpha=0.95,
    )
    for text_item in legend.get_texts():
        text_item.set_color("#E2E8F0")
    axis.axis("off")
    axis.set_aspect("equal", adjustable="datalim")
    figure.tight_layout(rect=(0.015, 0.04, 0.985, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=190, facecolor=figure.get_facecolor(), bbox_inches="tight")
    plt.close(figure)


def _serializable_graph(
    graph: nx.DiGraph,
    positions: dict[str, tuple[float, float]],
) -> dict[str, Any]:
    x_values = [point[0] for point in positions.values()]
    y_values = [point[1] for point in positions.values()]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    x_span = max(1e-9, x_max - x_min)
    y_span = max(1e-9, y_max - y_min)
    nodes = []
    for node, attributes in graph.nodes(data=True):
        x, y = positions[node]
        nodes.append({
            "id": node,
            "label": node,
            "type": str(attributes.get("type", "未知类型")),
            "description": str(attributes.get("description", "")),
            "degree": int(graph.degree(node)),
            "community": int(attributes.get("community", -1)),
            "color": TYPE_COLORS.get(str(attributes.get("type", "")), DEFAULT_TYPE_COLOR),
            "border": COMMUNITY_COLORS.get(int(attributes.get("community", -1)), "#CBD5E1"),
            "x": round(100 + (x - x_min) / x_span * 1400, 3),
            "y": round(120 + (y_max - y) / y_span * 760, 3),
        })
    edges = []
    for index, (source, target, attributes) in enumerate(graph.edges(data=True)):
        modality = _edge_modality(attributes)
        edges.append({
            "id": attributes.get("id") or f"edge-{index}",
            "source": source,
            "target": target,
            "relation": str(attributes.get("relation", "关系")),
            "normalized": str(attributes.get("relation_normalized", "")),
            "pages": attributes.get("pages", []),
            "modalities": attributes.get("modalities", []),
            "modality": modality,
            "color": MODALITY_COLORS.get(modality, DEFAULT_EDGE_COLOR),
            "confidence": float(attributes.get("confidence", 0.0)),
            "qualifiers": attributes.get("qualifiers", []),
            "evidence": attributes.get("evidence_texts", []),
        })
    return {"nodes": nodes, "edges": edges}


HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE_HTML__</title>
<style>
:root{color-scheme:dark;font-family:"Microsoft YaHei","PingFang SC",sans-serif;background:#07111f;color:#e2e8f0}
*{box-sizing:border-box}body{margin:0;background:#07111f}.app{display:grid;grid-template-columns:minmax(0,1fr) 340px;height:100vh}
main{position:relative;overflow:hidden}.topbar{position:absolute;z-index:3;left:20px;right:20px;top:16px;display:flex;gap:12px;align-items:center;flex-wrap:wrap;padding:12px 16px;border:1px solid #334155;border-radius:14px;background:#0f172ae8;box-shadow:0 12px 32px #0007}
h1{font-size:18px;margin:0 16px 0 0;color:#f8fafc}.stat{font-size:12px;color:#94a3b8}.control{display:flex;gap:6px;align-items:center;font-size:12px;color:#cbd5e1}select,button{background:#172033;color:#e2e8f0;border:1px solid #475569;border-radius:7px;padding:6px 8px}button{cursor:pointer}button:hover{border-color:#38bdf8}
svg{width:100%;height:100%;display:block;background:radial-gradient(circle at 45% 35%,#13233b,#07111f 65%)}.edge{opacity:.78;stroke-width:2}.edge-label{fill:#cbd5e1;font-size:9px;paint-order:stroke;stroke:#07111f;stroke-width:4px;stroke-linejoin:round;pointer-events:none}.node{cursor:pointer}.node circle{filter:drop-shadow(0 3px 4px #0009)}.node text{fill:#07111f;font-size:10px;font-weight:700;text-anchor:middle;pointer-events:auto;cursor:pointer;paint-order:stroke;stroke:#ffffff44;stroke-width:1px}.hidden{display:none}
aside{border-left:1px solid #334155;background:#0b1424;padding:20px;overflow:auto}aside h2{font-size:17px;color:#f8fafc;margin:0 0 12px}.hint{color:#94a3b8;font-size:13px;line-height:1.7}.pill{display:inline-block;border:1px solid #475569;border-radius:999px;padding:3px 8px;margin:2px;font-size:11px}.detail{font-size:13px;line-height:1.7;white-space:pre-wrap;overflow-wrap:anywhere}.detail strong{color:#f8fafc}.legend{margin-top:22px;padding-top:16px;border-top:1px solid #243249}.legend-row{display:flex;align-items:center;gap:8px;margin:7px 0;font-size:12px}.dot{width:12px;height:12px;border-radius:50%}.line{width:22px;height:3px;border-radius:2px}.foot{margin-top:20px;color:#64748b;font-size:11px;line-height:1.6}
@media(max-width:900px){.app{grid-template-columns:1fr;grid-template-rows:68vh 32vh}aside{border-left:0;border-top:1px solid #334155}.topbar{left:8px;right:8px;top:8px}}
</style>
</head>
<body><div class="app"><main>
<div class="topbar"><h1>__TITLE_HTML__</h1><span class="stat" id="stats"></span>
<label class="control">页码 <select id="pageFilter"><option value="all">全部</option></select></label>
<label class="control">模态 <select id="modalityFilter"><option value="all">全部</option></select></label>
<label class="control"><input id="labelToggle" type="checkbox" checked>关系标签</label><button id="reset">复位视图</button></div>
<svg id="canvas" viewBox="0 0 1600 980" aria-label="GraphRAG 知识图谱"><defs><marker id="arrow" viewBox="0 0 10 10" refX="13" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="context-stroke"/></marker></defs><g id="viewport"><g id="edges"></g><g id="edgeLabels"></g><g id="nodes"></g></g></svg>
</main><aside><h2>实体 / 关系详情</h2><div id="detail" class="hint">点击节点查看实体类型、GraphRAG 描述和度数；点击关系查看来源页、证据、条件与知识模态。滚轮缩放，拖动画布平移。</div><div class="legend" id="legend"></div><div class="foot">数据来源：Microsoft GraphRAG 2.7.2 entities.parquet、relationships.parquet、communities.parquet。自关系在质量层过滤，不在画布展示。</div></aside></div>
<script>
const graph=__GRAPH_DATA__;const metadata=__METADATA__;const svg=document.getElementById('canvas'),viewport=document.getElementById('viewport'),nodeLayer=document.getElementById('nodes'),edgeLayer=document.getElementById('edges'),labelLayer=document.getElementById('edgeLabels'),detail=document.getElementById('detail');
const byId=new Map(graph.nodes.map(n=>[n.id,n]));const ns='http://www.w3.org/2000/svg';const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const elem=(name,attrs={})=>{const el=document.createElementNS(ns,name);for(const [k,v] of Object.entries(attrs))el.setAttribute(k,v);return el};
graph.edges.forEach(edge=>{const a=byId.get(edge.source),b=byId.get(edge.target);const line=elem('line',{x1:a.x,y1:a.y,x2:b.x,y2:b.y,stroke:edge.color,'marker-end':'url(#arrow)',class:'edge'});line.dataset.id=edge.id;line.addEventListener('click',()=>showEdge(edge));edgeLayer.appendChild(line);const label=elem('text',{x:(a.x+b.x)/2,y:(a.y+b.y)/2,class:'edge-label'});label.dataset.id=edge.id;label.textContent=`${edge.relation}${edge.pages.length?' · p'+edge.pages.join(','):''}`;labelLayer.appendChild(label)});
graph.nodes.forEach(node=>{const g=elem('g',{class:'node',transform:`translate(${node.x},${node.y})`});g.dataset.id=node.id;const radius=15+Math.min(12,node.degree*2.2);g.appendChild(elem('circle',{r:radius,fill:node.color,stroke:node.border,'stroke-width':node.community>=0?4:2}));const text=elem('text',{y:radius+14});const label=node.label.length>16?node.label.slice(0,15)+'…':node.label;text.textContent=label;g.appendChild(text);g.addEventListener('click',()=>showNode(node));nodeLayer.appendChild(g)});
function showNode(n){detail.className='detail';detail.innerHTML=`<strong>${esc(n.label)}</strong>\n<span class="pill">${esc(n.type)}</span><span class="pill">度数 ${n.degree}</span>${n.community>=0?`<span class="pill">社区 ${n.community}</span>`:''}\n\n${esc(n.description||'暂无实体描述')}`}
function showEdge(e){detail.className='detail';detail.innerHTML=`<strong>${esc(e.source)} → ${esc(e.target)}</strong>\n<span class="pill">${esc(e.relation)}</span><span class="pill">${esc(e.modality)}</span><span class="pill">置信度 ${e.confidence.toFixed(2)}</span>\n\n<strong>规范关系：</strong>${esc(e.normalized)}\n<strong>来源页：</strong>${esc(e.pages.join(', ')||'未标注')}\n<strong>适用条件：</strong>${esc(e.qualifiers.join('；')||'无')}\n<strong>证据：</strong>${esc(e.evidence.join('\n')||'无')}`}
const pages=[...new Set(graph.edges.flatMap(e=>e.pages))].sort((a,b)=>a-b),mods=[...new Set(graph.edges.map(e=>e.modality))].sort();const pageFilter=document.getElementById('pageFilter'),modFilter=document.getElementById('modalityFilter');pages.forEach(p=>pageFilter.add(new Option('第 '+p+' 页',p)));mods.forEach(m=>modFilter.add(new Option(m,m)));
function filter(){const p=pageFilter.value,m=modFilter.value,active=new Set();graph.edges.forEach(e=>{const visible=(p==='all'||e.pages.includes(Number(p)))&&(m==='all'||e.modality===m);document.querySelectorAll(`[data-id="${CSS.escape(e.id)}"]`).forEach(el=>el.classList.toggle('hidden',!visible));if(visible){active.add(e.source);active.add(e.target)}});graph.nodes.forEach(n=>document.querySelector(`.node[data-id="${CSS.escape(n.id)}"]`).classList.toggle('hidden',!active.has(n.id)));document.getElementById('stats').textContent=`${active.size} 个实体 / ${graph.edges.filter(e=>(p==='all'||e.pages.includes(Number(p)))&&(m==='all'||e.modality===m)).length} 条关系`}
pageFilter.addEventListener('change',filter);modFilter.addEventListener('change',filter);document.getElementById('labelToggle').addEventListener('change',e=>labelLayer.style.display=e.target.checked?'':'none');
let scale=1,tx=0,ty=0,drag=false,last=null;function transform(){viewport.setAttribute('transform',`translate(${tx} ${ty}) scale(${scale})`)}svg.addEventListener('wheel',e=>{e.preventDefault();scale=Math.max(.35,Math.min(4,scale*(e.deltaY<0?1.12:.89)));transform()},{passive:false});svg.addEventListener('pointerdown',e=>{if(e.target.closest('.node')||e.target.classList.contains('edge'))return;drag=true;last=[e.clientX,e.clientY];svg.setPointerCapture(e.pointerId)});svg.addEventListener('pointermove',e=>{if(!drag)return;tx+=(e.clientX-last[0])/scale;ty+=(e.clientY-last[1])/scale;last=[e.clientX,e.clientY];transform()});svg.addEventListener('pointerup',()=>drag=false);document.getElementById('reset').addEventListener('click',()=>{scale=1;tx=0;ty=0;transform()});
const typeColors=Object.fromEntries(graph.nodes.map(n=>[n.type,n.color])),modColors=Object.fromEntries(graph.edges.map(e=>[e.modality,e.color]));document.getElementById('legend').innerHTML='<strong>实体类型</strong>'+Object.entries(typeColors).map(([k,v])=>`<div class="legend-row"><span class="dot" style="background:${v}"></span>${esc(k)}</div>`).join('')+'<br><strong>知识模态</strong>'+Object.entries(modColors).map(([k,v])=>`<div class="legend-row"><span class="line" style="background:${v}"></span>${esc(k)}</div>`).join('')+'<div class="legend-row"><span class="dot" style="background:#38bdf8;border:3px solid #fde047"></span>黄色外圈：Leiden 社区 0</div><div class="legend-row"><span class="dot" style="background:#38bdf8;border:3px solid #2dd4bf"></span>青色外圈：Leiden 社区 1</div>';
filter();
</script></body></html>"""


def render_interactive_graph(
    graph: nx.DiGraph,
    positions: dict[str, tuple[float, float]],
    output_path: Path,
    *,
    title: str,
    metadata: dict[str, Any],
) -> None:
    graph_data = _serializable_graph(graph, positions)
    graph_json = json.dumps(graph_data, ensure_ascii=False).replace("</", "<\\/")
    metadata_json = json.dumps(metadata, ensure_ascii=False).replace("</", "<\\/")
    rendered = (
        HTML_TEMPLATE.replace("__TITLE_HTML__", html.escape(title))
        .replace("__GRAPH_DATA__", graph_json)
        .replace("__METADATA__", metadata_json)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")


def write_graphml(graph: nx.DiGraph, output_path: Path) -> None:
    def graphml_value(value: Any) -> Any:
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value, ensure_ascii=False)
        if hasattr(value, "tolist"):
            return json.dumps(value.tolist(), ensure_ascii=False)
        return value

    serializable = nx.DiGraph()
    for node, attributes in graph.nodes(data=True):
        serializable.add_node(
            node,
            **{
                key: graphml_value(value)
                for key, value in attributes.items()
                if value is not None and not (isinstance(value, float) and math.isnan(value))
            },
        )
    for source, target, attributes in graph.edges(data=True):
        serializable.add_edge(
            source,
            target,
            **{
                key: graphml_value(value)
                for key, value in attributes.items()
                if value is not None
            },
        )
    # GraphRAG snapshots use NetworkX's GraphML generator as well.
    output_path.write_text("\n".join(nx.generate_graphml(serializable)), encoding="utf-8")


def generate_visualization(
    graphrag_output: Path,
    destination: Path,
    *,
    pages: str,
    title: str,
    include_isolated: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    graph, metadata = load_graphrag_graph(
        graphrag_output,
        include_isolated=include_isolated,
    )
    if graph.number_of_nodes() == 0:
        raise ValueError("GraphRAG 结果没有可绘制的关系实体")
    destination.mkdir(parents=True, exist_ok=True)
    positions = _graph_positions(graph, seed)
    png_path = destination / "graphrag-knowledge-graph.png"
    html_path = destination / "graphrag-knowledge-graph.html"
    graphml_path = destination / "graphrag-knowledge-graph.graphml"
    render_static_graph(graph, positions, png_path, title=title, pages=pages)
    render_interactive_graph(
        graph,
        positions,
        html_path,
        title=title,
        metadata=metadata,
    )
    write_graphml(graph, graphml_path)

    metadata.update({
        "pages": pages,
        "entity_types": dict(Counter(
            str(attributes.get("type", "未知类型"))
            for _, attributes in graph.nodes(data=True)
        )),
        "modalities": dict(Counter(
            _edge_modality(attributes)
            for _, _, attributes in graph.edges(data=True)
        )),
        "source_pages": sorted({
            int(page)
            for _, _, attributes in graph.edges(data=True)
            for page in attributes.get("pages", [])
        }),
        "outputs": {
            "png": str(png_path.resolve()),
            "html": str(html_path.resolve()),
            "graphml": str(graphml_path.resolve()),
        },
    })
    summary_path = destination / "graphrag-knowledge-graph-summary.json"
    metadata["outputs"]["summary"] = str(summary_path.resolve())
    summary_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(
        description="从 Microsoft GraphRAG 2.7.2 Parquet 产物绘制知识图谱"
    )
    parser.add_argument(
        "graphrag_output",
        type=Path,
        help="包含 entities.parquet/relationships.parquet 的 GraphRAG output 目录",
    )
    parser.add_argument("--output", type=Path, required=True, help="可视化输出目录")
    parser.add_argument("--pages", default="未指定", help="展示在图标题中的教材页码")
    parser.add_argument("--title", default="Microsoft GraphRAG 知识图谱", help="图标题")
    parser.add_argument(
        "--include-isolated",
        action="store_true",
        help="同时画出没有关系边、仅有属性事实的实体",
    )
    parser.add_argument("--seed", type=int, default=42, help="NetworkX 布局随机种子")
    args = parser.parse_args()
    summary = generate_visualization(
        args.graphrag_output,
        args.output,
        pages=args.pages,
        title=args.title,
        include_isolated=args.include_isolated,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
