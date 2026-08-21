from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.config import settings  # noqa: E402


TYPE_COLORS = {
    "电路": ("#F79A61", "#DF6C2D"),
    "电路参数": ("#8DCE91", "#55A862"),
    "器件与元件": ("#E8A4BE", "#C96B91"),
    "物理过程与效应": ("#C9A7DF", "#9C72BD"),
    "方法与模型": ("#F6C55B", "#DDA327"),
    "课程概念": ("#8EC8E8", "#4D9AC5"),
    "教材结构": ("#B8C0CC", "#7C8797"),
    "知识属性": ("#F2D67A", "#C9A83A"),
}
DEFAULT_COLOR = ("#B8C0CC", "#7C8797")
CANVAS = "#F6F8FA"
SIDEBAR = "#303240"
EDGE = "#AEB4BC"


def _font() -> font_manager.FontProperties:
    for path in (
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/msyhbd.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    ):
        if path.exists():
            return font_manager.FontProperties(fname=str(path))
    return font_manager.FontProperties(family="sans-serif")


def _display_name(properties: dict[str, Any], node_id: str) -> str:
    return str(
        properties.get("display_name")
        or properties.get("name")
        or properties.get("title")
        or node_id
    )


def load_neo4j_graph(knowledge_base: str) -> nx.MultiDiGraph:
    if not (settings.neo4j_uri and settings.neo4j_password):
        raise RuntimeError("NEO4J_URI/NEO4J_PASSWORD 未配置")
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )
    graph = nx.MultiDiGraph()
    try:
        driver.verify_connectivity()
        with driver.session(database=settings.neo4j_database) as session:
            node_rows = session.run(
                """
                MATCH (n:KnowledgeEntity {knowledge_base: $kb})
                RETURN n.id AS id, labels(n) AS labels, properties(n) AS properties
                """,
                kb=knowledge_base,
            ).data()
            edge_rows = session.run(
                """
                MATCH (a:KnowledgeEntity {knowledge_base: $kb})-[r]->
                      (b:KnowledgeEntity {knowledge_base: $kb})
                RETURN a.id AS source, b.id AS target, type(r) AS relationship_type,
                       properties(r) AS properties
                """,
                kb=knowledge_base,
            ).data()
    finally:
        driver.close()

    for row in node_rows:
        properties = dict(row.get("properties") or {})
        labels = [str(label) for label in row.get("labels", [])]
        entity_type = str(properties.get("entity_type", ""))
        if not entity_type:
            entity_type = next(
                (label for label in labels if label != "KnowledgeEntity"),
                "课程概念",
            )
        node_id = str(row["id"])
        node_attributes = {
            **properties,
            "entity_type": entity_type,
            "display_name": _display_name(properties, node_id),
            "labels": labels,
        }
        graph.add_node(node_id, **node_attributes)
    for row in edge_rows:
        source, target = str(row["source"]), str(row["target"])
        if source not in graph or target not in graph or source == target:
            continue
        properties = dict(row.get("properties") or {})
        edge_attributes = {
            **properties,
            "relationship_type": str(row["relationship_type"]),
        }
        graph.add_edge(source, target, **edge_attributes)
    graph.remove_nodes_from([node for node, degree in graph.degree() if degree == 0])
    return graph


def _hierarchical_positions(
    graph: nx.MultiDiGraph,
) -> dict[str, tuple[float, float]]:
    """Lay out a consolidated textbook graph by book/chapter/section scope."""

    positions: dict[str, tuple[float, float]] = {}
    scope_level = {
        node: str(attributes.get("scope_level", ""))
        for node, attributes in graph.nodes(data=True)
    }
    books = sorted(node for node, level in scope_level.items() if level == "book")
    chapters = sorted(
        (node for node, level in scope_level.items() if level == "chapter"),
        key=lambda node: str(graph.nodes[node].get("display_name", node)),
    )
    sections = sorted(
        (node for node, level in scope_level.items() if level == "section"),
        key=lambda node: str(graph.nodes[node].get("display_name", node)),
    )
    if not books or not chapters or not sections:
        return {}
    book = books[0]
    positions[book] = (0.0, 0.0)
    chapter_sections: dict[str, list[str]] = {chapter: [] for chapter in chapters}
    section_chapter: dict[str, str] = {}
    for source, target, attributes in graph.edges(data=True):
        if str(attributes.get("relationship_type", "")) != "HAS_SECTION":
            continue
        if source in chapter_sections and target in scope_level:
            chapter_sections[source].append(target)
            section_chapter[target] = source

    chapter_radius = 7.5
    for chapter_index, chapter in enumerate(chapters):
        chapter_angle = 2 * math.pi * chapter_index / max(1, len(chapters))
        chapter_x = chapter_radius * math.cos(chapter_angle)
        chapter_y = chapter_radius * math.sin(chapter_angle)
        positions[chapter] = (chapter_x, chapter_y)
        children = sorted(
            set(chapter_sections.get(chapter, [])),
            key=lambda node: str(graph.nodes[node].get("display_name", node)),
        )
        section_radius = 2.45 + 0.09 * math.sqrt(len(children))
        for section_index, section in enumerate(children):
            section_angle = chapter_angle + 2 * math.pi * section_index / max(1, len(children))
            positions[section] = (
                chapter_x + section_radius * math.cos(section_angle),
                chapter_y + section_radius * math.sin(section_angle),
            )

    section_members: dict[str, list[str]] = {section: [] for section in sections}
    attribute_parent: dict[str, str] = {}
    for source, target, attributes in graph.edges(data=True):
        relation = str(attributes.get("relationship_type", ""))
        if relation == "MENTIONED_IN" and target in section_members:
            section_members[target].append(source)
        if (
            relation in {"HAS_FORMULA", "HAS_VALUE"}
            and str(graph.nodes[target].get("entity_type", "")) == "知识属性"
        ):
            attribute_parent[target] = source
    golden_angle = math.pi * (3 - math.sqrt(5))
    for section, members in section_members.items():
        center = positions.get(section)
        if center is None:
            continue
        for member_index, member in enumerate(sorted(set(members))):
            radius = 0.11 * math.sqrt(member_index + 1)
            angle = member_index * golden_angle
            positions[member] = (
                center[0] + radius * math.cos(angle),
                center[1] + radius * math.sin(angle),
            )
    for attribute, parent in attribute_parent.items():
        if parent not in positions:
            continue
        phase = int(hashlib.sha1(attribute.encode("utf-8")).hexdigest()[:4], 16)
        angle = phase / 65535 * 2 * math.pi
        positions[attribute] = (
            positions[parent][0] + 0.13 * math.cos(angle),
            positions[parent][1] + 0.13 * math.sin(angle),
        )
    unplaced = sorted(set(graph.nodes()) - set(positions))
    outer_radius = chapter_radius + 4.8
    for index, node in enumerate(unplaced):
        angle = 2 * math.pi * index / max(1, len(unplaced))
        positions[node] = (
            outer_radius * math.cos(angle), outer_radius * math.sin(angle)
        )
    return positions


def _positions(
    graph: nx.MultiDiGraph,
    seed: int = 42,
) -> dict[str, tuple[float, float]]:
    undirected = graph.to_undirected()
    if not undirected:
        return {}
    if graph.number_of_nodes() > 800:
        hierarchical = _hierarchical_positions(graph)
        if hierarchical:
            return hierarchical
    raw = nx.spring_layout(
        undirected,
        seed=seed,
        weight="weight",
        k=3.35 / math.sqrt(max(2, undirected.number_of_nodes())),
        iterations=900,
        scale=1.0,
    )
    return {node: (float(point[0]), float(point[1])) for node, point in raw.items()}


def _short(text: str, limit: int = 13) -> str:
    value = str(text).strip()
    return value if len(value) <= limit else value[: limit - 1] + "…"


def render_png(
    graph: nx.MultiDiGraph,
    positions: dict[str, tuple[float, float]],
    destination: Path,
    knowledge_base: str,
) -> None:
    font = _font()
    figure = plt.figure(figsize=(27, 15), facecolor="#D7DCE3")
    sidebar = figure.add_axes((0.0, 0.0, 0.235, 1.0))
    canvas = figure.add_axes((0.245, 0.055, 0.745, 0.86))
    query_bar = figure.add_axes((0.245, 0.925, 0.745, 0.06))

    sidebar.set_facecolor(SIDEBAR)
    sidebar.set_xlim(0, 1)
    sidebar.set_ylim(0, 1)
    sidebar.axis("off")
    sidebar.patch.set_visible(True)
    sidebar.text(
        0.12, 0.955, "Database Information", color="white", fontsize=19,
        fontweight="bold", fontproperties=font, va="top",
    )
    sidebar.text(0.12, 0.89, "Use database", color="white", fontsize=12,
                 fontweight="bold", fontproperties=font)
    sidebar.add_patch(FancyBboxPatch(
        (0.12, 0.83), 0.76, 0.045, boxstyle="round,pad=0.004",
        facecolor="#F8FAFC", edgecolor="#D1D5DB",
    ))
    sidebar.text(0.145, 0.846, knowledge_base, color="#303240", fontsize=10,
                 fontproperties=font)

    type_counts = Counter(
        str(attributes.get("entity_type", "课程概念"))
        for _, attributes in graph.nodes(data=True)
    )
    relation_counts = Counter(
        str(attributes.get("relationship_type", "RELATED"))
        for _, _, attributes in graph.edges(data=True)
    )
    sidebar.text(0.12, 0.775, "Node Labels", color="white", fontsize=13,
                 fontweight="bold", fontproperties=font)
    y = 0.73
    for entity_type, count in type_counts.most_common():
        fill, border = TYPE_COLORS.get(entity_type, DEFAULT_COLOR)
        sidebar.add_patch(FancyBboxPatch(
            (0.12, y - 0.018), 0.38, 0.035, boxstyle="round,pad=0.004,rounding_size=0.018",
            facecolor=fill, edgecolor=border,
        ))
        sidebar.text(0.14, y - 0.006, f"{entity_type} ({count})", color="#26313D",
                     fontsize=8.3, fontproperties=font)
        y -= 0.048
    sidebar.text(0.12, y - 0.012, "Relationship Types", color="white", fontsize=13,
                 fontweight="bold", fontproperties=font)
    y -= 0.065
    for relation, count in relation_counts.most_common(12):
        sidebar.add_patch(FancyBboxPatch(
            (0.12, y - 0.015), 0.68, 0.03, boxstyle="round,pad=0.003",
            facecolor="#AEB3BF", edgecolor="#8F96A5",
        ))
        sidebar.text(0.14, y - 0.006, f"{_short(relation, 19)} ({count})",
                     color="#343846", fontsize=7.5, fontproperties=font)
        y -= 0.039

    query_bar.set_facecolor("#FDFDFE")
    query_bar.set_xlim(0, 1)
    query_bar.set_ylim(0, 1)
    query_bar.axis("off")
    query_bar.patch.set_visible(True)
    query_bar.text(
        0.02, 0.5,
        f"neo4j$ MATCH (a:KnowledgeEntity {{knowledge_base:'{knowledge_base}'}})-[r]->(b) RETURN a,r,b",
        color="#6B7280", fontsize=11, fontproperties=font, va="center",
    )

    canvas.set_facecolor(CANVAS)
    nodes = list(graph.nodes())
    dense = graph.number_of_nodes() > 350
    node_sizes = [
        24 + min(110, graph.degree(node) * 3)
        if dense else 850 + min(1250, graph.degree(node) * 150)
        for node in nodes
    ]
    fills = [
        TYPE_COLORS.get(str(graph.nodes[node].get("entity_type")), DEFAULT_COLOR)[0]
        for node in nodes
    ]
    borders = [
        TYPE_COLORS.get(str(graph.nodes[node].get("entity_type")), DEFAULT_COLOR)[1]
        for node in nodes
    ]
    nx.draw_networkx_edges(
        graph, positions, ax=canvas, edge_color=EDGE, width=1.05, alpha=0.9,
        arrows=True, arrowsize=13, arrowstyle="-|>", node_size=node_sizes,
        connectionstyle="arc3,rad=0.035",
    )
    nx.draw_networkx_nodes(
        graph, positions, ax=canvas, node_size=node_sizes, node_color=fills,
        edgecolors=borders, linewidths=1.8, alpha=0.98,
    )
    label_nodes = [
        node for node in nodes
        if not dense
        or str(graph.nodes[node].get("scope_level", "")) in {"book", "chapter"}
        or graph.degree(node) >= 10
    ]
    for node in label_nodes:
        x, y_pos = positions[node]
        canvas.text(
            x, y_pos, _short(str(graph.nodes[node].get("display_name", node)), 11),
            ha="center", va="center", fontsize=6.8, color="#374151",
            fontproperties=font, zorder=5,
        )
    for source, target, attributes in graph.edges(data=True):
        if dense:
            continue
        x1, y1 = positions[source]
        x2, y2 = positions[target]
        relation = _short(str(attributes.get("relationship_type", "RELATED")), 14)
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if angle > 90 or angle < -90:
            angle += 180
        canvas.text(
            (x1 + x2) / 2, (y1 + y2) / 2, relation,
            color="#737A84", fontsize=5.5, fontproperties=font,
            rotation=angle, rotation_mode="anchor", ha="center", va="bottom",
            bbox={"facecolor": CANVAS, "edgecolor": "none", "alpha": 0.78, "pad": 0.25},
            zorder=4,
        )
    canvas.set_title(
        f"{knowledge_base} 知识图谱  ·  {graph.number_of_nodes()} 个实体  ·  {graph.number_of_edges()} 条关系",
        color="#4B5563", fontsize=12, fontproperties=font, pad=12,
    )
    canvas.axis("off")
    canvas.patch.set_visible(True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, facecolor=figure.get_facecolor(), bbox_inches="tight")
    plt.close(figure)


HTML_TEMPLATE = r"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Neo4j - __KB__</title><style>
*{box-sizing:border-box}body{margin:0;font-family:"Microsoft YaHei","PingFang SC",sans-serif;background:#d7dce3;color:#343846}.app{display:grid;grid-template-columns:290px 1fr;height:100vh}.side{background:#303240;color:#fff;padding:25px 22px;overflow:auto}.side h1{font-size:22px;margin:0 0 30px}.side h2{font-size:15px;margin:23px 0 11px;border-bottom:1px solid #4a4d5b;padding-bottom:9px}.database{background:#fff;color:#303240;padding:11px;border-radius:2px}.chip{display:inline-block;padding:6px 10px;border-radius:18px;margin:3px 2px;font-size:12px;color:#343846}.relchip{display:inline-block;background:#aeb3bf;padding:6px 9px;border-radius:3px;margin:3px 2px;font-size:11px;color:#343846}.main{padding:14px 16px 16px;display:grid;grid-template-rows:56px 1fr 82px;gap:11px;min-width:0}.query{background:#fff;color:#7a808a;padding:16px 20px;font:15px Consolas,monospace;white-space:nowrap;overflow:hidden}.graph{position:relative;background:#f7f9fb;border:1px solid #cdd3da;overflow:hidden}.toolbar{position:absolute;z-index:4;left:16px;top:12px;display:flex;gap:7px;flex-wrap:wrap}.count{padding:6px 10px;border-radius:15px;color:#fff;font-size:11px}.relationship-count{background:#979daa}.node-count{background:#b97db6}.hint{position:absolute;right:13px;top:13px;color:#858c96;font-size:12px;z-index:4}.detail{background:#fff;border:1px solid #cdd3da;padding:12px 16px;font-size:13px;line-height:1.5;overflow:auto}.detail strong{color:#343846}svg{width:100%;height:100%;display:block}.edge{stroke:#9fa6af;stroke-width:1.45;opacity:.9;cursor:pointer}.edge-hit{stroke:transparent;stroke-width:14;cursor:pointer}.edge-label{fill:#666e79;font-size:10.5px;paint-order:stroke;stroke:#f7f9fb;stroke-width:4px;pointer-events:none}.node{cursor:pointer}.node circle{filter:drop-shadow(0 1px 1px #7775)}.node text{fill:#374151;font-size:11.5px;text-anchor:middle;dominant-baseline:middle;pointer-events:none}.hidden{display:none}@media(max-width:850px){.app{grid-template-columns:230px 1fr}.side{padding:20px 14px}}
</style></head><body><div class="app"><aside class="side"><h1>◉ Database Information</h1><h2>Use database</h2><div class="database">__KB__</div><h2>Node Labels</h2><div id="nodeLabels"></div><h2>Relationship Types</h2><div id="relationTypes"></div><h2>Property Keys</h2><div class="relchip">description</div><div class="relchip">source_pages</div><div class="relchip">modalities</div><div class="relchip">confidence</div></aside><main class="main"><div class="query">neo4j$ MATCH (a:KnowledgeEntity {knowledge_base:'__KB__'})-[r]->(b) RETURN a,r,b</div><section class="graph"><div class="toolbar"><span class="count node-count">*(__NODE_COUNT__)</span><span class="count relationship-count">*(__EDGE_COUNT__)</span></div><div class="hint">滚轮缩放 · 拖动画布 · 点击节点或关系查看证据</div><svg id="svg" viewBox="0 0 1000 620"><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#9fa6af"/></marker></defs><g id="viewport"><g id="edges"></g><g id="labels"></g><g id="nodes"></g></g></svg></section><div class="detail" id="detail"><strong>Neo4j 教材知识图谱</strong><br>点击节点查看实体描述；点击关系查看页码、模态、置信度和证据文本。</div></main></div><script>
const data=__DATA__,ns='http://www.w3.org/2000/svg',byId=new Map(data.nodes.map(n=>[n.id,n])),svg=document.getElementById('svg'),viewport=document.getElementById('viewport'),edges=document.getElementById('edges'),labels=document.getElementById('labels'),nodes=document.getElementById('nodes'),detail=document.getElementById('detail'),dense=data.nodes.length>350;const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const el=(n,a={})=>{const x=document.createElementNS(ns,n);for(const[k,v]of Object.entries(a))x.setAttribute(k,v);return x};const radius=n=>dense?(n.scopeLevel==='book'?10:n.scopeLevel==='chapter'?6:n.scopeLevel==='section'?4:2+Math.min(2,Math.sqrt(n.degree)*.28)):30+Math.min(17,n.degree*2.5);
if(!dense){for(let step=0;step<320;step++){const alpha=1-step/340;for(let i=0;i<data.nodes.length;i++)for(let j=i+1;j<data.nodes.length;j++){const a=data.nodes[i],b=data.nodes[j];let dx=b.x-a.x,dy=b.y-a.y,d=Math.hypot(dx,dy);if(d<.1){const q=(i*37+j*17)*.31;dx=Math.cos(q);dy=Math.sin(q);d=1}const gap=88;if(d<gap){const f=(gap-d)*.54*alpha/d;a.x-=dx*f;a.y-=dy*f;b.x+=dx*f;b.y+=dy*f}}for(const e of data.edges){const a=byId.get(e.source),b=byId.get(e.target),dx=b.x-a.x,dy=b.y-a.y,d=Math.max(1,Math.hypot(dx,dy)),f=(d-178)*.03*alpha;a.x+=dx/d*f;a.y+=dy/d*f;b.x-=dx/d*f;b.y-=dy/d*f}for(const n of data.nodes){n.x=Math.max(48,Math.min(952,n.x));n.y=Math.max(48,Math.min(572,n.y))}}}
data.edges.forEach(e=>{const a=byId.get(e.source),b=byId.get(e.target),dx=b.x-a.x,dy=b.y-a.y,d=Math.max(1,Math.hypot(dx,dy)),ra=radius(a),rb=radius(b),lineAttrs={x1:a.x+dx/d*ra,y1:a.y+dy/d*ra,x2:b.x-dx/d*(rb+(dense?1:7)),y2:b.y-dy/d*(rb+(dense?1:7))};const show=()=>detail.innerHTML=`<strong>${esc(a.name)} —[${esc(e.type)}]→ ${esc(b.name)}</strong><br>关系层：${esc(e.layer||'semantic_fact')}　来源页：${esc(e.pages.join(', ')||'未标注')}　模态：${esc(e.modalities.join(', ')||'未标注')}　置信度：${e.confidence.toFixed(2)}<br>${esc(e.description||'')}`;const line=el('line',{...lineAttrs,class:'edge','marker-end':dense?'':'url(#arrow)',stroke:e.layer==='document_structure'?'#c7cdd4':e.layer==='entity_link'?'#8bb7c9':'#9fa6af','stroke-width':dense?(e.layer==='semantic_fact'?.65:.35):1.45,opacity:dense?(e.layer==='semantic_fact'?.55:.24):.9});line.addEventListener('click',show);edges.appendChild(line);if(!dense){const hit=el('line',{...lineAttrs,class:'edge-hit'});hit.addEventListener('click',show);edges.appendChild(hit);const t=el('text',{x:(a.x+b.x)/2,y:(a.y+b.y)/2,class:'edge-label'});t.textContent=e.type;labels.appendChild(t)}});data.nodes.forEach(n=>{const g=el('g',{class:'node',transform:`translate(${n.x},${n.y})`}),r=radius(n);g.appendChild(el('circle',{r,fill:n.fill,stroke:n.border,'stroke-width':dense?.5:2.2}));if(!dense||n.scopeLevel==='book'||n.scopeLevel==='chapter'||n.degree>=10){const t=el('text',{'font-size':dense?5:11.5});t.textContent=n.name.length>9?n.name.slice(0,8)+'…':n.name;g.appendChild(t)}g.addEventListener('click',()=>detail.innerHTML=`<strong>${esc(n.name)}</strong>　${esc(n.type)}　度数 ${n.degree}<br>来源页：${esc(n.pages.join(', ')||'未标注')}<br>${esc(n.description||'暂无描述')}`);nodes.appendChild(g)});
document.getElementById('nodeLabels').innerHTML=Object.entries(data.typeCounts).map(([k,v])=>`<span class="chip" style="background:${data.typeColors[k]?.[0]||'#b8c0cc'}">${esc(k)} (${v})</span>`).join('');document.getElementById('relationTypes').innerHTML=Object.entries(data.relationCounts).map(([k,v])=>`<span class="relchip">${esc(k)} (${v})</span>`).join('');let scale=1,tx=0,ty=0,drag=false,last;const apply=()=>viewport.setAttribute('transform',`translate(${tx} ${ty}) scale(${scale})`);svg.addEventListener('wheel',e=>{e.preventDefault();scale=Math.max(.15,Math.min(dense?30:4,scale*(e.deltaY<0?1.12:.89)));apply()},{passive:false});svg.addEventListener('pointerdown',e=>{if(e.target.closest('.node'))return;drag=true;last=[e.clientX,e.clientY];svg.setPointerCapture(e.pointerId)});svg.addEventListener('pointermove',e=>{if(!drag)return;tx+=(e.clientX-last[0])/scale;ty+=(e.clientY-last[1])/scale;last=[e.clientX,e.clientY];apply()});svg.addEventListener('pointerup',()=>drag=false);
</script></body></html>"""


def _html_data(
    graph: nx.MultiDiGraph,
    positions: dict[str, tuple[float, float]],
) -> dict[str, Any]:
    xs = [point[0] for point in positions.values()]
    ys = [point[1] for point in positions.values()]
    x_min, x_max, y_min, y_max = min(xs), max(xs), min(ys), max(ys)
    x_span, y_span = max(1e-9, x_max - x_min), max(1e-9, y_max - y_min)
    nodes = []
    for node, attributes in graph.nodes(data=True):
        x, y = positions[node]
        entity_type = str(attributes.get("entity_type", "课程概念"))
        fill, border = TYPE_COLORS.get(entity_type, DEFAULT_COLOR)
        nodes.append({
            "id": node,
            "name": str(attributes.get("display_name", node)),
            "type": entity_type,
            "scopeLevel": str(attributes.get("scope_level", "")),
            "description": str(attributes.get("description", "")),
            "pages": list(attributes.get("source_pages") or attributes.get("pages") or []),
            "degree": int(graph.degree(node)),
            "fill": fill,
            "border": border,
            "x": round(55 + (x - x_min) / x_span * 890, 2),
            "y": round(55 + (y_max - y) / y_span * 510, 2),
        })
    edges = []
    for source, target, attributes in graph.edges(data=True):
        edges.append({
            "source": source,
            "target": target,
            "type": str(attributes.get("relationship_type", "RELATED")),
            "layer": str(attributes.get("edge_layer", "semantic_fact")),
            "description": str(attributes.get("description", "")),
            "pages": list(attributes.get("source_pages") or []),
            "modalities": list(attributes.get("modalities") or []),
            "confidence": float(attributes.get("confidence", 0.0) or 0.0),
        })
    return {
        "nodes": nodes,
        "edges": edges,
        "typeCounts": dict(Counter(node["type"] for node in nodes)),
        "relationCounts": dict(Counter(edge["type"] for edge in edges).most_common()),
        "typeColors": TYPE_COLORS,
    }


def _standalone_html(
    graph: nx.MultiDiGraph,
    positions: dict[str, tuple[float, float]],
    knowledge_base: str,
) -> str:
    data = json.dumps(_html_data(graph, positions), ensure_ascii=False).replace("</", "<\\/")
    return (
        HTML_TEMPLATE.replace("__KB__", html.escape(knowledge_base))
        .replace("__NODE_COUNT__", str(graph.number_of_nodes()))
        .replace("__EDGE_COUNT__", str(graph.number_of_edges()))
        .replace("__DATA__", data)
    )


def render_html(
    graph: nx.MultiDiGraph,
    positions: dict[str, tuple[float, float]],
    destination: Path,
    knowledge_base: str,
) -> None:
    content = _standalone_html(graph, positions, knowledge_base)
    destination.write_text(content, encoding="utf-8")


def render_fragment(
    graph: nx.MultiDiGraph,
    positions: dict[str, tuple[float, float]],
    destination: Path,
    knowledge_base: str,
) -> None:
    """生成 Codex 对话内可直接展示、且样式作用域隔离的 HTML 片段。"""
    content = _standalone_html(graph, positions, knowledge_base)
    style_match = re.search(r"<style>(.*?)</style>", content, flags=re.S)
    body_match = re.search(r"<body>(.*?)</body>", content, flags=re.S)
    if not (style_match and body_match):
        raise RuntimeError("无法从 Neo4j 可视化页面提取 HTML 片段")

    root_id = "neo4j-five-page-graph"
    css = style_match.group(1).replace(
        "*{box-sizing:border-box}body{",
        f"#{root_id},#{root_id} *{{box-sizing:border-box}}#{root_id}{{",
        1,
    ).replace("height:100vh", "height:680px", 1)
    selectors = (
        "app", "side", "database", "chip", "relchip", "main", "query",
        "graph", "toolbar", "count", "relationship-count", "node-count",
        "hint", "detail", "edge", "edge-hit", "edge-label", "node", "hidden",
    )
    for selector in selectors:
        css = re.sub(
            rf"(?<![\w-])\.{re.escape(selector)}(?![\w-])",
            f"#{root_id} .{selector}",
            css,
        )
    css = re.sub(r"(?<![\w-])svg(?=\{)", f"#{root_id} svg", css)
    css += (
        f"@media(max-width:620px){{#{root_id} .app{{grid-template-columns:1fr;height:auto}}"
        f"#{root_id} .side{{max-height:270px}}#{root_id} .main{{height:590px;padding:10px}}"
        f"#{root_id} .query,#{root_id} .hint{{display:none}}}}"
    )
    fragment = (
        f'<div id="{root_id}">{body_match.group(1)}</div>\n'
        f"<style>\n{css}\n</style>\n"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(fragment, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="将 Neo4j knowledge_base 导出为 Browser 风格图谱")
    parser.add_argument("--knowledge-base", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fragment-output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    graph = load_neo4j_graph(args.knowledge_base)
    if not graph:
        raise RuntimeError(f"Neo4j 中没有找到知识库：{args.knowledge_base}")
    args.output.mkdir(parents=True, exist_ok=True)
    positions = _positions(graph, args.seed)
    png_path = args.output / "neo4j-knowledge-graph.png"
    html_path = args.output / "neo4j-knowledge-graph.html"
    render_png(graph, positions, png_path, args.knowledge_base)
    render_html(graph, positions, html_path, args.knowledge_base)
    if args.fragment_output:
        render_fragment(graph, positions, args.fragment_output, args.knowledge_base)
    summary = {
        "knowledge_base": args.knowledge_base,
        "nodes": graph.number_of_nodes(),
        "relationships": graph.number_of_edges(),
        "components": nx.number_weakly_connected_components(graph),
        "png": str(png_path.resolve()),
        "html": str(html_path.resolve()),
        "fragment": str(args.fragment_output.resolve()) if args.fragment_output else None,
    }
    (args.output / "neo4j-knowledge-graph-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
