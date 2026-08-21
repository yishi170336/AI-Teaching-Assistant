from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from backend.app.rag.embedding_runtime import encode_texts


CONSOLIDATION_VERSION = "1.0-controlled-ontology-entity-linking"
GENERATED_BY = "graph_consolidation"
STRUCTURE_ENTITY_TYPE = "教材结构"
ATTRIBUTE_ENTITY_TYPE = "知识属性"


RELATION_ONTOLOGY: tuple[tuple[str, str, str], ...] = (
    ("ALIAS_OF", "别名", r"别名|别称|又称|也称|简称|同义|称为|命名为"),
    ("IS_A", "属于类别", r"属于|归类|分类为|类型归属|是一种|类别"),
    ("DEFINED_AS", "定义为", r"定义|术语|指代|^指$|表示为|描述对象"),
    ("HAS_PART", "包含组成", r"包含|组成|构成|集成有|内部含有|组件|元件"),
    ("HAS_PROPERTY", "具有属性", r"具有|具备|特性|特点|性质|属性|特征|状态为"),
    ("HAS_VALUE", "具有取值", r"等于|取值|数值|典型值|理论值|约为|约等于|范围是"),
    ("HAS_FORMULA", "具有公式", r"公式|表达式|函数关系|数学模型|计算关系"),
    ("CAUSES", "导致", r"导致|引起|引发|造成|因果|成因|产生原因|产生结果|后果"),
    ("PRODUCES", "产生", r"产生|生成|形成|输出至|感应形成"),
    ("DOES_NOT_AFFECT", "不影响", r"不影响|无关|独立于|不依赖"),
    ("DEPENDS_ON", "取决于", r"取决于|依赖|由.+决定|决定因素|受限于|制约"),
    ("AFFECTS", "影响", r"影响|作用于|作用效果|使.+变化|促进|提升|提高|改善"),
    ("CONTROLS", "控制", r"控制|受控于|驱动|触发"),
    ("REQUIRES", "需要条件", r"需要|要求|需满足|满足条件|前提|必要|约束条件|工作条件"),
    ("USED_FOR", "用于", r"用于|应用|用途|目的|旨在|解决|功能为|功能定义|执行功能"),
    ("IMPLEMENTS", "实现", r"实现|执行|完成|转换输入输出|运算功能"),
    ("APPLIES_TO", "适用于", r"适用|多用于|常用于|应用场景|应用场合"),
    ("SUPPORTS", "支持", r"支持|允许|提供功能|能够|可用于"),
    ("PREVENTS", "抑制消除", r"抑制|消除|抵消|避免|阻碍|保护"),
    ("CONNECTED_TO", "电气连接", r"连接|接地|并联|串联|短路|端口|节点|接至"),
    ("INPUT_OF", "输入到", r"输入端|从.+输入|输入变量|输入至"),
    ("OUTPUT_OF", "输出自", r"输出端|从.+输出|输出变量|输出位置|取自"),
    ("CALCULATED_BY", "计算方式", r"计算|求解|求和|叠加得到|参数计算"),
    ("EQUIVALENT_TO", "等效于", r"等效|等同|近似为|近似等于|相当于"),
    ("PROPORTIONAL_TO", "成正比", r"正比|正相关|随.+增加|越大.+越大"),
    ("INVERSELY_PROPORTIONAL_TO", "成反比", r"反比|负相关|越大.+越小|随.+减小"),
    ("GREATER_THAN", "大于", r"大于|高于|优于|更高|远大于|显著高于"),
    ("LESS_THAN", "小于", r"小于|低于|窄于|远小于|低于"),
    ("COMPARED_WITH", "比较", r"比较|相比|区别于|相反|相同|一致性|优先级"),
    ("CONVERTS_TO", "转换为", r"转换|转化|变为|修改为|替代"),
    ("INCREASES", "增大", r"增大|增加|扩展|升高"),
    ("DECREASES", "减小", r"减小|降低|缩小|缩短|衰减"),
    ("HAS_ADVANTAGE", "具有优点", r"优点|优势|性能提升"),
    ("HAS_DISADVANTAGE", "具有缺点", r"缺点|缺陷|不足|失真|副作用"),
    ("SEMANTICALLY_ALIGNED_WITH", "语义对齐", r"语义对齐"),
    ("HAS_CHAPTER", "包含章节", r"包含章节"),
    ("HAS_SECTION", "包含小节", r"包含小节"),
    ("MENTIONED_IN", "见于小节", r"见于小节"),
    ("RELATED_TO", "相关", r"相关|关系|关联|对应|体现|描述|呈现|反映|基于"),
)
CONTROLLED_RELATIONS = {item[0] for item in RELATION_ONTOLOGY}
RELATION_LABELS = {item[0]: item[1] for item in RELATION_ONTOLOGY}

_NORMALIZED_ALIASES = {
    "HAS_PROPERTY": "HAS_PROPERTY",
    "HAS_FORMULA": "HAS_FORMULA",
    "HAS_EXPRESSION": "HAS_FORMULA",
    "IS_A": "IS_A",
    "DEFINED_AS": "DEFINED_AS",
    "CAUSES": "CAUSES",
    "DEPENDS_ON": "DEPENDS_ON",
    "DETERMINES": "DEPENDS_ON",
    "CALCULATED_BY": "CALCULATED_BY",
    "COMPOSED_OF": "HAS_PART",
    "CONNECTED_AS": "CONNECTED_TO",
    "EQUALS": "HAS_VALUE",
    "IMPLEMENTS": "IMPLEMENTS",
    "IS_PROPORTIONAL_TO": "PROPORTIONAL_TO",
    "PROPORTIONAL_TO": "PROPORTIONAL_TO",
    "USED_FOR": "USED_FOR",
    "REQUIRES_CONDITION": "REQUIRES",
    "TRUTH_TABLE_RELATION": "HAS_VALUE",
}


def _stable_id(prefix: str, *values: object) -> str:
    raw = "|".join(str(value) for value in values)
    return f"{prefix}:" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, set, np.ndarray)):
        return list(value)
    return [value]


def _unique(values: Iterable[Any]) -> list[Any]:
    result: list[Any] = []
    seen: set[str] = set()
    for value in values:
        marker = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        if marker not in seen:
            seen.add(marker)
            result.append(value)
    return result


def controlled_relation(value: str, original: str = "") -> tuple[str, str]:
    raw = str(value or "").strip()
    if raw.upper() in _NORMALIZED_ALIASES:
        code = _NORMALIZED_ALIASES[raw.upper()]
        return code, RELATION_LABELS[code]
    text = re.sub(r"[_\s/]+", "", f"{original} {raw}").strip()
    for code, label, pattern in RELATION_ONTOLOGY:
        if re.search(pattern, text, flags=re.I):
            return code, label
    return "RELATED_TO", RELATION_LABELS["RELATED_TO"]


def _entity_signature(value: str) -> str:
    name = re.sub(r"[\s_{}()（）\[\]·,，。:：/\-]+", "", str(value)).casefold()
    replacements = {
        "共射极": "共射",
        "共集电极": "共集",
        "共基极": "共基",
        "场效应晶体管": "场效应管",
        "晶体三极管": "晶体管",
        "npn型": "npn",
        "pnp型": "pnp",
        "n沟道型": "n沟道",
        "p沟道型": "p沟道",
    }
    for source, target in replacements.items():
        name = name.replace(source, target)
    return name


def _types_compatible(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_type = str(left.get("entity_type", "课程概念"))
    right_type = str(right.get("entity_type", "课程概念"))
    return left_type == right_type or "课程概念" in {left_type, right_type}


def _remove_generated_artifacts(graph: dict[str, Any]) -> None:
    generated_node_ids = {
        str(node.get("id", ""))
        for node in graph.get("nodes", [])
        if node.get("generated_by") == GENERATED_BY
    }
    graph["nodes"] = [
        node for node in graph.get("nodes", [])
        if str(node.get("id", "")) not in generated_node_ids
    ]
    graph["edges"] = [
        edge for edge in graph.get("edges", [])
        if edge.get("generated_by") != GENERATED_BY
        and str(edge.get("source", "")) not in generated_node_ids
        and str(edge.get("target", "")) not in generated_node_ids
    ]
    graph["communities"] = [
        community for community in graph.get("communities", [])
        if community.get("algorithm")
        != "microsoft-graphrag-connected-component-fallback"
    ]
    graph["community_reports"] = [
        report for report in graph.get("community_reports", [])
        if report.get("algorithm")
        != "microsoft-graphrag-connected-component-fallback"
    ]


def _merge_safe_aliases(graph: dict[str, Any]) -> tuple[dict[str, str], int]:
    nodes = [node for node in graph.get("nodes", []) if isinstance(node, dict)]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for node in nodes:
        groups[_entity_signature(str(node.get("name", "")))].append(node)
    node_map = {str(node.get("id", "")): str(node.get("id", "")) for node in nodes}
    removed: set[str] = set()
    merged_count = 0
    for values in groups.values():
        compatible = [
            node for node in values
            if all(_types_compatible(node, other) for other in values)
        ]
        if len(compatible) < 2:
            continue
        representative = max(
            compatible,
            key=lambda node: (
                int(node.get("relationship_count", 0) or 0),
                int(node.get("evidence_count", 0) or 0),
                -len(str(node.get("name", ""))),
            ),
        )
        representative_id = str(representative["id"])
        for node in compatible:
            node_id = str(node["id"])
            if node_id == representative_id:
                continue
            node_map[node_id] = representative_id
            removed.add(node_id)
            merged_count += 1
            representative["aliases"] = _unique([
                *representative.get("aliases", []),
                str(node.get("name", "")),
                *node.get("aliases", []),
            ])
            for key in ("text_unit_ids", "pages", "source_pages", "raw_entity_types"):
                representative[key] = _unique([
                    *representative.get(key, []), *node.get(key, [])
                ])
            descriptions = _unique([
                str(representative.get("description", "")),
                str(node.get("description", "")),
            ])
            representative["description"] = " ".join(
                item for item in descriptions if item
            )[:2400]
    graph["nodes"] = [node for node in nodes if str(node.get("id", "")) not in removed]

    def mapped(value: Any) -> str:
        return node_map.get(str(value), str(value))

    for edge in graph.get("edges", []):
        edge["source"] = mapped(edge.get("source"))
        edge["target"] = mapped(edge.get("target"))
    node_lookup = {str(node["id"]): node for node in graph["nodes"]}
    for fact in graph.get("attribute_facts", []):
        fact["subject"] = mapped(fact.get("subject"))
        if fact["subject"] in node_lookup:
            fact["subject_name"] = str(node_lookup[fact["subject"]].get("name", ""))
    for mention in graph.get("relationship_mentions", []):
        mention["source"] = mapped(mention.get("source"))
        mention["target"] = mapped(mention.get("target"))
    for community in graph.get("communities", []):
        community["entity_ids"] = _unique(
            mapped(item) for item in community.get("entity_ids", [])
        )
    return node_map, merged_count


def _merge_relationships(graph: dict[str, Any]) -> tuple[dict[str, str], int, int]:
    node_ids = {str(node.get("id", "")) for node in graph.get("nodes", [])}
    merged: dict[tuple[str, str, str], dict[str, Any]] = {}
    edge_map: dict[str, str] = {}
    original_types: set[str] = set()
    dropped = 0
    for edge in graph.get("edges", []):
        source = str(edge.get("source", ""))
        target = str(edge.get("target", ""))
        if source not in node_ids or target not in node_ids or source == target:
            dropped += 1
            continue
        raw_normalized = str(
            edge.get("raw_relation_normalized")
            or edge.get("relation_normalized", "")
        )
        raw_original = str(edge.get("relation") or edge.get("type") or "相关")
        original_types.add(raw_normalized or raw_original)
        relation_code, relation_label = controlled_relation(
            raw_normalized, raw_original
        )
        key = (source, target, relation_code)
        edge_id = str(edge.get("id", ""))
        existing = merged.get(key)
        if existing is None:
            existing = {
                **edge,
                "id": edge_id or _stable_id("relationship", *key),
                "source": source,
                "target": target,
                "relation": raw_original,
                "type": raw_original,
                "raw_relation_normalized": raw_normalized,
                "relation_normalized": relation_code,
                "relation_category": relation_label,
                "relation_variants": _unique([
                    raw_original, raw_normalized,
                    *edge.get("relation_variants", []),
                ]),
                "edge_layer": edge.get("edge_layer", "semantic_fact"),
                "merged_edge_ids": [edge_id] if edge_id else [],
            }
            merged[key] = existing
        else:
            existing["merged_edge_ids"] = _unique([
                *existing.get("merged_edge_ids", []), edge_id
            ])
            existing["relation_variants"] = _unique([
                *existing.get("relation_variants", []), raw_original, raw_normalized
            ])
            for field in (
                "mention_ids", "text_unit_ids", "evidence_ids", "source_pages",
                "modalities", "qualifier_texts",
            ):
                existing[field] = _unique([
                    *existing.get(field, []), *edge.get(field, [])
                ])
            existing["description"] = " ".join(_unique([
                str(existing.get("description", "")),
                str(edge.get("description", "")),
            ]))[:3000]
            existing["weight"] = float(existing.get("weight", 0) or 0) + float(
                edge.get("weight", 0) or 0
            )
            existing["confidence"] = min(
                float(existing.get("confidence", 1) or 0),
                float(edge.get("confidence", 1) or 0),
            )
        if edge_id:
            edge_map[edge_id] = str(existing["id"])
    graph["edges"] = list(merged.values())
    for community in graph.get("communities", []):
        for field in ("relationship_ids", "official_relationship_ids"):
            community[field] = _unique(
                edge_map[str(item)]
                for item in community.get(field, [])
                if str(item) in edge_map
            )
    return edge_map, len(original_types), dropped


def _entity_link_candidates(nodes: list[dict[str, Any]]) -> list[tuple[str, str]]:
    signatures: dict[str, list[str]] = defaultdict(list)
    by_id = {str(node["id"]): node for node in nodes}
    for node in nodes:
        signature = _entity_signature(str(node.get("name", "")))
        if len(signature) >= 4:
            signatures[signature].append(str(node["id"]))
    candidates: set[tuple[str, str]] = set()
    for specific_signature, specific_ids in signatures.items():
        minimum = max(4, len(specific_signature) - 8)
        for length in range(minimum, len(specific_signature)):
            if length / len(specific_signature) < 0.62:
                continue
            for start in range(0, len(specific_signature) - length + 1):
                general_signature = specific_signature[start:start + length]
                for general_id in signatures.get(general_signature, []):
                    for specific_id in specific_ids:
                        if general_id == specific_id:
                            continue
                        if not _types_compatible(by_id[general_id], by_id[specific_id]):
                            continue
                        candidates.add((specific_id, general_id))
    return sorted(candidates)


def _add_entity_links(
    graph: dict[str, Any],
    embedding_model_path: Path | None,
    *,
    similarity_threshold: float = 0.88,
) -> tuple[int, int]:
    nodes = [
        node for node in graph.get("nodes", [])
        if node.get("generated_by") != GENERATED_BY
    ]
    candidates = _entity_link_candidates(nodes)
    if not candidates:
        return 0, 0
    by_id = {str(node["id"]): node for node in nodes}
    candidate_node_ids = sorted({item for pair in candidates for item in pair})
    similarities: dict[tuple[str, str], float] = {}
    if embedding_model_path is not None:
        embeddings = encode_texts(
            embedding_model_path,
            [str(by_id[node_id].get("name", "")) for node_id in candidate_node_ids],
            batch_size=4,
            show_progress_bar=False,
            purpose="document",
        )
        embedding_by_id = dict(zip(candidate_node_ids, embeddings, strict=True))
        similarities = {
            pair: float(np.dot(embedding_by_id[pair[0]], embedding_by_id[pair[1]]))
            for pair in candidates
        }
    existing_pairs = {
        (str(edge.get("source", "")), str(edge.get("target", "")))
        for edge in graph.get("edges", [])
    }
    links: list[dict[str, Any]] = []
    for specific_id, general_id in candidates:
        specific_name = str(by_id[specific_id].get("name", ""))
        general_name = str(by_id[general_id].get("name", ""))
        ratio = len(_entity_signature(general_name)) / max(
            1, len(_entity_signature(specific_name))
        )
        similarity = similarities.get((specific_id, general_id), 1.0)
        threshold = similarity_threshold if embedding_model_path is not None else 0.0
        if similarity < threshold or ratio < 0.62:
            continue
        if (specific_id, general_id) in existing_pairs or (
            general_id, specific_id
        ) in existing_pairs:
            continue
        text_unit_ids = _unique([
            *by_id[specific_id].get("text_unit_ids", []),
            *by_id[general_id].get("text_unit_ids", []),
        ])
        pages = sorted({
            int(page)
            for node_id in (specific_id, general_id)
            for page in by_id[node_id].get("source_pages", [])
            if str(page).isdigit()
        })
        links.append({
            "id": _stable_id("entity-link", specific_id, general_id),
            "source": specific_id,
            "target": general_id,
            "type": "语义对齐",
            "relation": "语义对齐",
            "relation_normalized": "SEMANTICALLY_ALIGNED_WITH",
            "relation_category": RELATION_LABELS["SEMANTICALLY_ALIGNED_WITH"],
            "description": (
                f"“{specific_name}”与“{general_name}”名称包含且语义向量高度相似；"
                "该边用于全书实体链接，不表示新增教材事实。"
            ),
            "text_unit_ids": text_unit_ids,
            "evidence_ids": text_unit_ids,
            "source_pages": pages,
            "modalities": ["entity_linking"],
            "confidence": round(similarity, 4),
            "weight": round(similarity, 4),
            "edge_layer": "entity_link",
            "derived": True,
            "generated_by": GENERATED_BY,
        })
        existing_pairs.add((specific_id, general_id))
    graph.setdefault("edges", []).extend(links)
    return len(candidates), len(links)


def _add_attribute_value_edges(graph: dict[str, Any]) -> tuple[int, int]:
    node_lookup = {str(node["id"]): node for node in graph.get("nodes", [])}
    generated_nodes: list[dict[str, Any]] = []
    generated_edges: list[dict[str, Any]] = []
    for fact in graph.get("attribute_facts", []):
        value_type = str(fact.get("value_type", "text"))
        if value_type not in {"formula", "quantity"}:
            continue
        subject_id = str(fact.get("subject", ""))
        if subject_id not in node_lookup:
            continue
        fact_id = str(fact.get("id", ""))
        subject_name = str(node_lookup[subject_id].get("name", "知识实体"))
        suffix = "公式" if value_type == "formula" else "参数值"
        display_name = f"{subject_name}的{suffix}"
        if len(display_name) > 30:
            display_name = f"{subject_name[:24]}…的{suffix}"
        value_node_id = _stable_id("attribute-value", fact_id)
        source_page = int(fact.get("source_page", 0) or 0)
        generated_nodes.append({
            "id": value_node_id,
            "type": "entity",
            "entity_type": ATTRIBUTE_ENTITY_TYPE,
            "name": display_name,
            "description": str(fact.get("evidence_text", "")),
            "value": str(fact.get("value", "")),
            "value_type": value_type,
            "text_unit_ids": _as_list(fact.get("text_unit_id")),
            "evidence_count": len(_as_list(fact.get("evidence_ids"))),
            "pages": [source_page] if source_page else [],
            "source_pages": [source_page] if source_page else [],
            "relationship_count": 1,
            "standalone": False,
            "generated_by": GENERATED_BY,
        })
        relation_code = "HAS_FORMULA" if value_type == "formula" else "HAS_VALUE"
        evidence_ids = _unique([
            *fact.get("evidence_ids", []), fact.get("text_unit_id", "")
        ])
        generated_edges.append({
            "id": _stable_id("attribute-edge", fact_id),
            "source": subject_id,
            "target": value_node_id,
            "type": str(fact.get("relation", suffix)),
            "relation": str(fact.get("relation", suffix)),
            "relation_normalized": relation_code,
            "relation_category": RELATION_LABELS[relation_code],
            "description": str(fact.get("evidence_text", "")),
            "text_unit_ids": _as_list(fact.get("text_unit_id")),
            "evidence_ids": [item for item in evidence_ids if item],
            "source_pages": [source_page] if source_page else [],
            "modalities": [str(fact.get("modality", value_type))],
            "confidence": float(fact.get("confidence", 1.0) or 0.0),
            "weight": 1.0,
            "edge_layer": "attribute_fact",
            "generated_by": GENERATED_BY,
        })
    graph.setdefault("nodes", []).extend(generated_nodes)
    graph.setdefault("edges", []).extend(generated_edges)
    return len(generated_nodes), len(generated_edges)


def _add_document_structure(graph: dict[str, Any]) -> tuple[int, int]:
    text_units = {
        str(item.get("id", "")): item for item in graph.get("text_units", [])
    }
    original_nodes = [
        node for node in graph.get("nodes", [])
        if node.get("generated_by") != GENERATED_BY
    ]
    book_id = _stable_id("scope-book", "电子电路基础")
    scope_nodes: dict[str, dict[str, Any]] = {
        book_id: {
            "id": book_id,
            "type": "entity",
            "entity_type": STRUCTURE_ENTITY_TYPE,
            "name": "电子电路基础",
            "description": "整本教材的知识结构根节点。",
            "text_unit_ids": list(text_units),
            "evidence_count": len(text_units),
            "pages": [],
            "source_pages": [],
            "relationship_count": 0,
            "standalone": False,
            "scope_level": "book",
            "generated_by": GENERATED_BY,
        }
    }
    scope_edges: dict[str, dict[str, Any]] = {}

    def add_scope_edge(
        source: str,
        target: str,
        relation_code: str,
        relation: str,
        text_unit_ids: list[str],
        pages: list[int],
    ) -> None:
        edge_id = _stable_id("scope-edge", source, target, relation_code)
        existing = scope_edges.get(edge_id)
        if existing is not None:
            existing["text_unit_ids"] = _unique([
                *existing.get("text_unit_ids", []), *text_unit_ids
            ])
            existing["evidence_ids"] = list(existing["text_unit_ids"])
            existing["source_pages"] = sorted({
                *existing.get("source_pages", []), *pages
            })
            return
        scope_edges[edge_id] = {
            "id": edge_id,
            "source": source,
            "target": target,
            "type": relation,
            "relation": relation,
            "relation_normalized": relation_code,
            "relation_category": RELATION_LABELS[relation_code],
            "description": f"教材目录与证据位置表明该实体{relation}。",
            "text_unit_ids": text_unit_ids,
            "evidence_ids": text_unit_ids,
            "source_pages": pages,
            "modalities": ["document_structure"],
            "confidence": 1.0,
            "weight": 1.0,
            "edge_layer": "document_structure",
            "generated_by": GENERATED_BY,
        }

    all_pages: set[int] = set()
    for node in original_nodes:
        node_text_units = [
            text_units[item]
            for item in node.get("text_unit_ids", [])
            if item in text_units
        ]
        node_pages = sorted({
            int(item.get("page_start", 0) or 0)
            for item in node_text_units
            if item.get("page_start")
        })
        all_pages.update(node_pages)
        memberships: list[tuple[int, str, str, str]] = []
        for text_unit in node_text_units:
            chapter = str(text_unit.get("chapter", "")).strip() or "未归类章节"
            section = str(text_unit.get("section", "")).strip() or chapter
            memberships.append((
                int(text_unit.get("page_start", 0) or 0),
                chapter,
                section,
                str(text_unit.get("id", "")),
            ))
        if not memberships:
            memberships = [(0, "未归类章节", "未归类小节", "")]
        memberships.sort()
        primary_page, chapter, section, primary_text_unit = memberships[0]
        chapter_id = _stable_id("scope-chapter", chapter)
        section_id = _stable_id("scope-section", chapter, section)
        chapter_text_units = _unique(
            item[3] for item in memberships if item[1] == chapter and item[3]
        )
        section_text_units = _unique(
            item[3] for item in memberships
            if item[1] == chapter and item[2] == section and item[3]
        )
        chapter_pages = sorted({item[0] for item in memberships if item[1] == chapter and item[0]})
        section_pages = sorted({
            item[0] for item in memberships
            if item[1] == chapter and item[2] == section and item[0]
        })
        scope_nodes.setdefault(chapter_id, {
            "id": chapter_id,
            "type": "entity",
            "entity_type": STRUCTURE_ENTITY_TYPE,
            "name": chapter,
            "description": f"教材章节：{chapter}",
            "text_unit_ids": chapter_text_units,
            "evidence_count": len(chapter_text_units),
            "pages": chapter_pages,
            "source_pages": chapter_pages,
            "relationship_count": 0,
            "standalone": False,
            "scope_level": "chapter",
            "generated_by": GENERATED_BY,
        })
        chapter_node = scope_nodes[chapter_id]
        chapter_node["text_unit_ids"] = _unique([
            *chapter_node.get("text_unit_ids", []), *chapter_text_units
        ])
        chapter_node["evidence_count"] = len(chapter_node["text_unit_ids"])
        chapter_node["pages"] = sorted({
            *chapter_node.get("pages", []), *chapter_pages
        })
        chapter_node["source_pages"] = list(chapter_node["pages"])
        scope_nodes.setdefault(section_id, {
            "id": section_id,
            "type": "entity",
            "entity_type": STRUCTURE_ENTITY_TYPE,
            "name": section,
            "description": f"教材小节：{section}",
            "text_unit_ids": section_text_units,
            "evidence_count": len(section_text_units),
            "pages": section_pages,
            "source_pages": section_pages,
            "relationship_count": 0,
            "standalone": False,
            "scope_level": "section",
            "generated_by": GENERATED_BY,
        })
        section_node = scope_nodes[section_id]
        section_node["text_unit_ids"] = _unique([
            *section_node.get("text_unit_ids", []), *section_text_units
        ])
        section_node["evidence_count"] = len(section_node["text_unit_ids"])
        section_node["pages"] = sorted({
            *section_node.get("pages", []), *section_pages
        })
        section_node["source_pages"] = list(section_node["pages"])
        add_scope_edge(
            book_id, chapter_id, "HAS_CHAPTER", "包含章节",
            chapter_text_units, chapter_pages,
        )
        add_scope_edge(
            chapter_id, section_id, "HAS_SECTION", "包含小节",
            section_text_units, section_pages,
        )
        node_id = str(node["id"])
        node["primary_scope_id"] = section_id
        node["scope_ids"] = _unique(
            _stable_id("scope-section", item[1], item[2]) for item in memberships
        )
        add_scope_edge(
            node_id, section_id, "MENTIONED_IN", "见于小节",
            [primary_text_unit] if primary_text_unit else [],
            [primary_page] if primary_page else [],
        )
    scope_nodes[book_id]["pages"] = sorted(all_pages)
    scope_nodes[book_id]["source_pages"] = sorted(all_pages)
    graph.setdefault("nodes", []).extend(scope_nodes.values())
    graph.setdefault("edges", []).extend(scope_edges.values())
    return len(scope_nodes), len(scope_edges)


def _refresh_graph_indexes(graph: dict[str, Any]) -> None:
    node_lookup = {str(node["id"]): node for node in graph.get("nodes", [])}
    degree: Counter[str] = Counter()
    for edge in graph.get("edges", []):
        degree[str(edge.get("source", ""))] += 1
        degree[str(edge.get("target", ""))] += 1
    for node_id, node in node_lookup.items():
        node["relationship_count"] = degree[node_id]
        node["standalone"] = degree[node_id] == 0
    links: dict[tuple[str, str], dict[str, Any]] = {}
    for edge in graph.get("edges", []):
        source = str(edge["source"])
        target = str(edge["target"])
        key = (source, target)
        link = links.setdefault(key, {
            "id": _stable_id("relationship-link", source, target),
            "source": source,
            "target": target,
            "mention_ids": [],
            "relationship_ids": [],
            "text_unit_ids": [],
            "weight": 0.0,
        })
        link["mention_ids"] = _unique([
            *link["mention_ids"], *edge.get("mention_ids", [])
        ])
        link["relationship_ids"] = _unique([
            *link["relationship_ids"], str(edge["id"])
        ])
        link["text_unit_ids"] = _unique([
            *link["text_unit_ids"], *edge.get("text_unit_ids", [])
        ])
        link["weight"] += float(edge.get("weight", 1.0) or 0.0)
    graph["relationship_links"] = list(links.values())


def audit_consolidated_graph(graph: dict[str, Any]) -> dict[str, Any]:
    import networkx as nx

    node_ids = {str(node.get("id", "")) for node in graph.get("nodes", [])}
    network = nx.Graph()
    network.add_nodes_from(node_ids)
    for edge in graph.get("edges", []):
        source = str(edge.get("source", ""))
        target = str(edge.get("target", ""))
        if source in node_ids and target in node_ids:
            network.add_edge(source, target)
    components = nx.number_connected_components(network) if node_ids else 0
    isolated = sum(network.degree(node_id) == 0 for node_id in node_ids)
    uncontrolled = sum(
        str(edge.get("relation_normalized", "")) not in CONTROLLED_RELATIONS
        for edge in graph.get("edges", [])
    )
    critical = isolated + uncontrolled + max(0, components - 1)
    layers = Counter(
        str(edge.get("edge_layer", "semantic_fact"))
        for edge in graph.get("edges", [])
    )
    issues = []
    for code, count in (
        ("disconnected_components", max(0, components - 1)),
        ("isolated_entities", isolated),
        ("uncontrolled_relation_types", uncontrolled),
    ):
        if count:
            issues.append({"severity": "critical", "code": code, "count": count})
    return {
        "schema_version": "1.0-consolidated-graph-quality",
        "status": "passed" if critical == 0 else "failed",
        "critical_issues": critical,
        "warning_issues": 0,
        "metrics": {
            "nodes": len(node_ids),
            "edges": len(graph.get("edges", [])),
            "connected_components": components,
            "isolated_entities": isolated,
            "average_degree": round(
                sum(dict(network.degree()).values()) / len(node_ids)
                if node_ids else 0.0,
                4,
            ),
            "relation_types": len({
                str(edge.get("relation_normalized", ""))
                for edge in graph.get("edges", [])
            }),
            "edge_layers": dict(sorted(layers.items())),
        },
        "issues": issues,
    }


def consolidate_semantic_graph(
    graph: dict[str, Any],
    embedding_model_path: Path | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Consolidate an extracted graph without asking an LLM for new facts."""

    previous_stats = dict(graph.get("stats", {}))
    _remove_generated_artifacts(graph)
    _, merged_entities = _merge_safe_aliases(graph)
    _, original_relation_types, dropped_edges = _merge_relationships(graph)
    entity_link_candidates, entity_links = _add_entity_links(
        graph, embedding_model_path
    )
    attribute_nodes, attribute_edges = _add_attribute_value_edges(graph)
    scope_nodes, scope_edges = _add_document_structure(graph)
    _refresh_graph_indexes(graph)

    from backend.app.rag.graphrag_adapter import repair_community_relationship_mapping

    community_mapping = repair_community_relationship_mapping(graph)
    _refresh_graph_indexes(graph)
    audit = audit_consolidated_graph(graph)
    graph.setdefault("stats", {}).update({
        "consolidation_version": CONSOLIDATION_VERSION,
        "entities": len(graph.get("nodes", [])),
        "relationships": len(graph.get("edges", [])),
        "original_relation_types": original_relation_types,
        "controlled_relation_types": audit["metrics"]["relation_types"],
        "merged_alias_entities": max(
            merged_entities,
            int(previous_stats.get("merged_alias_entities", 0) or 0),
        ),
        "dropped_or_self_loop_edges": max(
            dropped_edges,
            int(previous_stats.get("dropped_or_self_loop_edges", 0) or 0),
        ),
        "entity_link_candidates": entity_link_candidates,
        "entity_links": entity_links,
        "materialized_attribute_nodes": attribute_nodes,
        "materialized_attribute_edges": attribute_edges,
        "structure_nodes": scope_nodes,
        "structure_edges": scope_edges,
        "community_mapping": community_mapping,
    })
    return graph, audit
