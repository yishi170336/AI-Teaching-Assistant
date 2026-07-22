from __future__ import annotations

import hashlib
import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Any


BASE_SCORE_PER_MISTAKE = 10
REPEAT_SCORE_BONUS = 5
SOURCE_DIVERSITY_BONUS = 2
MODERATE_THRESHOLD = 20
SEVERE_THRESHOLD = 35
MIN_RELIABLE_SAMPLE = 3
APPROXIMATE_MATCH_THRESHOLD = 0.72


def _normalized_name(value: str) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", value.casefold())


def _custom_tag_id(name: str) -> str:
    return "custom:" + hashlib.sha1(name.encode("utf-8")).hexdigest()[:16]


def _similarity(left: str, right: str) -> float:
    normalized_left, normalized_right = _normalized_name(left), _normalized_name(right)
    if not normalized_left or not normalized_right:
        return 0.0
    ratio = SequenceMatcher(None, normalized_left, normalized_right).ratio()
    if normalized_left in normalized_right or normalized_right in normalized_left:
        coverage = min(len(normalized_left), len(normalized_right)) / max(
            len(normalized_left), len(normalized_right)
        )
        ratio = max(ratio, 0.70 + 0.25 * coverage)
    return min(1.0, ratio)


class MistakeKnowledgeService:
    """Read-only adapter from mistakes to the existing course knowledge graph."""

    EXPLICIT_PREREQUISITE_RELATIONS = {
        "PREREQUISITE",
        "PREREQUISITE_OF",
        "PRECEDES",
        "FOUNDATION_OF",
    }
    DEPENDENCY_RELATIONS = {"DEPENDS_ON", "REQUIRES"}

    def __init__(self, knowledge_bases: Any) -> None:
        self.knowledge_bases = knowledge_bases

    @staticmethod
    def _concept_nodes(graph: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            node
            for node in graph.get("nodes", [])
            if isinstance(node, dict)
            and node.get("type") == "concept"
            and str(node.get("name", "")).strip()
        ]

    @staticmethod
    def _unmatched_tag(point: str) -> dict[str, Any]:
        return {
            "tag_id": _custom_tag_id(point),
            "tag_name": point,
            "tag_source": "custom",
            "knowledge_node_id": None,
            "match_type": "unmatched",
            "confidence": 0.0,
            "is_exact": False,
            "needs_confirmation": True,
        }

    def _match_tag(
        self, point: str, concepts: list[dict[str, Any]]
    ) -> dict[str, Any]:
        normalized_point = _normalized_name(point)
        exact = next(
            (
                node
                for node in concepts
                if _normalized_name(str(node.get("name", ""))) == normalized_point
            ),
            None,
        )
        if exact:
            return {
                "tag_id": str(exact.get("id")),
                "tag_name": str(exact.get("name")),
                "tag_source": "knowledge_graph",
                "knowledge_node_id": str(exact.get("id")),
                "match_type": "exact",
                "confidence": 1.0,
                "is_exact": True,
                "needs_confirmation": False,
            }
        candidates = sorted(
            (
                (_similarity(point, str(node.get("name", ""))), node)
                for node in concepts
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        if candidates and candidates[0][0] >= APPROXIMATE_MATCH_THRESHOLD:
            confidence, matched = candidates[0]
            return {
                "tag_id": str(matched.get("id")),
                "tag_name": str(matched.get("name")),
                "tag_source": "knowledge_graph",
                "knowledge_node_id": str(matched.get("id")),
                "match_type": "approximate",
                "confidence": round(confidence, 3),
                "is_exact": False,
                "needs_confirmation": confidence < 0.86,
            }
        return self._unmatched_tag(point)

    @staticmethod
    def _chapter_for_tags(
        graph: dict[str, Any], matched_names: set[str]
    ) -> dict[str, Any] | None:
        ranked: list[tuple[int, int, dict[str, Any]]] = []
        for chapter in graph.get("chapters", []):
            if not isinstance(chapter, dict):
                continue
            concept_names = {
                _normalized_name(str(concept.get("name", "")))
                for concept in chapter.get("concepts", [])
                if isinstance(concept, dict)
            }
            matches = len(concept_names & matched_names)
            if matches:
                ranked.append((matches, -int(chapter.get("order", 0) or 0), chapter))
        return max(ranked, default=(0, 0, None), key=lambda item: (item[0], item[1]))[2]

    def _chunk_location(
        self,
        knowledge_base: str,
        matched_names: set[str],
        preferred_chapter: str = "",
    ) -> tuple[str, str, float]:
        try:
            chunks = self.knowledge_bases.get(knowledge_base).chunks
        except (AttributeError, RuntimeError, ValueError):
            return "", "", 0.0
        evidence: dict[tuple[str, str], Counter[str]] = {}
        for chunk in chunks:
            tags = {
                _normalized_name(str(tag))
                for tag in getattr(chunk, "knowledge_tags", [])
                if str(tag).strip()
            }
            matched = tags & matched_names
            if not matched:
                continue
            chapter = str(getattr(chunk, "chapter", "") or "").strip()
            section = str(getattr(chunk, "section", "") or "").strip()
            if chapter or section:
                evidence.setdefault((chapter, section), Counter()).update(matched)
        if not evidence:
            return "", "", 0.0
        preferred = {
            location: matches
            for location, matches in evidence.items()
            if preferred_chapter and location[0] == preferred_chapter
        }
        candidates = preferred or evidence
        (chapter, section), matches = max(
            candidates.items(),
            key=lambda item: (
                len(item[1]),
                sum(item[1].values()),
                -len(item[0][1]),
                item[0],
            ),
        )
        coverage = len(matches) / max(1, len(matched_names))
        confidence = min(0.95, 0.55 + 0.25 * coverage + 0.02 * sum(matches.values()))
        return chapter, section, round(confidence, 3)

    def _prerequisites(
        self,
        graph: dict[str, Any],
        matched_ids: set[str],
        chapter: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        nodes = {
            str(node.get("id")): node
            for node in graph.get("nodes", [])
            if isinstance(node, dict) and node.get("id")
        }
        result: list[dict[str, Any]] = []
        for edge in graph.get("edges", []):
            if not isinstance(edge, dict):
                continue
            source, target = str(edge.get("source", "")), str(edge.get("target", ""))
            relation = str(edge.get("type", "")).upper()
            prerequisite_id = ""
            if target in matched_ids and relation in self.EXPLICIT_PREREQUISITE_RELATIONS:
                prerequisite_id = source
            elif source in matched_ids and relation in self.DEPENDENCY_RELATIONS:
                prerequisite_id = target
            node = nodes.get(prerequisite_id)
            if node and node.get("type") == "concept":
                result.append(
                    {
                        "knowledge_node_id": prerequisite_id,
                        "name": str(node.get("name", "")),
                        "source": "knowledge_graph",
                        "relation": relation.lower(),
                        "confidence": 0.95,
                    }
                )
        if result:
            unique = {str(item["knowledge_node_id"]): item for item in result}
            return list(unique.values())[:5]

        if chapter:
            order = int(chapter.get("order", 0) or 0)
            previous = next(
                (
                    item
                    for item in graph.get("chapters", [])
                    if isinstance(item, dict) and int(item.get("order", 0) or 0) == order - 1
                ),
                None,
            )
            if previous:
                return [
                    {
                        "knowledge_node_id": str(concept.get("id", "")) or None,
                        "name": str(concept.get("name", "")),
                        "source": "chapter_order",
                        "relation": "previous_chapter",
                        "confidence": 0.45,
                    }
                    for concept in previous.get("concepts", [])[:3]
                    if isinstance(concept, dict) and concept.get("name")
                ]
        return []

    def align(
        self, knowledge_base: str, knowledge_points: list[str]
    ) -> dict[str, Any]:
        points = list(dict.fromkeys(point.strip() for point in knowledge_points if point.strip()))[:12]
        try:
            graph = self.knowledge_bases.graph(knowledge_base)
            concepts = self._concept_nodes(graph)
        except (OSError, RuntimeError, TypeError, ValueError):
            return {
                "knowledge_tags": [self._unmatched_tag(point) for point in points],
                "location": {
                    "chapter": "暂未确定",
                    "section": "暂未确定",
                    "source": "unavailable",
                    "confidence": 0.0,
                },
                "prerequisites": [],
            }

        tags = [self._match_tag(point, concepts) for point in points]
        matched_names = {
            _normalized_name(str(tag["tag_name"]))
            for tag in tags
            if tag["tag_source"] == "knowledge_graph"
        }
        matched_ids = {
            str(tag["knowledge_node_id"])
            for tag in tags
            if tag.get("knowledge_node_id")
        }
        chapter_summary = self._chapter_for_tags(graph, matched_names)
        chunk_chapter, chunk_section, chunk_confidence = self._chunk_location(
            knowledge_base,
            matched_names,
            str((chapter_summary or {}).get("name", "")),
        )
        chapter_name = chunk_chapter or str((chapter_summary or {}).get("name", ""))
        section_name = chunk_section
        if chapter_name or section_name:
            location = {
                "chapter": chapter_name or "暂未确定",
                "section": section_name or "暂未确定",
                "source": "knowledge_graph",
                "confidence": chunk_confidence or 0.7,
            }
        else:
            location = {
                "chapter": "暂未确定",
                "section": "暂未确定",
                "source": "unmatched",
                "confidence": 0.0,
            }
        return {
            "knowledge_tags": tags,
            "location": location,
            "prerequisites": self._prerequisites(graph, matched_ids, chapter_summary),
        }

    @staticmethod
    def _severity(score: int) -> str:
        if score >= SEVERE_THRESHOLD:
            return "重度薄弱"
        if score >= MODERATE_THRESHOLD:
            return "中度薄弱"
        return "轻度薄弱"

    def analyze(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        source_counts: Counter[str] = Counter()
        chapter_counts: Counter[str] = Counter()
        tag_records: dict[str, dict[str, Any]] = {}
        annotation_count = 0

        for item in items:
            source_counts[str(item.get("source") or "user_uploaded")] += 1
            location = item.get("location") if isinstance(item.get("location"), dict) else {}
            chapter = str(location.get("chapter") or "暂未确定")
            chapter_counts[chapter] += 1
            annotations = item.get("annotations")
            annotation_count += len(annotations) if isinstance(annotations, list) else 0
            tags = item.get("knowledge_tags") if isinstance(item.get("knowledge_tags"), list) else []
            names = [
                str(tag.get("tag_name", "")).strip()
                for tag in tags
                if isinstance(tag, dict) and str(tag.get("tag_name", "")).strip()
            ] or [str(point).strip() for point in item.get("knowledge_points", []) if str(point).strip()]
            for name in dict.fromkeys(names):
                record = tag_records.setdefault(
                    name,
                    {
                        "name": name,
                        "mistake_ids": [],
                        "sources": set(),
                        "chapters": Counter(),
                        "sections": Counter(),
                        "prerequisites": {},
                    },
                )
                record["mistake_ids"].append(str(item.get("id", "")))
                record["sources"].add(str(item.get("source") or "user_uploaded"))
                record["chapters"][chapter] += 1
                section = str(location.get("section") or "暂未确定")
                record["sections"][section] += 1
                for prerequisite in item.get("prerequisites", []):
                    if isinstance(prerequisite, dict) and prerequisite.get("name"):
                        record["prerequisites"][str(prerequisite["name"])] = prerequisite

        weak_areas: list[dict[str, Any]] = []
        for record in tag_records.values():
            count = len(record["mistake_ids"])
            source_diversity = len(record["sources"])
            score = (
                count * BASE_SCORE_PER_MISTAKE
                + max(0, count - 1) * REPEAT_SCORE_BONUS
                + max(0, source_diversity - 1) * SOURCE_DIVERSITY_BONUS
            )
            weak_areas.append(
                {
                    "knowledge_point": record["name"],
                    "mistake_count": count,
                    "mistake_ids": record["mistake_ids"],
                    "source_count": source_diversity,
                    "score": score,
                    "severity": self._severity(score),
                    "chapter": record["chapters"].most_common(1)[0][0],
                    "section": record["sections"].most_common(1)[0][0],
                    "prerequisites": list(record["prerequisites"].values())[:3],
                }
            )
        weak_areas.sort(key=lambda item: (-int(item["score"]), str(item["knowledge_point"])))
        total = len(items)
        data_sufficient = total >= MIN_RELIABLE_SAMPLE
        return {
            "total_mistakes": total,
            "data_sufficient": data_sufficient,
            "notice": (
                "分析基于当前错题的确定性统计。"
                if data_sufficient
                else "当前错题数据较少，分析结果仅供参考。"
            ),
            "source_counts": dict(source_counts),
            "chapter_counts": dict(chapter_counts),
            "annotation_count": annotation_count,
            "weak_areas": weak_areas,
            "recommended_order": [
                {
                    "priority": index + 1,
                    **area,
                }
                for index, area in enumerate(weak_areas[:6])
            ],
            "scoring_rule": {
                "base_per_mistake": BASE_SCORE_PER_MISTAKE,
                "repeat_bonus": REPEAT_SCORE_BONUS,
                "source_diversity_bonus": SOURCE_DIVERSITY_BONUS,
                "moderate_threshold": MODERATE_THRESHOLD,
                "severe_threshold": SEVERE_THRESHOLD,
                "minimum_reliable_sample": MIN_RELIABLE_SAMPLE,
            },
        }
