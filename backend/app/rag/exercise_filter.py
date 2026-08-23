from __future__ import annotations

import re
from dataclasses import replace
from typing import Any, Iterable, TypeVar

from backend.app.rag.models import PageDocument


EXERCISE_FILTER_POLICY_VERSION = "1.0-exercise-section-noise"

_EXERCISE_TITLE_PATTERN = re.compile(
    r"^(?:部分)?(?:本章|章末|课后)?(?:习题|练习题|复习题|思考题|自测题|测试题)"
    r"(?:参考答案|答案|解答|提示)?$"
)
_EXERCISE_ALTERNATE_TITLES = {
    "思考与练习",
    "自我检测",
    "答案与提示",
}


def _compact_title(value: Any) -> str:
    text = re.sub(r"\s+", "", str(value or "")).strip()
    text = re.sub(r"^[第]?[一二三四五六七八九十百零0-9]+章", "", text)
    text = re.sub(r"^\d+(?:\.\d+){0,3}", "", text)
    return text.strip("-—_：:、.．()（）[]【】")


def is_exercise_section_title(value: Any) -> bool:
    """Return whether a structural title denotes exercises rather than an example."""

    title = _compact_title(value)
    if not title or title.startswith(("例题", "例", "习题课")):
        return False
    return bool(
        _EXERCISE_TITLE_PATTERN.fullmatch(title)
        or title in _EXERCISE_ALTERNATE_TITLES
    )


def is_exercise_context(*values: Any) -> bool:
    return any(is_exercise_section_title(value) for value in values)


def exercise_section_ids(sections: Iterable[dict[str, Any]]) -> set[str]:
    """Return exercise section ids and all of their structural descendants."""

    values = [dict(item) for item in sections if isinstance(item, dict)]
    removed = {
        str(item.get("id"))
        for item in values
        if item.get("id")
        and is_exercise_context(item.get("title"), item.get("name"))
    }
    changed = True
    while changed:
        changed = False
        for item in values:
            item_id = str(item.get("id", ""))
            if (
                item_id
                and item_id not in removed
                and str(item.get("parent", "")) in removed
            ):
                removed.add(item_id)
                changed = True
    return removed


def _normalized_bbox(block: dict[str, Any]) -> list[float]:
    bbox = block.get("bbox", [])
    if not isinstance(bbox, list) or len(bbox) != 4:
        return []
    try:
        return [float(value) for value in bbox]
    except (TypeError, ValueError):
        return []


def _exercise_ranges(
    blocks: list[dict[str, Any]], removed_indexes: set[int]
) -> list[list[float]]:
    """Build normalized vertical exclusion bands for mixed-content pages."""

    if not removed_indexes:
        return []
    ranges: list[list[float]] = []
    ordered = sorted(
        enumerate(blocks),
        key=lambda item: (int(item[1].get("reading_order", item[0]) or item[0]), item[0]),
    )
    position = 0
    while position < len(ordered):
        source_index, block = ordered[position]
        if source_index not in removed_indexes:
            position += 1
            continue
        group: list[dict[str, Any]] = []
        while position < len(ordered) and ordered[position][0] in removed_indexes:
            group.append(ordered[position][1])
            position += 1
        boxes = [bbox for item in group if (bbox := _normalized_bbox(item))]
        if not boxes:
            continue
        start_y = max(0.0, min(box[1] for box in boxes))
        end_y = min(1000.0, max(box[3] for box in boxes))
        if position >= len(ordered):
            end_y = 1000.0
        else:
            next_box = _normalized_bbox(ordered[position][1])
            if next_box and next_box[1] >= start_y:
                end_y = max(end_y, min(1000.0, next_box[1]))
        ranges.append([start_y, max(start_y, end_y)])
    return ranges


def filter_exercise_documents(
    documents: Iterable[PageDocument],
) -> tuple[list[PageDocument], list[dict[str, Any]]]:
    """Remove exercise-scoped Paddle blocks while retaining mixed-page course text."""

    kept: list[PageDocument] = []
    audit: list[dict[str, Any]] = []
    for document in documents:
        extra = dict(document.extra or {})
        raw_blocks = extra.get("text_blocks", [])
        blocks = [dict(item) for item in raw_blocks if isinstance(item, dict)]
        removed_indexes = {
            index
            for index, block in enumerate(blocks)
            if is_exercise_context(
                block.get("chapter", document.chapter),
                block.get("section", document.section),
            )
        }
        whole_document_exercise = is_exercise_context(
            document.chapter, document.section
        )
        if not blocks:
            if whole_document_exercise:
                audit.append({
                    "page": document.page,
                    "source_page": document.source_page or document.page,
                    "keep": False,
                    "page_type": "exercise",
                    "removed_blocks": 0,
                    "kept_blocks": 0,
                    "reason": "章节标题识别为习题区",
                    "policy_version": EXERCISE_FILTER_POLICY_VERSION,
                })
                continue
            kept.append(document)
            audit.append({
                "page": document.page,
                "source_page": document.source_page or document.page,
                "keep": True,
                "page_type": "course_content",
                "removed_blocks": 0,
                "kept_blocks": 0,
                "reason": "未发现习题章节块",
                "policy_version": EXERCISE_FILTER_POLICY_VERSION,
            })
            continue

        retained_blocks = [
            block for index, block in enumerate(blocks) if index not in removed_indexes
        ]
        substantive = [
            block
            for block in retained_blocks
            if str(block.get("type", "")).lower()
            not in {"page_header", "page_footer", "noise"}
            and str(block.get("text", "")).strip()
        ]
        if removed_indexes and not substantive:
            audit.append({
                "page": document.page,
                "source_page": document.source_page or document.page,
                "keep": False,
                "page_type": "exercise",
                "removed_blocks": len(removed_indexes),
                "kept_blocks": len(retained_blocks),
                "reason": "本页有效内容全部属于习题章节",
                "policy_version": EXERCISE_FILTER_POLICY_VERSION,
            })
            continue

        ranges = _exercise_ranges(blocks, removed_indexes)
        extra["text_blocks"] = retained_blocks
        extra["exercise_filter"] = {
            "policy_version": EXERCISE_FILTER_POLICY_VERSION,
            "removed_block_ids": [
                str(blocks[index].get("id", ""))
                for index in sorted(removed_indexes)
                if blocks[index].get("id")
            ],
            "excluded_normalized_y_ranges": ranges,
        }
        context_block = substantive[-1] if substantive else {}
        text = "\n".join(
            str(block.get("text", "")).strip()
            for block in retained_blocks
            if str(block.get("text", "")).strip()
        ).strip()
        kept.append(replace(
            document,
            text=text or document.text,
            chapter=str(context_block.get("chapter", document.chapter)) or document.chapter,
            section=str(context_block.get("section", document.section)) or document.section,
            extra=extra,
        ))
        audit.append({
            "page": document.page,
            "source_page": document.source_page or document.page,
            "keep": True,
            "page_type": "mixed" if removed_indexes else "course_content",
            "removed_blocks": len(removed_indexes),
            "kept_blocks": len(retained_blocks),
            "reason": (
                "已按章节边界移除同页习题块并保留课程正文"
                if removed_indexes
                else "未发现习题章节块"
            ),
            "policy_version": EXERCISE_FILTER_POLICY_VERSION,
        })
    return kept, audit


T = TypeVar("T")


def filter_exercise_items(items: Iterable[T]) -> list[T]:
    """Filter PageDocument/LayoutElement/TextChunk-like objects by context."""

    return [
        item
        for item in items
        if not is_exercise_context(
            getattr(item, "chapter", ""), getattr(item, "section", "")
        )
    ]


def bbox_in_exercise_range(
    document: PageDocument,
    bbox: Iterable[float],
    *,
    page_height: float,
) -> bool:
    """Check a PDF-coordinate bbox against mixed-page exercise exclusion bands."""

    values = list(bbox)
    if len(values) != 4 or page_height <= 0:
        return False
    ranges = (
        ((document.extra or {}).get("exercise_filter") or {}).get(
            "excluded_normalized_y_ranges", []
        )
    )
    center_y = ((float(values[1]) + float(values[3])) / 2.0) / page_height * 1000.0
    return any(
        isinstance(item, list)
        and len(item) == 2
        and float(item[0]) <= center_y <= float(item[1])
        for item in ranges
    )
