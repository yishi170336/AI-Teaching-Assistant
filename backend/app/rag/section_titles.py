from __future__ import annotations

import re
from typing import Any


_SECTION_SENTENCE_FRAGMENTS = (
    "如图", "所示", "试求", "试画", "试分析", "已知", "求出", "判断",
    "可得", "因此", "所以", "其中", "这时", "由此",
)

_STRUCTURAL_SECTION_ALIASES = {
    "本章小结": "本章小结",
    "本章总结": "本章小结",
    "小结": "小结",
    "本章习题": "习题",
    "章末习题": "习题",
    "习题": "习题",
    "复习题": "复习题",
    "思考题": "思考题",
    "自测题": "自测题",
    "练习题": "练习题",
    "习题答案": "习题答案",
    "参考答案": "参考答案",
    "部分习题参考答案": "部分习题参考答案",
}

_MEASUREMENT_UNIT = (
    r"(?:[fpnumkMGTμµ]?(?:A|V|W|F|H|S|Ω|Ω|欧|安|伏|瓦|法|亨|西)"
    r"|(?:m|k|M|G)?Hz|dB|℃|°C|K|rpm|rad/s|V/V|A/V|V/A)"
)
_MEASUREMENT_SECTION_PATTERN = re.compile(
    rf"^\s*\d{{1,3}}(?:\s*[.．]\s*\d{{1,3}}){{1,3}}\s*"
    rf"{_MEASUREMENT_UNIT}(?:\s*[/·×]\s*{_MEASUREMENT_UNIT})?\s*$"
)


def normalize_structural_section(value: Any) -> str:
    """Return a canonical chapter-tail heading only when the whole line is one."""

    line = str(value or "").strip()
    line = re.sub(r"^(?:#{1,6}\s*|[•●▪·]\s*)", "", line)
    compact = re.sub(r"[\s　:：、.．—_-]+", "", line)
    return _STRUCTURAL_SECTION_ALIASES.get(compact, "")


def visible_structural_section(text: str) -> str:
    for raw_line in text.splitlines():
        section = normalize_structural_section(raw_line)
        if section:
            return section
        compact = re.sub(r"[\s　]+", "", raw_line).lstrip("#•●▪·")
        # Scanned-page OCR occasionally joins the heading and first paragraph.
        if compact.startswith(("本章小结", "本章总结")):
            return "本章小结"
    return ""


def is_measurement_section(value: Any) -> bool:
    """Detect OCR values such as ``1.0 mA`` that resemble section numbers."""

    heading = str(value or "").strip()
    if _MEASUREMENT_SECTION_PATTERN.fullmatch(heading):
        return True
    match = re.fullmatch(
        r"\s*\d{1,3}(?:\s*[.．]\s*\d{1,3}){1,3}\s+(.+?)\s*",
        heading,
    )
    if not match:
        return False
    title = re.sub(r"\s+", "", match.group(1))
    return bool(re.fullmatch(rf"{_MEASUREMENT_UNIT}(?:[/·×]{_MEASUREMENT_UNIT})?", title))


def normalize_numbered_section(value: Any) -> str:
    heading = re.sub(r"\s+", " ", str(value or "")).strip()
    if is_measurement_section(heading):
        return ""
    match = re.fullmatch(
        r"(\d{1,2}(?:\s*[.．]\s*\d{1,2}){1,3})\s+([^=。；，,!?！？]{2,60})",
        heading,
    )
    if not match:
        return ""
    title = match.group(2).strip(" .．、:：-")
    if any(fragment in title for fragment in _SECTION_SENTENCE_FRAGMENTS):
        return ""
    number = re.sub(r"\s*[.．]\s*", ".", match.group(1))
    return f"{number} {title}"


def numbered_section_parts(value: Any) -> tuple[tuple[int, ...], str]:
    normalized = normalize_numbered_section(value)
    if not normalized:
        return (), ""
    number, title = normalized.split(" ", 1)
    return tuple(int(part) for part in number.split(".")), title


def visible_numbered_sections(text: str) -> list[str]:
    headings: list[str] = []
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line or re.match(r"^(?:图|表|式|例|【例)", line):
            continue
        section = normalize_numbered_section(line)
        if section:
            headings.append(section)
    return headings


def _chinese_integer(value: str) -> int | None:
    if value.isdigit():
        return int(value)
    digits = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
              "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    if value == "十":
        return 10
    if "十" in value:
        tens, ones = value.split("十", 1)
        tens_value = digits.get(tens, 1) if tens else 1
        ones_value = digits.get(ones, 0) if ones else 0
        return tens_value * 10 + ones_value
    if all(character in digits for character in value):
        result = 0
        for character in value:
            result = result * 10 + digits[character]
        return result
    return None


def chapter_number(value: Any) -> int | None:
    match = re.search(r"第\s*([零〇一二三四五六七八九十两0-9]+)\s*章", str(value or ""))
    return _chinese_integer(match.group(1)) if match else None


def section_matches_chapter(section: Any, chapter: Any) -> bool:
    parts, _ = numbered_section_parts(section)
    expected = chapter_number(chapter)
    return not parts or expected is None or parts[0] == expected


def display_section(section: Any, chapter: Any, text: str) -> str:
    """Correct legacy index metadata at read time without mutating the index."""

    structural = visible_structural_section(text)
    if structural:
        return structural

    visible = [
        candidate
        for candidate in visible_numbered_sections(text)
        if section_matches_chapter(candidate, chapter)
    ]
    if visible:
        return visible[-1]

    stored_structural = normalize_structural_section(section)
    if stored_structural:
        return stored_structural
    stored = normalize_numbered_section(section)
    if stored and section_matches_chapter(stored, chapter):
        return stored
    return ""


def repair_legacy_chunk_sections(chunks: list[Any]) -> int:
    """Repair section metadata in memory by using sibling chunks from the same page."""

    grouped: dict[tuple[str, int], list[Any]] = {}
    for chunk in chunks:
        page = getattr(chunk, "page_start", None)
        if page is None:
            continue
        grouped.setdefault((str(getattr(chunk, "source", "")), int(page)), []).append(chunk)

    repaired = 0
    for page_chunks in grouped.values():
        structural = next(
            (
                candidate
                for chunk in page_chunks
                if (candidate := visible_structural_section(str(getattr(chunk, "text", ""))))
            ),
            "",
        )
        visible: list[str] = []
        for chunk in page_chunks:
            chapter = getattr(chunk, "chapter", "")
            visible.extend(
                candidate
                for candidate in visible_numbered_sections(str(getattr(chunk, "text", "")))
                if section_matches_chapter(candidate, chapter)
            )
        canonical = structural or (visible[-1] if visible else "")
        if not canonical:
            continue
        canonical_number, _ = numbered_section_parts(canonical)
        for chunk in page_chunks:
            stored = str(getattr(chunk, "section", ""))
            stored_number, _ = numbered_section_parts(stored)
            should_repair = bool(structural) or (
                is_measurement_section(stored)
                or not normalize_numbered_section(stored)
                or not section_matches_chapter(stored, getattr(chunk, "chapter", ""))
                or (
                    canonical_number
                    and stored_number == canonical_number
                    and stored != canonical
                )
            )
            if should_repair and stored != canonical:
                chunk.section = canonical
                repaired += 1
    return repaired
