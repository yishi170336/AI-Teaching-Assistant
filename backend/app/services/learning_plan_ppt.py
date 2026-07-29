from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import tempfile
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

from backend.app.rag.section_titles import is_measurement_section


logger = logging.getLogger(__name__)


SLIDE_WIDTH = Inches(13.333333)
SLIDE_HEIGHT = Inches(7.5)
FONT_CN = "Microsoft YaHei"
FONT_LATIN = "Aptos"

INK = "163E3D"
INK_SOFT = "3B5E5B"
TEAL_DARK = "0C5F5B"
TEAL = "0F766E"
TEAL_BRIGHT = "2AA695"
MINT = "DFF2ED"
MINT_LIGHT = "F2F8F6"
CREAM = "FBFAF5"
WHITE = "FFFFFF"
SAND = "EEE8DA"
CORAL = "E98268"
GOLD = "D5A943"
GRAY = "718784"
LINE = "D7E5E1"

_HEADING_RE = re.compile(r"^(#{1,4})\s+(.+?)\s*$", re.MULTILINE)
_LIST_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)、]\s+|[（(]?[一二三四五六七八九十]+[）)、.]\s*)")
_STAGE_RE = re.compile(
    r"(?:阶段\s*[一二三四五六七八九十\d]+|第\s*[一二三四五六七八九十\d]+\s*(?:阶段|步|模块)|"
    r"模块\s*[一二三四五六七八九十\d]+)",
    re.IGNORECASE,
)
_NON_LEARNING_STAGE_RE = re.compile(
    r"复盘|验收|迁移(?:应用)?|资料索引|参考资料|检索依据",
    re.IGNORECASE,
)
_DURATION_RE = re.compile(
    r"\d+(?:\.\d+)?(?:\s*[~～\-–—至]\s*\d+(?:\.\d+)?)?\s*(?:分钟|小时|课时|课次|天|周)",
    re.IGNORECASE,
)
_LATEX_SPAN_RE = re.compile(
    r"\$\$(.+?)\$\$|(?<!\\)\$(.+?)(?<!\\)\$|\\\[(.+?)\\\]|\\\((.+?)\\\)",
    re.DOTALL,
)
_MATH_FONT_REGULAR = Path("C:/Windows/Fonts/msyh.ttc")
_MATH_FONT_BOLD = Path("C:/Windows/Fonts/msyhbd.ttc")


@dataclass
class PlanSection:
    title: str
    body: str
    level: int = 2


@dataclass
class PlanStage:
    title: str
    goal: str
    concepts: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    practices: list[str] = field(default_factory=list)
    standards: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)


@dataclass
class ParsedLearningPlan:
    title: str
    goal: str
    summary: str
    stages: list[PlanStage]
    metrics: list[str]
    principles: list[str]
    references: list[str]


def _mathtext_safe(value: str) -> str:
    value = _normalize_latex(_plain(value))

    def normalize_formula(match: re.Match[str]) -> str:
        body = next((part for part in match.groups() if part is not None), "")
        body = body.replace(r"\displaystyle", "")
        body = re.sub(r"\\begin\{(?:aligned|align\*?|array)\}", "", body)
        body = re.sub(r"\\end\{(?:aligned|align\*?|array)\}", "", body)
        body = body.replace(r"\\", r"\quad ")
        body = re.sub(r"\\operatorname\{([^{}]+)\}", r"\\mathrm{\1}", body)
        body = re.sub(
            r"\\text\{([A-Za-z0-9 ./%°μΩ-]+)\}",
            r"\\mathrm{\1}",
            body,
        )
        return f"${body.strip()}$"

    return _LATEX_SPAN_RE.sub(normalize_formula, value)


def _wrap_rich_text(value: str, capacity: int) -> str:
    value = _mathtext_safe(value)
    capacity = max(6, capacity)
    lines: list[str] = []
    current: list[str] = []
    current_length = 0
    cursor = 0

    def add_plain(chunk: str) -> None:
        nonlocal current, current_length
        for char in chunk:
            if char == "\n":
                lines.append("".join(current).rstrip())
                current = []
                current_length = 0
                continue
            if current_length >= capacity:
                lines.append("".join(current).rstrip())
                current = []
                current_length = 0
            current.append(char)
            current_length += 1

    for match in _LATEX_SPAN_RE.finditer(value):
        add_plain(value[cursor : match.start()])
        formula = match.group(0)
        formula_length = max(3, len(_clean_math(formula)))
        if current and current_length + formula_length > capacity:
            lines.append("".join(current).rstrip())
            current = []
            current_length = 0
        current.append(formula)
        current_length += min(formula_length, capacity)
        cursor = match.end()
    add_plain(value[cursor:])
    if current or not lines:
        lines.append("".join(current).rstrip())
    return "\n".join(line for line in lines if line)


def _fit_rich_text(value: str, width: float, height: float, size: float) -> str:
    capacity = max(7, int(width * 72 / max(size * 0.9, 1)))
    max_lines = max(1, int(height * 72 / max(size * 1.02, 1)))
    visible_length = len(_plain(value, preserve_latex=False))
    capacity = max(capacity, math.ceil(visible_length / max_lines))
    return _wrap_rich_text(value, capacity)


class LatexTextRenderer:
    def __init__(self, scratch_dir: Path):
        self.scratch_dir = scratch_dir
        self.scratch_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _font(bold: bool):
        from matplotlib.font_manager import FontProperties

        font_path = _MATH_FONT_BOLD if bold and _MATH_FONT_BOLD.exists() else _MATH_FONT_REGULAR
        if font_path.exists():
            return FontProperties(fname=str(font_path))
        return FontProperties(family=FONT_CN, weight="bold" if bold else "normal")

    @staticmethod
    def _figure(width: float, height: float):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        figure = Figure(figsize=(max(width, 0.2), max(height, 0.2)), dpi=200)
        figure.patch.set_alpha(0)
        FigureCanvasAgg(figure)
        return figure

    def render_text(
        self,
        value: str,
        width: float,
        height: float,
        *,
        size: float,
        color: str,
        bold: bool,
        align: Any,
        valign: Any,
        line_spacing: float,
    ) -> Path:
        payload = _fit_rich_text(value, width, height, size)
        key = hashlib.sha256(
            repr(("text", payload, width, height, size, color, bold, str(align), str(valign))).encode("utf-8")
        ).hexdigest()[:24]
        output = self.scratch_dir / f"formula-{key}.png"
        if output.exists():
            return output

        from matplotlib import rc_context

        horizontal = "center" if align == PP_ALIGN.CENTER else "right" if align == PP_ALIGN.RIGHT else "left"
        x = 0.5 if horizontal == "center" else 0.99 if horizontal == "right" else 0.01
        if valign == MSO_ANCHOR.MIDDLE:
            vertical, y = "center", 0.5
        elif valign == MSO_ANCHOR.BOTTOM:
            vertical, y = "bottom", 0.02
        else:
            vertical, y = "top", 0.98
        figure = self._figure(width, height)
        with rc_context(
            {
                "mathtext.fontset": "stix",
                "mathtext.default": "bf" if bold else "it",
                "axes.unicode_minus": False,
            }
        ):
            figure.text(
                x,
                y,
                payload,
                color=f"#{color}",
                fontsize=size,
                fontproperties=self._font(bold),
                horizontalalignment=horizontal,
                verticalalignment=vertical,
                linespacing=line_spacing,
            )
            figure.canvas.print_png(str(output))
        return output

    def render_lines(
        self,
        lines: list[tuple[str, str]],
        width: float,
        height: float,
        *,
        size: float,
        color: str,
        marker_color: str,
    ) -> Path:
        row_height = height / max(len(lines), 1)
        fitted = [
            (marker, _fit_rich_text(value, width - 0.54, row_height, size))
            for marker, value in lines
        ]
        key = hashlib.sha256(
            repr(("lines", fitted, width, height, size, color, marker_color)).encode("utf-8")
        ).hexdigest()[:24]
        output = self.scratch_dir / f"formula-list-{key}.png"
        if output.exists():
            return output

        from matplotlib import rc_context

        figure = self._figure(width, height)
        marker_x = 0.01
        text_x = min(0.14, 0.52 / max(width, 0.1))
        with rc_context({"mathtext.fontset": "stix", "axes.unicode_minus": False}):
            for index, (marker, value) in enumerate(fitted):
                y = 1 - (index + 0.5) / max(len(fitted), 1)
                figure.text(
                    marker_x,
                    y,
                    marker,
                    color=f"#{marker_color}",
                    fontsize=size,
                    fontproperties=self._font(True),
                    verticalalignment="center",
                )
                figure.text(
                    text_x,
                    y,
                    value,
                    color=f"#{color}",
                    fontsize=size,
                    fontproperties=self._font(False),
                    verticalalignment="center",
                    linespacing=1.08,
                )
            figure.canvas.print_png(str(output))
        return output


def _rgb(value: str) -> RGBColor:
    return RGBColor.from_string(value)


def _normalize_latex(value: str) -> str:
    def canonical(match: re.Match[str]) -> str:
        body = next((part for part in match.groups() if part is not None), "")
        body = re.sub(r"\s+", " ", body).strip()
        return f"${body}$"

    return _LATEX_SPAN_RE.sub(canonical, value)


def _has_latex(value: str) -> bool:
    return bool(_LATEX_SPAN_RE.search(_normalize_latex(value)))


def _clean_math(value: str) -> str:
    replacements = {
        r"\geq": "≥",
        r"\ge": "≥",
        r"\leq": "≤",
        r"\le": "≤",
        r"\approx": "≈",
        r"\times": "×",
        r"\cdot": "·",
        r"\beta": "β",
        r"\alpha": "α",
        r"\pi": "π",
        r"\theta": "θ",
        r"\lambda": "λ",
        r"\Delta": "Δ",
        r"\Omega": "Ω",
        r"\mu": "μ",
        r"\text": "",
    }
    for source, target in replacements.items():
        value = value.replace(source, target)
    def braced(text: str, start: int) -> tuple[str, int] | None:
        if start >= len(text) or text[start] != "{":
            return None
        depth = 0
        for cursor in range(start, len(text)):
            if text[cursor] == "{":
                depth += 1
            elif text[cursor] == "}":
                depth -= 1
                if depth == 0:
                    return text[start + 1 : cursor], cursor + 1
        return None

    while r"\frac" in value:
        start = value.find(r"\frac")
        numerator = braced(value, start + len(r"\frac"))
        if numerator is None:
            break
        denominator = braced(value, numerator[1])
        if denominator is None:
            break
        replacement = f"({numerator[0]})/({denominator[0]})"
        value = value[:start] + replacement + value[denominator[1] :]

    value = re.sub(r"_\{([^{}]+)\}", r"_\1", value)
    value = re.sub(r"\^\{([^{}]+)\}", r"^\1", value)
    value = re.sub(r"([_^])([A-Za-z0-9])", r"\1\2", value)
    value = re.sub(r"\\[A-Za-z]+", "", value)
    value = value.replace("{", "").replace("}", "").replace("$", "")
    return value


def _plain(value: str, preserve_latex: bool = True) -> str:
    value = _normalize_latex(value)
    formulas: list[str] = []

    def protect(match: re.Match[str]) -> str:
        formulas.append(match.group(0))
        return f"ZXQLATEX{len(formulas) - 1}QXZ"

    value = _LATEX_SPAN_RE.sub(protect, value)
    value = re.sub(r"!\[([^]]*)\]\([^)]+\)", r"\1", value)
    value = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", value)
    value = re.sub(r"<[^>]+>", "", value)
    value = value.replace("**", "").replace("__", "").replace("`", "").replace("*", "")
    value = re.sub(r"^\s*>\s*", "", value)
    value = _LIST_RE.sub("", value.strip())
    value = re.sub(r"\s+", " ", value).strip(" \t\r\n|—–-：:")
    for index, formula in enumerate(formulas):
        replacement = formula if preserve_latex else _clean_math(formula)
        value = value.replace(f"ZXQLATEX{index}QXZ", replacement)
    return value


def _shorten(value: str, limit: int) -> str:
    value = _plain(value)
    visible = _plain(value, preserve_latex=False)
    if len(visible) <= limit:
        return value
    content_limit = max(1, limit - 1)
    result: list[str] = []
    consumed = 0
    cursor = 0
    for match in _LATEX_SPAN_RE.finditer(value):
        prefix = value[cursor : match.start()]
        remaining = content_limit - consumed
        if remaining <= 0:
            break
        if len(prefix) > remaining:
            result.append(prefix[:remaining])
            consumed = content_limit
            break
        result.append(prefix)
        consumed += len(prefix)
        formula = match.group(0)
        formula_length = max(3, len(_clean_math(formula)))
        if consumed + formula_length > content_limit:
            break
        result.append(formula)
        consumed += formula_length
        cursor = match.end()
    else:
        remaining = content_limit - consumed
        if remaining > 0:
            result.append(value[cursor : cursor + remaining])
    clipped = "".join(result).rstrip("，、；：,. ")
    return f"{clipped}…"


def _editorial_clip(value: str, limit: int, *, preserve_latex: bool = True) -> str:
    """Shorten audience copy at a semantic boundary without visible ellipses."""

    cleaned = _plain(value, preserve_latex=preserve_latex)
    cleaned = re.sub(r"\bschedule_guidance\b[^。；]*[。；]?", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(_plain(cleaned, preserve_latex=False)) <= limit:
        return cleaned

    without_examples = re.sub(r"[（(](?:如|例如|比如)[^）)]*[）)]", "", cleaned)
    without_examples = re.sub(r"(?:例如|比如)[:：][^。；]*", "", without_examples)
    without_examples = re.sub(r"[:：]\s*[,，；;]", "：", without_examples)
    without_examples = re.sub(r"[,，]\s*[；;]", "；", without_examples)
    without_examples = re.sub(r"\s+([，。；：])", r"\1", without_examples)
    without_examples = re.sub(r"\s+", " ", without_examples).strip()
    if len(_plain(without_examples, preserve_latex=False)) <= limit:
        return without_examples.rstrip("，、；：,. ")

    parts = [
        part.strip()
        for part in re.split(r"(?<=[。；])|(?=；)|(?<=：)", without_examples)
        if part.strip()
    ]
    selected = ""
    for part in parts:
        candidate = f"{selected}{part}".strip()
        if len(_plain(candidate, preserve_latex=False)) > limit:
            break
        selected = candidate
    if selected:
        return selected.rstrip("，、；：,. ")

    visible = _plain(without_examples, preserve_latex=False)
    boundary = max(
        without_examples.rfind(marker, 0, limit)
        for marker in ("；", "。", "，", "、", "：")
    )
    clipped = without_examples[: boundary if boundary >= max(8, limit // 2) else limit]
    clipped = clipped.rstrip("，、；：,. （(【[")
    return clipped or visible[:limit].rstrip("，、；：,. （(【[")


def _remove_time_arrangements(value: str) -> str:
    """Remove scheduling promises while keeping the learning action itself."""

    value = re.sub(
        r"\bDay\s*\d+(?:\s*[～~\-–—至]\s*\d+)?\s*[:：]?\s*",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"\d+\s*小时后", "完成首次订正后", value)
    value = re.sub(r"(?:能)?在?\s*\d+\s*分钟内", "能独立", value)
    value = re.sub(
        r"[^。；\n]*(?:建议投入|预计用时|建议时长|阶段时长|推荐总投入|学习课次|"
        r"按天日历|按周日历|每天|每周|跨\s*\d+(?:\s*[～~\-–—至]\s*\d+)?\s*周)"
        r"[^。；\n]*[。；]?",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"\s+", " ", value)
    return value.strip(" \t\r\n，。；、|")


def _latexify_inline_symbols(value: str) -> str:
    """Wrap common circuit variables in LaTeX without touching existing formulas."""

    formulas: list[str] = []

    def protect(match: re.Match[str]) -> str:
        formulas.append(match.group(0))
        return f"ZXQFORMULA{len(formulas) - 1}QXZ"

    value = _LATEX_SPAN_RE.sub(protect, _normalize_latex(value))
    greek = {"β": r"\beta", "π": r"\pi", "θ": r"\theta", "λ": r"\lambda", "Δ": r"\Delta"}
    value = re.sub(
        r"(?<![A-Za-z0-9\\])([A-Za-z])_([βπθλΔ])(?![A-Za-z0-9])",
        lambda match: f"${match.group(1)}_{greek[match.group(2)]}$",
        value,
    )
    value = re.sub(
        r"(?<![A-Za-z0-9\\])([A-Za-z])_([A-Za-z0-9]+)(?![A-Za-z0-9])",
        lambda match: (
            f"${match.group(1)}_{match.group(2)}$"
            if len(match.group(2)) == 1
            else f"${match.group(1)}_{{{match.group(2)}}}$"
        ),
        value,
    )
    value = re.sub(
        r"(?<![A-Za-z0-9\\])([A-Za-z])_\{([^{}]+)\}(?![A-Za-z0-9])",
        r"$\1_{\2}$",
        value,
    )
    value = re.sub(r"(?<![A-Za-z0-9])Q(?=\s*点)", r"$Q$", value)
    for symbol, latex in greek.items():
        value = value.replace(symbol, f"${latex}$")
    for index, formula in enumerate(formulas):
        value = value.replace(f"ZXQFORMULA{index}QXZ", formula)
    return value


def _full_plan_copy(value: str) -> str:
    value = _remove_time_arrangements(_plain(value, preserve_latex=True))
    value = re.sub(r"\s+([，。；：！？])", r"\1", value)
    return _latexify_inline_symbols(value.strip(" \t\r\n|"))


def _reference_copy(value: str) -> str:
    """Hide OCR measurements that were stored as legacy section titles."""

    cleaned = _full_plan_copy(value)
    parts = [part.strip() for part in re.split(r"\s*[·|]\s*", cleaned) if part.strip()]
    return " · ".join(part for part in parts if not is_measurement_section(part))


def _full_standard_copy(value: str) -> str:
    cleaned = _full_plan_copy(value)
    if "共射电压放大能力" in cleaned and "共集输入电阻高" in cleaned:
        return (
            r"能分别写出共射电压增益 $|A_v|\approx\beta R'_L/r_{be}$ 与共集输入电阻 "
            r"$R_i\approx r_\pi+(1+\beta)R_E$，并说明各参数的作用"
        )
    if "知识图谱覆盖全部" in cleaned and "个知识点" in cleaned:
        return re.sub(r"覆盖全部\s*\d+\s*个知识点", "覆盖全部学习模块", cleaned)
    return cleaned


def _compact_action(value: str) -> str:
    """Turn verbose generated instructions into complete, slide-sized actions."""

    value = re.sub(r"\[资料\s*\d+\]", "", value)
    value = re.sub(r"[（(](?:如|例如|比如|见下表)[^）)]*[）)]", "", value)
    value = re.sub(r"(?:例如|比如)[:：][^；。]*", "", value)
    value = re.sub(r"\s+", " ", _plain(value, preserve_latex=False)).strip()
    rewrite_rules = (
        (
            r"^对照错题本逐条标注",
            "逐条标注错题对应的前置知识缺口，并记录需要回顾的模型或概念",
        ),
        (
            r"^精读",
            "精读反馈概念与分类资料，绘制反馈引入条件思维导图",
        ),
        (
            r"^完成\s*2\s*个基础题",
            "完成直流/交流通路绘制与反馈类型判断各 1 题",
        ),
        (
            r"^梳理.*三元矩阵",
            "建立“反馈类型—性能改善—电路实现”三元矩阵，并用典型电路验证",
        ),
        (
            r"^对比三种组态",
            "用表格对比三种组态的阻抗、增益、相位、高频特性和典型应用",
        ),
        (
            r"^完成\s*3\s*类典型题",
            "练习反馈类型判断、旁路电容影响分析和局部反馈设计",
        ),
        (
            r"^精做\s*3\s*类题型",
            "精做反馈极性、组态对比和旁路电容变化三类题",
        ),
        (
            r"^自主设计\s*2\s*个反例验证",
            "设计正反馈自激与含电容反馈两个反例并完成验证",
        ),
        (
            r"^进行“?综合电路分析”?模拟测试",
            "完成含多级反馈与旁路电容的综合电路分析测试",
        ),
        (
            r"^写一份“?反馈机制知识图谱”?",
            "绘制反馈机制知识图谱，覆盖反馈类型、性能影响、放大组态和关键公式",
        ),
        (
            r"^回顾错题本，撰写\s*300\s*字反思报告",
            "撰写 300 字错题反思，归纳常见混淆与规避方法",
        ),
    )
    for pattern, replacement in rewrite_rules:
        if re.search(pattern, value):
            return replacement

    value = re.sub(r"[:：]\s*[①1][^。；]*", "", value)
    value = value.rstrip("；。")
    return _editorial_clip(value, 64, preserve_latex=False)


def _compact_standard(value: str) -> str:
    """Keep acceptance criteria measurable and complete on a slide."""

    value = _plain(value, preserve_latex=False)
    if "直流反馈用于稳定" in value and "交流反馈改善性能" in value:
        return "能区分直流反馈稳定 Q 点与交流反馈改善性能"
    if "在任意电路图中" in value and "反馈网络" in value:
        return "能在电路图中标出反馈网络、输出取样点和输入求和方式"
    if "瞬时极性法误判" in value:
        return "能解释瞬时极性法误判的原因"
    if "深度负反馈" in value and "闭环增益" in value:
        return "能说明闭环增益近似式及其适用条件"
    if "共基输入电阻小" in value:
        return "能解释共基低输入电阻和良好高频特性的原因"
    if "输出量影响输入量" in value:
        return "能把“输出量影响输入量”转化为输入端求和关系"
    if "10 分钟内" in value and "反馈类型判定" in value:
        return "10分钟内完成反馈类型、取样与求和方式判定"
    if "共射电压增益" in value and "共集输入电阻高" in value:
        return "能说明共射电压增益与共集高输入电阻的定量依据"
    if "错题本中 90%" in value:
        return "同类错题重做正确率 ≥95%，且连续 3 次答对"
    if "综合题平均得分" in value:
        return "综合题得分 ≥85%，并能解释反馈类型判断依据"
    if "知识图谱覆盖全部" in value:
        return "知识图谱覆盖全部学习模块，且关键关系与公式书写规范"
    if "概念混淆" in value and "正确率" in value:
        return "概念混淆类错题正确率 100%"
    value = re.sub(r"[（(](?:如|例如|比如)[^）)]*[）)]", "", value)
    value = value.rstrip("；。")
    return _editorial_clip(value, 52, preserve_latex=False)


def _dedupe(values: list[str], limit: int = 99) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = _plain(value)
        key = re.sub(r"[\s，。；、,.]", "", _plain(cleaned, preserve_latex=False)).lower()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
        if len(result) >= limit:
            break
    return result


def _split_sections(markdown: str) -> tuple[str, list[PlanSection]]:
    matches = list(_HEADING_RE.finditer(markdown))
    if not matches:
        return "", [PlanSection(title="学习方案", body=markdown, level=2)]
    preamble = markdown[: matches[0].start()].strip()
    sections: list[PlanSection] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        sections.append(
            PlanSection(
                title=_plain(match.group(2)),
                body=markdown[match.end() : end].strip(),
                level=len(match.group(1)),
            )
        )
    return preamble, sections


def _content_items(value: str, limit: int = 12) -> list[str]:
    items: list[str] = []
    raw_lines = value.splitlines()
    for line_index, raw_line in enumerate(raw_lines):
        line = raw_line.strip()
        if not line or line.startswith("#") or re.fullmatch(r"[-*_]{3,}", line):
            continue
        if line.startswith("|") and line.endswith("|"):
            next_line = raw_lines[line_index + 1].strip() if line_index + 1 < len(raw_lines) else ""
            if next_line.startswith("|") and re.fullmatch(
                r"\|?\s*:?-{2,}:?(?:\s*\|\s*:?-{2,}:?)+\s*\|?",
                next_line,
            ):
                continue
            cells = [_plain(cell) for cell in line.strip("|").split("|")]
            if cells and not all(re.fullmatch(r":?-{2,}:?", cell or "--") for cell in cells):
                line = " · ".join(cell for cell in cells if cell)
            else:
                continue
        cleaned = _plain(line)
        if cleaned:
            items.append(cleaned)
    if len(items) <= 1:
        prose = _plain(value)
        items.extend(
            part.strip()
            for part in re.split(r"(?<=[。；！？])", prose)
            if part.strip()
        )
    return _dedupe(items, limit)


def _strip_label(value: str) -> tuple[str, str]:
    match = re.match(
        r"^(目标|阶段目标|学习目标|核心内容|学习内容|知识要点|重点难点|"
        r"行动|具体行动|学习任务|任务|练习|巩固练习|练习与复盘|复盘任务|"
        r"完成标准|验收标准|完成标志|资料依据|依据|参考资料|"
        r"建议投入|建议时长|预计用时|阶段时长|时间安排|时间)\s*[:：]?\s*(.*)$",
        value,
    )
    if not match:
        return "", value
    return match.group(1), match.group(2).strip()


def _stage_details(section: PlanSection) -> PlanStage:
    buckets: dict[str, list[str]] = {
        "goal": [],
        "concepts": [],
        "actions": [],
        "practices": [],
        "standards": [],
        "sources": [],
    }
    current = ""
    general: list[str] = []
    for item in _content_items(section.body, 99):
        label, payload = _strip_label(item)
        if label:
            if "目标" in label:
                current = "goal"
            elif label in {"核心内容", "学习内容", "知识要点", "重点难点"}:
                current = "concepts"
            elif label in {"练习", "巩固练习", "练习与复盘", "复盘任务"}:
                current = "practices"
            elif label in {
                "行动",
                "具体行动",
                "学习任务",
                "任务",
            }:
                current = "actions"
            elif "标准" in label or "标志" in label:
                current = "standards"
            elif "依据" in label or "资料" in label:
                current = "sources"
            elif (
                "投入" in label
                or "时长" in label
                or "用时" in label
                or "时间" in label
            ):
                current = ""
            if payload and current:
                buckets[current].append(payload)
            continue
        if re.search(r"\[资料\s*\d+\]", item) or item.startswith("资料"):
            if current:
                buckets[current].append(item)
            else:
                buckets["sources"].append(item)
        elif current:
            buckets[current].append(item)
        else:
            general.append(item)

    title = re.sub(r"^[一二三四五六七八九十\d]+[、.．]\s*", "", section.title)
    title = re.sub(
        r"^(?:阶段\s*[一二三四五六七八九十\d]+|模块\s*[一二三四五六七八九十\d]+|"
        r"第\s*[一二三四五六七八九十\d]+\s*(?:阶段|模块))\s*[:：·|-]?\s*",
        "",
        title,
    )
    title = _DURATION_RE.sub("", title)
    title = re.sub(r"[（(]\s*[,，、;/]*\s*[）)]", "", title)
    title = title.strip(" ：:·|-—（）()，,")
    goal_candidates = buckets["goal"] or general[:1]
    actions = buckets["actions"] or general[1:] or general
    practices = buckets["practices"]
    if not practices:
        retained_actions: list[str] = []
        for action in actions:
            if re.match(
                r"^(?:完成.*(?:题|测试)|精做|练习|进行.*测试|自主设计.*反例|"
                r"重做|回顾.*(?:反思|复盘))",
                _plain(action, preserve_latex=False),
            ):
                practices.append(action)
            else:
                retained_actions.append(action)
        actions = retained_actions
    concepts = buckets["concepts"]
    standards = buckets["standards"]
    if not standards:
        standards = [item for item in general if re.search(r"完成|正确率|能够|独立|通过|达到", item)]
    goal = _full_plan_copy(goal_candidates[0] if goal_candidates else title)
    concepts = [_full_plan_copy(item) for item in _dedupe(concepts, 99)]
    actions = [_full_plan_copy(item) for item in _dedupe(actions, 99)]
    practices = [_full_plan_copy(item) for item in _dedupe(practices, 99)]
    standards = [_full_standard_copy(item) for item in _dedupe(standards, 99)]
    sources = [_reference_copy(item) for item in _dedupe(buckets["sources"], 99)]
    return PlanStage(
        title=_full_plan_copy(title) or "学习任务",
        goal=goal,
        concepts=[item for item in concepts if item],
        actions=[item for item in actions if item],
        practices=[item for item in practices if item],
        standards=[item for item in standards if item],
        sources=[item for item in sources if item],
    )


def _generic_title(value: str) -> bool:
    compact = re.sub(r"[\s：:·|—-]", "", value)
    return not compact or compact in {
        "学习规划",
        "学习计划",
        "学习规划方案",
        "个性化学习规划",
        "学习路线",
        "学习方案",
    }


def _derive_title(sections: list[PlanSection], goal: str, topic: str) -> str:
    for section in sections[:3]:
        if _STAGE_RE.search(section.title) or re.search(r"日程|安排|验收|指标", section.title):
            continue
        if section.level <= 2 and not _generic_title(section.title):
            title = re.sub(r"^(学习规划|学习计划|学习路线)\s*[:：·|-]\s*", "", section.title)
            if title:
                return _editorial_clip(title, 22)
    topic_text = _plain(topic)
    weak_points = re.search(r"薄弱知识点\s*[:：]\s*([^\n]+)", topic_text)
    if weak_points:
        points = [point.strip() for point in re.split(r"[、,，]", weak_points.group(1)) if point.strip()]
        modules: list[str] = []
        module_patterns = (
            ("旁路电容", ("旁路电容", "发射极电阻", "交流短路", "直流开路")),
            ("三种基本组态", ("共射", "共集", "共基", "三种接法", "输入电阻", "输出电阻", "相位关系")),
            ("反馈机制", ("反馈", "静态工作点", "输出量影响输入量")),
        )
        for point in points:
            module = next(
                (
                    label
                    for label, patterns in module_patterns
                    if any(pattern in point for pattern in patterns)
                ),
                "",
            )
            if module and module not in modules:
                modules.append(module)
        if modules:
            return "、".join(modules[:3])
    if topic_text:
        topic_text = re.sub(r"^(请|帮我|请帮我|我想|我需要)", "", topic_text)
        topic_text = re.sub(r"(制定|生成|做)(一份|一个)?(详细的)?(学习规划|学习计划).*$", "", topic_text)
        if topic_text and not re.search(r"错题本|薄弱知识点|错题摘要", topic_text):
            return _editorial_clip(topic_text, 22)
    return _editorial_clip(goal, 22) or "个性化学习规划"


def parse_learning_plan(markdown: str, topic: str = "") -> ParsedLearningPlan:
    preamble, sections = _split_sections(markdown)
    reference_sections = [
        section
        for section in sections
        if re.search(r"检索依据|参考文献|引用资料|sources?", section.title, re.IGNORECASE)
    ]
    references = [
        _reference_copy(item)
        for section in reference_sections
        for item in _content_items(section.body, 99)
    ]
    references = [item for item in _dedupe(references, 99) if item]
    relevant = [
        section
        for section in sections
        if not re.search(
            r"检索依据|参考文献|引用资料|sources?|日程|时间表|每日|每周|"
            r"\d+\s*天|Day\s*\d+",
            section.title,
            re.IGNORECASE,
        )
    ]
    stage_sections = [section for section in relevant if _STAGE_RE.search(section.title)]
    if not stage_sections:
        stage_sections = [
            section
            for section in relevant
            if re.search(r"目标|行动|任务|完成标准|学习内容|练习", section.body)
            and not re.search(r"总体|验收|指标|日程|安排|时间表", section.title)
        ]
    if not stage_sections:
        stage_sections = [section for section in relevant if section.body][:5]

    stages = [_stage_details(section) for section in stage_sections]
    stages = [
        stage
        for stage in stages
        if (stage.title or stage.actions)
        and not _NON_LEARNING_STAGE_RE.search(stage.title)
    ]
    if not stages:
        fallback = _content_items(markdown, 8)
        stages = [
            PlanStage(
                title="核心学习任务",
                goal=_full_plan_copy(fallback[0] if fallback else "完成本次学习目标"),
                actions=[_full_plan_copy(item) for item in fallback[1:]],
                standards=["能独立复述核心概念，并完成一次自测与纠错"],
            )
        ]

    goal_candidates: list[str] = []
    searchable = "\n".join([preamble] + [section.body for section in relevant[:3]])
    goal_match = re.search(
        r"(?:总体目标|学习目标|规划目标|目标)\s*[:：]\s*([^\n]{4,180})",
        searchable,
    )
    if goal_match:
        goal_candidates.append(goal_match.group(1))
    goal_candidates.extend(stage.goal for stage in stages[:1])
    goal_candidates.extend(_content_items(preamble, 2))
    goal = _full_plan_copy(
        next((item for item in goal_candidates if _plain(item)), "完成本次学习目标")
    )

    metric_sections = [
        section
        for section in relevant
        if re.search(r"验收|指标|完成标准|自测|复盘|检查", section.title)
        and section not in stage_sections
    ]
    metrics = [item for section in metric_sections for item in _content_items(section.body, 99)]
    metrics = [
        item
        for item in metrics
        if not re.fullmatch(r"指标项\s*·\s*具体内容\s*·\s*达标阈值", item)
        and not re.search(
            r"本规划严格遵循|不生成日历|schedule_guidance|plan_guidance|"
            r"建议投入|总投入|按天|按周|课次",
            item,
            re.I,
        )
    ]
    if not metrics:
        metrics.extend(item for stage in stages for item in stage.standards)
    normalized_metrics: list[str] = []
    for item in _dedupe(metrics, 99):
        if item.startswith("知识覆盖度"):
            item = "模块覆盖度 · 每个薄弱模块均有学习任务与对应自测 · 100%覆盖"
        elif item.startswith("公式应用准确性"):
            item = "公式应用准确性 · 能匹配公式、适用条件与电路场景 · 正确率≥95%"
        normalized = _full_plan_copy(item)
        if normalized:
            normalized_metrics.append(normalized)
    metrics = normalized_metrics
    if not metrics:
        metrics = [
            "能不看资料复述核心概念与关键关系",
            "完成代表性练习并解释每一步依据",
            "记录错因，完成一次针对性复练",
            "用自测结果决定是否进入下一阶段",
        ]

    overview_sections = [
        section
        for section in relevant
        if re.search(r"评估|诊断|原则|说明|节奏|现状", section.title)
        and section not in stage_sections
    ]
    principles = [item for section in overview_sections for item in _content_items(section.body, 99)]
    normalized_principles: list[str] = []
    for item in _dedupe(principles, 99):
        if item.startswith("窄范围合并"):
            item = "把关联知识放在同一模块中理解"
        elif item.startswith("系统范围拆分"):
            item = "按知识依赖关系分阶段推进"
        elif item.startswith("所有阶段严格遵循"):
            item = "达到本模块掌握要求后再进入下一模块"
        normalized = _full_plan_copy(item)
        if normalized:
            normalized_principles.append(normalized)
    principles = normalized_principles
    default_principles = [
        "难点用例题与错题双向验证",
        "根据自测结果动态调整学习强度",
        "通过练习、纠错与巩固加深理解",
    ]
    principles = _dedupe(principles, 99) if principles else default_principles

    summary_candidates = _content_items(preamble, 2)
    if not summary_candidates:
        for section in relevant:
            if section not in stage_sections:
                summary_candidates.extend(_content_items(section.body, 1))
                if summary_candidates:
                    break
    title = _derive_title(sections, goal, topic)
    if not title.endswith(("规划", "计划", "路线")):
        title = f"{title} · 学习规划"
    if "错题本" in topic:
        title_core = re.sub(r"\s*·\s*学习规划$", "", title)
        goal = f"围绕{title_core}补全知识链，通过针对性学习和巩固练习消除高频错因"
        summary = f"聚焦{title_core}，按“补前置—建联系—做练习—达要求”的顺序推进"
    else:
        summary = _full_plan_copy(summary_candidates[0] if summary_candidates else goal)

    return ParsedLearningPlan(
        title=title,
        goal=goal,
        summary=summary,
        stages=stages,
        metrics=metrics,
        principles=principles,
        references=references,
    )


def _set_background(slide, color: str) -> None:
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = _rgb(color)


def _shape(
    slide,
    shape_type,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    fill: str,
    line: str | None = None,
    line_width: float = 1,
    radius_adjust: float | None = None,
):
    shape = slide.shapes.add_shape(shape_type, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = _rgb(fill)
    shape.line.color.rgb = _rgb(line or fill)
    shape.line.width = Pt(line_width)
    if radius_adjust is not None and shape.adjustments:
        shape.adjustments[0] = radius_adjust
    return shape


def _line(
    slide,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    color: str = LINE,
    width: float = 1.4,
    dash: bool = False,
):
    line = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT,
        Inches(x1),
        Inches(y1),
        Inches(x2),
        Inches(y2),
    )
    line.line.color.rgb = _rgb(color)
    line.line.width = Pt(width)
    if dash:
        line.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    return line


def _text(
    slide,
    value: str,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    size: float = 18,
    color: str = INK,
    bold: bool = False,
    font: str = FONT_CN,
    align=PP_ALIGN.LEFT,
    valign=MSO_ANCHOR.TOP,
    margin: float = 0,
    line_spacing: float = 1.08,
):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.clear()
    frame.margin_left = Inches(margin)
    frame.margin_right = Inches(margin)
    frame.margin_top = Inches(margin)
    frame.margin_bottom = Inches(margin)
    frame.vertical_anchor = valign
    frame.word_wrap = True
    paragraph = frame.paragraphs[0]
    paragraph.alignment = align
    paragraph.line_spacing = line_spacing
    run = paragraph.add_run()
    run.text = value
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = _rgb(color)
    return box


def _picture_alt_text(picture, value: str) -> None:
    try:
        properties = picture._element.nvPicPr.cNvPr
        properties.set("name", "LaTeX formula")
        properties.set("descr", f"LaTeX rendered formula: {_plain(value, preserve_latex=False)}")
    except (AttributeError, TypeError):
        logger.debug("Unable to attach alternative text to rendered formula", exc_info=True)


def _display_text(
    slide,
    value: str,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    renderer: LatexTextRenderer | None,
    size: float = 18,
    color: str = INK,
    bold: bool = False,
    font: str = FONT_CN,
    align=PP_ALIGN.LEFT,
    valign=MSO_ANCHOR.TOP,
    margin: float = 0,
    line_spacing: float = 1.08,
):
    if renderer is not None and _has_latex(value):
        try:
            image_path = renderer.render_text(
                value,
                w,
                h,
                size=size,
                color=color,
                bold=bold,
                align=align,
                valign=valign,
                line_spacing=line_spacing,
            )
            picture = slide.shapes.add_picture(
                str(image_path),
                Inches(x),
                Inches(y),
                width=Inches(w),
                height=Inches(h),
            )
            _picture_alt_text(picture, value)
            return picture
        except Exception:
            logger.warning("LaTeX rendering failed; using readable text fallback", exc_info=True)
            value = _plain(value, preserve_latex=False)
    return _text(
        slide,
        value,
        x,
        y,
        w,
        h,
        size=size,
        color=color,
        bold=bold,
        font=font,
        align=align,
        valign=valign,
        margin=margin,
        line_spacing=line_spacing,
    )


def _rich_lines(
    slide,
    lines: list[tuple[str, str]],
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    renderer: LatexTextRenderer | None = None,
    size: float = 17,
    color: str = INK_SOFT,
    bullet_color: str = TEAL,
    gap: float = 8,
):
    if renderer is not None and any(_has_latex(value) for _, value in lines):
        try:
            image_path = renderer.render_lines(
                lines,
                w,
                h,
                size=size,
                color=color,
                marker_color=bullet_color,
            )
            picture = slide.shapes.add_picture(
                str(image_path),
                Inches(x),
                Inches(y),
                width=Inches(w),
                height=Inches(h),
            )
            _picture_alt_text(picture, " / ".join(value for _, value in lines))
            return picture
        except Exception:
            logger.warning("LaTeX list rendering failed; using readable text fallback", exc_info=True)
            lines = [(marker, _plain(value, preserve_latex=False)) for marker, value in lines]
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = frame.margin_right = Inches(0)
    frame.margin_top = frame.margin_bottom = Inches(0)
    for index, (marker, value) in enumerate(lines):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.space_after = Pt(gap)
        paragraph.line_spacing = 1.12
        marker_run = paragraph.add_run()
        marker_run.text = f"{marker}  "
        marker_run.font.name = FONT_LATIN
        marker_run.font.size = Pt(size)
        marker_run.font.bold = True
        marker_run.font.color.rgb = _rgb(bullet_color)
        text_run = paragraph.add_run()
        text_run.text = value
        text_run.font.name = FONT_CN
        text_run.font.size = Pt(size)
        text_run.font.color.rgb = _rgb(color)
    return box


def _brand(slide, *, dark: bool = False) -> None:
    color = "A9CFC8" if dark else TEAL
    _shape(slide, MSO_SHAPE.OVAL, 0.72, 0.46, 0.16, 0.16, fill=CORAL, line=CORAL)
    _text(
        slide,
        "CIRCUITMIND  /  LEARNING LAB",
        0.98,
        0.42,
        3.5,
        0.25,
        size=9,
        color=color,
        bold=True,
        font=FONT_LATIN,
    )


def _footer(slide, page_number: int, *, dark: bool = False) -> None:
    color = "8DB5AF" if dark else "8CA09D"
    _text(slide, "本次学习规划 · 内容与图形均可编辑", 0.72, 7.04, 5.2, 0.22, size=9, color=color)
    _text(
        slide,
        f"{page_number:02d}",
        12.0,
        7.02,
        0.58,
        0.24,
        size=9,
        color=color,
        bold=True,
        font=FONT_LATIN,
        align=PP_ALIGN.RIGHT,
    )


def _slide_title(slide, kicker: str, title: str, page_number: int, *, dark: bool = False) -> None:
    _brand(slide, dark=dark)
    _text(slide, kicker.upper(), 0.72, 0.98, 3.2, 0.24, size=10, color=CORAL, bold=True, font=FONT_LATIN)
    _text(slide, title, 0.72, 1.23, 11.7, 0.72, size=35, color=WHITE if dark else INK, bold=True)
    _footer(slide, page_number, dark=dark)


def _list_rows(
    slide,
    values: list[str],
    x: float,
    y: float,
    w: float,
    *,
    row_height: float,
    size: float,
    color: str = INK_SOFT,
    accent: str = TEAL,
) -> None:
    for index, value in enumerate(values[:3]):
        row_y = y + index * row_height
        marker_size = 0.34
        _shape(
            slide,
            MSO_SHAPE.ROUNDED_RECTANGLE,
            x,
            row_y + 0.06,
            marker_size,
            marker_size,
            fill=accent if index == 0 else MINT,
            line=accent if index == 0 else MINT,
        )
        _text(
            slide,
            f"{index + 1:02d}",
            x,
            row_y + 0.06,
            marker_size,
            marker_size,
            size=9,
            color=WHITE if index == 0 else TEAL_DARK,
            bold=True,
            font=FONT_LATIN,
            align=PP_ALIGN.CENTER,
            valign=MSO_ANCHOR.MIDDLE,
        )
        _text(
            slide,
            _plain(value, preserve_latex=False),
            x + 0.52,
            row_y,
            w - 0.52,
            row_height - 0.04,
            size=size,
            color=color,
            line_spacing=1.04,
        )


def _compact_source(value: str) -> str:
    matches = re.findall(
        r"\[(资料\s*\d+)\][^；、/]{0,36}?第?\s*(\d+)\s*页",
        _plain(value, preserve_latex=False),
    )
    if matches:
        return "；".join(f"[{label.replace(' ', '')}] 第 {page} 页" for label, page in matches)
    return _editorial_clip(value, 100, preserve_latex=False)


def _metric_parts(value: str) -> tuple[str, str, str]:
    parts = [part.strip() for part in re.split(r"\s*·\s*", _plain(value, preserve_latex=False)) if part.strip()]
    if len(parts) >= 3:
        return parts[0], " · ".join(parts[1:-1]), parts[-1]
    if len(parts) == 2:
        return parts[0], parts[1], "完成后自检"
    return parts[0] if parts else "学习结果", "对照阶段完成标准检查", "达到要求"


def _title_slide(prs: Presentation, plan: ParsedLearningPlan, renderer: LatexTextRenderer) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_background(slide, TEAL_DARK)
    _shape(slide, MSO_SHAPE.RECTANGLE, 0, 0, 0.18, 7.5, fill=CORAL, line=CORAL)
    _brand(slide, dark=True)
    _text(slide, "PERSONAL LEARNING BLUEPRINT", 0.78, 1.18, 5.0, 0.28, size=11, color="8CCDC3", bold=True, font=FONT_LATIN)
    title_display = plan.title
    if title_display.count("、") >= 2:
        title_display = "\n".join(title_display.rsplit("、", 1))
    _text(slide, title_display, 0.75, 1.68, 7.65, 1.78, size=50, color=WHITE, bold=True, line_spacing=0.94)
    _display_text(
        slide,
        plan.goal,
        0.78,
        3.78,
        7.25,
        1.18,
        renderer=renderer,
        size=19,
        color="D4EAE6",
        line_spacing=1.12,
    )
    _line(slide, 0.78, 5.42, 7.55, 5.42, color="347A73", width=1.2)
    _text(
        slide,
        f"{len(plan.stages)} 个学习模块  ·  按目录逐项完成",
        0.78,
        5.7,
        5.4,
        0.34,
        size=13,
        color="A9CFC8",
        bold=True,
    )
    _text(slide, date.today().strftime("%Y.%m.%d"), 0.8, 6.76, 1.5, 0.24, size=9, color="86AAA5", bold=True, font=FONT_LATIN)

    _text(slide, "LEARNING PATH", 9.18, 1.28, 2.6, 0.26, size=11, color=CORAL, bold=True, font=FONT_LATIN)
    _line(slide, 9.43, 1.9, 9.43, 5.95, color="3C817A", width=1.4)
    stages = plan.stages[:5]
    step_gap = 3.55 / max(len(stages), 1)
    for index, stage in enumerate(stages):
        y = 1.92 + index * step_gap
        _shape(slide, MSO_SHAPE.OVAL, 9.27, y, 0.32, 0.32, fill=CORAL if index == 0 else MINT, line=TEAL_DARK, line_width=2)
        _text(slide, f"{index + 1:02d}", 9.84, y - 0.01, 0.48, 0.24, size=10, color="8CCDC3", bold=True, font=FONT_LATIN)
        _text(slide, stage.title, 10.42, y - 0.06, 2.18, 0.54, size=15, color=WHITE, bold=True)


def _overview_slide(
    prs: Presentation,
    plan: ParsedLearningPlan,
    page: int,
    renderer: LatexTextRenderer,
) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_background(slide, CREAM)
    _slide_title(slide, "LEARNING GOAL", "学习目标与方法", page)
    _text(slide, "学习目标", 0.78, 2.22, 1.4, 0.28, size=12, color=CORAL, bold=True)
    _display_text(
        slide,
        plan.goal,
        0.78,
        2.7,
        6.65,
        1.72,
        renderer=renderer,
        size=27,
        color=INK,
        bold=True,
        line_spacing=1.02,
    )
    _line(slide, 0.78, 4.72, 7.3, 4.72, color=LINE, width=1.3)
    _text(slide, "学习范围", 0.78, 5.08, 1.1, 0.26, size=12, color=TEAL, bold=True)
    title_core = re.sub(r"\s*·\s*(?:学习规划|学习计划|学习路线)$", "", plan.title)
    modules = [item.strip() for item in title_core.split("、") if item.strip()][:4]
    cursor_x = 0.78
    for module in modules:
        width = min(2.2, 0.62 + len(module) * 0.23)
        _shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, cursor_x, 5.58, width, 0.52, fill=MINT, line=MINT)
        _text(slide, module, cursor_x, 5.58, width, 0.52, size=13, color=TEAL_DARK, bold=True, align=PP_ALIGN.CENTER, valign=MSO_ANCHOR.MIDDLE)
        cursor_x += width + 0.18

    _line(slide, 8.02, 2.15, 8.02, 6.38, color="C7DCD7", width=1.4)
    _text(slide, "学习方法", 8.48, 2.18, 1.6, 0.3, size=14, color=TEAL, bold=True)
    principles = plan.principles[:3]
    for index, principle in enumerate(principles):
        y = 2.92 + index * 1.12
        _text(slide, f"0{index + 1}", 8.48, y, 0.56, 0.34, size=17, color=CORAL, bold=True, font=FONT_LATIN)
        _text(slide, principle, 9.28, y - 0.02, 3.18, 0.7, size=17, color=INK_SOFT, bold=True, line_spacing=1.05)


def _roadmap_slide(
    prs: Presentation,
    plan: ParsedLearningPlan,
    page: int,
    renderer: LatexTextRenderer,
) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_background(slide, MINT_LIGHT)
    _slide_title(slide, "CONTENTS", "学习目录", page)
    items = ["学习目标与方法", *(stage.title for stage in plan.stages)]
    rows = 1 if len(items) <= 4 else 2
    per_row = min(4, len(items)) if rows == 1 else (len(items) + 1) // 2
    for row in range(rows):
        row_items = items[row * per_row : (row + 1) * per_row]
        if not row_items:
            continue
        gap = 0.35
        available = 11.72
        width = (available - gap * (len(row_items) - 1)) / len(row_items)
        total = width * len(row_items) + gap * (len(row_items) - 1)
        start_x = (13.333 - total) / 2
        y = 3.0 if rows == 1 else 2.55 + row * 2.25
        _line(slide, start_x + 0.32, y + 0.31, start_x + total - 0.32, y + 0.31, color="A8D1CA", width=2.2)
        for index, item in enumerate(row_items):
            global_index = row * per_row + index
            x = start_x + index * (width + gap)
            _shape(slide, MSO_SHAPE.OVAL, x + width / 2 - 0.32, y, 0.64, 0.64, fill=CORAL if global_index == 0 else TEAL, line=MINT_LIGHT, line_width=3)
            _text(slide, str(global_index + 1), x + width / 2 - 0.32, y, 0.64, 0.64, size=15, color=WHITE, bold=True, font=FONT_LATIN, align=PP_ALIGN.CENTER, valign=MSO_ANCHOR.MIDDLE)
            _text(slide, item, x + 0.08, y + 0.94, width - 0.16, 1.0, size=20, color=INK, bold=True, align=PP_ALIGN.CENTER, line_spacing=1.0)
    _text(slide, "正文按目录顺序展开，每个模块都包含目标、内容、行动、练习与掌握要求", 2.55, 6.24, 8.23, 0.36, size=13, color=TEAL_DARK, bold=True, align=PP_ALIGN.CENTER)


def _split_complete_text(value: str, max_chars: int = 112) -> list[str]:
    """Split long copy without dropping characters or adding ellipses."""

    value = _full_plan_copy(value)
    if not value:
        return []
    fragments = [
        fragment
        for fragment in re.split(r"(?<=[。；！？])", value)
        if fragment
    ]
    chunks: list[str] = []
    current = ""
    for fragment in fragments:
        if len(current) + len(fragment) <= max_chars:
            current += fragment
            continue
        if current:
            chunks.append(current.strip())
            current = ""
        while len(fragment) > max_chars:
            boundary = max(
                fragment.rfind(marker, 0, max_chars)
                for marker in ("；", "，", "、", "：")
            )
            cut = boundary + 1 if boundary >= max_chars // 2 else max_chars
            chunks.append(fragment[:cut].strip())
            fragment = fragment[cut:].strip()
        current = fragment
    if current:
        chunks.append(current.strip())
    return [chunk for chunk in chunks if chunk]


def _entry_units(value: str) -> int:
    return max(1, math.ceil(len(_plain(value, preserve_latex=False)) / 52))


def _expanded_entries(entries: list[tuple[str, str]]) -> list[tuple[str, str, int]]:
    expanded: list[tuple[str, str, int]] = []
    for label, value in entries:
        parts = _split_complete_text(value)
        for index, part in enumerate(parts):
            part_label = label if index == 0 else f"{label}（续）"
            expanded.append((part_label, part, _entry_units(part)))
    return expanded


def _paginate_entries(
    entries: list[tuple[str, str]],
    *,
    capacity: int,
) -> list[list[tuple[str, str, int]]]:
    pages: list[list[tuple[str, str, int]]] = []
    current: list[tuple[str, str, int]] = []
    used = 0
    for entry in _expanded_entries(entries):
        cost = entry[2] + 1
        if current and used + cost > capacity:
            pages.append(current)
            current = []
            used = 0
        current.append(entry)
        used += cost
    if current:
        pages.append(current)
    return pages or [[("学习任务", "完成本阶段核心学习任务并记录结果", 1)]]


def _detail_rows(
    slide,
    entries: list[tuple[str, str, int]],
    *,
    renderer: LatexTextRenderer | None = None,
    y: float = 2.28,
    height: float = 4.22,
) -> None:
    gaps = 0.1 * max(len(entries) - 1, 0)
    total_units = sum(units + 0.45 for _, _, units in entries)
    unit_height = (height - gaps) / max(total_units, 1)
    cursor = y
    for index, (label, value, units) in enumerate(entries):
        row_height = unit_height * (units + 0.45)
        accent = CORAL if label == "阶段目标" else TEAL
        _shape(
            slide,
            MSO_SHAPE.RECTANGLE,
            0.78,
            cursor + 0.06,
            0.06,
            max(0.34, row_height - 0.12),
            fill=accent,
            line=accent,
        )
        _text(
            slide,
            label,
            1.02,
            cursor + 0.05,
            1.22,
            row_height - 0.08,
            size=13,
            color=accent,
            bold=True,
            valign=MSO_ANCHOR.MIDDLE,
        )
        _display_text(
            slide,
            value,
            2.35,
            cursor + 0.02,
            10.12,
            row_height - 0.04,
            renderer=renderer,
            size=16.5,
            color=INK if label == "阶段目标" else INK_SOFT,
            bold=label == "阶段目标",
            valign=MSO_ANCHOR.MIDDLE,
            line_spacing=1.05,
        )
        cursor += row_height
        if index < len(entries) - 1:
            _line(slide, 1.02, cursor + 0.04, 12.48, cursor + 0.04, color=LINE, width=0.8)
            cursor += 0.1


def _stage_detail_slide(
    prs: Presentation,
    stage: PlanStage,
    stage_index: int,
    stage_total: int,
    page: int,
    entries: list[tuple[str, str, int]],
    *,
    section: str,
    part_index: int,
    part_total: int,
    renderer: LatexTextRenderer,
) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_background(slide, CREAM if stage_index % 2 else MINT_LIGHT)
    _brand(slide)
    _footer(slide, page)
    _text(
        slide,
        f"STAGE {stage_index:02d} / {stage_total:02d}",
        0.78,
        0.99,
        2.2,
        0.26,
        size=10,
        color=CORAL,
        bold=True,
        font=FONT_LATIN,
    )
    _text(slide, stage.title, 0.75, 1.28, 8.9, 0.72, size=35, color=INK, bold=True)
    part = f"  {part_index}/{part_total}" if part_total > 1 else ""
    _text(
        slide,
        f"{section}{part}",
        9.58,
        1.46,
        2.92,
        0.3,
        size=13,
        color=TEAL,
        bold=True,
        align=PP_ALIGN.RIGHT,
    )
    _line(slide, 0.78, 2.02, 12.52, 2.02, color="BDD9D3", width=1.2)
    _detail_rows(slide, entries, renderer=renderer)


def _stage_slides(
    prs: Presentation,
    stage: PlanStage,
    stage_index: int,
    stage_total: int,
    page: int,
    renderer: LatexTextRenderer,
) -> int:
    learning_entries = [("阶段目标", stage.goal)]
    learning_entries.extend(("核心内容", item) for item in stage.concepts)
    actions = stage.actions or ["根据本阶段目标整理知识要点，并形成可复查的学习产出"]
    learning_entries.extend(("学习步骤", item) for item in actions)
    learning_pages = _paginate_entries(learning_entries, capacity=14)

    standards = stage.standards or ["能独立讲清本阶段核心内容，并完成自测与纠错"]
    practice_entries = [("巩固练习", item) for item in stage.practices]
    practice_entries.extend(("掌握要求", item) for item in standards)
    standard_pages = _paginate_entries(
        practice_entries,
        capacity=14,
    )
    added = 0
    for part_index, entries in enumerate(learning_pages, start=1):
        _stage_detail_slide(
            prs,
            stage,
            stage_index,
            stage_total,
            page + added,
            entries,
            section="学什么，怎么做",
            part_index=part_index,
            part_total=len(learning_pages),
            renderer=renderer,
        )
        added += 1
    for part_index, entries in enumerate(standard_pages, start=1):
        _stage_detail_slide(
            prs,
            stage,
            stage_index,
            stage_total,
            page + added,
            entries,
            section="练什么，达到什么要求",
            part_index=part_index,
            part_total=len(standard_pages),
            renderer=renderer,
        )
        added += 1
    return added


def _principle_slides(
    prs: Presentation,
    principles: list[str],
    page: int,
    renderer: LatexTextRenderer,
) -> int:
    remaining = principles[3:]
    if not remaining:
        return 0
    pages = _paginate_entries(
        [("学习方法", principle) for principle in remaining],
        capacity=12,
    )
    for part_index, entries in enumerate(pages, start=1):
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        _set_background(slide, CREAM)
        kicker = f"LEARNING GOAL / {part_index + 1:02d}"
        _slide_title(slide, kicker, "学习目标与方法", page + part_index - 1)
        _detail_rows(slide, entries, renderer=renderer, y=2.28, height=4.18)
    return len(pages)


def validate_learning_plan_ppt(
    output_path: Path,
    plan: ParsedLearningPlan | None = None,
) -> list[str]:
    """Return audience-facing and structural issues found in a generated deck."""

    presentation = Presentation(str(output_path))
    issues: list[str] = []
    all_text: list[str] = []
    slide_texts: list[str] = []
    for slide_index, slide in enumerate(presentation.slides, start=1):
        slide_text: list[str] = []
        for shape in slide.shapes:
            if (
                shape.left < 0
                or shape.top < 0
                or shape.left + shape.width > presentation.slide_width + 2
                or shape.top + shape.height > presentation.slide_height + 2
            ):
                issues.append(f"第 {slide_index} 页存在超出画布的对象")
            if getattr(shape, "has_text_frame", False) and shape.text.strip():
                slide_text.append(shape.text)
            else:
                try:
                    description = shape._element.nvPicPr.cNvPr.get("descr", "")
                except AttributeError:
                    description = ""
                if description:
                    slide_text.append(description)
        if not slide_text:
            issues.append(f"第 {slide_index} 页没有可编辑文字")
        all_text.extend(slide_text)
        slide_texts.append("\n".join(slide_text))

    visible_text = "\n".join(all_text)
    forbidden_patterns = (
        (r"schedule_guidance", "可见内容包含内部字段 schedule_guidance"),
        (r"plan_guidance", "可见内容包含内部字段 plan_guidance"),
        (r"(?m)^\s*>", "可见内容包含 Markdown 引用标记"),
        (r"\\(?:frac|dfrac|tfrac|begin|end)\b", "可见内容包含未转换的 LaTeX 命令"),
        (r"(?<!\\)\$", "可见内容包含未转换的 LaTeX 定界符"),
        (
            r"建议投入|预计用时|建议时长|阶段时长|时间安排|学习日程|"
            r"\bDay\s*\d+|学习课次|总投入|每天|每周|\d+\s*(?:课次|小时)",
            "可见内容仍包含学习时间安排",
        ),
        (
            r"资料索引|检索依据|对应资料|复盘验收|综合验收|迁移应用|"
            r"什么时候算真正学会|现在就开始",
            "可见内容包含目录之外的资料、验收、迁移或通用结尾页面",
        ),
    )
    for pattern, message in forbidden_patterns:
        if re.search(pattern, visible_text, re.IGNORECASE):
            issues.append(message)
    required_sections = ("学习目录", "学习目标与方法", "学什么，怎么做", "练什么，达到什么要求")
    for section in required_sections:
        if section not in visible_text:
            issues.append(f"缺少必要页面：{section}")
    if plan is not None:
        compact_visible = re.sub(
            r"\s+",
            "",
            _plain(visible_text, preserve_latex=False),
        )
        required_copy: list[str] = [plan.goal]
        required_copy.extend(plan.principles)
        for stage in plan.stages:
            required_copy.extend([stage.title, stage.goal])
            required_copy.extend(stage.concepts)
            required_copy.extend(stage.actions)
            required_copy.extend(stage.practices)
            required_copy.extend(stage.standards)
        for value in required_copy:
            for chunk in _split_complete_text(value):
                compact_chunk = re.sub(
                    r"\s+",
                    "",
                    _plain(chunk, preserve_latex=False),
                )
                if compact_chunk not in compact_visible:
                    issues.append(f"规划内容未完整写入 PPT：{chunk[:24]}")
        outline_items = ["学习目标与方法", *(stage.title for stage in plan.stages)]
        contents_text = re.sub(r"\s+", "", slide_texts[1] if len(slide_texts) > 1 else "")
        body_text = re.sub(r"\s+", "", "\n".join(slide_texts[2:]))
        for item in outline_items:
            compact_item = re.sub(r"\s+", "", item)
            if compact_item not in contents_text:
                issues.append(f"学习目录缺少项目：{item}")
            if compact_item not in body_text:
                issues.append(f"目录项目没有对应正文：{item}")
    return _dedupe(issues)


def build_learning_plan_ppt(markdown: str, output_path: Path, topic: str = "") -> tuple[ParsedLearningPlan, int]:
    plan = parse_learning_plan(markdown, topic)
    presentation = Presentation()
    presentation.slide_width = SLIDE_WIDTH
    presentation.slide_height = SLIDE_HEIGHT
    core = presentation.core_properties
    core.title = plan.title
    core.subject = "学习规划演示文稿"
    core.author = "CircuitMind"
    core.comments = "根据本次学习规划回答自动生成"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="circuitmind-latex-") as latex_dir:
        renderer = LatexTextRenderer(Path(latex_dir))
        _title_slide(presentation, plan, renderer)
        page = 2
        _roadmap_slide(presentation, plan, page, renderer)
        page += 1
        _overview_slide(presentation, plan, page, renderer)
        page += 1
        page += _principle_slides(presentation, plan.principles, page, renderer)
        for index, stage in enumerate(plan.stages, start=1):
            page += _stage_slides(
                presentation,
                stage,
                index,
                len(plan.stages),
                page,
                renderer,
            )
        presentation.save(str(output_path))
        quality_issues = validate_learning_plan_ppt(output_path, plan)
        if quality_issues:
            raise ValueError("学习规划 PPT 质检失败：" + "；".join(quality_issues))

    return plan, len(presentation.slides)


def learning_plan_ppt_path(root_dir: Path, session_id: str, markdown: str, topic: str = "") -> Path:
    digest = hashlib.sha256(f"v6-latex-outline\0{topic}\0{markdown}".encode("utf-8")).hexdigest()[:20]
    return root_dir / "data" / "presentations" / session_id / f"learning-plan-{digest}.pptx"


def generate_learning_plan_ppt(
    root_dir: Path,
    session_id: str,
    markdown: str,
    topic: str = "",
) -> tuple[Path, str, int]:
    output_path = learning_plan_ppt_path(root_dir, session_id, markdown, topic)
    plan = parse_learning_plan(markdown, topic)
    if output_path.exists() and output_path.stat().st_size > 10_000:
        slide_count = len(Presentation(str(output_path)).slides)
        return output_path, plan.title, slide_count

    output_path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix="learning-plan-",
        suffix=".pptx",
        dir=output_path.parent,
    )
    os.close(handle)
    temporary_path = Path(temporary_name)
    try:
        _, slide_count = build_learning_plan_ppt(markdown, temporary_path, topic)
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return output_path, plan.title, slide_count


def presentation_filename(title: str) -> str:
    safe = re.sub(
        r"[\\/:*?\"<>|\r\n]+",
        "-",
        _plain(title, preserve_latex=False),
    ).strip(" .-")
    safe = _shorten(safe, 48) or "学习规划"
    return f"{safe}.pptx"
