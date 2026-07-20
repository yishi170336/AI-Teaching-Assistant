from __future__ import annotations

import re


COURSE_CONCEPTS = (
    "半导体", "本征半导体", "杂质半导体", "N型半导体", "P型半导体", "自由电子",
    "空穴", "载流子", "多数载流子", "少数载流子", "扩散运动", "漂移运动", "复合",
    "空间电荷区", "耗尽层", "PN结", "内建电场", "正向偏置", "反向偏置",
    "二极管", "稳压二极管", "二极管等效模型", "恒压降模型", "理想二极管模型",
    "晶体管", "场效应管", "静态工作点", "共射放大电路", "共集放大电路",
    "共基放大电路", "偏置电路", "射极偏置电路", "放大区", "截止区", "饱和区",
    "伏安特性", "单向导电性", "反向击穿", "齐纳击穿", "雪崩击穿", "温度特性",
    "反向饱和电流", "电压放大", "电流放大", "电压放大倍数", "输入电阻",
    "输出电阻", "非线性失真", "信号反相", "直流电源", "电阻", "电容", "电感",
    "欧姆定律", "基尔霍夫定律", "戴维南定理", "诺顿定理", "叠加定理",
    "节点电压法", "网孔电流法", "KCL", "KVL", "正弦稳态", "相量", "复阻抗",
    "感抗", "容抗", "RLC", "谐振", "有功功率", "无功功率", "视在功率",
    "功率因数", "运算放大器", "负反馈",
)

COMPONENT_CONCEPTS = {
    "resistor": "电阻",
    "capacitor": "电容",
    "inductor": "电感",
    "diode": "二极管",
    "zener": "稳压二极管",
    "npn": "晶体管",
    "pnp": "晶体管",
    "bjt": "晶体管",
    "bipolar_junction_transistor": "晶体管",
    "mosfet": "场效应管",
    "vsource": "直流电源",
    "voltage_source": "直流电源",
}

COMPONENT_TYPE_LABELS = {
    "resistor": "电阻",
    "capacitor": "电容",
    "inductor": "电感",
    "diode": "二极管",
    "zener": "稳压二极管",
    "npn": "晶体管",
    "pnp": "晶体管",
    "bjt": "晶体管",
    "bipolar_junction_transistor": "晶体管",
    "mosfet": "场效应管",
    "vsource": "电压源",
    "voltage_source": "电压源",
    "isource": "电流源",
    "current_source": "电流源",
    "switch": "开关",
    "port": "信号端口",
}

COMPONENT_SYMBOL_ROLES = (
    (r"^rl\d*$", "负载电阻"),
    (r"^rb[12]$", "基极分压电阻"),
    (r"^rb\d*$", "基极偏置电阻"),
    (r"^rc\d*$", "集电极电阻"),
    (r"^re\d*$", "发射极电阻"),
    (r"^rs\d*$", "信号源内阻"),
    (r"^rg\d*$", "栅极电阻"),
    (r"^rd\d*$", "漏极电阻"),
    (r"^rds\d*$", "漏源等效电阻"),
    (r"^rbe\d*$", "基极-发射极等效电阻"),
    (r"^ri\d*$", "输入电阻"),
    (r"^(?:ro|r0)\d*$", "输出电阻"),
    (r"^rf\d*$", "反馈电阻"),
    (r"^ce\d*$", "发射极旁路电容"),
    (r"^cs\d*$", "源极旁路电容"),
    (r"^(?:ci|c1)$", "输入耦合电容"),
    (r"^(?:co|c2)$", "输出耦合电容"),
    (r"^vbb\d*$", "基极偏置电源"),
    (r"^vcc\d*$", "集电极直流电源"),
    (r"^vgg\d*$", "栅极偏置电源"),
    (r"^vdd\d*$", "漏极直流电源"),
    (r"^vee\d*$", "发射极负电源"),
    (r"^vaa\d*$", "直流电源"),
    (r"^(?:us|vs)\d*$", "输入信号源"),
    (r"^(?:ui|vi)\d*$", "输入电压"),
    (r"^(?:uo|vo)\d*$", "输出电压"),
    (r"^dz\d*$", "稳压二极管"),
    (r"^[qt]\d*$", "晶体管"),
    (r"^m\d*$", "场效应管"),
    (r"^d\d*$", "二极管"),
    (r"^s\d*$", "开关"),
)

COMPONENT_CONTEXT_ROLES = (
    "基极-发射极等效电阻", "发射极旁路电容", "源极旁路电容",
    "基极分压电阻", "基极偏置电阻", "集电极直流电源", "基极偏置电源",
    "栅极偏置电源", "漏极直流电源", "输入耦合电容", "输出耦合电容",
    "信号源内阻", "集电极电阻", "发射极电阻", "负载电阻", "反馈电阻",
    "输入电阻", "输出电阻", "栅极电阻", "漏极电阻", "晶体管",
    "场效应管", "稳压二极管", "二极管", "直流电源", "输入信号源",
)


def _canonical_component_symbol(value: str) -> str:
    """Normalize common textbook subscript spellings without changing the label."""

    return re.sub(r"[\s_{}'’′-]", "", value).lower()


def component_role(
    symbol: str,
    component_type: str = "",
    *,
    explicit_role: str = "",
    context: str = "",
) -> str:
    """Return the circuit-specific meaning of a component reference designator."""

    reference = str(symbol).strip()
    role = str(explicit_role or "").strip()
    if reference and role:
        role = re.sub(re.escape(reference), "", role, flags=re.I).strip(" ：:，,（）()")
    if role.lower() in {"", "null", "none", "unknown", "component", "元件", "器件"}:
        role = ""
    if role and len(role) <= 24:
        return role

    if reference and context:
        for candidate in COMPONENT_CONTEXT_ROLES:
            before = rf"{re.escape(candidate)}\s*{re.escape(reference)}"
            after = rf"{re.escape(reference)}\s*(?:为|是|作为|表示|：|:)\s*{re.escape(candidate)}"
            if re.search(before, context, re.I) or re.search(after, context, re.I):
                return candidate

    canonical = _canonical_component_symbol(reference)
    for pattern, candidate in COMPONENT_SYMBOL_ROLES:
        if re.fullmatch(pattern, canonical, re.I):
            return candidate
    return COMPONENT_TYPE_LABELS.get(str(component_type).strip().lower(), "电路元件")


def component_display_name(
    symbol: str,
    component_type: str = "",
    *,
    explicit_role: str = "",
    context: str = "",
) -> str:
    """Format a graph label as a semantic role followed by its original symbol."""

    reference = str(symbol).strip() or "未标注"
    role = component_role(
        reference,
        component_type,
        explicit_role=explicit_role,
        context=context,
    )
    return f"{role} {reference}"

NON_CONCEPT_TAGS = {
    "text", "image", "figure", "formula", "table", "circuit", "multimodal",
    "component", "net", "port", "vsource", "isource", "resistor", "capacitor",
    "inductor", "npn", "pnp", "bjt", "mosfet",
}


def normalize_concept_name(value: str) -> str:
    """Remove noisy chapter-number prefixes emitted by PDF layout parsing."""

    normalized = re.sub(r"\s+", " ", value).strip()
    numbered = re.match(
        r"^[.．、]?\s*(?:\d+\s*[.．]\s*)*\d+\s*(.+)$",
        normalized,
    )
    if numbered and re.search(r"[\u4e00-\u9fff]", numbered.group(1)):
        normalized = numbered.group(1)
    return normalized.strip(" .．、:：-")


def meaningful_section(section: str) -> str:
    value = normalize_concept_name(section)
    if not value or len(value) > 60:
        return ""
    if re.search(r"(?:pages?|页)[_-]?\d+[_-]\d+", value, re.I):
        return ""
    if "_" in value and not re.search(r"[\u4e00-\u9fff]", value):
        return ""
    return value if re.search(r"[\u4e00-\u9fff]", value) else ""


def extract_course_concepts(text: str, section: str = "") -> list[str]:
    lowered = text.lower()
    concepts = [concept for concept in COURSE_CONCEPTS if concept.lower() in lowered]
    section_name = meaningful_section(section)
    if section_name and section_name not in concepts:
        concepts.insert(0, section_name)
    return list(dict.fromkeys(concepts))[:12]


def extract_formula_concepts(text: str) -> list[str]:
    """Map canonical circuit symbols to the concepts they mathematically define."""

    normalized = re.sub(r"[\\{}_\s]", "", text).lower()
    concepts: list[str] = []
    if any(symbol in normalized for symbol in ("ibq", "icq", "ubeq", "uceq")):
        concepts.append("静态工作点")
    if "beta" in normalized or "β" in normalized:
        concepts.append("电流放大")
    if any(symbol in normalized for symbol in ("vcc", "vbb")):
        concepts.append("直流电源")
    if any(symbol in normalized for symbol in ("rb", "rc")):
        concepts.append("电阻")
    return list(dict.fromkeys(concepts))


def is_course_concept(value: str) -> bool:
    normalized = value.strip()
    if not normalized or normalized.lower() in NON_CONCEPT_TAGS:
        return False
    if re.search(r"(?:pages?|页)[_-]?\d+[_-]\d+", normalized, re.I):
        return False
    if "_" in normalized and not re.search(r"[\u4e00-\u9fff]", normalized):
        return False
    return normalized in COURSE_CONCEPTS or bool(meaningful_section(normalized))
