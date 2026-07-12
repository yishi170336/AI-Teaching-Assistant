from __future__ import annotations

import re


COURSE_CONCEPTS = (
    "本征半导体", "N型半导体", "P型半导体", "PN结", "二极管", "稳压二极管",
    "晶体管", "场效应管", "静态工作点", "共射放大电路", "偏置电路", "放大区",
    "截止区", "饱和区", "伏安特性", "单向导电性", "反向击穿", "电压放大",
    "电流放大", "非线性失真", "信号反相", "直流电源", "电阻", "电容", "电感",
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
