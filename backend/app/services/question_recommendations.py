from __future__ import annotations

import json
import math
import re
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.app.config import settings
from backend.app.rag.section_titles import chapter_number, numbered_section_parts
from backend.app.services.homework import HomeworkStore


RECOMMENDATION_BANK_TITLE = "电子电路基础学习指导书"
PROFILE_VERSION = "2026.07.4"
DIFFICULTY_LABELS = {
    "basic": "基础",
    "intermediate": "进阶",
    "advanced": "挑战",
}
ALIASES = {
    "共发射极": "共射放大电路",
    "共射": "共射放大电路",
    "运算放大器": "集成运算放大器",
    "运放": "集成运算放大器",
    "op amp": "集成运算放大器",
    "场效应管": "场效应管",
    "fet": "场效应管",
    "三极管": "晶体三极管",
    "bjt": "晶体三极管",
    "二极管": "二极管",
    "负反馈": "负反馈",
    "反馈": "反馈",
    "波形": "波形分析",
    "限幅": "限幅电路",
}
SKILL_KEYWORDS = {
    "概念判断": ("判断", "说明", "为什么", "概念", "选择", "正确", "错误"),
    "参数计算": ("计算", "求", "估算", "数值", "参数"),
    "电路分析": ("分析电路", "工作原理", "静态工作点", "增益", "放大"),
    "波形分析": ("波形", "画出", "输入输出", "传输特性"),
    "电路设计": ("设计", "选择元件", "确定参数", "满足指标"),
}
COMPONENT_KEYWORDS = (
    "二极管", "稳压二极管", "晶体三极管", "场效应管", "运算放大器",
    "电阻", "电容", "电感", "变压器", "滤波器", "振荡器",
)
METHOD_KEYWORDS = (
    "KCL", "KVL", "叠加定理", "戴维南定理", "诺顿定理", "小信号模型",
    "图解法", "节点电压法", "网孔电流法", "相量法", "虚短虚断",
    "直流等效分析", "交流小信号分析", "分段线性法", "负载线法", "瞬时极性法",
)
FUNCTION_KEYWORDS = {
    "整流电路": ("整流", "桥式整流"),
    "限幅与钳位": ("限幅", "钳位"),
    "稳压电路": ("稳压", "基准电压"),
    "基本放大电路": ("放大电路", "共射", "共集", "共基"),
    "差分放大电路": ("差分", "差动放大"),
    "功率放大电路": ("功率放大", "互补对称", "交越失真"),
    "反馈放大电路": ("反馈放大", "反馈组态", "反馈深度"),
    "运算电路": ("比例运算", "加法运算", "积分电路", "微分电路"),
    "电压比较器": ("比较器", "滞回比较"),
    "波形发生电路": ("波形发生", "方波", "锯齿波", "三角波"),
    "滤波电路": ("滤波", "低通", "高通", "带通", "带阻"),
    "振荡电路": ("振荡", "正弦波发生"),
}
TASK_KEYWORDS = {
    "工作状态判断": ("导通", "截止", "工作区", "状态", "是否工作"),
    "静态工作点": ("静态工作点", "偏置", "IBQ", "ICQ", "UCEQ"),
    "增益计算": ("增益", "放大倍数", "电压放大", "电流放大"),
    "输入输出特性": ("输入电阻", "输出电阻", "传输特性", "输入输出"),
    "波形与时序": ("波形", "周期", "频率", "相位", "时间常数"),
    "反馈判别": ("反馈类型", "反馈组态", "正反馈", "负反馈", "反馈深度"),
    "参数设计": ("设计", "选择参数", "确定参数", "满足指标"),
    "故障与误差分析": ("故障", "误差", "失真", "异常", "改进"),
}
_PAPER_BANK_PATTERN = re.compile(
    r"(?:测试题|考试|测验|模拟卷|期中|期末|[a-zＡ-Ｚ]卷)",
    re.IGNORECASE,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_list(value: Any, limit: int = 12) -> list[str]:
    values = value if isinstance(value, list) else []
    result: list[str] = []
    for item in values:
        text = str(item or "").strip()
        if text and text not in result:
            result.append(text[:80])
    return result[:limit]


def _normalize_topic(value: str) -> str:
    text = re.sub(r"\s+", "", value).casefold()
    for alias, canonical in ALIASES.items():
        if alias.casefold() in text:
            return canonical
    return value.strip()


def _tokens(value: str) -> list[str]:
    latin = re.findall(r"[a-zA-Z][a-zA-Z0-9+-]*|\d+(?:\.\d+)?", value.casefold())
    chinese = re.findall(r"[\u4e00-\u9fff]{2,8}", value)
    ngrams: list[str] = []
    for chunk in chinese:
        ngrams.extend(chunk[index:index + 2] for index in range(max(1, len(chunk) - 1)))
    return latin + ngrams


def _chapter_key(value: Any) -> str:
    """Return a stable numeric chapter key shared by KB chunks and questions."""

    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        return ""
    number = chapter_number(text)
    if number is not None:
        return f"chapter-{number}"
    section_parts, _title = numbered_section_parts(text)
    if section_parts:
        return f"chapter-{section_parts[0]}"
    match = re.search(r"(?:^|[^a-z])chapter[-_\s]*(\d{1,2})(?:$|[^\d])", text, re.I)
    if match:
        return f"chapter-{int(match.group(1))}"
    numeric = re.match(r"^\s*(\d{1,2})(?:\s*[.．]\s*\d{1,2})+", text)
    if numeric:
        return f"chapter-{int(numeric.group(1))}"
    return ""


def _normalized_chapter_scope(value: Any) -> list[dict[str, Any]]:
    scope = value if isinstance(value, list) else []
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in scope:
        item = raw if isinstance(raw, dict) else {"chapter": raw}
        chapter = str(item.get("chapter") or item.get("title") or "").strip()[:160]
        key = str(item.get("chapter_key") or "").strip() or _chapter_key(chapter)
        if not key or key in seen:
            continue
        seen.add(key)
        try:
            score = max(0.0, min(1.0, float(item.get("score") or 0.0)))
        except (TypeError, ValueError):
            score = 0.0
        try:
            evidence_count = max(0, int(item.get("evidence_count") or 0))
        except (TypeError, ValueError):
            evidence_count = 0
        relevance = str(item.get("relevance") or "weak").strip().lower()
        normalized.append({
            "chapter_key": key,
            "chapter": chapter or key,
            "relevance": "strong" if relevance == "strong" else "weak",
            "score": round(score, 4),
            "evidence_count": evidence_count,
            "sections": _clean_list(item.get("sections"), 4),
        })
    return normalized


def select_relevant_chapters(
    hits: list[dict[str, Any]],
    *,
    explicit_chapter: str = "",
    max_strong: int = 2,
) -> list[dict[str, Any]]:
    """Aggregate KB hits into a narrow chapter scope with one weak fallback.

    At most two strong chapters are kept.  One weaker but still plausible
    chapter may be added so cross-chapter concepts are not lost, while low
    confidence tail chapters never expand the question search indiscriminately.
    """

    explicit = str(explicit_chapter or "").strip()
    if explicit and _chapter_key(explicit):
        return [{
            "chapter_key": _chapter_key(explicit),
            "chapter": explicit[:160],
            "relevance": "strong",
            "score": 1.0,
            "evidence_count": 0,
            "sections": [],
        }]

    grouped: dict[str, dict[str, Any]] = {}
    for hit in hits:
        if not isinstance(hit, dict):
            continue
        chapter = str(hit.get("chapter") or "").strip()
        section = str(hit.get("section") or "").strip()
        key = _chapter_key(chapter) or _chapter_key(section)
        if not key:
            continue
        chapter_label = chapter if _chapter_key(chapter) else section
        try:
            score = max(0.0, min(1.0, float(hit.get("score") or 0.0)))
        except (TypeError, ValueError):
            score = 0.0
        item = grouped.setdefault(key, {
            "chapter_key": key,
            "chapter": chapter_label[:160],
            "best_score": 0.0,
            "evidence_count": 0,
            "sections": [],
        })
        if score > item["best_score"]:
            item["best_score"] = score
            item["chapter"] = chapter_label[:160]
        item["evidence_count"] += 1
        if section and section not in item["sections"]:
            item["sections"].append(section[:160])

    ranked = sorted(
        grouped.values(),
        key=lambda item: (item["best_score"], item["evidence_count"]),
        reverse=True,
    )
    if not ranked:
        return []
    top_score = float(ranked[0]["best_score"])
    if top_score <= 0:
        ranked = ranked[:1]
    strong_cutoff = max(0.18, top_score * 0.72)
    weak_cutoff = max(0.10, top_score * 0.45)
    strong = [item for item in ranked if item["best_score"] >= strong_cutoff][
        : max(1, max_strong)
    ]
    if not strong:
        strong = ranked[:1]
    selected_keys = {item["chapter_key"] for item in strong}
    weak = next(
        (
            item for item in ranked
            if item["chapter_key"] not in selected_keys
            and item["best_score"] >= weak_cutoff
        ),
        None,
    )
    result = [
        {
            "chapter_key": item["chapter_key"],
            "chapter": item["chapter"],
            "relevance": "strong",
            "score": round(float(item["best_score"]), 4),
            "evidence_count": item["evidence_count"],
            "sections": item["sections"][:4],
        }
        for item in strong
    ]
    if weak is not None:
        result.append({
            "chapter_key": weak["chapter_key"],
            "chapter": weak["chapter"],
            "relevance": "weak",
            "score": round(float(weak["best_score"]), 4),
            "evidence_count": weak["evidence_count"],
            "sections": weak["sections"][:4],
        })
    return result


def _question_chapter_key(
    question: dict[str, Any], profile: dict[str, Any] | None = None
) -> str:
    profile = profile if isinstance(profile, dict) else question.get("retrieval_profile")
    profile = profile if isinstance(profile, dict) else {}
    location = question.get("location")
    location = location if isinstance(location, dict) else {}
    for value in (
        profile.get("chapter"),
        profile.get("section"),
        location.get("chapter"),
        question.get("section_title"),
        question.get("section_key"),
    ):
        key = _chapter_key(value)
        if key:
            return key
    return ""


def _profile_matches_chapter(profile: dict[str, Any], requested: str) -> bool:
    requested_key = _chapter_key(requested)
    profile_key = _chapter_key(profile.get("chapter")) or _chapter_key(profile.get("section"))
    if requested_key and profile_key:
        return requested_key == profile_key
    return bool(requested and requested in str(profile.get("chapter", "")))


def default_recommendation_enabled(bank: dict[str, Any]) -> bool:
    explicit = bank.get("recommendation_enabled")
    if isinstance(explicit, bool):
        return explicit
    return (
        str(bank.get("title", "")).strip() == RECOMMENDATION_BANK_TITLE
        and len([item for item in bank.get("questions", []) if isinstance(item, dict)]) >= 300
    )


def has_reference_answer(question: dict[str, Any]) -> bool:
    """Whether a recommended question can later be checked against server evidence."""
    return bool(
        str(question.get("answer", "")).strip()
        or _clean_list(question.get("answer_subquestions"), 1)
        or _clean_list(question.get("answer_figures"), 1)
    )


def build_retrieval_profile(question: dict[str, Any]) -> dict[str, Any]:
    existing = question.get("retrieval_profile")
    existing = existing if isinstance(existing, dict) else {}
    manual_fields = _clean_list(existing.get("manual_fields"), 16)
    prompt = str(question.get("prompt", ""))
    section = str(question.get("section_title", ""))
    content = "\n".join(
        [
            section,
            prompt,
            " ".join(str(item.get("text", "")) for item in question.get("subquestions", []) if isinstance(item, dict)),
        ]
    )
    knowledge_points = [
        _normalize_topic(item)
        for item in _clean_list(question.get("knowledge_points"), 12)
    ]
    for alias, canonical in ALIASES.items():
        if alias.casefold() in content.casefold() and canonical not in knowledge_points:
            knowledge_points.append(canonical)
    question_type = str(question.get("question_type") or "other")
    skills = [
        skill
        for skill, keywords in SKILL_KEYWORDS.items()
        if any(keyword in content for keyword in keywords)
    ]
    if not skills:
        skills = ["参数计算"] if question_type == "calculation" else ["概念判断"]
    components = [item for item in COMPONENT_KEYWORDS if item in content]
    methods = [item for item in METHOD_KEYWORDS if item.casefold() in content.casefold()]
    if "集成运算放大器" in knowledge_points and any(word in content for word in ("理想", "运放")):
        methods.append("虚短虚断")
    if any(word in content for word in ("静态工作点", "偏置", "直流通路")):
        methods.append("直流等效分析")
    if any(word in content for word in ("电压增益", "放大倍数", "小信号")) and any(
        item in content for item in ("晶体管", "三极管", "场效应管", "共射", "共源")
    ):
        methods.append("交流小信号分析")
    if "二极管" in content and any(word in content for word in ("导通", "截止", "输出电压", "传输特性")):
        methods.append("分段线性法")
    if any(word in content for word in ("反馈类型", "反馈组态", "反馈极性")):
        methods.append("瞬时极性法")
    methods = list(dict.fromkeys(methods))
    circuit_functions = [
        function
        for function, keywords in FUNCTION_KEYWORDS.items()
        if any(keyword.casefold() in content.casefold() for keyword in keywords)
    ]
    tasks = [
        task
        for task, keywords in TASK_KEYWORDS.items()
        if any(keyword.casefold() in content.casefold() for keyword in keywords)
    ]
    subquestion_count = len(question.get("subquestions", []))
    if any(word in content for word in ("设计", "证明", "综合", "推导")) or subquestion_count >= 4:
        difficulty = "advanced"
    elif question_type in {"calculation", "design"} or subquestion_count >= 2:
        difficulty = "intermediate"
    else:
        difficulty = "basic"
    location = question.get("location") if isinstance(question.get("location"), dict) else {}
    chapter = section or str(location.get("chapter", ""))
    profile = {
        "knowledge_points": list(dict.fromkeys(knowledge_points))[:12],
        "question_type": question_type,
        "difficulty": difficulty,
        "skills": skills[:8],
        "components": components[:8],
        "methods": methods[:8],
        "circuit_functions": circuit_functions[:8],
        "tasks": tasks[:8],
        "chapter": chapter[:160],
        "section": section[:160],
        "source": str(existing.get("source") or "auto"),
        "version": str(existing.get("version") or PROFILE_VERSION),
        "confidence": float(existing.get("confidence") or (0.82 if knowledge_points else 0.58)),
        "status": str(existing.get("status") or ("auto" if knowledge_points else "needs_review")),
        "manual_fields": manual_fields,
    }
    for field in manual_fields:
        if field in existing:
            profile[field] = existing[field]
    return profile


class QuestionRecommendationService:
    """Small durable hybrid index for server-authoritative question recommendations."""

    def __init__(self, homework_store: HomeworkStore, root: Path | None = None) -> None:
        self.homework_store = homework_store
        self.root = (root or settings.root_dir / "data" / "question_recommendations").resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.history_path = self.root / "history.json"
        self._lock = threading.RLock()

    def _read_history(self) -> dict[str, list[dict[str, Any]]]:
        try:
            value = json.loads(self.history_path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError, json.JSONDecodeError):
            return {}

    def _write_history(self, value: dict[str, list[dict[str, Any]]]) -> None:
        temporary = self.history_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.history_path)

    def _recommendation_banks(
        self, *, student_id: str, knowledge_base: str = ""
    ) -> list[dict[str, Any]]:
        banks = self.homework_store.list_recommendation_banks(student_id=student_id)
        target = str(knowledge_base or "").strip()
        if not target:
            return banks
        return [
            bank for bank in banks
            if str(bank.get("knowledge_base") or "default") == target
        ]

    @staticmethod
    def _scoped_questions(
        bank: dict[str, Any],
        questions: list[dict[str, Any]],
        chapter_scope: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not chapter_scope:
            return questions
        bank_label = " ".join((
            str(bank.get("title") or ""),
            str(bank.get("source_name") or ""),
        ))
        if (
            str(bank.get("source_origin") or "").strip().lower() == "photo_answer"
            or _PAPER_BANK_PATTERN.search(bank_label)
        ):
            return questions
        available_keys = {
            key for question in questions
            if (key := _question_chapter_key(question))
        }
        if not available_keys:
            # Papers and legacy flat banks have no trustworthy chapter boundary.
            return questions
        scope_keys = {
            str(item.get("chapter_key") or "") for item in chapter_scope
            if str(item.get("chapter_key") or "")
        }
        return [
            question for question in questions
            if _question_chapter_key(question) in scope_keys
        ]

    def metadata(self, *, student_id: str) -> dict[str, Any]:
        """Return server-authoritative inventory statistics for accessible banks."""
        banks = self.homework_store.list_question_banks(
            include_questions=True,
            student_id=student_id,
        )
        ready_banks = [bank for bank in banks if bank.get("status") == "ready"]
        enabled_banks = self.homework_store.list_recommendation_banks(
            student_id=student_id
        )
        total_questions = sum(
            len([item for item in bank.get("questions", []) if isinstance(item, dict)])
            for bank in banks
        )
        ready_questions = sum(
            len([item for item in bank.get("questions", []) if isinstance(item, dict)])
            for bank in ready_banks
        )
        recommendable_questions = 0
        for bank in enabled_banks:
            warnings = list(bank.get("processing_warnings", []))
            recommendable_questions += sum(
                1
                for question in bank.get("questions", [])
                if isinstance(question, dict)
                and self.homework_store._question_answer_readiness(question, warnings).get("status") == "ready"
                and has_reference_answer(question)
            )
        return {
            "bank_count": len(banks),
            "ready_bank_count": len(ready_banks),
            "recommendation_bank_count": len(enabled_banks),
            "question_count": total_questions,
            "ready_question_count": ready_questions,
            "recommendable_question_count": recommendable_questions,
        }

    def _parse_query(self, query: str, inherited: dict[str, Any] | None = None) -> dict[str, Any]:
        inherited = inherited or {}
        allow_relaxation = any(keyword in query for keyword in ("放宽", "最接近", "不限"))
        if re.search(r"再来|换一|下一道|放宽", query) and inherited:
            requirements = dict(inherited)
            requirements["query"] = query.strip()[:1000]
            requirements["allow_relaxation"] = allow_relaxation
            return requirements
        topics = list(dict.fromkeys(
            canonical for alias, canonical in ALIASES.items()
            if alias.casefold() in query.casefold()
        ))
        difficulty = ""
        if any(word in query for word in ("基础", "简单", "入门")):
            difficulty = "basic"
        elif any(word in query for word in ("困难", "挑战", "综合", "高难")):
            difficulty = "advanced"
        elif any(word in query for word in ("进阶", "中等")):
            difficulty = "intermediate"
        question_type = ""
        for label, keywords in {
            "calculation": ("计算", "求解", "数值"),
            "choice": ("选择",),
            "true_false": ("判断",),
            "design": ("设计",),
            "short_answer": ("简答", "说明"),
        }.items():
            if any(word in query for word in keywords):
                question_type = label
                break
        skills = [skill for skill, words in SKILL_KEYWORDS.items() if any(word in query for word in words)]
        components = [item for item in COMPONENT_KEYWORDS if item.casefold() in query.casefold()]
        methods = [item for item in METHOD_KEYWORDS if item.casefold() in query.casefold()]
        circuit_functions = [
            function
            for function, keywords in FUNCTION_KEYWORDS.items()
            if any(keyword.casefold() in query.casefold() for keyword in keywords)
        ]
        tasks = [
            task
            for task, keywords in TASK_KEYWORDS.items()
            if any(keyword.casefold() in query.casefold() for keyword in keywords)
        ]
        chapter_match = re.search(r"第\s*([一二三四五六七八九十\d]+)\s*章", query)
        requires_figure: bool | None = None
        if any(word in query for word in ("包含题图", "带图", "电路图题")):
            requires_figure = True
        elif any(word in query for word in ("纯文字题", "不要题图", "无图题")):
            requires_figure = False
        return {
            "query": query.strip()[:1000],
            "knowledge_points": topics,
            "question_type": question_type,
            "difficulty": difficulty,
            "skills": skills,
            "components": components,
            "methods": methods,
            "circuit_functions": circuit_functions,
            "tasks": tasks,
            "chapter": chapter_match.group(0) if chapter_match else "",
            "requires_figure": requires_figure,
            "allow_relaxation": allow_relaxation,
        }

    @staticmethod
    def _score(query: str, requirements: dict[str, Any], question: dict[str, Any], profile: dict[str, Any]) -> tuple[float, dict[str, float]]:
        agent_analysis = (
            requirements.get("agent_analysis")
            if isinstance(requirements.get("agent_analysis"), dict)
            else {}
        )
        haystack = " ".join([
            str(question.get("prompt", "")),
            str(question.get("section_title", "")),
            " ".join(profile.get("knowledge_points", [])),
            " ".join(profile.get("skills", [])),
            " ".join(profile.get("components", [])),
            " ".join(profile.get("methods", [])),
            " ".join(profile.get("circuit_functions", [])),
            " ".join(profile.get("tasks", [])),
        ])
        query_tokens = Counter(_tokens(query))
        document_tokens = Counter(_tokens(haystack))
        overlap = sum(min(count, document_tokens.get(token, 0)) for token, count in query_tokens.items())
        lexical = overlap / max(1, sum(query_tokens.values()))
        requested_topics = set(requirements.get("knowledge_points", []))
        profile_topics = set(profile.get("knowledge_points", []))
        topic = (
            len(requested_topics.intersection(profile_topics)) / len(requested_topics)
            if requested_topics else 0.0
        )
        type_score = 1.0 if requirements.get("question_type") and requirements["question_type"] == profile.get("question_type") else 0.0
        difficulty = 1.0 if requirements.get("difficulty") and requirements["difficulty"] == profile.get("difficulty") else 0.0
        skills = set(requirements.get("skills", []))
        skill = len(skills.intersection(profile.get("skills", []))) / max(1, len(skills))
        chapter = (
            1.0
            if requirements.get("chapter")
            and _profile_matches_chapter(profile, str(requirements["chapter"]))
            else 0.0
        )
        requested_components = set(requirements.get("components", []))
        component = len(requested_components.intersection(profile.get("components", []))) / max(1, len(requested_components))
        requested_methods = set(requirements.get("methods", []))
        method = len(requested_methods.intersection(profile.get("methods", []))) / max(1, len(requested_methods))
        requested_functions = set(requirements.get("circuit_functions", []))
        circuit_function = len(requested_functions.intersection(profile.get("circuit_functions", []))) / max(1, len(requested_functions))
        requested_tasks = set(requirements.get("tasks", []))
        task = len(requested_tasks.intersection(profile.get("tasks", []))) / max(1, len(requested_tasks))
        inferred_terms = {
            str(item).strip()
            for field in (
                "knowledge_points",
                "components",
                "methods",
                "circuit_functions",
                "tasks",
                "reasoning_focus",
            )
            for item in (
                agent_analysis.get(field, [])
                if isinstance(agent_analysis.get(field), list)
                else [agent_analysis.get(field, "")]
            )
            if str(item).strip()
        }
        inferred_match = (
            sum(1 for item in inferred_terms if item.casefold() in haystack.casefold())
            / max(1, len(inferred_terms))
        )
        components = {
            "semantic": round(lexical, 4),
            "knowledge": round(topic, 4),
            "question_type": round(type_score, 4),
            "difficulty": round(difficulty, 4),
            "skill": round(skill, 4),
            "chapter": round(chapter, 4),
            "component": round(component, 4),
            "method": round(method, 4),
            "circuit_function": round(circuit_function, 4),
            "task": round(task, 4),
            "agent_semantic": round(inferred_match, 4),
        }
        weighted = (
            lexical * 0.20
            + topic * 0.28
            + type_score * 0.12
            + difficulty * 0.10
            + skill * 0.07
            + chapter * 0.03
            + component * 0.07
            + method * 0.05
            + circuit_function * 0.07
            + task * 0.05
            + inferred_match * 0.13
        )
        if requested_topics and topic < 1:
            weighted -= (1 - topic) * 0.35
        if requirements.get("question_type") and not type_score:
            weighted -= 0.25
        if requirements.get("difficulty") and not difficulty:
            weighted -= 0.3
        if not requirements.get("knowledge_points") and not requirements.get("question_type") and not requirements.get("difficulty"):
            weighted += math.log1p(max(0, len(str(question.get("prompt", ""))))) / 100
        return weighted, components

    @staticmethod
    def _sanitize_agent_analysis(value: dict[str, Any] | None) -> dict[str, Any]:
        value = value if isinstance(value, dict) else {}
        result: dict[str, Any] = {
            "intent_summary": str(value.get("intent_summary", "")).strip()[:240],
            "reasoning_focus": str(value.get("reasoning_focus", "")).strip()[:240],
        }
        for field in (
            "knowledge_points",
            "components",
            "methods",
            "circuit_functions",
            "tasks",
            "skills",
            "soft_preferences",
            "avoid",
            "tradeoffs",
            "fit_dimensions",
        ):
            result[field] = _clean_list(value.get(field), 10)
        raw_constraints = value.get("explicit_constraints")
        if isinstance(raw_constraints, dict):
            question_type = str(raw_constraints.get("question_type", "")).strip()
            difficulty = str(raw_constraints.get("difficulty", "")).strip()
            requires_figure = raw_constraints.get("requires_figure")
            result["explicit_constraints"] = {
                "question_type": question_type if question_type in {
                    "calculation", "choice", "true_false", "design", "short_answer",
                } else "",
                "difficulty": difficulty if difficulty in {
                    "basic", "intermediate", "advanced",
                } else "",
                "chapter": str(raw_constraints.get("chapter", "")).strip()[:120],
                "requires_figure": requires_figure if isinstance(requires_figure, bool) else None,
            }
        return result

    @staticmethod
    def _apply_agent_requirements(
        requirements: dict[str, Any], agent_analysis: dict[str, Any]
    ) -> dict[str, Any]:
        """Make model-understood semantics primary while preserving hard constraints."""

        updated = dict(requirements)
        for field in (
            "knowledge_points", "skills", "components", "methods",
            "circuit_functions", "tasks",
        ):
            semantic_values = _clean_list(agent_analysis.get(field), 12)
            if semantic_values:
                # These are semantic retrieval requirements, not hard exclusion
                # gates. They replace sparse keyword extraction from a deictic
                # request such as "按这个去题库找一道".
                updated[field] = semantic_values
        constraints = agent_analysis.get("explicit_constraints")
        if not isinstance(constraints, dict):
            return updated
        for field in ("question_type", "difficulty", "chapter"):
            if constraints.get(field):
                updated[field] = constraints[field]
            elif field in {"question_type", "difficulty"}:
                # A model that read the request and found no explicit constraint
                # clears accidental keyword-derived hard filters.
                updated[field] = ""
        if constraints.get("requires_figure") is not None:
            updated["requires_figure"] = constraints["requires_figure"]
        return updated

    def shortlist(
        self,
        *,
        query: str,
        constraint_query: str | None = None,
        student_id: str,
        inherited_requirements: dict[str, Any] | None = None,
        agent_analysis: dict[str, Any] | None = None,
        excluded_question_ids: set[str] | None = None,
        knowledge_base: str = "",
        chapter_scope: list[dict[str, Any]] | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        # `query` may include the full bound source question for semantic
        # retrieval. Only the student's own request may create hard filters;
        # otherwise words such as "选择题" inside the source are mistaken for
        # an explicit request to return the same presentation type.
        requirements = self._parse_query(
            constraint_query if constraint_query is not None else query,
            inherited_requirements,
        )
        sanitized_analysis = self._sanitize_agent_analysis(agent_analysis)
        requirements = self._apply_agent_requirements(requirements, sanitized_analysis)
        requirements["agent_analysis"] = sanitized_analysis
        normalized_scope = _normalized_chapter_scope(chapter_scope)
        requirements["chapter_scope"] = normalized_scope
        excluded = {str(item) for item in (excluded_question_ids or set()) if str(item)}
        candidates: list[dict[str, Any]] = []
        scope_by_key = {item["chapter_key"]: item for item in normalized_scope}
        for bank in self._recommendation_banks(
            student_id=student_id, knowledge_base=knowledge_base
        ):
            questions = [
                question for question in bank.get("questions", [])
                if isinstance(question, dict)
            ]
            for question in self._scoped_questions(bank, questions, normalized_scope):
                if str(question.get("id", "")) in excluded:
                    continue
                readiness = self.homework_store._question_answer_readiness(
                    question, list(bank.get("processing_warnings", []))
                )
                if readiness.get("status") != "ready":
                    continue
                if not has_reference_answer(question):
                    continue
                profile = build_retrieval_profile(question)
                score, score_components = self._score(query, requirements, question, profile)
                scope_match = scope_by_key.get(_question_chapter_key(question, profile), {})
                candidates.append({
                    "question_id": str(question.get("id", "")),
                    "question_bank_id": str(bank.get("id", "")),
                    "number": str(question.get("number", "")),
                    "chapter": profile.get("chapter", ""),
                    "prompt": str(question.get("prompt", ""))[:1200],
                    "subquestions": [
                        str(item.get("text", ""))[:300]
                        for item in question.get("subquestions", [])
                        if isinstance(item, dict)
                    ][:6],
                    "question_type": profile.get("question_type", ""),
                    "difficulty": profile.get("difficulty", ""),
                    "knowledge_points": profile.get("knowledge_points", []),
                    "skills": profile.get("skills", []),
                    "components": profile.get("components", []),
                    "methods": profile.get("methods", []),
                    "circuit_functions": profile.get("circuit_functions", []),
                    "tasks": profile.get("tasks", []),
                    "has_figure": bool(question.get("figures")),
                    "reference_available": True,
                    "chapter_relevance": scope_match.get("relevance", ""),
                    "retrieval_score": round(score, 5),
                    "score_components": score_components,
                })
        candidates.sort(key=lambda item: item["retrieval_score"], reverse=True)
        # The Agent may request a broad catalog and semantically read stems in
        # batches. Ranking remains useful ordering, but no longer hard-caps the
        # model to the first 20 keyword/profile matches.
        return candidates[: max(1, min(limit, 500))]

    def recommend(
        self,
        *,
        query: str,
        constraint_query: str | None = None,
        student_id: str,
        inherited_requirements: dict[str, Any] | None = None,
        agent_analysis: dict[str, Any] | None = None,
        preferred_question_id: str = "",
        agent_reason: str = "",
        agent_evidence: list[str] | None = None,
        allowed_question_ids: set[str] | None = None,
        excluded_question_ids: set[str] | None = None,
        knowledge_base: str = "",
        chapter_scope: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        requirements = self._parse_query(
            constraint_query if constraint_query is not None else query,
            inherited_requirements,
        )
        sanitized_analysis = self._sanitize_agent_analysis(agent_analysis)
        requirements = self._apply_agent_requirements(requirements, sanitized_analysis)
        requirements["agent_analysis"] = sanitized_analysis
        normalized_scope = _normalized_chapter_scope(chapter_scope)
        requirements["chapter_scope"] = normalized_scope
        excluded = {str(item) for item in (excluded_question_ids or set()) if str(item)}
        allowed = (
            {str(item) for item in allowed_question_ids if str(item)}
            if allowed_question_ids is not None
            else None
        )
        candidates: list[dict[str, Any]] = []
        for bank in self._recommendation_banks(
            student_id=student_id, knowledge_base=knowledge_base
        ):
            questions = [
                question for question in bank.get("questions", [])
                if isinstance(question, dict)
            ]
            for question in self._scoped_questions(bank, questions, normalized_scope):
                question_id = str(question.get("id", ""))
                if question_id in excluded or (
                    allowed is not None and question_id not in allowed
                ):
                    continue
                readiness = self.homework_store._question_answer_readiness(
                    question, list(bank.get("processing_warnings", []))
                )
                if readiness.get("status") != "ready":
                    continue
                if not has_reference_answer(question):
                    continue
                profile = build_retrieval_profile(question)
                score, score_components = self._score(query, requirements, question, profile)
                candidates.append({
                    "bank": bank,
                    "question": question,
                    "profile": profile,
                    "score": score,
                    "score_components": score_components,
                })
        if allowed_question_ids is not None:
            candidates = [
                item
                for item in candidates
                if str(item["question"].get("id", "")) in allowed
            ]
        if not candidates:
            if allowed_question_ids is not None:
                raise LookupError("题库中没有与当前原题核心结构足够相似的可用题目")
            raise LookupError("当前没有结构完整且已启用推荐的题目")
        requested_topics = set(requirements.get("knowledge_points", []))
        requested_type = str(requirements.get("question_type") or "")
        requested_difficulty = str(requirements.get("difficulty") or "")
        requested_skills = set(requirements.get("skills", []))
        requested_components = set(requirements.get("components", []))
        requested_methods = set(requirements.get("methods", []))
        requested_functions = set(requirements.get("circuit_functions", []))
        requested_tasks = set(requirements.get("tasks", []))
        requested_chapter = str(requirements.get("chapter") or "")
        requested_figure = requirements.get("requires_figure")

        def matches(
            item: dict[str, Any],
            *,
            require_all_topics: bool,
            require_type: bool,
            require_difficulty: bool,
            require_all_details: bool,
        ) -> bool:
            profile = item["profile"]
            profile_topics = set(profile.get("knowledge_points", []))
            topic_ok = (
                not requested_topics
                or (
                    requested_topics.issubset(profile_topics)
                    if require_all_topics
                    else bool(requested_topics.intersection(profile_topics))
                )
            )
            type_ok = not requested_type or not require_type or profile.get("question_type") == requested_type
            difficulty_ok = (
                not requested_difficulty
                or not require_difficulty
                or profile.get("difficulty") == requested_difficulty
            )
            detail_pairs = [
                (requested_skills, set(profile.get("skills", []))),
                (requested_components, set(profile.get("components", []))),
                (requested_methods, set(profile.get("methods", []))),
                (requested_functions, set(profile.get("circuit_functions", []))),
                (requested_tasks, set(profile.get("tasks", []))),
            ]
            active_detail_pairs = [pair for pair in detail_pairs if pair[0]]
            if require_all_details:
                details_ok = all(requested.issubset(actual) for requested, actual in active_detail_pairs)
            else:
                # Auto tags are often incomplete. A near candidate only needs
                # to match a meaningful share of the requested detail groups,
                # instead of carrying every skill/method/task tag verbatim.
                matched_detail_groups = sum(
                    1 for requested, actual in active_detail_pairs
                    if requested.intersection(actual)
                )
                minimum_detail_groups = max(1, math.ceil(len(active_detail_pairs) * 0.34))
                details_ok = (
                    not active_detail_pairs
                    or matched_detail_groups >= minimum_detail_groups
                )
            chapter_ok = (
                not requested_chapter
                or _profile_matches_chapter(profile, requested_chapter)
            )
            figure_ok = (
                requested_figure is None
                or bool(item["question"].get("figures")) is bool(requested_figure)
            )
            return (
                topic_ok and type_ok and difficulty_ok and details_ok
                and chapter_ok and figure_ok
            )

        strict_candidates = [
            item for item in candidates
            if matches(
                item,
                require_all_topics=True,
                require_type=True,
                require_difficulty=True,
                require_all_details=True,
            )
        ]
        relaxed_conditions: list[str] = []
        eligible = strict_candidates
        if not eligible:
            eligible = [
                item for item in candidates
                if matches(
                    item,
                    require_all_topics=True,
                    require_type=True,
                    require_difficulty=False,
                    require_all_details=False,
                )
            ]
        if not eligible:
            eligible = [
                item for item in candidates
                if matches(
                    item,
                    require_all_topics=False,
                    require_type=True,
                    require_difficulty=False,
                    require_all_details=False,
                )
            ]
        if not eligible:
            eligible = candidates

        # Only objectively verifiable user constraints may veto an Agent choice.
        # Knowledge/topic/difficulty profiles are extraction hints and must not
        # overrule an Agent that read the actual candidate stems.
        hard_constraints_present = bool(
            requested_type or requested_chapter or requested_figure is not None
        )
        hard_candidates = [
            item for item in candidates
            if (
                (not requested_type or item["profile"].get("question_type") == requested_type)
                and (
                    not requested_chapter
                    or _profile_matches_chapter(item["profile"], requested_chapter)
                )
                and (
                    requested_figure is None
                    or bool(item["question"].get("figures")) is bool(requested_figure)
                )
            )
        ] if hard_constraints_present else candidates

        with self._lock:
            history = self._read_history()
            recent = history.get(student_id, [])[-100:]
            positions = {
                f"{item.get('question_bank_id')}:{item.get('question_id')}": index
                for index, item in enumerate(recent)
            }
            unseen = [
                item for item in eligible
                if f"{item['bank']['id']}:{item['question']['id']}" not in positions
            ]
            pool = unseen or sorted(
                eligible,
                key=lambda item: positions.get(f"{item['bank']['id']}:{item['question']['id']}", -1),
            )
            preferred = next(
                (
                    item for item in hard_candidates
                    if str(item["question"].get("id", "")) == preferred_question_id
                ),
                None,
            )
            selected = preferred or max(
                pool,
                key=lambda item: (
                    item["score"],
                    -int(item["question"].get("sequence") or 0),
                ),
            )
            bank = selected["bank"]
            question = selected["question"]
            if matches(
                selected,
                require_all_topics=True,
                require_type=True,
                require_difficulty=True,
                require_all_details=True,
            ):
                match_status = "exact"
            elif matches(
                selected,
                require_all_topics=True,
                require_type=True,
                require_difficulty=False,
                require_all_details=False,
            ):
                match_status = "relaxed"
            elif matches(
                selected,
                require_all_topics=False,
                require_type=True,
                require_difficulty=False,
                require_all_details=False,
            ):
                match_status = "partial"
            else:
                match_status = "fallback"
            public_question = self.homework_store.public_question_for_recommendation(
                str(bank["id"]), str(question["id"]), student_id=student_id
            )
            selected_profile = selected["profile"]
            relaxed_conditions = []
            if requested_topics and not requested_topics.issubset(set(selected_profile.get("knowledge_points", []))):
                relaxed_conditions.append("knowledge_points")
            if requested_type and selected_profile.get("question_type") != requested_type:
                relaxed_conditions.append("question_type")
            if requested_difficulty and selected_profile.get("difficulty") != requested_difficulty:
                relaxed_conditions.append("difficulty")
            for field, requested_values in (
                ("skills", requested_skills),
                ("components", requested_components),
                ("methods", requested_methods),
                ("circuit_functions", requested_functions),
                ("tasks", requested_tasks),
            ):
                if requested_values and not requested_values.issubset(set(selected_profile.get(field, []))):
                    relaxed_conditions.append(field)
            if requested_chapter and not _profile_matches_chapter(
                selected_profile, requested_chapter
            ):
                relaxed_conditions.append("chapter")
            if requested_figure is not None and bool(question.get("figures")) is not bool(requested_figure):
                relaxed_conditions.append("requires_figure")
            record = {
                "question_bank_id": bank["id"],
                "question_id": question["id"],
                "requirements": requirements,
                "recommended_at": _now(),
            }
            history[student_id] = [*recent, record][-100:]
            self._write_history(history)
        profile = selected["profile"]
        matched = [
            item for item in requirements.get("knowledge_points", [])
            if item in profile.get("knowledge_points", [])
        ] or profile.get("knowledge_points", [])[:3]
        evidence: list[str] = []
        if normalized_scope:
            evidence.append(
                "检索章节：" + "、".join(
                    str(item.get("chapter") or item.get("chapter_key"))
                    for item in normalized_scope
                )
            )
        if matched:
            evidence.append("知识点：" + "、".join(matched))
        if requirements.get("question_type") == profile.get("question_type"):
            evidence.append("题型符合要求")
        if requirements.get("difficulty") == profile.get("difficulty"):
            evidence.append("难度符合要求")
        if profile.get("chapter"):
            evidence.append("原书章节：" + str(profile["chapter"]))
        intent_label = str(
            requirements.get("agent_analysis", {}).get("intent_summary", "")
        ).strip() or query.strip().splitlines()[0][:80]
        coverage = (
            "实际共同知识点为" + "、".join(matched)
            if matched
            else "当前只能确认它是完整上下文语义检索中的最高分候选"
        )
        deterministic_reason = (
            (
                f"这道题完整满足“{intent_label[:80]}”的已识别条件，"
                if match_status == "exact"
                else f"题库中没有完全满足全部条件的可用题目，本题是当前最接近“{intent_label[:80]}”的候选，"
            )
            + coverage
            + "。"
        )
        return {
            "question_ref": {
                "kind": "question_bank",
                "question_bank_id": bank["id"],
                "question_id": question["id"],
            },
            "question": public_question,
            "reference_available": True,
            "source": {
                "book_title": bank.get("title", ""),
                "source_name": bank.get("source_name", ""),
                "number": question.get("number", ""),
                "page_start": question.get("page_start"),
                "page_end": question.get("page_end"),
                "chapter": profile.get("chapter", ""),
                "section": profile.get("section", ""),
            },
            "profile": profile,
            "requirements": requirements,
            "chapter_scope": normalized_scope,
            "reason": agent_reason.strip()[:500] if preferred and agent_reason.strip() else deterministic_reason,
            "evidence": _clean_list(agent_evidence, 8) if preferred and agent_evidence else evidence,
            "score_summary": selected["score_components"],
            "match_status": match_status,
            "relaxed_conditions": list(dict.fromkeys(relaxed_conditions)),
            "selection_method": "agent_rerank" if preferred else "deterministic_fallback",
            "agent_analysis": requirements["agent_analysis"],
        }
