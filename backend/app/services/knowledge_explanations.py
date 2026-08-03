from __future__ import annotations

import hashlib
import json
import re
import shutil
import threading
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse, urlunparse
from uuid import uuid4

import httpx

from backend.app.config import settings


ACTIVE_STATUSES = {"planning", "generating"}
TERMINAL_STATUSES = {"completed", "cancelled", "error"}


class KnowledgeExplanationError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def qwen_image_endpoint(base_url: str) -> str:
    """Convert a DashScope compatible-mode URL to the native image endpoint."""

    value = base_url.strip()
    if not value:
        return settings.qwen_image_endpoint
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Qwen Image API Base URL 必须是有效的 HTTP(S) 地址")
    if parsed.path.rstrip("/").endswith(
        "/services/aigc/multimodal-generation/generation"
    ):
        return value.rstrip("/")
    return urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            "/api/v1/services/aigc/multimodal-generation/generation",
            "",
            "",
            "",
        )
    )


class QwenImageClient:
    """Small async client for the DashScope native Qwen Image 2.0 API."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = settings.qwen_image_model,
        endpoint: str = settings.qwen_image_endpoint,
        timeout: float = settings.qwen_image_timeout_seconds,
        transport: httpx.AsyncBaseTransport | None = None,
        download_transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("知识讲解需要配置 Qwen API Key")
        if not model.startswith("qwen-image-2.0"):
            raise ValueError("知识讲解当前仅支持 Qwen Image 2.0 系列模型")
        self.model = model.strip()
        self.endpoint = qwen_image_endpoint(endpoint)
        self._api_key = api_key.strip()
        request_timeout = httpx.Timeout(timeout, connect=min(timeout, 20.0))
        self._client = httpx.AsyncClient(
            timeout=request_timeout,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            transport=transport,
        )
        # Never forward the DashScope API key to the temporary OSS download host.
        self._download_client = httpx.AsyncClient(
            timeout=request_timeout,
            follow_redirects=True,
            transport=download_transport,
        )

    async def close(self) -> None:
        await self._client.aclose()
        await self._download_client.aclose()

    async def generate(
        self,
        prompt: str,
        *,
        seed: int,
        size: str = settings.qwen_image_size,
    ) -> bytes:
        payload = {
            "model": self.model,
            "input": {
                "messages": [
                    {"role": "user", "content": [{"text": prompt}]},
                ]
            },
            "parameters": {
                "negative_prompt": (
                    "低分辨率，文字模糊，错别字，乱码，伪文字，公式错误，文字重叠，"
                    "内容被裁切，重复模块，错误编号，多余卡片，空白卡片，擅自添加教学目标，"
                    "拥挤杂乱，过度留白，低对比度，3D写实人物，水印，品牌标志，二维码"
                ),
                "prompt_extend": False,
                "watermark": False,
                "size": size,
                "n": 1,
                "seed": seed,
            },
        }
        try:
            response = await self._client.post(self.endpoint, json=payload)
        except httpx.HTTPError as exc:
            raise KnowledgeExplanationError(f"无法连接 Qwen Image 服务：{exc}") from exc
        if response.is_error:
            raise KnowledgeExplanationError(
                f"Qwen Image 请求失败 ({response.status_code})：{self._error_detail(response)}"
            )
        try:
            body = response.json()
            choices = body["output"]["choices"]
            content = choices[0]["message"]["content"]
            image_url = next(
                str(item["image"])
                for item in content
                if isinstance(item, dict) and item.get("image")
            )
        except (KeyError, IndexError, StopIteration, TypeError, ValueError) as exc:
            raise KnowledgeExplanationError("Qwen Image 返回了无法识别的图片结果") from exc
        if not image_url.startswith("https://"):
            raise KnowledgeExplanationError("Qwen Image 返回了不安全的图片下载地址")
        try:
            image_response = await self._download_client.get(image_url)
            image_response.raise_for_status()
        except httpx.HTTPError as exc:
            raise KnowledgeExplanationError(f"生成成功，但下载讲解页失败：{exc}") from exc
        image_bytes = image_response.content
        if not image_bytes or len(image_bytes) > 30 * 1024 * 1024:
            raise KnowledgeExplanationError("生成的讲解页文件为空或超过 30 MB")
        content_type = image_response.headers.get("content-type", "").lower()
        if "image" not in content_type and not image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            raise KnowledgeExplanationError("Qwen Image 下载结果不是有效图片")
        return image_bytes

    @staticmethod
    def _error_detail(response: httpx.Response) -> str:
        try:
            body = response.json()
            return str(body.get("message") or body.get("code") or "未知错误")[:400]
        except (TypeError, ValueError):
            return (response.text or response.reason_phrase)[:400]


class KnowledgeExplanationStore:
    def __init__(self, root_dir: Path = settings.root_dir) -> None:
        self.root = root_dir / "data" / "knowledge_explanations"
        self._lock = threading.RLock()

    @staticmethod
    def _safe_id(value: str, *, label: str) -> str:
        normalized = value.strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,96}", normalized):
            raise ValueError(f"{label}不合法")
        return normalized

    def _task_dir(self, task_id: str) -> Path:
        normalized = self._safe_id(task_id, label="讲解任务标识")
        return self.root / normalized

    def _manifest_path(self, task_id: str) -> Path:
        return self._task_dir(task_id) / "manifest.json"

    def create(
        self,
        *,
        student_id: str,
        question: str,
        requested_page_count: int,
        text_model: str,
        image_model: str,
    ) -> dict[str, Any]:
        student_id = self._safe_id(student_id, label="学生标识")
        task_id = uuid4().hex
        now = _utc_now()
        record: dict[str, Any] = {
            "id": task_id,
            "student_id": student_id,
            "question": question.strip(),
            "status": "planning",
            "progress": 2,
            "message": "正在分析问题并规划讲解大纲…",
            "title": "",
            "subtitle": "",
            "requested_page_count": requested_page_count,
            "page_count": 0,
            "text_model": text_model,
            "image_model": image_model,
            "pages": [],
            "error": "",
            "created_at": now,
            "updated_at": now,
        }
        with self._lock:
            self._write(record)
        return self.public_record(record)

    def get(self, task_id: str, student_id: str = "") -> dict[str, Any]:
        with self._lock:
            record = self._read(task_id)
        if student_id and record.get("student_id") != self._safe_id(
            student_id, label="学生标识"
        ):
            raise FileNotFoundError(task_id)
        return self.public_record(record)

    def raw(self, task_id: str) -> dict[str, Any]:
        with self._lock:
            return self._read(task_id)

    def list(self, student_id: str, *, limit: int = 12) -> list[dict[str, Any]]:
        student_id = self._safe_id(student_id, label="学生标识")
        with self._lock:
            if not self.root.exists():
                return []
            records: list[dict[str, Any]] = []
            for manifest in self.root.glob("*/manifest.json"):
                try:
                    record = json.loads(manifest.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError, TypeError):
                    continue
                if record.get("student_id") == student_id:
                    records.append(record)
        records.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
        return [self.public_record(item) for item in records[: max(1, min(limit, 50))]]

    def update(self, task_id: str, **changes: Any) -> dict[str, Any]:
        with self._lock:
            record = self._read(task_id)
            record.update(changes)
            record["updated_at"] = _utc_now()
            self._write(record)
        return self.public_record(record)

    def update_page(self, task_id: str, page_index: int, **changes: Any) -> dict[str, Any]:
        with self._lock:
            record = self._read(task_id)
            pages = record.get("pages") or []
            if page_index < 1 or page_index > len(pages):
                raise IndexError("讲解页序号不存在")
            pages[page_index - 1].update(changes)
            record["pages"] = pages
            record["updated_at"] = _utc_now()
            self._write(record)
        return self.public_record(record)

    def save_page_image(self, task_id: str, page_index: int, image_bytes: bytes) -> str:
        task_dir = self._task_dir(task_id)
        task_dir.mkdir(parents=True, exist_ok=True)
        filename = f"page-{page_index:02d}.png"
        target = task_dir / filename
        temporary = task_dir / f".{filename}.tmp"
        temporary.write_bytes(image_bytes)
        temporary.replace(target)
        return filename

    def page_file(self, task_id: str, page_index: int, student_id: str) -> Path:
        record = self.get(task_id, student_id)
        pages = record.get("pages") or []
        if page_index < 1 or page_index > len(pages):
            raise FileNotFoundError(task_id)
        filename = str(pages[page_index - 1].get("image_file", ""))
        if not re.fullmatch(r"page-\d{2}\.png", filename):
            raise FileNotFoundError(task_id)
        path = self._task_dir(task_id) / filename
        if not path.is_file():
            raise FileNotFoundError(task_id)
        return path

    def recover_interrupted(self) -> None:
        with self._lock:
            if not self.root.exists():
                return
            for manifest in self.root.glob("*/manifest.json"):
                try:
                    record = json.loads(manifest.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError, TypeError):
                    continue
                if record.get("status") not in ACTIVE_STATUSES:
                    continue
                record.update(
                    {
                        "status": "error",
                        "message": "服务重启导致生成中断，请重新生成",
                        "error": "服务重启导致生成中断",
                        "updated_at": _utc_now(),
                    }
                )
                self._write(record)

    def delete(self, task_id: str, student_id: str) -> None:
        self.get(task_id, student_id)
        task_dir = self._task_dir(task_id).resolve()
        root = self.root.resolve()
        if task_dir.parent != root:
            raise ValueError("讲解任务路径不合法")
        with self._lock:
            shutil.rmtree(task_dir)

    def public_record(self, value: dict[str, Any]) -> dict[str, Any]:
        record = deepcopy(value)
        task_id = quote(str(record.get("id", "")), safe="")
        student_id = quote(str(record.get("student_id", "")), safe="")
        for page in record.get("pages") or []:
            page_index = int(page.get("index", 0) or 0)
            if page.get("image_file") and page_index:
                page["image_url"] = (
                    f"/api/knowledge-explanations/{task_id}/pages/{page_index}"
                    f"?student_id={student_id}"
                )
            else:
                page["image_url"] = ""
        return record

    def _read(self, task_id: str) -> dict[str, Any]:
        path = self._manifest_path(task_id)
        if not path.is_file():
            raise FileNotFoundError(task_id)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            raise FileNotFoundError(task_id) from exc
        if not isinstance(value, dict):
            raise FileNotFoundError(task_id)
        return value

    def _write(self, record: dict[str, Any]) -> None:
        task_dir = self._task_dir(str(record["id"]))
        task_dir.mkdir(parents=True, exist_ok=True)
        target = task_dir / "manifest.json"
        temporary = task_dir / ".manifest.json.tmp"
        temporary.write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(target)


class KnowledgeExplanationService:
    def __init__(self, store: KnowledgeExplanationStore) -> None:
        self.store = store

    async def generate(
        self,
        task_id: str,
        *,
        question: str,
        requested_page_count: int,
        text_client: Any,
        image_client: QwenImageClient,
    ) -> dict[str, Any]:
        self.store.update(
            task_id,
            status="planning",
            progress=4,
            message="正在根据问题难度划分讲解模块…",
        )
        plan = await self._create_plan(question, requested_page_count, text_client)
        pages = [
            {
                "index": index,
                "title": item["title"],
                "subtitle": item["subtitle"],
                "learning_goal": item["learning_goal"],
                "sections": [],
                "key_takeaway": "",
                "status": "pending",
                "message": "等待生成",
                "image_file": "",
            }
            for index, item in enumerate(plan["pages"], start=1)
        ]
        self.store.update(
            task_id,
            title=plan["title"],
            subtitle=plan["subtitle"],
            page_count=len(pages),
            pages=pages,
            status="generating",
            progress=10,
            message=f"大纲已完成，开始生成第 1/{len(pages)} 页…",
        )
        seed = int(hashlib.sha256(question.encode("utf-8")).hexdigest()[:8], 16)
        for page in pages:
            page_index = int(page["index"])
            base_progress = 10 + int((page_index - 1) * 88 / len(pages))
            self.store.update_page(
                task_id,
                page_index,
                status="writing",
                message="正在组织本页内容与图示…",
            )
            self.store.update(
                task_id,
                progress=base_progress,
                message=f"正在编排第 {page_index}/{len(pages)} 页内容…",
            )
            detail = await self._create_page_detail(
                question=question,
                plan=plan,
                page=page,
                text_client=text_client,
            )
            self.store.update_page(
                task_id,
                page_index,
                sections=detail["sections"],
                key_takeaway=detail["key_takeaway"],
                status="drawing",
                message="Qwen Image 2.0 正在绘制…",
            )
            self.store.update(
                task_id,
                progress=min(96, base_progress + 4),
                message=f"Qwen Image 2.0 正在生成第 {page_index}/{len(pages)} 页…",
            )
            prompt = build_page_prompt(
                lesson_title=plan["title"],
                lesson_subtitle=plan["subtitle"],
                page_count=len(pages),
                page={**page, **detail},
            )
            image_bytes = await image_client.generate(
                prompt,
                seed=(seed + page_index * 7919) % 2_147_483_647,
            )
            filename = self.store.save_page_image(task_id, page_index, image_bytes)
            self.store.update_page(
                task_id,
                page_index,
                status="ready",
                message="已生成",
                image_file=filename,
            )
            self.store.update(
                task_id,
                progress=min(98, 10 + int(page_index * 88 / len(pages))),
                message=f"已完成 {page_index}/{len(pages)} 页",
            )
        return self.store.update(
            task_id,
            status="completed",
            progress=100,
            message=f"{len(pages)} 页知识讲解已生成",
        )

    async def _create_plan(
        self, question: str, requested_page_count: int, text_client: Any
    ) -> dict[str, Any]:
        count_instruction = (
            f"必须恰好规划 {requested_page_count} 页。"
            if requested_page_count
            else "请按问题复杂度自行规划 4 到 7 页；简单概念用 4 页，含推导、例题或应用时用 5 到 7 页。"
        )
        prompt = f"""
你是一名擅长知识可视化的课程设计师。请把用户的问题规划成一组连续的中文知识讲解信息图。

用户问题：{question}

要求：
1. {count_instruction}
2. 模块必须针对这个问题动态划分，形成从直觉/背景到核心原理，再到推导、例子、应用或误区的学习闭环；不要机械套模板。
3. 每页只承担一个清晰教学任务，标题短而准确，适合 16:9 紧凑信息图。
4. 最后一页必须完成总结、适用边界或迁移应用，但名称仍要贴合具体主题。
5. 只输出 JSON，不要 Markdown。

JSON 结构：
{{
  "title": "整组讲解总标题，不超过24字",
  "subtitle": "一句话说明学习主线，不超过42字",
  "pages": [
    {{"title": "本页标题，不超过18字", "subtitle": "本页副标题，不超过30字", "learning_goal": "学完本页能回答什么，不超过48字"}}
  ]
}}
""".strip()
        raw = await text_client.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.1,
            json_mode=True,
            reasoning_budget=768,
        )
        return normalize_plan(_json_object(raw), question, requested_page_count)

    async def _create_page_detail(
        self,
        *,
        question: str,
        plan: dict[str, Any],
        page: dict[str, Any],
        text_client: Any,
    ) -> dict[str, Any]:
        outline = "；".join(
            f"{index + 1}.{item['title']}"
            for index, item in enumerate(plan["pages"])
        )
        prompt = f"""
你正在为一组中文知识讲解信息图撰写第 {page['index']}/{len(plan['pages'])} 页的精确文案。

用户问题：{question}
整组标题：{plan['title']}
讲解顺序：{outline}
本页标题：{page['title']}
本页副标题：{page['subtitle']}
本页目标：{page['learning_goal']}

请给出 3 到 5 个紧凑内容模块。模块类型应随知识点变化，可使用概念、公式、图解、对比、步骤、例题、工程意义、误区等；不要重复前后页内容。
每个正文最多 58 个汉字，优先使用准确术语、必要公式和单位；公式写成普通可读文本或 LaTeX，不要编造数值和结论。
visual 要具体描述适合本模块的简洁图示，例如坐标曲线、流程箭头、结构剖面、对比表或图标，不要只写“配图”。
只输出 JSON，不要 Markdown：
{{
  "sections": [
    {{"heading": "模块标题，不超过12字", "body": "1至2句准确讲解", "visual": "图示内容，不超过36字", "accent": "blue|green|orange|red"}}
  ],
  "key_takeaway": "页底一句话总结，不超过52字"
}}
""".strip()
        raw = await text_client.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.1,
            json_mode=True,
            reasoning_budget=640,
        )
        return normalize_page_detail(_json_object(raw), page)


def _json_object(raw: str) -> dict[str, Any]:
    text = str(raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise KnowledgeExplanationError("讲解大纲模型未返回有效 JSON")
        try:
            value = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise KnowledgeExplanationError("讲解大纲模型未返回有效 JSON") from exc
    if not isinstance(value, dict):
        raise KnowledgeExplanationError("讲解大纲必须是 JSON 对象")
    return value


def _short(value: Any, limit: int, fallback: str) -> str:
    normalized = re.sub(r"\s+", " ", str(value or "")).strip()
    return (normalized or fallback)[:limit]


def normalize_plan(
    value: dict[str, Any], question: str, requested_page_count: int
) -> dict[str, Any]:
    raw_pages = value.get("pages")
    if not isinstance(raw_pages, list):
        raw_pages = []
    target = requested_page_count or max(4, min(7, len(raw_pages) or 5))
    defaults = [
        ("问题与核心概念", "先明确问题在问什么", "能说清关键对象、条件和目标"),
        ("原理如何运作", "建立直观机制图景", "能用因果链解释核心机制"),
        ("关键关系与推导", "把直觉连接到准确表达", "能读懂关键公式、关系或步骤"),
        ("典型情境演示", "用具体情境检验理解", "能把原理用于一个代表性例子"),
        ("易错点与边界", "区分相近概念和适用条件", "能避开常见误解并判断适用范围"),
        ("应用与迁移", "把知识连接到真实问题", "能迁移到新的问题或工程情境"),
        ("总结与自检", "收束整条学习主线", "能用自己的话复述并完成自检"),
        ("进阶思考", "从结论继续向外延伸", "能提出一个合理的进阶问题"),
    ]
    pages: list[dict[str, str]] = []
    for index in range(target):
        raw = raw_pages[index] if index < len(raw_pages) and isinstance(raw_pages[index], dict) else {}
        default = defaults[min(index, len(defaults) - 1)]
        pages.append(
            {
                "title": _short(raw.get("title"), 18, default[0]),
                "subtitle": _short(raw.get("subtitle"), 30, default[1]),
                "learning_goal": _short(raw.get("learning_goal"), 48, default[2]),
            }
        )
    topic = _short(question, 24, "知识讲解")
    return {
        "title": _short(value.get("title"), 24, f"{topic}详解"),
        "subtitle": _short(value.get("subtitle"), 42, "从核心问题出发，建立可迁移的理解框架"),
        "pages": pages,
    }


def normalize_page_detail(
    value: dict[str, Any], page: dict[str, Any]
) -> dict[str, Any]:
    raw_sections = value.get("sections")
    if not isinstance(raw_sections, list):
        raw_sections = []
    accents = {"blue", "green", "orange", "red"}
    sections: list[dict[str, str]] = []
    for index, raw in enumerate(raw_sections[:5]):
        if not isinstance(raw, dict):
            continue
        body = _short(raw.get("body"), 90, "")
        if not body:
            continue
        accent = str(raw.get("accent", "blue")).lower()
        sections.append(
            {
                "heading": _short(raw.get("heading"), 12, f"要点 {index + 1}"),
                "body": body,
                "visual": _short(raw.get("visual"), 48, "简洁概念关系图"),
                "accent": accent if accent in accents else "blue",
            }
        )
    if len(sections) < 3:
        raise KnowledgeExplanationError(f"“{page['title']}”的页面文案不完整，请重新生成")
    return {
        "sections": sections,
        "key_takeaway": _short(
            value.get("key_takeaway"),
            64,
            f"掌握{page['title']}，就能回答：{page['learning_goal']}",
        ),
    }


def build_page_prompt(
    *,
    lesson_title: str,
    lesson_subtitle: str,
    page_count: int,
    page: dict[str, Any],
) -> str:
    chinese_ordinals = "一二三四五六七八九十"
    page_index = int(page["index"])
    ordinal = chinese_ordinals[page_index - 1] if 1 <= page_index <= 10 else str(page_index)
    visible_modules = "\n".join(
        (
            f"模块{index}：标题“{section['heading']}”；正文“{section['body']}”。"
        )
        for index, section in enumerate(page["sections"], start=1)
    )
    visual_modules = "\n".join(
        (
            f"模块{index}配图：{section['visual']}；{section['accent']}标题条。"
        )
        for index, section in enumerate(page["sections"], start=1)
    )
    module_count = len(page["sections"])
    return f"""
【01 最终成品】
横向 16:9 中文理工科知识信息图，单页教学幻灯片；紧凑、清晰、图文并茂，不是网页截图。

【02 可见信息与层级】
以下是画面中唯一允许出现的文案白名单，其他简报说明不得入图。须逐字准确，不得改写、增删、重复或造字。

一级信息：左上角页码“{page_index}/{page_count}”；顶部居中主标题“{lesson_title}（{ordinal}）：{page['title']}”。
二级信息：主标题正下方副标题“{page['subtitle']}”。
三级信息：中部必须恰好放置 {module_count} 张内容卡片，每张卡片只出现一次且编号连续：
{visible_modules}
四级信息：底部通栏总结“{page['key_takeaway']}”。

位置：标题区顶部 14%，卡片区中部 72%，深蓝总结条底部 10%。不得增加“教学目标”“总结”“10%”等标签或额外卡片。

【03 语言和文字优先级】
只用简体中文，准确性高于装饰。优先保证主标题、页码、编号、公式和总结，再保证卡片文字。中文用清晰无衬线体；公式保留符号、上下标与括号。正文不小于视觉 18px。图内仅写明示变量、坐标和数值，不生成伪文字。

【04 视觉方向与必须保留的细节】
白到浅蓝灰背景，海军蓝主标题、钴蓝分区标题、细蓝边圆角卡片、12 栏网格。{module_count} 个模块排成 2 至 3 列并对齐；少量红、绿、橙强调。

每张卡片须有标题、正文及相邻二维矢量图；图约 35%，文字约 65%。绘图说明不得整句入图：
{visual_modules}

背景仅可有淡线稿。不要照片、人物、3D、品牌、水印、二维码或版权信息；不得裁字、压字。
""".strip()
