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
PAGE_LAYOUTS = (
    "concept-map",
    "process-flow",
    "comparison",
    "formula-focus",
    "system-diagram",
    "case-walkthrough",
)
PAGE_LAYOUT_INSTRUCTIONS = {
    "concept-map": "以一个核心概念为视觉锚点，用分支、连线和邻近注释表达概念之间的关系。",
    "process-flow": "采用清晰的方向性信息流，用步骤节点、因果箭头或状态变化表现过程。",
    "comparison": "采用左右或上下对照结构，突出共同条件、关键差异和判断结论。",
    "formula-focus": "以关键公式、曲线或坐标图为主视觉，变量解释和推导关系环绕主视觉展开。",
    "system-diagram": "用占据主要画面的结构图、剖面图或系统框图承载知识，文字作为精确标注。",
    "case-walkthrough": "从具体情境、例题或工程现象切入，按问题、分析、结果组织视觉叙事。",
}
VISUAL_TYPES = ("general", "circuit", "curve", "formula-derivation")
VISUAL_HIERARCHIES = ("primary", "secondary", "supporting")
CONTENT_GENERATION_ATTEMPTS = 2
DANGLING_TEXT_ENDINGS = (
    "的",
    "之",
    "并",
    "且",
    "但",
    "为",
    "将",
    "把",
    "被",
    "由",
    "向",
    "从",
    "在",
    "如",
    "例如",
    "即",
    "则",
    "若",
    "使",
    "让",
    "旁标",
    "标注",
    "指向",
    "包含",
    "包括",
    "通过",
    "如下",
    "分别为",
    "依次为",
)
SPECIALIZED_VISUAL_INSTRUCTIONS = {
    "circuit": (
        "电路图约束：只绘制文案或绘图说明明确给出的器件、节点和连接关系；使用规范二维电路符号、"
        "直线导线与实心连接点，导线交叉但不连接时不得加连接点。电源、地、器件极性以及电压/电流"
        "箭头方向必须前后一致。若拓扑信息不足，改画功能框图，不得猜测或补造完整电路。电路内部"
        "只标说明中已经出现的标准变量和元件代号，不添加说明性中文或伪文字。"
    ),
    "curve": (
        "曲线图约束：横轴、纵轴分别对应什么物理量及单位必须明确且互不混用；仅使用文案给出的"
        "数值、拐点和阈值。若未给出轴变量、单位或关键数值，只画定性坐标箭头和趋势，不得自行添加"
        "3 dB、fL、峰值或刻度。曲线的单调性、极值、渐近线和关键区域必须与正文公式及结论一致，"
        "图例颜色与曲线一一对应。"
    ),
    "formula-derivation": (
        "公式推导约束：公式必须逐字符忠实复制正文，保留负号、分式、括号、上下标和希腊字母；"
        "按“已知关系→等价变形→结果”的方向排列，每个等号或箭头两侧必须数学等价。不得引入正文"
        "未定义的变量、数值或中间结论；若正文没有提供完整中间步骤，只展示关系和结果，不得伪造"
        "推导。公式字号高于普通正文，变量说明紧邻首次出现的位置。"
    ),
}


class KnowledgeExplanationError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def qwen_image_model_label(model: str) -> str:
    return "Qwen Image 2.0 Pro" if model.strip().endswith("-pro") else "Qwen Image 2.0"


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
                    "内容被裁切，重复内容，错误编号，多余卡片，空白卡片，擅自添加教学目标，"
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

    def mark_terminal(
        self,
        task_id: str,
        *,
        status: str,
        message: str,
        error: str = "",
    ) -> dict[str, Any]:
        if status not in {"cancelled", "error"}:
            raise ValueError("讲解任务终止状态不合法")
        page_message = "生成已取消" if status == "cancelled" else "本页未能生成"
        with self._lock:
            record = self._read(task_id)
            self._mark_unfinished_pages(record, status=status, message=page_message)
            record.update(
                {
                    "status": status,
                    "message": message,
                    "error": error,
                    "updated_at": _utc_now(),
                }
            )
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
                self._mark_unfinished_pages(
                    record,
                    status="error",
                    message="服务重启，本页未能生成",
                )
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
            # The full prompt remains in the manifest for generation audits, but
            # returning it with every polling/history response would be wasteful.
            page.pop("image_prompt", None)
            page_index = int(page.get("index", 0) or 0)
            page.setdefault(
                "layout", PAGE_LAYOUTS[(max(page_index, 1) - 1) % len(PAGE_LAYOUTS)]
            )
            if page.get("image_file") and page_index:
                page["image_url"] = (
                    f"/api/knowledge-explanations/{task_id}/pages/{page_index}"
                    f"?student_id={student_id}"
                )
            else:
                page["image_url"] = ""
        return record

    @staticmethod
    def _mark_unfinished_pages(
        record: dict[str, Any], *, status: str, message: str
    ) -> None:
        pages = record.get("pages") or []
        for page in pages:
            if page.get("status") == "ready":
                continue
            page["status"] = status
            page["message"] = message
        record["pages"] = pages

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
            message="正在理解问题并规划讲解内容…",
        )
        plan = await self._create_plan(question, requested_page_count, text_client)
        pages = [
            {
                "index": index,
                "title": item["title"],
                "subtitle": item["subtitle"],
                "learning_goal": item["learning_goal"],
                "content_brief": item["content_brief"],
                "visual_focus": item["visual_focus"],
                "covers": item["covers"],
                "layout": item["layout"],
                "sections": [],
                "key_takeaway": "",
                "visual_layout": {},
                "image_prompt": "",
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
            requirements=plan["requirements"],
            page_count=len(pages),
            pages=pages,
            status="generating",
            progress=10,
            message=f"大纲已完成，开始规划第 1/{len(pages)} 页具体内容…",
        )
        # Finish the instructional content for every page before any visual layout
        # or image prompt is designed. This keeps the later stages from silently
        # changing the teaching plan to fit an early visual idea.
        for page in pages:
            page_index = int(page["index"])
            base_progress = 10 + int((page_index - 1) * 30 / len(pages))
            self.store.update_page(
                task_id,
                page_index,
                status="writing",
                message="正在组织本页内容与图示…",
            )
            self.store.update(
                task_id,
                progress=base_progress,
                message=f"正在规划第 {page_index}/{len(pages)} 页具体内容…",
            )
            detail = await self._create_page_detail(
                question=question,
                plan=plan,
                page=page,
                text_client=text_client,
            )
            page.update(detail)
            self.store.update_page(
                task_id,
                page_index,
                sections=detail["sections"],
                key_takeaway=detail["key_takeaway"],
                message="本页内容已确认，等待整组布局设计…",
            )
            self.store.update(
                task_id,
                progress=10 + int(page_index * 30 / len(pages)),
                message=f"已完成 {page_index}/{len(pages)} 页内容规划",
            )

        self.store.update(
            task_id,
            progress=42,
            message="页面内容已确认，正在设计整组图片布局…",
        )
        for page in pages:
            self.store.update_page(
                task_id,
                int(page["index"]),
                status="writing",
                message="正在设计主视觉、内容分区与阅读动线…",
            )
        visual_layouts = await self._create_visual_layouts(
            question=question,
            plan=plan,
            pages=pages,
            text_client=text_client,
        )
        for page, visual_layout in zip(pages, visual_layouts):
            page["visual_layout"] = visual_layout
            self.store.update_page(
                task_id,
                int(page["index"]),
                visual_layout=visual_layout,
                message="图片布局已确认，等待生成专属提示词…",
            )
        self.store.update(
            task_id,
            progress=48,
            message="整组图片布局已完成，开始编译逐页提示词…",
        )

        seed = int(hashlib.sha256(question.encode("utf-8")).hexdigest()[:8], 16)
        image_model_label = qwen_image_model_label(image_client.model)
        for page in pages:
            page_index = int(page["index"])
            base_progress = 48 + int((page_index - 1) * 50 / len(pages))
            self.store.update_page(
                task_id,
                page_index,
                status="writing",
                message="正在按已确认布局编写本页专属生图提示词…",
            )
            self.store.update(
                task_id,
                progress=base_progress,
                message=f"正在生成第 {page_index}/{len(pages)} 页专属提示词…",
            )
            drafted_prompt = await self._create_image_prompt(
                question=question,
                plan=plan,
                page=page,
                text_client=text_client,
            )
            prompt = build_page_prompt(
                lesson_title=plan["title"],
                lesson_subtitle=plan["subtitle"],
                page_count=len(pages),
                page=page,
                drafted_prompt=drafted_prompt,
            )
            self.store.update_page(
                task_id,
                page_index,
                image_prompt=prompt,
                status="drawing",
                message=f"{image_model_label} 正在绘制…",
            )
            self.store.update(
                task_id,
                progress=min(96, base_progress + 4),
                message=f"{image_model_label} 正在生成第 {page_index}/{len(pages)} 页…",
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
                progress=min(98, 48 + int(page_index * 50 / len(pages))),
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
            else "请按回答这个问题实际需要的内容自行规划 4 到 7 页；以讲清问题所需的最少页数为准，不为凑页数填充无关内容。"
        )
        prompt = f"""
你是一名擅长知识可视化的课程内容设计师。请先理解用户真正想弄清的问题，再规划一组连续的中文知识讲解信息图。

用户问题：{question}

要求：
1. {count_instruction}
2. 讲解内容、先后顺序和每页任务完全依据用户问题与知识依赖决定。不要默认套用“概念→原理→公式→案例→工程应用→总结”，也不要为了结构完整加入问题不需要的背景、应用、误区或例题。
3. 先找出回答问题不可缺少的知识关系，再把它们组合为页面；允许多页连续推导、连续对比或连续解释同一机制，只要这是最清楚的讲法。
4. requirements 必须逐项拆出用户明确要求解释的对象、关系、条件、观察角度和输出形式，使用 R1、R2……编号；不能把“结合直流通路、交流等效电路和频率响应”等要求合并或遗漏。每页 covers 必须列出实际覆盖的 requirements id，所有覆盖项至少被一页实质性讲解。
5. 每页只承担一个明确任务。content_brief 要写清本页必须讲到的具体结论、条件、公式或关系，并实质覆盖 covers 对应内容；visual_focus 要写清最能帮助理解的主视觉，而不是泛泛写“配图”。
6. 标题短而准确，适合 16:9 紧凑信息图。总标题和页面标题必须是完整短语或完整句子；若需要缩短，应完整改写，不得截断词尾或保留半句话。页面标题只写本页具体主题，不重复总标题、不自行添加“（一）”等页序。
7. 从 concept-map、process-flow、comparison、formula-focus、system-diagram、case-walkthrough 中选择真正适合本页内容的 layout；相邻页可以相同，不为追求变化而牺牲内容表达。
8. 所有字段必须在规定字数内写成完整语句或完整短语，不得以“的、之、与、旁标、标注、显示、绘制”等残缺词语结尾。
9. 只输出 JSON，不要 Markdown。

JSON 结构：
{{
  "title": "整组讲解总标题，建议不超过24字且语义完整",
  "subtitle": "一句话说明这组页面如何回答用户问题，不超过42字",
  "requirements": [
    {{"id": "R1", "content": "用户问题中一个不可遗漏的具体要求，不超过80字"}}
  ],
  "pages": [
    {{"title": "本页标题，建议不超过18字且语义完整", "subtitle": "本页副标题，不超过30字", "learning_goal": "学完本页能回答什么，不超过48字", "content_brief": "本页必须覆盖的具体内容、条件与结论，不超过120字", "visual_focus": "本页最重要的主视觉及其信息关系，不超过80字", "covers": ["R1"], "layout": "六种 layout 之一"}}
  ]
}}
""".strip()
        feedback = ""
        for attempt in range(CONTENT_GENERATION_ATTEMPTS):
            retry_instruction = (
                "\n\n上一次结果未通过校验，必须针对以下问题完整重写，不要只做局部补丁：\n"
                + feedback
                if feedback
                else ""
            )
            raw = await text_client.chat(
                [{"role": "user", "content": prompt + retry_instruction}],
                temperature=0.1,
                json_mode=True,
                reasoning_budget=768,
            )
            try:
                plan = normalize_plan(
                    _json_object(raw), question, requested_page_count
                )
            except KnowledgeExplanationError as exc:
                feedback = str(exc)
                continue
            try:
                passed, feedback = await self._review_plan_coverage(
                    question=question,
                    plan=plan,
                    text_client=text_client,
                )
            except KnowledgeExplanationError as exc:
                feedback = f"审查结果无效：{exc}"
                continue
            if passed:
                return plan
        raise KnowledgeExplanationError(
            "讲解内容规划连续两次未通过问题覆盖与语义完整性校验：" + feedback
        )

    async def _review_plan_coverage(
        self,
        *,
        question: str,
        plan: dict[str, Any],
        text_client: Any,
    ) -> tuple[bool, str]:
        prompt = f"""
你是独立的知识讲解内容审查员。不要替规划辩护，要逐字核对用户问题中的每一个对象、因果关系、限定条件、要求采用的分析视角和输出要求，判断规划是否都有实质性页面内容承接。

用户问题：{question}
待审规划：
{json.dumps(plan, ensure_ascii=False, indent=2)}

审查标准：
1. requirements 是否完整拆出用户的全部明确要求，不能用宽泛词语掩盖遗漏。
2. 每个 requirement 是否至少被一页 covers 引用，且对应 content_brief 确实说明了要讲什么，而不是只在标题或 covers 中挂名。
3. 指定页数较少时仍必须覆盖整个问题；如果一页装不下，应在该页内重新组织，而不是擅自只回答其中一部分。
4. 标题、副标题、学习目标、content_brief 和 visual_focus 必须语义完整，无截断、半句、悬空连接词或未闭合公式。
5. 不因没有套用固定教学顺序而判错，只检查用户问题覆盖、知识依赖和内容完整性。

只输出 JSON：
{{"passed": true, "missing_requirements": [], "issues": []}}
未通过时 passed=false，并用 missing_requirements 列出缺少的具体问题要求，用 issues 列出需要重写的页面和原因。
""".strip()
        raw = await text_client.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            json_mode=True,
            reasoning_budget=512,
        )
        return _review_feedback(_json_object(raw))

    async def _create_page_detail(
        self,
        *,
        question: str,
        plan: dict[str, Any],
        page: dict[str, Any],
        text_client: Any,
    ) -> dict[str, Any]:
        requirement_lookup = {
            item["id"]: item["content"] for item in plan["requirements"]
        }
        page_requirements = [
            {"id": requirement_id, "content": requirement_lookup[requirement_id]}
            for requirement_id in page["covers"]
        ]
        outline = "；".join(
            f"{index + 1}.{item['title']}（{item['content_brief']}）"
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
本页内容边界：{page['content_brief']}
本页主视觉：{page['visual_focus']}
本页必须实质覆盖的问题要点：{json.dumps(page_requirements, ensure_ascii=False)}
本页版式：{page['layout']}（{PAGE_LAYOUT_INSTRUCTIONS[page['layout']]}）

严格在本页内容边界内规划 2 到 6 个内容分区，数量由内容决定，不得为凑版式拆分或补充无关模块。分区可以是推导步骤、对比对象、机制环节、条件、图解、案例数据或结论，不能默认套用概念、原理、应用的固定顺序。
每个正文最多 90 个汉字，优先使用准确术语、必要公式、条件和单位；公式写成普通可读文本或 LaTeX，不要编造数值和结论。
heading 必须是与主题直接相关的自然标题，不得使用“模块1”“要点2”“总结3”等通用编号。
visual 要像给制图人员的指令一样具体，写明对象、空间关系、箭头、标签、坐标轴、公式或强调位置，不要只写“配图”。
visual_type 必须选择 general、circuit、curve、formula-derivation 之一。绘制电路时，visual 必须写清器件、节点、连接、极性和箭头方向，信息不足则选择功能框图；绘制曲线时，必须写清横纵轴物理量、单位、趋势和正文明确给出的关键点；绘制公式推导时，必须写清起始关系、中间等价变形和结果，正文没有中间步骤时不得补造。
所有 heading、body、visual 和 key_takeaway 都必须是完整语句或完整短语，不得因字数限制按字符截断，不得以“的、之、与、旁标、标注、显示、绘制”等残缺词语结尾；括号、引号、公式必须闭合。
只输出 JSON，不要 Markdown：
{{
  "sections": [
    {{"heading": "知识点标题，不超过16字", "body": "1至2句准确讲解", "visual": "可直接执行的详细制图说明，不超过100字", "visual_type": "general|circuit|curve|formula-derivation", "accent": "blue|green|orange|red"}}
  ],
  "key_takeaway": "本页最重要的一句话结论，不超过52字"
}}
""".strip()
        feedback = ""
        for attempt in range(CONTENT_GENERATION_ATTEMPTS):
            retry_instruction = (
                "\n\n上一次本页文案未通过校验，必须针对以下问题完整重写：\n"
                + feedback
                if feedback
                else ""
            )
            raw = await text_client.chat(
                [{"role": "user", "content": prompt + retry_instruction}],
                temperature=0.1,
                json_mode=True,
                reasoning_budget=640,
            )
            try:
                detail = normalize_page_detail(_json_object(raw), page)
            except KnowledgeExplanationError as exc:
                feedback = str(exc)
                continue
            try:
                passed, feedback = await self._review_page_detail(
                    question=question,
                    plan=plan,
                    page=page,
                    detail=detail,
                    page_requirements=page_requirements,
                    text_client=text_client,
                )
            except KnowledgeExplanationError as exc:
                feedback = f"审查结果无效：{exc}"
                continue
            if passed:
                return detail
        raise KnowledgeExplanationError(
            f"“{page['title']}”连续两次未通过内容覆盖与语义完整性校验：{feedback}"
        )

    async def _review_page_detail(
        self,
        *,
        question: str,
        plan: dict[str, Any],
        page: dict[str, Any],
        detail: dict[str, Any],
        page_requirements: list[dict[str, str]],
        text_client: Any,
    ) -> tuple[bool, str]:
        review_payload = {
            "question": question,
            "lesson_title": plan["title"],
            "page_title": page["title"],
            "page_subtitle": page["subtitle"],
            "content_brief": page["content_brief"],
            "visual_focus": page["visual_focus"],
            "required_coverage": page_requirements,
            "detail": detail,
        }
        prompt = f"""
你是独立的单页教学内容审查员。核对下面页面是否真正覆盖本页承担的问题要点，并检查每一句正文和绘图说明是否完整、准确、可执行。

待审内容：
{json.dumps(review_payload, ensure_ascii=False, indent=2)}

审查标准：
1. required_coverage 中每一项都必须在 sections 的正文或关键结论中得到实质解释，不能只出现名词。
2. detail 必须符合 content_brief，不遗漏指定条件、因果链、公式、分析视角或对比关系，也不能越界重复其他页任务。
3. heading、body、visual、key_takeaway 不得有截断、半句、悬空动词或连接词；括号、引号、公式和因果箭头必须闭合。
4. visual 必须给出足够执行的信息。出现“旁标、标注、显示、绘制、指向”等动作时，必须继续写清标什么、显示什么或指向什么。
5. 电路拓扑、曲线坐标轴、公式变量若信息不足，应明确要求简化示意，不能暗示生图模型自行补造。

只输出 JSON：
{{"passed": true, "missing_requirements": [], "issues": []}}
未通过时 passed=false，并具体指出缺失要点或残缺字段，供下一次完整重写。
""".strip()
        raw = await text_client.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            json_mode=True,
            reasoning_budget=512,
        )
        return _review_feedback(_json_object(raw))

    async def _create_visual_layouts(
        self,
        *,
        question: str,
        plan: dict[str, Any],
        pages: list[dict[str, Any]],
        text_client: Any,
    ) -> list[dict[str, Any]]:
        layout_inputs = [
            {
                "index": page["index"],
                "title": page["title"],
                "subtitle": page["subtitle"],
                "layout_archetype": page["layout"],
                "layout_archetype_instruction": PAGE_LAYOUT_INSTRUCTIONS[page["layout"]],
                "visual_focus": page["visual_focus"],
                "sections": page["sections"],
                "key_takeaway": page["key_takeaway"],
            }
            for page in pages
        ]
        prompt = f"""
你是一名专业的中文知识信息图视觉布局设计师。教学内容已经确认；你现在只负责为整组横向 16:9 页面设计画面结构，不能改写、删减、合并或补充任何知识内容，也不能提前撰写生图提示词。

用户原问题：{question}
整组视觉主线：{plan['subtitle']}
已确认的页面内容：
{json.dumps(layout_inputs, ensure_ascii=False, indent=2)}

布局设计要求：
1. 同时观察整组页面，让页码、标题区、字体、线条和基础色彩保持统一；各页主体构图必须由本页知识关系决定，避免连续使用相同的等宽卡片阵列。
2. 每个 section 必须且只能映射到一个 region，section_index 与输入顺序严格一致。不能在布局阶段创造新分区或遗漏已有分区。
3. composition 写清整页主次结构；reading_flow 写清读者视线从哪里开始、经过哪些区域、在哪里收束。
4. 每个 region 必须写清位置、画面占比、视觉层级、该区具体呈现方式以及与其他区域的箭头、连线或邻接关系。视觉层级只能为 primary、secondary、supporting；每页至少有一个且最多两个 primary。
5. 电路、曲线、公式推导或结构示意应获得与理解难度匹配的空间，不能为了排版把主图压缩成装饰；并列比较可使用两个 primary 区域。
6. takeaway_placement 与 takeaway_treatment 要自然收束本页逻辑，不默认使用固定底部通栏。palette_strategy 只描述颜色如何承担语义，decoration 只允许与主题直接相关且不干扰正文的轻量元素。
7. 所有字段必须是完整、可执行的中文短句，不得使用“适当布局”“合理排版”“美观呈现”等空泛表述。
8. 只输出 JSON，不要 Markdown。

JSON 结构：
{{
  "pages": [
    {{
      "index": 1,
      "composition": "整页主体构图与主次关系，不超过100字",
      "reading_flow": "明确的阅读起点、路径与收束点，不超过80字",
      "regions": [
        {{
          "section_index": 1,
          "position": "画布中的具体位置，不超过30字",
          "proportion": "宽高或主体占比，不超过30字",
          "hierarchy": "primary|secondary|supporting",
          "presentation": "本区图文如何组合，不超过120字",
          "connection": "与其他区域的关系或明确说明独立呈现，不超过80字"
        }}
      ],
      "takeaway_placement": "结论放置位置，不超过40字",
      "takeaway_treatment": "结论强调方式，不超过80字",
      "palette_strategy": "本页颜色的语义分工，不超过80字",
      "decoration": "主题装饰及避让原则，不超过80字"
    }}
  ]
}}
""".strip()
        feedback = ""
        for _attempt in range(CONTENT_GENERATION_ATTEMPTS):
            retry_instruction = (
                "\n\n上一次布局未通过结构校验，必须根据以下问题重新设计整组布局：\n"
                + feedback
                if feedback
                else ""
            )
            raw = await text_client.chat(
                [{"role": "user", "content": prompt + retry_instruction}],
                temperature=0.2,
                json_mode=True,
                reasoning_budget=768,
            )
            try:
                return normalize_visual_layouts(_json_object(raw), pages)
            except KnowledgeExplanationError as exc:
                feedback = str(exc)
        raise KnowledgeExplanationError(
            "整组图片布局连续两次未通过结构与内容映射校验：" + feedback
        )

    async def _create_image_prompt(
        self,
        *,
        question: str,
        plan: dict[str, Any],
        page: dict[str, Any],
        text_client: Any,
    ) -> str:
        page_count = len(plan["pages"])
        display_title = page_display_title(
            plan["title"], page["title"], int(page["index"]), page_count
        )
        page_payload = {
            "page_number": f"{page['index']}/{page_count}",
            "display_title": display_title,
            "subtitle": page["subtitle"],
            "content_brief": page["content_brief"],
            "visual_focus": page["visual_focus"],
            "layout": page["layout"],
            "layout_instruction": PAGE_LAYOUT_INSTRUCTIONS[page["layout"]],
            "sections": page["sections"],
            "key_takeaway": page["key_takeaway"],
            "visual_layout": page["visual_layout"],
        }
        prompt = f"""
你是一名中文教育信息图生图提示词工程师。教学内容和视觉布局都已经确认，请把它们编译成一份可直接交给生图模型的完整中文提示词。你只能忠实展开既定方案，不得重新规划、增删或改写教学内容，也不得自行改变区域位置、占比、视觉层级或阅读动线。

用户原问题：{question}
整组讲解主线：{plan['subtitle']}
本页确定内容：
{json.dumps(page_payload, ensure_ascii=False, indent=2)}

输出要求：
1. 只输出最终生图提示词，不要解释、不要 JSON、不要代码围栏。
2. 第一段采用“生成一张「所属领域知识讲解」风格的横向 16:9 信息图，主题为《……》，副标题为‘……’”的形式，并说明主辅色、专业气质和本页布局。
3. 必须依次使用以下 Markdown 结构，内容具体程度参照专业信息图制作说明：
   ### 1. 顶部区域
   明确页码、主标题、副标题、与主题相关但不喧宾夺主的装饰元素。
   ### 2. 主体内容
   严格按 visual_layout 的 composition、reading_flow 和 regions 编排。按本页 sections 的实际数量逐区写成“#### （1）真实知识点标题”，并用 section_index 将 region 与 section 一一对应。每一区都要逐字写入既定 position、proportion、hierarchy、presentation 和 connection，再展开可见标题、可见正文、图形对象、箭头/连线/坐标/公式与颜色强调；不得重新设计成上三下二或等宽卡片。
   ### 3. 结论区
   严格使用 visual_layout 的 takeaway_placement 与 takeaway_treatment 呈现 key_takeaway，不强制底部通栏。
   ### 风格要求
   忠实写入 visual_layout 的 palette_strategy 与 decoration，并明确字体层级、间距、扁平化矢量质感和可读性。
4. display_title、subtitle、每个 section 的 heading/body 以及 key_takeaway 必须逐字出现在提示词中；visual 只能被展开为更具体的绘图说明，不能改变其中的知识关系。
5. visual_layout 的结构说明只用于控制构图，不得作为画面文字。学习目标、content_brief、visual_focus、layout 名称和本段元指令同样不得成为画面文字。不得使用“模块1”“总结4”等机械标签；需要编号时，编号必须与真实知识点标题组合。
6. 电路、曲线和公式推导必须严格服从本页已经给出的对象、拓扑、坐标、变量与数学关系；信息不足时要求画简化示意，不得让生图模型自行补造。
""".strip()
        return await text_client.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.1,
            reasoning_budget=768,
        )


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


def semantic_text_issue(
    value: Any, *, label: str, check_dangling_words: bool = True
) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        return f"{label}为空"
    if text.endswith(
        ("，", ",", "、", "：", ":", "；", ";", "→", "=", "+", "-", "−", "/")
    ):
        return f"{label}以未完成的标点或运算符结尾"
    if check_dangling_words:
        for ending in sorted(DANGLING_TEXT_ENDINGS, key=len, reverse=True):
            if text.endswith(ending):
                return f"{label}以“{ending}”结尾，语义不完整"
    delimiter_pairs = (("（", "）"), ("(", ")"), ("【", "】"), ("[", "]"), ("《", "》"))
    for opening, closing in delimiter_pairs:
        if text.count(opening) != text.count(closing):
            return f"{label}中的括号或书名号不成对"
    if text.count("“") != text.count("”") or text.count("‘") != text.count("’"):
        return f"{label}中的引号不成对"
    return ""


def complete_text(
    value: Any,
    *,
    label: str,
    max_length: int,
    fallback: str = "",
    check_dangling_words: bool = True,
) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip() or fallback
    if len(text) > max_length:
        raise KnowledgeExplanationError(
            f"{label}超过 {max_length} 字，必须完整改写，不能按字符截断"
        )
    issue = semantic_text_issue(
        text, label=label, check_dangling_words=check_dangling_words
    )
    if issue:
        raise KnowledgeExplanationError(issue)
    return text


def _review_feedback(value: Any) -> tuple[bool, str]:
    review = value if isinstance(value, dict) else {}
    missing = review.get("missing_requirements")
    issues = review.get("issues")
    messages = [
        str(item).strip()
        for collection in (missing, issues)
        if isinstance(collection, list)
        for item in collection
        if str(item).strip()
    ]
    passed = review.get("passed") is True and not messages
    return passed, "；".join(messages) or "审查未通过，但审查器未给出具体原因"


def _detected_specialized_visual_types(section: dict[str, Any]) -> list[str]:
    visual_description = " ".join(
        str(section.get(key) or "") for key in ("heading", "visual")
    ).lower()
    formula_description = " ".join(
        str(section.get(key) or "") for key in ("body", "visual")
    ).lower()
    detected: list[str] = []
    if re.search(r"电路|原理图|接线|拓扑|节点|支路|等效图|运放|晶体管|二极管", visual_description):
        detected.append("circuit")
    if re.search(r"曲线|坐标|波形|频响|频率响应|特性图|趋势图|折线|散点", visual_description):
        detected.append("curve")
    if re.search(r"公式|推导|方程|等式|恒等|化简|代入|求解|[=≈∝]|\\frac", formula_description):
        detected.append("formula-derivation")
    return detected


def normalize_visual_type(value: Any, section: dict[str, Any]) -> str:
    candidate = str(value or "").strip().lower().replace("_", "-")
    aliases = {
        "formula": "formula-derivation",
        "derivation": "formula-derivation",
        "schematic": "circuit",
        "plot": "curve",
        "chart": "curve",
    }
    candidate = aliases.get(candidate, candidate)
    if candidate in VISUAL_TYPES and candidate != "general":
        return candidate
    detected = _detected_specialized_visual_types(section)
    return detected[0] if detected else "general"


def normalize_page_layout(value: Any, page_index: int, previous: str = "") -> str:
    del previous  # Kept in the signature for callers that still pass legacy data.
    candidate = str(value or "").strip().lower()
    if candidate not in PAGE_LAYOUTS:
        candidate = PAGE_LAYOUTS[(page_index - 1) % len(PAGE_LAYOUTS)]
    return candidate


def normalize_plan(
    value: dict[str, Any], question: str, requested_page_count: int
) -> dict[str, Any]:
    raw_requirements = value.get("requirements")
    if not isinstance(raw_requirements, list) or not raw_requirements:
        raise KnowledgeExplanationError("讲解大纲缺少用户问题覆盖清单")
    requirements: list[dict[str, str]] = []
    requirement_ids: set[str] = set()
    for index, raw_requirement in enumerate(raw_requirements, start=1):
        if not isinstance(raw_requirement, dict):
            raise KnowledgeExplanationError(f"问题覆盖项 R{index} 格式不正确")
        requirement_id = str(raw_requirement.get("id") or "").strip().upper()
        if not re.fullmatch(r"R[1-9][0-9]?", requirement_id):
            raise KnowledgeExplanationError(f"问题覆盖项 R{index} 缺少合法 id")
        if requirement_id in requirement_ids:
            raise KnowledgeExplanationError(f"问题覆盖项 {requirement_id} 重复")
        requirement_ids.add(requirement_id)
        requirements.append(
            {
                "id": requirement_id,
                "content": complete_text(
                    raw_requirement.get("content"),
                    label=f"问题覆盖项 {requirement_id}",
                    max_length=80,
                ),
            }
        )

    raw_pages = value.get("pages")
    if not isinstance(raw_pages, list):
        raw_pages = []
    if not raw_pages:
        raise KnowledgeExplanationError("讲解大纲没有返回任何页面，请重新生成")
    target = requested_page_count or max(4, min(7, len(raw_pages)))
    if len(raw_pages) < target:
        raise KnowledgeExplanationError(
            f"讲解大纲只返回 {len(raw_pages)} 页，少于要求的 {target} 页，请重新生成"
        )
    pages: list[dict[str, Any]] = []
    covered_requirement_ids: set[str] = set()
    for index in range(target):
        raw = raw_pages[index]
        if not isinstance(raw, dict):
            raise KnowledgeExplanationError(f"讲解大纲第 {index + 1} 页格式不正确，请重新生成")
        page_number = index + 1
        title = complete_text(
            raw.get("title"),
            label=f"第 {page_number} 页标题",
            max_length=22,
            fallback="本页重点",
            check_dangling_words=False,
        )
        subtitle = complete_text(
            raw.get("subtitle"),
            label=f"第 {page_number} 页副标题",
            max_length=30,
            fallback=f"聚焦{title}",
        )
        learning_goal = complete_text(
            raw.get("learning_goal"),
            label=f"第 {page_number} 页学习目标",
            max_length=48,
            fallback=f"理解{title}",
        )
        covers = raw.get("covers")
        if not isinstance(covers, list) or not covers:
            raise KnowledgeExplanationError(f"第 {page_number} 页没有声明覆盖哪些问题要点")
        normalized_covers = list(
            dict.fromkeys(str(item).strip().upper() for item in covers if str(item).strip())
        )
        unknown_covers = set(normalized_covers) - requirement_ids
        if unknown_covers:
            raise KnowledgeExplanationError(
                f"第 {page_number} 页引用了未知覆盖项：{', '.join(sorted(unknown_covers))}"
            )
        covered_requirement_ids.update(normalized_covers)
        layout = normalize_page_layout(raw.get("layout"), index + 1)
        pages.append(
            {
                "title": title,
                "subtitle": subtitle,
                "learning_goal": learning_goal,
                "content_brief": complete_text(
                    raw.get("content_brief"),
                    label=f"第 {page_number} 页内容边界",
                    max_length=120,
                    fallback=learning_goal,
                ),
                "visual_focus": complete_text(
                    raw.get("visual_focus"),
                    label=f"第 {page_number} 页主视觉",
                    max_length=100,
                    fallback=subtitle,
                ),
                "covers": normalized_covers,
                "layout": layout,
            }
        )
    missing_requirement_ids = requirement_ids - covered_requirement_ids
    if missing_requirement_ids:
        raise KnowledgeExplanationError(
            "讲解大纲没有安排以下问题覆盖项："
            + ", ".join(sorted(missing_requirement_ids))
        )
    return {
        "title": complete_text(
            value.get("title"),
            label="整组讲解总标题",
            max_length=24,
            fallback="知识讲解",
            check_dangling_words=False,
        ),
        "subtitle": complete_text(
            value.get("subtitle"),
            label="整组讲解副标题",
            max_length=42,
            fallback="紧扣用户问题组织讲解内容",
        ),
        "requirements": requirements,
        "pages": pages,
    }


def normalize_page_detail(
    value: dict[str, Any], page: dict[str, Any]
) -> dict[str, Any]:
    raw_sections = value.get("sections")
    if not isinstance(raw_sections, list):
        raw_sections = []
    accents = {"blue", "green", "orange", "red"}
    fallback_headings = ("关键内容", "必要关系", "图示说明", "条件判断", "结果解释", "补充说明")
    sections: list[dict[str, str]] = []
    for index, raw in enumerate(raw_sections[:6]):
        if not isinstance(raw, dict):
            continue
        section_number = index + 1
        body = complete_text(
            raw.get("body"),
            label=f"“{page['title']}”第 {section_number} 个分区正文",
            max_length=110,
        )
        accent = str(raw.get("accent", "blue")).lower()
        raw_heading = complete_text(
            raw.get("heading"),
            label=f"“{page['title']}”第 {section_number} 个分区标题",
            max_length=20,
            fallback=fallback_headings[index],
            check_dangling_words=False,
        )
        heading = re.sub(
            r"^(?:模块|要点|总结|部分)\s*[0-9一二三四五六七八九十]*\s*[:：、.\-]\s*",
            "",
            raw_heading,
        ).strip()
        if re.fullmatch(
            r"(?:模块|要点|总结|部分)\s*[0-9一二三四五六七八九十]*",
            heading,
        ):
            heading = ""
        sections.append(
            {
                "heading": heading or fallback_headings[index],
                "body": body,
                "visual": complete_text(
                    raw.get("visual"),
                    label=f"“{page['title']}”第 {section_number} 个分区绘图说明",
                    max_length=140,
                    fallback="围绕正文关系绘制简洁示意图",
                ),
                "visual_type": normalize_visual_type(raw.get("visual_type"), raw),
                "accent": accent if accent in accents else "blue",
            }
        )
    if len(sections) < 2:
        raise KnowledgeExplanationError(f"“{page['title']}”的页面文案不完整，请重新生成")
    return {
        "sections": sections,
        "key_takeaway": complete_text(
            value.get("key_takeaway"),
            label=f"“{page['title']}”关键结论",
            max_length=64,
            fallback=f"掌握{page['title']}，就能回答本页聚焦的问题",
        ),
    }


def normalize_visual_layouts(
    value: dict[str, Any], pages: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    raw_pages = value.get("pages")
    if not isinstance(raw_pages, list) or len(raw_pages) != len(pages):
        raise KnowledgeExplanationError(
            f"图片布局必须完整返回 {len(pages)} 页，当前返回数量不一致"
        )
    page_lookup = {int(page["index"]): page for page in pages}
    layouts_by_index: dict[int, dict[str, Any]] = {}
    for raw_page in raw_pages:
        if not isinstance(raw_page, dict):
            raise KnowledgeExplanationError("图片布局页面格式不正确")
        try:
            page_index = int(raw_page.get("index"))
        except (TypeError, ValueError) as exc:
            raise KnowledgeExplanationError("图片布局缺少合法页码") from exc
        if page_index not in page_lookup:
            raise KnowledgeExplanationError(f"图片布局引用了未知页码 {page_index}")
        if page_index in layouts_by_index:
            raise KnowledgeExplanationError(f"第 {page_index} 页图片布局重复")

        page = page_lookup[page_index]
        raw_regions = raw_page.get("regions")
        section_count = len(page["sections"])
        if not isinstance(raw_regions, list) or len(raw_regions) != section_count:
            raise KnowledgeExplanationError(
                f"第 {page_index} 页布局必须为 {section_count} 个内容分区逐一指定区域"
            )
        regions_by_section: dict[int, dict[str, Any]] = {}
        primary_count = 0
        for raw_region in raw_regions:
            if not isinstance(raw_region, dict):
                raise KnowledgeExplanationError(f"第 {page_index} 页存在无效布局区域")
            try:
                section_index = int(raw_region.get("section_index"))
            except (TypeError, ValueError) as exc:
                raise KnowledgeExplanationError(
                    f"第 {page_index} 页布局区域缺少合法 section_index"
                ) from exc
            if not 1 <= section_index <= section_count:
                raise KnowledgeExplanationError(
                    f"第 {page_index} 页布局引用了未知内容分区 {section_index}"
                )
            if section_index in regions_by_section:
                raise KnowledgeExplanationError(
                    f"第 {page_index} 页内容分区 {section_index} 被重复布局"
                )
            hierarchy = str(raw_region.get("hierarchy") or "").strip().lower()
            if hierarchy not in VISUAL_HIERARCHIES:
                raise KnowledgeExplanationError(
                    f"第 {page_index} 页内容分区 {section_index} 的视觉层级无效"
                )
            if hierarchy == "primary":
                primary_count += 1
            regions_by_section[section_index] = {
                "section_index": section_index,
                "position": complete_text(
                    raw_region.get("position"),
                    label=f"第 {page_index} 页分区 {section_index} 位置",
                    max_length=30,
                ),
                "proportion": complete_text(
                    raw_region.get("proportion"),
                    label=f"第 {page_index} 页分区 {section_index} 占比",
                    max_length=30,
                ),
                "hierarchy": hierarchy,
                "presentation": complete_text(
                    raw_region.get("presentation"),
                    label=f"第 {page_index} 页分区 {section_index} 呈现方式",
                    max_length=120,
                ),
                "connection": complete_text(
                    raw_region.get("connection"),
                    label=f"第 {page_index} 页分区 {section_index} 连接关系",
                    max_length=80,
                ),
            }
        expected_section_indexes = set(range(1, section_count + 1))
        if set(regions_by_section) != expected_section_indexes:
            raise KnowledgeExplanationError(
                f"第 {page_index} 页布局没有完整映射所有内容分区"
            )
        if not 1 <= primary_count <= 2:
            raise KnowledgeExplanationError(
                f"第 {page_index} 页必须设置一至两个 primary 主视觉区域"
            )
        layouts_by_index[page_index] = {
            "composition": complete_text(
                raw_page.get("composition"),
                label=f"第 {page_index} 页整体构图",
                max_length=100,
            ),
            "reading_flow": complete_text(
                raw_page.get("reading_flow"),
                label=f"第 {page_index} 页阅读动线",
                max_length=80,
            ),
            "regions": [
                regions_by_section[index] for index in range(1, section_count + 1)
            ],
            "takeaway_placement": complete_text(
                raw_page.get("takeaway_placement"),
                label=f"第 {page_index} 页结论位置",
                max_length=40,
            ),
            "takeaway_treatment": complete_text(
                raw_page.get("takeaway_treatment"),
                label=f"第 {page_index} 页结论呈现方式",
                max_length=80,
            ),
            "palette_strategy": complete_text(
                raw_page.get("palette_strategy"),
                label=f"第 {page_index} 页配色策略",
                max_length=80,
            ),
            "decoration": complete_text(
                raw_page.get("decoration"),
                label=f"第 {page_index} 页装饰策略",
                max_length=80,
            ),
        }
    if set(layouts_by_index) != set(page_lookup):
        raise KnowledgeExplanationError("图片布局没有覆盖全部讲解页面")
    return [layouts_by_index[int(page["index"])] for page in pages]


def page_display_title(
    lesson_title: str, page_title: str, page_index: int, page_count: int
) -> str:
    chinese_ordinals = "一二三四五六七八九十"
    ordinal = chinese_ordinals[page_index - 1] if 1 <= page_index <= 10 else str(page_index)
    return (
        f"{lesson_title}：{page_title}"
        if page_count == 1
        else f"{lesson_title}（{ordinal}）：{page_title}"
    )


def _clean_drafted_prompt(value: Any) -> str:
    prompt = str(value or "").strip()
    if prompt.startswith("```"):
        prompt = re.sub(r"^```(?:markdown|md|text)?\s*|\s*```$", "", prompt, flags=re.IGNORECASE)
    return prompt.strip()


def _drafted_prompt_is_complete(
    prompt: str, *, title: str, page: dict[str, Any]
) -> bool:
    if not 400 <= len(prompt) <= 16_000:
        return False
    required_structure = (
        "### 1. 顶部区域",
        "### 2. 主体内容",
        "### 3. 结论区",
        "### 风格要求",
    )
    required_copy = [title, str(page["subtitle"]), str(page["key_takeaway"])]
    for section in page["sections"]:
        required_copy.extend((str(section["heading"]), str(section["body"])))
    visual_layout = page.get("visual_layout")
    if isinstance(visual_layout, dict) and visual_layout:
        layout_keys = (
            "composition",
            "reading_flow",
            "takeaway_placement",
            "takeaway_treatment",
            "palette_strategy",
            "decoration",
        )
        if any(not visual_layout.get(key) for key in layout_keys):
            return False
        required_copy.extend(
            str(visual_layout[key]) for key in layout_keys
        )
        regions = visual_layout.get("regions")
        if not isinstance(regions, list) or len(regions) != len(page["sections"]):
            return False
        for region in regions:
            if isinstance(region, dict):
                region_keys = ("position", "proportion", "presentation", "connection")
                if any(not region.get(key) for key in region_keys):
                    return False
                required_copy.extend(
                    str(region[key]) for key in region_keys
                )
            else:
                return False
    return all(item in prompt for item in (*required_structure, *required_copy))


def build_page_prompt(
    *,
    lesson_title: str,
    lesson_subtitle: str,
    page_count: int,
    page: dict[str, Any],
    drafted_prompt: str = "",
) -> str:
    page_index = int(page["index"])
    title = page_display_title(lesson_title, str(page["title"]), page_index, page_count)
    visual_types: list[str] = []
    section_blocks: list[str] = []
    accent_labels = {
        "blue": "蓝色",
        "green": "绿色",
        "orange": "橙色",
        "red": "红色",
    }
    visual_layout = page.get("visual_layout")
    visual_layout = visual_layout if isinstance(visual_layout, dict) else {}
    layout_regions = {
        int(region["section_index"]): region
        for region in visual_layout.get("regions") or []
        if isinstance(region, dict) and str(region.get("section_index", "")).isdigit()
    }
    hierarchy_labels = {
        "primary": "主视觉",
        "secondary": "次级说明",
        "supporting": "辅助信息",
    }
    for section_index, section in enumerate(page["sections"], start=1):
        visual_type = normalize_visual_type(section.get("visual_type"), section)
        section_visual_types = [
            visual_type,
            *_detected_specialized_visual_types(section),
        ]
        for detected_type in section_visual_types:
            if detected_type not in visual_types:
                visual_types.append(detected_type)
        accent = accent_labels.get(str(section.get("accent")), "蓝色")
        region = layout_regions.get(section_index, {})
        required_region_keys = ("position", "proportion", "presentation", "connection")
        region_instruction = (
            f"位于{region['position']}，{region['proportion']}，视觉层级为"
            f"{hierarchy_labels.get(str(region.get('hierarchy')), '内容分区')}；"
            f"{region['presentation']}；{region['connection']}"
            if region and all(region.get(key) for key in required_region_keys)
            else "依据整页信息关系分配空间，不使用机械等宽卡片"
        )
        section_blocks.append(
            f"""#### （{section_index}）{section['heading']}
- 位置与形式：{region_instruction}；本区以{accent}作少量语义强调。
- 标题：清晰显示「{section['heading']}」。
- 文字：逐字显示「{section['body']}」。
- 示意图：{section['visual']}
- 图文关系：图形与对应文字就近对齐，用必要的箭头、连线或标注明确阅读方向。"""
        )
    specialized_rules = "\n".join(
        f"- {SPECIALIZED_VISUAL_INSTRUCTIONS[visual_type]}"
        for visual_type in visual_types
        if visual_type in SPECIALIZED_VISUAL_INSTRUCTIONS
    ) or "- 本页没有电路、曲线或公式推导专项图示；所有图形仍须忠实表达提示词中的知识关系。"
    layout = normalize_page_layout(page.get("layout"), page_index)
    layout_instruction = PAGE_LAYOUT_INSTRUCTIONS[layout]
    composition = visual_layout.get("composition") or layout_instruction
    reading_flow = visual_layout.get("reading_flow") or "按知识依赖从主视觉依次阅读各分区"
    takeaway_placement = visual_layout.get("takeaway_placement") or "与当前版式协调的视觉收束区"
    takeaway_treatment = visual_layout.get("takeaway_treatment") or "使用强调框、批注或中心结论自然收束"
    palette_strategy = visual_layout.get("palette_strategy") or "蓝色承担主线，绿、橙、红仅用于明确语义强调"
    decoration = visual_layout.get("decoration") or "只使用与主题直接相关的浅蓝色轻量线稿，并避让正文"
    fallback_prompt = f"""
生成一张「专业知识讲解」风格的横向 16:9 中文信息图，主题为《{title}》，副标题为“{page['subtitle']}”。整体采用蓝白为主色调，风格简洁专业、逻辑分层清楚。已确认的整体构图：{composition}

### 1. 顶部区域
- 左上角：蓝色圆角页码标签，白色文字「{page_index}/{page_count}」。
- 主标题：深蓝色粗体大字「{title}」。
- 副标题：浅蓝灰色常规字体「{page['subtitle']}」。
- 装饰元素：{decoration}

### 2. 主体内容
- 整体布局：{composition}
- 阅读动线：{reading_flow}

{chr(10).join(section_blocks)}

### 3. 结论区
- 结论位置：{takeaway_placement}
- 呈现方式：{takeaway_treatment}
- 不强制底部通栏，必须服从上述已确认的结论位置与呈现方式。
- 结论文字逐字显示：「{page['key_takeaway']}」。

### 风格要求
- 色彩：白到浅蓝灰背景，深蓝标题；{palette_strategy}
- 排版：标题、正文、注释层级分明，分区间留白均匀，图标、公式和文字严格对齐。
- 质感：扁平化二维矢量设计，无复杂阴影、无写实 3D；突出教学信息可读性。
- 整组一致性：围绕“{lesson_subtitle}”保持页码、字体与线条体系一致，但本句不是画面文字。
""".strip()
    candidate = _clean_drafted_prompt(drafted_prompt)
    prompt = (
        candidate
        if _drafted_prompt_is_complete(candidate, title=title, page=page)
        else fallback_prompt
    )
    return f"""{prompt}

### 内容准确性要求
- 页面可见文字只能来自本提示词明确给出的页码、标题、副标题、分区标题、正文、图内标签和结论；结构说明本身不得画入页面。
- 所有指定文字逐字准确，只出现一次，不改写、不漏字、不造字；不得添加“模块1”“要点2”“总结3”等机械标签。
- 只使用简体中文和提示词明确给出的标准变量、单位、坐标与数值。优先保证标题、公式、坐标轴和结论可读，不生成伪文字。

### 专业图示准确性
{specialized_rules}

### 最终输出限制
- 横向 16:9 单页教学信息图，不是网页截图。不得裁字、压字、文字重叠或重复内容。
- 不要照片、人物、写实 3D、品牌、水印、二维码或版权信息；背景装饰不能干扰正文。
""".strip()
