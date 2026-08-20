from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import re
import shutil
from pathlib import Path
from typing import Any, AsyncGenerator, Generator, Iterable

import httpx
import pandas as pd

from backend.app.rag.embedding_runtime import encode_texts
from backend.app.rag.knowledge_document import KnowledgeUnit
from backend.app.rag.multimodal import BuildModelConfig


MICROSOFT_GRAPHRAG_SCHEMA_VERSION = "3.4-microsoft-graphrag-2.7"
QWEN_CHAT_TYPE = "circuitmind_qwen_chat"
LOCAL_EMBEDDING_TYPE = "circuitmind_local_qwen3_embedding"


def _json_value(raw: str) -> Any:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            from json_repair import repair_json

            return json.loads(repair_json(text))
        except Exception:
            match = re.search(r"[\[{].*[\]}]", text, flags=re.S)
            return json.loads(match.group(0)) if match else {}


def _normalize_graphrag_records(content: str) -> str:
    """Repair Qwen tuple lines when the model omits GraphRAG's ## delimiter."""

    if "##" in content:
        return content
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    records = [
        line
        for line in lines
        if line.startswith("(")
        and line.endswith(")")
        and re.match(r'^\("(?:entity|relationship)"', line, re.I)
    ]
    if len(records) < 2:
        return content
    suffix = "\n<|COMPLETE|>" if "<|COMPLETE|>" in content else ""
    return "\n##\n".join(records) + suffix


class QwenGraphRagChatModel:
    """Direct DashScope OpenAI-compatible adapter for Microsoft GraphRAG."""

    def __init__(self, name: str, config: Any, cache: Any | None = None, **_: Any) -> None:
        self.name = name
        self.config = config
        self.cache = cache
        self.endpoint = f"{str(config.api_base).rstrip('/')}/chat/completions"

    def _request_payload(
        self, prompt: str, history: list | None, kwargs: dict[str, Any]
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        messages = [dict(item) for item in (history or [])]
        messages.append({"role": "user", "content": prompt})
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": int(self.config.max_tokens or 4096),
            "enable_thinking": False,
        }
        if kwargs.get("json") or kwargs.get("json_model"):
            payload["response_format"] = {"type": "json_object"}
        return payload, messages

    def _response(self, response: httpx.Response, messages: list[dict[str, Any]], kwargs: dict[str, Any]) -> Any:
        from graphrag.language_model.response.base import BaseModelOutput, BaseModelResponse

        value = response.json()
        choices = value.get("choices") or []
        content = str(choices[0].get("message", {}).get("content", "")) if choices else ""
        content = _normalize_graphrag_records(content)
        messages = [*messages, {"role": "assistant", "content": content}]
        parsed = None
        json_model = kwargs.get("json_model")
        if kwargs.get("json") or json_model:
            parsed_value = _json_value(content)
            if inspect.isclass(json_model):
                parsed = json_model.model_validate(parsed_value)
        return BaseModelResponse(
            output=BaseModelOutput(content=content, full_response=value),
            parsed_response=parsed,
            history=messages,
        )

    def chat(self, prompt: str, history: list | None = None, **kwargs: Any) -> Any:
        payload, messages = self._request_payload(prompt, history, kwargs)
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with httpx.Client(timeout=httpx.Timeout(240, connect=20)) as client:
                    response = client.post(self.endpoint, headers=headers, json=payload)
                    if response.status_code == 400 and "response_format" in payload:
                        payload.pop("response_format", None)
                        response = client.post(self.endpoint, headers=headers, json=payload)
                    response.raise_for_status()
                return self._response(response, messages, kwargs)
            except Exception as exc:
                last_error = exc
                if attempt == 2:
                    raise
        raise RuntimeError("Qwen GraphRAG request failed") from last_error

    async def achat(self, prompt: str, history: list | None = None, **kwargs: Any) -> Any:
        payload, messages = self._request_payload(prompt, history, kwargs)
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(240, connect=20)) as client:
                    response = await client.post(self.endpoint, headers=headers, json=payload)
                    if response.status_code == 400 and "response_format" in payload:
                        payload.pop("response_format", None)
                        response = await client.post(self.endpoint, headers=headers, json=payload)
                    response.raise_for_status()
                return self._response(response, messages, kwargs)
            except Exception as exc:
                last_error = exc
                if attempt == 2:
                    raise
        raise RuntimeError("Qwen GraphRAG request failed") from last_error

    def chat_stream(
        self, prompt: str, history: list | None = None, **kwargs: Any
    ) -> Generator[str, None, None]:
        yield self.chat(prompt, history, **kwargs).output.content

    async def achat_stream(
        self, prompt: str, history: list | None = None, **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        response = await self.achat(prompt, history, **kwargs)
        yield response.output.content


class LocalQwen3EmbeddingModel:
    """SentenceTransformer adapter used by GraphRAG indexing and search."""

    def __init__(self, name: str, config: Any, cache: Any | None = None, **_: Any) -> None:
        self.name = name
        self.config = config
        self.cache = cache
        self.model_path = Path(config.model)

    def embed_batch(self, text_list: list[str], **_: Any) -> list[list[float]]:
        return encode_texts(
            self.model_path, text_list, batch_size=4, purpose="document"
        ).tolist()

    def embed(self, text: str, **_: Any) -> list[float]:
        return encode_texts(
            self.model_path, [text], batch_size=1, purpose="query"
        )[0].tolist()

    async def aembed_batch(self, text_list: list[str], **kwargs: Any) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_batch, text_list, **kwargs)

    async def aembed(self, text: str, **kwargs: Any) -> list[float]:
        return await asyncio.to_thread(self.embed, text, **kwargs)


def register_graphrag_models() -> None:
    from graphrag.language_model.factory import ModelFactory

    ModelFactory.register_chat(QWEN_CHAT_TYPE, lambda **kwargs: QwenGraphRagChatModel(**kwargs))
    ModelFactory.register_embedding(
        LOCAL_EMBEDDING_TYPE,
        lambda **kwargs: LocalQwen3EmbeddingModel(**kwargs),
    )


async def _run_textbook_extract_graph(config: Any, context: Any) -> Any:
    """Strict-JSON textbook extractor feeding the official downstream workflows."""

    from graphrag.index.typing.workflow import WorkflowFunctionOutput
    from graphrag.utils.storage import load_table_from_storage, write_table_to_storage

    from backend.app.rag.semantic_graph import (
        SemanticTextUnit,
        extract_text_unit_graphs,
    )

    text_units_df = await load_table_from_storage("text_units", context.output_storage)
    units = [
        SemanticTextUnit(
            id=str(row["id"]),
            text=str(row.get("text", "")),
            source="",
            page_start=0,
            page_end=0,
            chapter="",
            section="",
            block_ids=[],
            modality="multimodal_knowledge",
        )
        for _, row in text_units_df.iterrows()
    ]
    additional = context.state.get("additional_context", {})
    client = additional.get("textbook_graph_client")
    cache_path = additional.get("textbook_extraction_cache")
    if client is None:
        raise RuntimeError("教材 GraphRAG 抽取器缺少 Qwen 客户端")
    extractions = await asyncio.to_thread(
        extract_text_unit_graphs,
        units,
        client,
        cache_path=Path(cache_path) if cache_path else None,
        batch_size=1,
        max_workers=3,
    )

    entity_values: dict[tuple[str, str], dict[str, Any]] = {}
    relationship_values: dict[tuple[str, str], dict[str, Any]] = {}
    for unit in units:
        extraction = extractions.get(unit.id, {})
        descriptions_by_name: dict[str, list[str]] = {}
        for relation in extraction.get("relationships", []):
            if not isinstance(relation, dict):
                continue
            source = str(relation.get("source", "")).strip()
            target = str(relation.get("target", "")).strip()
            evidence = str(relation.get("evidence_text", "")).strip()
            if source:
                descriptions_by_name.setdefault(source, []).append(evidence)
            if target:
                descriptions_by_name.setdefault(target, []).append(evidence)
            if not source or not target or not evidence:
                continue
            key = (source, target)
            value = relationship_values.setdefault(key, {
                "source": source,
                "target": target,
                "descriptions": [],
                "relations": [],
                "text_unit_ids": [],
                "weight": 0.0,
            })
            if evidence not in value["descriptions"]:
                value["descriptions"].append(evidence)
            relation_original = str(relation.get("relation_original", "")).strip()
            if relation_original and relation_original not in value["relations"]:
                value["relations"].append(relation_original)
            value["text_unit_ids"].append(unit.id)
            value["weight"] += float(relation.get("strength", 1.0) or 1.0)
        for fact in extraction.get("attribute_facts", []):
            if not isinstance(fact, dict):
                continue
            subject = str(fact.get("subject", "")).strip()
            evidence = str(fact.get("evidence_text", "")).strip()
            if subject and evidence:
                descriptions_by_name.setdefault(subject, []).append(evidence)
        for entity in extraction.get("entities", []):
            if not isinstance(entity, dict):
                continue
            title = str(entity.get("name", "")).strip()
            entity_type = str(entity.get("type", "")).strip()
            if not title or not entity_type:
                continue
            key = (title, entity_type)
            value = entity_values.setdefault(key, {
                "title": title,
                "type": entity_type,
                "descriptions": [],
                "text_unit_ids": [],
            })
            candidates = [
                str(entity.get("description", "")).strip(),
                *descriptions_by_name.get(title, []),
            ]
            for description in candidates:
                if description and description not in value["descriptions"]:
                    value["descriptions"].append(description)
            value["text_unit_ids"].append(unit.id)

    connected_titles = {
        name
        for value in relationship_values.values()
        for name in (value["source"], value["target"])
    }
    entities = pd.DataFrame([
        {
            "title": value["title"],
            "type": value["type"],
            "description": " ".join(value["descriptions"][:4]) or value["title"],
            "text_unit_ids": list(dict.fromkeys(value["text_unit_ids"])),
            "frequency": len(set(value["text_unit_ids"])),
        }
        for value in entity_values.values()
        if value["title"] in connected_titles
    ])
    relationships = pd.DataFrame([
        {
            "source": value["source"],
            "target": value["target"],
            "description": json.dumps({
                "relation_original": value["relations"][0] if value["relations"] else "相关",
                "relation_variants": value["relations"],
                "evidence_texts": value["descriptions"][:4],
            }, ensure_ascii=False),
            "text_unit_ids": list(dict.fromkeys(value["text_unit_ids"])),
            "weight": value["weight"],
        }
        for value in relationship_values.values()
    ])
    if entities.empty or relationships.empty:
        raise RuntimeError("教材 GraphRAG JSON 抽取没有产生可用实体关系")
    await write_table_to_storage(entities, "entities", context.output_storage)
    await write_table_to_storage(relationships, "relationships", context.output_storage)
    return WorkflowFunctionOutput(result={
        "entities": entities,
        "relationships": relationships,
    })


async def _run_textbook_community_reports(config: Any, context: Any) -> Any:
    """Create Qwen JSON community reports in GraphRAG's final table schema."""

    from graphrag.data_model.schemas import COMMUNITY_REPORTS_FINAL_COLUMNS
    from graphrag.index.typing.workflow import WorkflowFunctionOutput
    from graphrag.utils.storage import load_table_from_storage, write_table_to_storage

    entities = await load_table_from_storage("entities", context.output_storage)
    relationships = await load_table_from_storage("relationships", context.output_storage)
    communities = await load_table_from_storage("communities", context.output_storage)
    client = context.state.get("additional_context", {}).get("textbook_graph_client")
    if client is None:
        raise RuntimeError("教材 GraphRAG 社区报告缺少 Qwen 客户端")

    rows: list[dict[str, Any]] = []
    for _, community in communities.iterrows():
        entity_ids = set(_as_list(community.get("entity_ids")))
        relationship_ids = set(_as_list(community.get("relationship_ids")))
        entity_records = entities.loc[entities["id"].astype(str).isin(entity_ids)]
        relationship_records = relationships.loc[
            relationships["id"].astype(str).isin(relationship_ids)
        ]
        context_value = {
            "entities": [
                {
                    "name": str(item.get("title", "")),
                    "type": str(item.get("type", "")),
                    "description": str(item.get("description", ""))[:600],
                }
                for item in entity_records.to_dict("records")
            ],
            "relationships": [
                {
                    "source": str(item.get("source", "")),
                    "target": str(item.get("target", "")),
                    "description": str(item.get("description", ""))[:600],
                }
                for item in relationship_records.to_dict("records")
            ],
        }
        prompt = (
            "你是电子电路教材知识社区总结器。只根据给定实体和关系生成中文报告，"
            "不得补充教材之外的事实。返回 JSON："
            '{"title":"社区标题","summary":"摘要",'
            '"findings":[{"summary":"发现标题","explanation":"有证据的说明"}],'
            '"rating":0.0,"rating_explanation":"重要性依据"}。\n数据：'
            + json.dumps(context_value, ensure_ascii=False)
        )
        response = await asyncio.to_thread(client.complete_json, prompt)
        title = str(response.get("title", "")).strip() or str(community.get("title", "知识社区"))
        summary = str(response.get("summary", "")).strip()
        findings = response.get("findings", [])
        if not isinstance(findings, list):
            findings = []
        normalized_findings = [
            {
                "summary": str(item.get("summary", "")).strip(),
                "explanation": str(item.get("explanation", "")).strip(),
            }
            for item in findings
            if isinstance(item, dict)
            and (str(item.get("summary", "")).strip() or str(item.get("explanation", "")).strip())
        ]
        if not summary:
            summary = "；".join(
                str(item.get("description", "")).strip()
                for item in context_value["entities"][:4]
                if str(item.get("description", "")).strip()
            ) or "本社区汇集相互关联的电子电路教材概念。"
        if not normalized_findings:
            normalized_findings = [
                {
                    "summary": str(item["name"]),
                    "explanation": str(item["description"]),
                }
                for item in context_value["entities"][:5]
            ]
        try:
            rating = float(response.get("rating", 5.0))
        except (TypeError, ValueError):
            match = re.search(r"\d+(?:\.\d+)?", str(response.get("rating", "")))
            rating = float(match.group(0)) if match else 5.0
        full_content = "\n\n".join([
            f"# {title}",
            summary,
            *[
                f"## {item['summary']}\n\n{item['explanation']}"
                for item in normalized_findings
            ],
        ])
        community_id = int(community.get("community", 0) or 0)
        rows.append({
            "id": _stable_graph_id("community-report", community_id, title, summary),
            "human_readable_id": community_id,
            "community": community_id,
            "level": int(community.get("level", 0) or 0),
            "parent": community.get("parent"),
            "children": community.get("children", []),
            "title": title,
            "summary": summary,
            "full_content": full_content,
            "rank": rating,
            "rating_explanation": str(response.get("rating_explanation", "")).strip(),
            "findings": normalized_findings,
            "full_content_json": json.dumps({
                "title": title,
                "summary": summary,
                "findings": normalized_findings,
                "rating": rating,
                "rating_explanation": str(response.get("rating_explanation", "")).strip(),
            }, ensure_ascii=False),
            "period": str(community.get("period", "")),
            "size": int(community.get("size", len(entity_ids)) or len(entity_ids)),
        })
    reports = pd.DataFrame(rows, columns=COMMUNITY_REPORTS_FINAL_COLUMNS)
    await write_table_to_storage(reports, "community_reports", context.output_storage)
    return WorkflowFunctionOutput(result=reports)


def _graphrag_config(
    root_dir: Path,
    model_config: BuildModelConfig,
    embedding_model_path: Path,
) -> Any:
    register_graphrag_models()
    from graphrag.config.create_graphrag_config import create_graphrag_config

    vector_uri = str((root_dir / "vector_store").resolve())
    return create_graphrag_config(
        {
            "models": {
                "default_chat_model": {
                    "type": QWEN_CHAT_TYPE,
                    "auth_type": "api_key",
                    "api_key": model_config.api_key,
                    "api_base": model_config.base_url,
                    "model": "qwen3.7-flash",
                    "encoding_model": "cl100k_base",
                    "model_supports_json": True,
                    "async_mode": "asyncio",
                    "concurrent_requests": 2,
                    "retry_strategy": "none",
                    "max_tokens": 4096,
                },
                "default_embedding_model": {
                    "type": LOCAL_EMBEDDING_TYPE,
                    "auth_type": "api_key",
                    "api_key": "local-checkpoint",
                    "model": str(embedding_model_path.resolve()),
                    "encoding_model": "cl100k_base",
                    "async_mode": "asyncio",
                    "concurrent_requests": 1,
                    "retry_strategy": "none",
                },
            },
            "input": {
                "storage": {"type": "file", "base_dir": "input"},
                "file_type": "json",
            },
            "chunks": {
                "strategy": "tokens",
                "size": 900,
                "overlap": 0,
                "group_by_columns": ["id"],
                "prepend_metadata": True,
                "chunk_size_includes_metadata": True,
            },
            "output": {"type": "file", "base_dir": "output"},
            "cache": {"type": "file", "base_dir": "cache"},
            "reporting": {"type": "file", "base_dir": "logs"},
            "vector_store": {
                "default_vector_store": {
                    "type": "lancedb",
                    "db_uri": vector_uri,
                    "container_name": "default",
                }
            },
            "extract_graph": {
                "entity_types": [
                    "课程概念",
                    "电路",
                    "器件与元件",
                    "电路参数",
                    "物理过程与性质",
                    "方法",
                    "公式语义",
                    "表格结论",
                ],
                "max_gleanings": 0,
            },
            "community_reports": {"max_length": 1200, "max_input_length": 8000},
            "snapshots": {"graphml": False, "embeddings": False},
        },
        root_dir=str(root_dir),
    )


def _input_documents(units: Iterable[KnowledgeUnit]) -> pd.DataFrame:
    rows = []
    for unit in units:
        rows.append({
            "id": unit.id,
            "title": " / ".join(unit.title_path) or unit.source,
            "text": unit.text,
            "creation_date": "",
            "metadata": json.dumps({
                "source": unit.source,
                "chapter": unit.chapter,
                "section": unit.section,
                "page_start": unit.page_start,
                "page_end": unit.page_end,
                "evidence_ids": unit.evidence_ids,
            }, ensure_ascii=False),
        })
    return pd.DataFrame(rows)


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        if value.startswith("["):
            try:
                return [str(item) for item in json.loads(value)]
            except json.JSONDecodeError:
                pass
        return [value] if value else []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def _json_safe(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if hasattr(value, "item") and not isinstance(value, (str, bytes, dict, list, tuple)):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _entity_key(value: str) -> str:
    return re.sub(r"[\s_{}()（）\[\]]+", "", str(value)).casefold()


def _canonical_entity_name(value: str) -> str:
    name = re.sub(r"\s+", " ", str(value)).strip(" \"'，。；：")
    name = re.sub(
        r"(?<=[\u4e00-\u9fff])\s*[TRCQLDU]_?\d+\s*的\s*",
        "的",
        name,
        flags=re.I,
    )
    name = re.sub(
        r"(?<=[\u4e00-\u9fff])\s*[TRCQLDU]_?\d+\s*(?=型|管|电路|$)",
        "",
        name,
        flags=re.I,
    )
    if re.fullmatch(r"[A-Za-zΑ-Ωα-ω0-9_{}()（）\[\].+\-/\s]+", name):
        name = re.sub(r"\s+", "", name)
    symbol_suffix = re.fullmatch(
        r"([\u4e00-\u9fff]{2,20})\s*[（(]?"
        r"[A-Za-zΑ-Ωα-ω][A-Za-zΑ-Ωα-ω0-9_{}()\s]*[）)]?",
        name,
    )
    if symbol_suffix and symbol_suffix.group(1).endswith(
        ("电阻", "电流", "电压", "增益", "放大倍数", "稳定性", "温度特性")
    ):
        name = symbol_suffix.group(1)
    if re.search(r"[TRCQLDU]\s*\d+|[A-Za-zΑ-Ωα-ω]_?[A-Za-z0-9]+", name, re.I):
        for property_name in (
            "输出电阻", "输入电阻", "输出电流", "参考电流", "集电极电流",
            "基极电流", "发射极电流", "输出电压", "输入电压",
        ):
            if property_name in name:
                name = property_name
                break
    if "温度稳定性" in name and re.search(r"[A-Za-zΑ-Ωα-ω0-9_]", name):
        name = "温度稳定性"
    if "晶体管" in name and "β" in name:
        name = "晶体管电流放大系数"
    if name.endswith("输出电阻"):
        name = "输出电阻"
    name = re.sub(r"^(.{2,12}(?:管|电阻|电容|电感))[A-Za-z]\d+$", r"\1", name)
    return name


def _invalid_entity(name: str, entity_type: str) -> bool:
    if not name or not entity_type or len(name) > 48:
        return True
    if re.match(r"^(?:图|表|式|第?\d+页)", name):
        return True
    if re.fullmatch(r"[TRCQLDU]_?\d+", name, re.I):
        return True
    if re.match(r"^[TRCQLDU]_?\s*\d+", name, re.I):
        return True
    if re.match(r"^节点\s*[A-Za-z]?\d*$", name, re.I):
        return True
    if name in {"简单", "元件少", "很小", "很大", "较高", "较低", "流电阻大"}:
        return True
    if re.search(r"(?:稍有上翘|做不到|不能做到|不可能|可得)", name):
        return True
    if name.endswith(("测量点", "测试点")):
        return True
    if not re.search(r"[\u4e00-\u9fff]", name):
        if len(name) <= 2 or re.search(r"[_0-9Α-Ωα-ω\u0300-\u036f]", name):
            return True
    if re.fullmatch(r"[A-Za-zΑ-Ωα-ω][A-Za-zΑ-Ωα-ω0-9_{}()\-+/]*", name):
        return True
    if re.search(r"[=≈≠≤≥∑∫√]", name):
        return True
    return False


def _relationship_content(value: str) -> tuple[str, str]:
    description = re.sub(r"\s+", " ", str(value)).strip()
    try:
        payload = json.loads(description)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        relation = str(payload.get("relation_original", "")).strip() or "相关"
        evidence_values = payload.get("evidence_texts", [])
        if isinstance(evidence_values, list):
            evidence = " ".join(
                str(item).strip() for item in evidence_values if str(item).strip()
            )
        else:
            evidence = str(evidence_values).strip()
        return relation[:80], evidence or relation
    relation = re.split(r"[。；;]", description, maxsplit=1)[0][:80] or "相关"
    return relation, description


def _stable_graph_id(prefix: str, *values: object) -> str:
    raw = "|".join(str(value) for value in values)
    return f"{prefix}:" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def convert_graphrag_outputs(
    output_path: Path,
    units: Iterable[KnowledgeUnit],
) -> dict[str, Any]:
    """Convert official Parquet outputs to the project's evidence-rich graph API."""

    unit_map = {unit.id: unit for unit in units}
    entities_df = pd.read_parquet(output_path / "entities.parquet")
    relationships_df = pd.read_parquet(output_path / "relationships.parquet")
    text_units_df = pd.read_parquet(output_path / "text_units.parquet")
    communities_df = pd.read_parquet(output_path / "communities.parquet")
    reports_df = pd.read_parquet(output_path / "community_reports.parquet")

    text_units: list[dict[str, Any]] = []
    text_unit_context: dict[str, dict[str, Any]] = {}
    for _, row in text_units_df.iterrows():
        document_ids = _as_list(row.get("document_ids"))
        unit = next((unit_map[item] for item in document_ids if item in unit_map), None)
        context = {
            "source": unit.source if unit else "",
            "page_start": unit.page_start if unit else 0,
            "page_end": unit.page_end if unit else 0,
            "chapter": unit.chapter if unit else "",
            "section": unit.section if unit else "",
            "block_ids": unit.evidence_ids if unit else [],
        }
        text_unit = {
            "id": str(row["id"]),
            "text": str(row.get("text", "")),
            **context,
            "modality": "multimodal_knowledge",
            "image_path": None,
        }
        text_units.append(text_unit)
        text_unit_context[text_unit["id"]] = text_unit

    nodes: list[dict[str, Any]] = []
    node_by_key: dict[str, dict[str, Any]] = {}
    official_to_node: dict[str, str] = {}
    title_to_node: dict[str, str] = {}
    for _, row in entities_df.iterrows():
        raw_name = str(row.get("title", "")).strip()
        name = _canonical_entity_name(raw_name)
        entity_type = str(row.get("type", "")).strip()
        if _invalid_entity(name, entity_type):
            continue
        key = _entity_key(name)
        text_unit_ids = _as_list(row.get("text_unit_ids"))
        pages = sorted({
            int(text_unit_context[item]["page_start"])
            for item in text_unit_ids
            if item in text_unit_context and text_unit_context[item]["page_start"]
        })
        existing = node_by_key.get(key)
        if existing is None:
            existing = {
                "id": _stable_graph_id("entity", key),
                "type": "entity",
                "entity_type": entity_type,
                "name": name,
                "description": str(row.get("description", "")).strip(),
                "aliases": [raw_name] if raw_name != name else [],
                "raw_entity_types": [entity_type],
                "text_unit_ids": text_unit_ids,
                "evidence_count": len(text_unit_ids),
                "pages": pages,
                "relationship_count": 0,
                "standalone": False,
            }
            nodes.append(existing)
            node_by_key[key] = existing
        else:
            existing["text_unit_ids"] = list(dict.fromkeys([
                *existing["text_unit_ids"], *text_unit_ids
            ]))
            existing["pages"] = sorted(set([*existing["pages"], *pages]))
            if raw_name != existing["name"] and raw_name not in existing["aliases"]:
                existing["aliases"].append(raw_name)
        official_to_node[str(row.get("id", ""))] = existing["id"]
        title_to_node[_entity_key(raw_name)] = existing["id"]

    edges: list[dict[str, Any]] = []
    mentions: list[dict[str, Any]] = []
    links: list[dict[str, Any]] = []
    official_to_edge: dict[str, str] = {}
    node_lookup = {node["id"]: node for node in nodes}
    seen_edges: set[tuple[str, str, str]] = set()
    for _, row in relationships_df.iterrows():
        source = title_to_node.get(_entity_key(str(row.get("source", ""))))
        target = title_to_node.get(_entity_key(str(row.get("target", ""))))
        relation, description = _relationship_content(str(row.get("description", "")))
        if not source or not target or source == target or not description:
            continue
        edge_key = (source, target, description)
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)
        text_unit_ids = _as_list(row.get("text_unit_ids"))
        edge_id = _stable_graph_id("relationship", source, target, description)
        mention_ids: list[str] = []
        for text_unit_id in text_unit_ids or [""]:
            context = text_unit_context.get(text_unit_id, {})
            mention_id = _stable_graph_id("relationship-mention", edge_id, text_unit_id)
            mention_ids.append(mention_id)
            mentions.append({
                "id": mention_id,
                "source": source,
                "target": target,
                "source_name": node_lookup[source]["name"],
                "target_name": node_lookup[target]["name"],
                "source_mention": node_lookup[source]["name"],
                "target_mention": node_lookup[target]["name"],
                "relation": relation,
                "evidence_text": description,
                "strength": float(row.get("weight", 1.0) or 1.0),
                "text_unit_id": text_unit_id,
                "evidence_id": text_unit_id,
                "source_modality": "multimodal_knowledge",
                "source_page": int(context.get("page_start", 0) or 0),
                "qualifier_text": "",
            })
        edge = {
            "id": edge_id,
            "source": source,
            "target": target,
            "type": relation,
            "relation": relation,
            "description": description,
            "mention_ids": mention_ids,
            "text_unit_ids": text_unit_ids,
            "evidence_ids": text_unit_ids,
            "evidence_count": len(text_unit_ids),
            "weight": float(row.get("weight", 1.0) or 1.0),
            "qualifier_texts": [],
        }
        edges.append(edge)
        official_to_edge[str(row.get("id", ""))] = edge_id
        node_lookup[source]["relationship_count"] += 1
        node_lookup[target]["relationship_count"] += 1
        links.append({
            "id": _stable_graph_id("relationship-link", source, target),
            "source": source,
            "target": target,
            "mention_ids": mention_ids,
            "relationship_ids": [edge_id],
            "text_unit_ids": text_unit_ids,
            "weight": float(row.get("weight", 1.0) or 1.0),
        })

    communities: list[dict[str, Any]] = []
    for _, row in communities_df.iterrows():
        entity_ids = [
            official_to_node[item]
            for item in _as_list(row.get("entity_ids"))
            if item in official_to_node
        ]
        relationship_ids = [
            official_to_edge[item]
            for item in _as_list(row.get("relationship_ids"))
            if item in official_to_edge
        ]
        communities.append({
            "id": str(row.get("id", "")),
            "community": int(row.get("community", 0) or 0),
            "parent": _json_safe(row.get("parent")),
            "children": _as_list(row.get("children")),
            "level": int(row.get("level", 0) or 0),
            "title": str(row.get("title", "")),
            "entity_ids": entity_ids,
            "relationship_ids": relationship_ids,
            "text_unit_ids": _as_list(row.get("text_unit_ids")),
            "size": int(row.get("size", len(entity_ids)) or len(entity_ids)),
            "algorithm": "microsoft-graphrag-hierarchical-leiden",
        })

    reports = [
        {
            "id": str(row.get("id", "")),
            "community": int(row.get("community", 0) or 0),
            "level": int(row.get("level", 0) or 0),
            "title": str(row.get("title", "")),
            "summary": str(row.get("summary", "")),
            "full_content": str(row.get("full_content", "")),
            "rank": float(row.get("rank", 0.0) or 0.0),
            "findings": _json_safe(row.get("findings", [])),
        }
        for _, row in reports_df.iterrows()
    ]
    evidence = [
        {
            **text_unit,
            "knowledge_elements": (
                next(
                    (
                        unit.knowledge_elements
                        for unit in unit_map.values()
                        if unit.source == text_unit["source"]
                        and unit.page_start <= int(text_unit["page_start"]) <= unit.page_end
                    ),
                    [],
                )
            ),
        }
        for text_unit in text_units
    ]
    return {
        "schema_version": MICROSOFT_GRAPHRAG_SCHEMA_VERSION,
        "nodes": nodes,
        "edges": edges,
        "relationship_mentions": mentions,
        "relationship_links": links,
        "attribute_facts": [],
        "text_units": text_units,
        "evidence": evidence,
        "communities": communities,
        "community_reports": reports,
        "symbol_definitions": [],
        "stats": {
            "entities": len(nodes),
            "relationships": len(edges),
            "relationship_mentions": len(mentions),
            "attribute_facts": 0,
            "text_units": len(text_units),
            "communities": len(communities),
            "extraction_method": "microsoft_graphrag",
            "extraction_model": "qwen3.7-flash",
            "embedding_model": "Qwen3-Embedding-0.6B",
        },
    }


def audit_microsoft_graphrag(graph: dict[str, Any]) -> dict[str, Any]:
    node_ids = {str(node.get("id", "")) for node in graph.get("nodes", [])}
    untyped = sum(
        not str(node.get("entity_type", "")).strip() for node in graph.get("nodes", [])
    )
    invalid_names = sum(
        _invalid_entity(str(node.get("name", "")), str(node.get("entity_type", "")))
        for node in graph.get("nodes", [])
    )
    dangling = sum(
        str(edge.get("source", "")) not in node_ids
        or str(edge.get("target", "")) not in node_ids
        for edge in graph.get("edges", [])
    )
    unsupported = sum(
        not edge.get("evidence_ids") for edge in graph.get("edges", [])
    )
    critical = untyped + invalid_names + dangling + unsupported
    issues = []
    for code, count in (
        ("untyped_entities", untyped),
        ("invalid_entity_names", invalid_names),
        ("dangling_relationships", dangling),
        ("relationships_without_evidence", unsupported),
    ):
        if count:
            issues.append({"severity": "critical", "code": code, "count": count})
    return {
        "schema_version": "1.0-microsoft-graphrag-quality",
        "status": "passed" if critical == 0 else "failed",
        "critical_issues": critical,
        "warning_issues": 0,
        "metrics": {
            "entities": len(node_ids),
            "relationships": len(graph.get("edges", [])),
            "text_units": len(graph.get("text_units", [])),
            "communities": len(graph.get("communities", [])),
        },
        "issues": issues,
    }


def run_microsoft_graphrag(
    units: Iterable[KnowledgeUnit],
    output_dir: Path,
    model_config: BuildModelConfig,
    embedding_model_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run Microsoft GraphRAG as the authoritative semantic index builder."""

    unit_values = list(units)
    if not unit_values:
        raise RuntimeError("Microsoft GraphRAG 没有收到可索引的知识单元")
    if model_config.model != "qwen3.7-flash":
        model_config = BuildModelConfig(
            provider="qwen",
            model="qwen3.7-flash",
            api_key=model_config.api_key,
            base_url=model_config.base_url,
            enable_thinking=False,
        )
    root_dir = (output_dir / "graphrag").resolve()
    output_root = output_dir.resolve()
    if output_root not in root_dir.parents:
        raise RuntimeError("GraphRAG 输出目录越界")
    for child_name in ("output", "vector_store", "logs"):
        child = (root_dir / child_name).resolve()
        if child.exists() and root_dir in child.parents:
            shutil.rmtree(child)
    root_dir.mkdir(parents=True, exist_ok=True)
    config = _graphrag_config(root_dir, model_config, embedding_model_path)

    async def build() -> None:
        from graphrag.api import build_index
        from graphrag.index.workflows.factory import PipelineFactory

        from backend.app.rag.multimodal import CompatibleMultimodalClient

        PipelineFactory.register("extract_graph", _run_textbook_extract_graph)
        PipelineFactory.register(
            "create_community_reports", _run_textbook_community_reports
        )
        graph_client = CompatibleMultimodalClient(model_config)

        results = await build_index(
            config,
            method="standard",
            input_documents=_input_documents(unit_values),
            additional_context={
                "textbook_graph_client": graph_client,
                "textbook_extraction_cache": str(root_dir / "semantic_extractions.jsonl"),
            },
        )
        errors = [
            (result.workflow, error)
            for result in results
            for error in (result.errors or [])
        ]
        if errors:
            workflow, error = errors[0]
            raise RuntimeError(
                f"Microsoft GraphRAG 构建失败（{workflow}）：{error}"
            )

    asyncio.run(build())
    graph = convert_graphrag_outputs(root_dir / "output", unit_values)
    audit = audit_microsoft_graphrag(graph)
    if audit["status"] != "passed":
        raise RuntimeError(
            "Microsoft GraphRAG 质量门禁失败："
            f"{audit['critical_issues']} 个关键问题"
        )
    return graph, audit
