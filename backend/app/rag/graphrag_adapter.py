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
from backend.app.rag.knowledge_document import KnowledgeStatement, KnowledgeUnit
from backend.app.rag.multimodal import BuildModelConfig


MICROSOFT_GRAPHRAG_SCHEMA_VERSION = "3.5-microsoft-graphrag-atomic-facts"
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
    """Feed normalized atomic facts into the official downstream workflows."""

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
    statement_map = additional.get("textbook_statements", {})
    extractions: dict[str, dict[str, Any]] = {}
    if isinstance(statement_map, dict) and statement_map:
        for _, row in text_units_df.iterrows():
            unit_id = str(row["id"])
            statements = [
                statement_map[document_id]
                for document_id in _as_list(row.get("document_ids"))
                if document_id in statement_map
            ]
            relationships: list[dict[str, Any]] = []
            attribute_facts: list[dict[str, Any]] = []
            entities: dict[str, dict[str, Any]] = {}
            for statement in statements:
                if not isinstance(statement, dict):
                    continue
                subject = str(statement.get("subject", "")).strip()
                evidence = str(statement.get("evidence_text", "")).strip()
                if not subject or not evidence:
                    continue
                entities.setdefault(subject, {
                    "name": subject,
                    "type": str(statement.get("subject_type", "课程概念")),
                    "description": evidence,
                })
                if statement.get("statement_type") == "attribute":
                    attribute_facts.append({
                        "subject": subject,
                        "subject_type": str(statement.get("subject_type", "课程概念")),
                        "relation_original": str(statement.get("predicate_original", "")),
                        "relation_normalized": str(statement.get("predicate_normalized", "")),
                        "value": str(statement.get("value", "")),
                        "value_type": str(statement.get("value_type", "text")),
                        "qualifiers": _as_list(statement.get("qualifiers")),
                        "evidence_text": evidence,
                        "evidence_id": str(statement.get("evidence_id", "")),
                        "source_page": int(statement.get("source_page", 0) or 0),
                        "modality": str(statement.get("modality", "text")),
                        "confidence": float(statement.get("confidence", 0) or 0),
                    })
                    continue
                target = str(statement.get("object", "")).strip()
                if not target:
                    continue
                entities.setdefault(target, {
                    "name": target,
                    "type": str(statement.get("object_type", "课程概念")),
                    "description": evidence,
                })
                relationships.append({
                    "source": subject,
                    "target": target,
                    "relation_original": str(statement.get("predicate_original", "")),
                    "relation_normalized": str(statement.get("predicate_normalized", "")),
                    "qualifiers": _as_list(statement.get("qualifiers")),
                    "evidence_text": evidence,
                    "evidence_id": str(statement.get("evidence_id", "")),
                    "source_page": int(statement.get("source_page", 0) or 0),
                    "modality": str(statement.get("modality", "text")),
                    "confidence": float(statement.get("confidence", 0) or 0),
                    "strength": 1.0,
                })
            extractions[unit_id] = {
                "entities": list(entities.values()),
                "relationships": relationships,
                "attribute_facts": attribute_facts,
            }
    else:
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
    relationship_values: dict[tuple[str, str, str], dict[str, Any]] = {}
    attribute_values: list[dict[str, Any]] = []
    for unit in units:
        extraction = extractions.get(unit.id, {})
        descriptions_by_name: dict[str, list[str]] = {}
        for relation in extraction.get("relationships", []):
            if not isinstance(relation, dict):
                continue
            raw_source = str(relation.get("source", "")).strip()
            raw_target = str(relation.get("target", "")).strip()
            source = _canonical_entity_name(raw_source)
            target = _canonical_entity_name(raw_target)
            evidence = str(relation.get("evidence_text", "")).strip()
            if source:
                descriptions_by_name.setdefault(source, []).append(evidence)
            if target:
                descriptions_by_name.setdefault(target, []).append(evidence)
            if not source or not target or not evidence:
                continue
            relation_normalized = str(
                relation.get("relation_normalized", relation.get("relation_original", ""))
            ).strip()
            if (
                _invalid_entity(source, str(next((item.get("type") for item in extraction.get("entities", []) if item.get("name") == raw_source), "课程概念")))
                or _invalid_entity(target, str(next((item.get("type") for item in extraction.get("entities", []) if item.get("name") == raw_target), "课程概念")))
            ):
                continue
            key = (source, target, relation_normalized)
            value = relationship_values.setdefault(key, {
                "source": source,
                "target": target,
                "descriptions": [],
                "relations": [],
                "relation_normalized": relation_normalized,
                "qualifiers": [],
                "evidence_ids": [],
                "source_pages": [],
                "modalities": [],
                "confidences": [],
                "text_unit_ids": [],
                "weight": 0.0,
            })
            if evidence not in value["descriptions"]:
                value["descriptions"].append(evidence)
            relation_original = str(relation.get("relation_original", "")).strip()
            if relation_original and relation_original not in value["relations"]:
                value["relations"].append(relation_original)
            for qualifier in _as_list(relation.get("qualifiers")):
                if qualifier and qualifier not in value["qualifiers"]:
                    value["qualifiers"].append(qualifier)
            if relation.get("evidence_id"):
                value["evidence_ids"].append(str(relation["evidence_id"]))
            if relation.get("source_page"):
                value["source_pages"].append(int(relation["source_page"]))
            if relation.get("modality"):
                value["modalities"].append(str(relation["modality"]))
            if relation.get("confidence") is not None:
                value["confidences"].append(float(relation.get("confidence", 0) or 0))
            value["text_unit_ids"].append(unit.id)
            value["weight"] += float(relation.get("strength", 1.0) or 1.0)
        for fact in extraction.get("attribute_facts", []):
            if not isinstance(fact, dict):
                continue
            subject = str(fact.get("subject", "")).strip()
            evidence = str(fact.get("evidence_text", "")).strip()
            if subject and evidence:
                descriptions_by_name.setdefault(subject, []).append(evidence)
                canonical_subject = _canonical_entity_name(subject)
                subject_type = str(fact.get("subject_type", "课程概念"))
                if not _invalid_entity(canonical_subject, subject_type):
                    attribute_values.append({
                        **fact,
                        "subject": canonical_subject,
                        "text_unit_id": unit.id,
                    })
        for entity in extraction.get("entities", []):
            if not isinstance(entity, dict):
                continue
            title = _canonical_entity_name(str(entity.get("name", "")).strip())
            entity_type = str(entity.get("type", "")).strip()
            if _invalid_entity(title, entity_type):
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
    connected_titles.update(str(value["subject"]) for value in attribute_values)
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
                "relation_normalized": value["relation_normalized"],
                "evidence_texts": value["descriptions"][:4],
                "qualifiers": value["qualifiers"],
                "evidence_ids": list(dict.fromkeys(value["evidence_ids"])),
                "source_pages": sorted(set(value["source_pages"])),
                "modalities": list(dict.fromkeys(value["modalities"])),
                "confidence": min(value["confidences"]) if value["confidences"] else 1.0,
            }, ensure_ascii=False),
            "text_unit_ids": list(dict.fromkeys(value["text_unit_ids"])),
            "weight": value["weight"],
        }
        for value in relationship_values.values()
    ])
    attributes = pd.DataFrame(attribute_values)
    if entities.empty or (relationships.empty and attributes.empty):
        raise RuntimeError("教材 GraphRAG JSON 抽取没有产生可用实体关系")
    await write_table_to_storage(entities, "entities", context.output_storage)
    await write_table_to_storage(relationships, "relationships", context.output_storage)
    await write_table_to_storage(attributes, "attribute_facts", context.output_storage)
    return WorkflowFunctionOutput(result={
        "entities": entities,
        "relationships": relationships,
        "attribute_facts": attributes,
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
        entity_titles = set(entity_records["title"].astype(str))
        relationship_records = relationships.loc[
            relationships["id"].astype(str).isin(relationship_ids)
            | relationships["source"].astype(str).isin(entity_titles)
            | relationships["target"].astype(str).isin(entity_titles)
        ].copy()
        relationship_records["boundary"] = ~relationship_records["id"].astype(str).isin(
            relationship_ids
        )
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
                    "boundary": bool(item.get("boundary", False)),
                }
                for item in relationship_records.head(60).to_dict("records")
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
                # Each input document is already one evidence-backed atomic fact.
                # Metadata must not consume the chunk budget or become extraction text.
                "prepend_metadata": False,
                "chunk_size_includes_metadata": False,
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
        title = " / ".join(unit.title_path) or unit.source
        common_metadata = {
            "knowledge_unit_id": unit.id,
            "source": unit.source,
            "chapter": unit.chapter,
            "section": unit.section,
            "page_start": unit.page_start,
            "page_end": unit.page_end,
            "evidence_ids": unit.evidence_ids,
        }
        if unit.statements:
            for statement in unit.statements:
                rows.append({
                    "id": statement.id,
                    "title": title,
                    "text": statement.graph_text(),
                    "creation_date": "",
                    "metadata": json.dumps({
                        **common_metadata,
                        "statement_id": statement.id,
                        "statement_type": statement.statement_type,
                        "statement_modality": statement.modality,
                        "statement_evidence_id": statement.evidence_id,
                        "statement_source_page": statement.source_page,
                    }, ensure_ascii=False),
                })
            continue
        # Compatibility fallback for old indexes that have not generated statements.
        rows.append({
            "id": unit.id,
            "title": title,
            "text": unit.text,
            "creation_date": "",
            "metadata": json.dumps(common_metadata, ensure_ascii=False),
        })
    return pd.DataFrame(rows)


def _statement_map(units: Iterable[KnowledgeUnit]) -> dict[str, dict[str, Any]]:
    return {
        statement.id: statement.to_dict()
        for unit in units
        for statement in unit.statements
    }


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
    if not name or not entity_type or len(name) > 32:
        return True
    if re.match(r"^(?:图|表|式|第?\d+页)", name):
        return True
    if re.fullmatch(r"[TRCQLDU]_?\d+", name, re.I):
        return True
    if re.match(r"^[TRCQLDU]_?\s*\d+", name, re.I):
        return True
    if re.search(r"[TRCQLDU]_?\d+", name, re.I):
        return True
    if re.match(r"^节点\s*[A-Za-z]?\d*$", name, re.I):
        return True
    if name in {"简单", "元件少", "很小", "很大", "较高", "较低", "流电阻大"}:
        return True
    if re.search(r"(?:稍有上翘|做不到|不能做到|不可能|可得)", name):
        return True
    if name.endswith(("的连接方式", "的限流", "的形状")):
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


def _relationship_payload(value: str) -> dict[str, Any]:
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
        return {
            "relation": relation[:80],
            "relation_normalized": str(
                payload.get("relation_normalized", relation)
            ).strip() or relation,
            "description": evidence or relation,
            "qualifiers": _as_list(payload.get("qualifiers")),
            "evidence_ids": _as_list(payload.get("evidence_ids")),
            "source_pages": [
                int(item) for item in _as_list(payload.get("source_pages"))
                if str(item).isdigit()
            ],
            "modalities": _as_list(payload.get("modalities")),
            "confidence": float(payload.get("confidence", 1.0) or 0.0),
        }
    relation = re.split(r"[。；;]", description, maxsplit=1)[0][:80] or "相关"
    return {
        "relation": relation,
        "relation_normalized": relation,
        "description": description,
        "qualifiers": [],
        "evidence_ids": [],
        "source_pages": [],
        "modalities": [],
        "confidence": 1.0,
    }


def _relationship_content(value: str) -> tuple[str, str]:
    """Compatibility wrapper retained for existing callers and tests."""

    payload = _relationship_payload(value)
    return str(payload["relation"]), str(payload["description"])


def _stable_graph_id(prefix: str, *values: object) -> str:
    raw = "|".join(str(value) for value in values)
    return f"{prefix}:" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def repair_community_relationship_mapping(graph: dict[str, Any]) -> dict[str, int]:
    """Give every relationship a valid, evidence-neutral community context.

    GraphRAG's hierarchical Leiden workflow intentionally emits communities for
    the largest connected component. Textbook graphs commonly contain many
    smaller, valid fact components, so those relationships otherwise have no
    community mapping at all. Preserve Leiden communities and represent every
    uncovered connected component as an explicit deterministic fallback
    community. This only classifies existing nodes and relationships; it never
    creates or rewrites a semantic fact.
    """

    nodes = [item for item in graph.get("nodes", []) if isinstance(item, dict)]
    edges = [item for item in graph.get("edges", []) if isinstance(item, dict)]
    node_ids = {str(item.get("id", "")) for item in nodes if item.get("id")}
    edge_by_id = {
        str(item.get("id", "")): item for item in edges if item.get("id")
    }
    node_by_id = {
        str(item.get("id", "")): item for item in nodes if item.get("id")
    }

    official_communities = [
        item
        for item in graph.get("communities", [])
        if isinstance(item, dict)
        and item.get("algorithm") != "microsoft-graphrag-connected-component-fallback"
    ]
    official_reports = [
        item
        for item in graph.get("community_reports", [])
        if isinstance(item, dict)
        and item.get("algorithm") != "microsoft-graphrag-connected-component-fallback"
    ]

    edge_communities: dict[str, list[str]] = {}
    edge_boundary_communities: dict[str, list[str]] = {}
    for community in official_communities:
        community_id = str(community.get("id", ""))
        entity_ids = list(dict.fromkeys(
            str(item) for item in _as_list(community.get("entity_ids"))
            if str(item) in node_ids
        ))
        original_relationship_ids = community.get(
            "official_relationship_ids", community.get("relationship_ids", [])
        )
        official_relationship_ids = list(dict.fromkeys(
            str(item) for item in _as_list(original_relationship_ids)
            if str(item) in edge_by_id
        ))
        entity_set = set(entity_ids)
        official_relationship_set = set(official_relationship_ids)
        inferred_relationship_ids = [
            edge_id
            for edge_id, edge in edge_by_id.items()
            if edge_id not in official_relationship_set
            and str(edge.get("source", "")) in entity_set
            and str(edge.get("target", "")) in entity_set
        ]
        relationship_ids = list(dict.fromkeys([
            *official_relationship_ids, *inferred_relationship_ids
        ]))
        relationship_set = set(relationship_ids)
        boundary_relationship_ids = [
            edge_id
            for edge_id, edge in edge_by_id.items()
            if edge_id not in relationship_set
            and (
                str(edge.get("source", "")) in entity_set
                or str(edge.get("target", "")) in entity_set
            )
        ]
        community["entity_ids"] = entity_ids
        community["official_relationship_ids"] = official_relationship_ids
        community["inferred_relationship_ids"] = inferred_relationship_ids
        community["relationship_ids"] = relationship_ids
        community["boundary_relationship_ids"] = boundary_relationship_ids
        community["size"] = len(entity_ids)
        for edge_id in relationship_ids:
            edge_communities.setdefault(edge_id, []).append(community_id)
        for edge_id in boundary_relationship_ids:
            edge_boundary_communities.setdefault(edge_id, []).append(community_id)

    uncovered_edge_ids = {
        edge_id
        for edge_id in edge_by_id
        if not edge_communities.get(edge_id)
        and not edge_boundary_communities.get(edge_id)
    }
    adjacency: dict[str, set[str]] = {}
    incident_edges: dict[str, set[str]] = {}
    for edge_id in uncovered_edge_ids:
        edge = edge_by_id[edge_id]
        source = str(edge.get("source", ""))
        target = str(edge.get("target", ""))
        if source not in node_ids or target not in node_ids:
            continue
        adjacency.setdefault(source, set()).add(target)
        adjacency.setdefault(target, set()).add(source)
        incident_edges.setdefault(source, set()).add(edge_id)
        incident_edges.setdefault(target, set()).add(edge_id)

    numeric_communities = [
        int(item.get("community", 0) or 0)
        for item in official_communities
        if str(item.get("community", "")).lstrip("-").isdigit()
    ]
    next_community_number = max(numeric_communities, default=-1) + 1
    fallback_communities: list[dict[str, Any]] = []
    fallback_reports: list[dict[str, Any]] = []
    visited: set[str] = set()
    for start in sorted(adjacency):
        if start in visited:
            continue
        component_nodes: set[str] = set()
        stack = [start]
        while stack:
            node_id = stack.pop()
            if node_id in visited:
                continue
            visited.add(node_id)
            component_nodes.add(node_id)
            stack.extend(adjacency.get(node_id, set()) - visited)
        component_edges = sorted({
            edge_id
            for node_id in component_nodes
            for edge_id in incident_edges.get(node_id, set())
        })
        if not component_edges:
            continue
        component_node_ids = sorted(component_nodes)
        community_id = _stable_graph_id(
            "community-fallback", *component_edges
        )
        degree = {
            node_id: len(incident_edges.get(node_id, set()))
            for node_id in component_node_ids
        }
        representative_ids = sorted(
            component_node_ids,
            key=lambda node_id: (
                -degree[node_id],
                -int(node_by_id[node_id].get("evidence_count", 0) or 0),
                str(node_by_id[node_id].get("name", "")),
            ),
        )
        representative_names = [
            str(node_by_id[node_id].get("name", node_id))
            for node_id in representative_ids[:3]
        ]
        title = "、".join(representative_names)
        if len(component_node_ids) > len(representative_names):
            title += "等概念"
        text_unit_ids = list(dict.fromkeys(
            str(text_unit_id)
            for edge_id in component_edges
            for text_unit_id in _as_list(
                edge_by_id[edge_id].get("text_unit_ids")
            )
            if str(text_unit_id)
        ))
        source_pages = sorted({
            int(page)
            for edge_id in component_edges
            for page in _as_list(edge_by_id[edge_id].get("source_pages"))
            if str(page).isdigit()
        })
        community_number = next_community_number + len(fallback_communities)
        fallback_community = {
            "id": community_id,
            "community": community_number,
            "parent": None,
            "children": [],
            "level": 0,
            "title": title,
            "entity_ids": component_node_ids,
            "official_relationship_ids": [],
            "inferred_relationship_ids": component_edges,
            "relationship_ids": component_edges,
            "boundary_relationship_ids": [],
            "text_unit_ids": text_unit_ids,
            "source_pages": source_pages,
            "size": len(component_node_ids),
            "algorithm": "microsoft-graphrag-connected-component-fallback",
            "fallback_reason": "not_emitted_by_hierarchical_leiden",
        }
        fallback_communities.append(fallback_community)
        finding_edges = [edge_by_id[edge_id] for edge_id in component_edges[:5]]
        fallback_reports.append({
            "id": _stable_graph_id("community-report-fallback", community_id),
            "community": community_number,
            "level": 0,
            "title": title,
            "summary": (
                f"该连通分量包含 {len(component_node_ids)} 个实体和 "
                f"{len(component_edges)} 条已有证据关系。"
            ),
            "full_content": "\n".join(
                str(edge.get("description", "")) for edge in finding_edges
                if str(edge.get("description", "")).strip()
            ),
            "rank": float(len(component_edges)),
            "findings": [
                {
                    "summary": (
                        f"{node_by_id[str(edge['source'])].get('name', edge['source'])} "
                        f"— {edge.get('relation', edge.get('type', '相关'))} — "
                        f"{node_by_id[str(edge['target'])].get('name', edge['target'])}"
                    ),
                    "explanation": str(edge.get("description", "")),
                }
                for edge in finding_edges
            ],
            "algorithm": "microsoft-graphrag-connected-component-fallback",
        })
        for edge_id in component_edges:
            edge_communities.setdefault(edge_id, []).append(community_id)

    graph["communities"] = [*official_communities, *fallback_communities]
    graph["community_reports"] = [*official_reports, *fallback_reports]
    for edge_id, edge in edge_by_id.items():
        community_ids = list(dict.fromkeys(edge_communities.get(edge_id, [])))
        boundary_ids = list(dict.fromkeys(
            edge_boundary_communities.get(edge_id, [])
        ))
        edge["community_ids"] = community_ids
        edge["boundary_community_ids"] = boundary_ids
        edge["cross_community"] = bool(boundary_ids)
        if any(item.startswith("community-fallback:") for item in community_ids):
            edge["community_mapping_source"] = "connected_component_fallback"
        elif community_ids:
            edge["community_mapping_source"] = "hierarchical_leiden"
        elif boundary_ids:
            edge["community_mapping_source"] = "hierarchical_leiden_boundary"
        else:
            edge["community_mapping_source"] = "unclassified"

    summary = {
        "official_communities": len(official_communities),
        "fallback_communities": len(fallback_communities),
        "communities": len(graph["communities"]),
        "relationships": len(edges),
        "intra_community_relationships": sum(
            bool(item.get("community_ids")) for item in edges
        ),
        "boundary_relationships": sum(
            not item.get("community_ids")
            and bool(item.get("boundary_community_ids"))
            for item in edges
        ),
        "unclassified_relationships": sum(
            not item.get("community_ids")
            and not item.get("boundary_community_ids")
            for item in edges
        ),
    }
    graph.setdefault("stats", {}).update(summary)
    return summary


def convert_graphrag_outputs(
    output_path: Path,
    units: Iterable[KnowledgeUnit],
) -> dict[str, Any]:
    """Convert official Parquet outputs to the project's evidence-rich graph API."""

    unit_values = list(units)
    unit_map = {unit.id: unit for unit in unit_values}
    statement_map = {
        statement.id: (unit, statement)
        for unit in unit_values
        for statement in unit.statements
    }
    entities_df = pd.read_parquet(output_path / "entities.parquet")
    relationships_df = pd.read_parquet(output_path / "relationships.parquet")
    text_units_df = pd.read_parquet(output_path / "text_units.parquet")
    communities_df = pd.read_parquet(output_path / "communities.parquet")
    reports_df = pd.read_parquet(output_path / "community_reports.parquet")
    attributes_path = output_path / "attribute_facts.parquet"
    attributes_df = (
        pd.read_parquet(attributes_path)
        if attributes_path.exists()
        else pd.DataFrame()
    )

    text_units: list[dict[str, Any]] = []
    text_unit_context: dict[str, dict[str, Any]] = {}
    for _, row in text_units_df.iterrows():
        document_ids = _as_list(row.get("document_ids"))
        statement_pair = next(
            (statement_map[item] for item in document_ids if item in statement_map),
            None,
        )
        statement = statement_pair[1] if statement_pair else None
        unit = statement_pair[0] if statement_pair else next(
            (unit_map[item] for item in document_ids if item in unit_map), None
        )
        page_start = (
            int(statement.source_page)
            if statement and statement.source_page
            else (unit.page_start if unit else 0)
        )
        page_end = page_start or (unit.page_end if unit else 0)
        evidence_ids = (
            [statement.evidence_id]
            if statement and statement.evidence_id
            else (unit.evidence_ids if unit else [])
        )
        context = {
            "source": unit.source if unit else "",
            "page_start": page_start,
            "page_end": page_end,
            "chapter": unit.chapter if unit else "",
            "section": unit.section if unit else "",
            "block_ids": evidence_ids,
            "knowledge_unit_id": unit.id if unit else "",
            "statement_ids": [
                item for item in document_ids if item in statement_map
            ],
            "evidence_metadata": (
                statement.evidence_metadata if statement else {}
            ),
        }
        text_unit = {
            "id": str(row["id"]),
            "text": str(row.get("text", "")),
            **context,
            "modality": statement.modality if statement else "multimodal_knowledge",
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
                "source_pages": pages,
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
            existing["source_pages"] = existing["pages"]
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
        payload = _relationship_payload(str(row.get("description", "")))
        relation = str(payload["relation"])
        relation_normalized = str(payload["relation_normalized"])
        description = str(payload["description"])
        if not source or not target or source == target or not description:
            continue
        edge_key = (source, target, relation_normalized)
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)
        text_unit_ids = _as_list(row.get("text_unit_ids"))
        contextual_evidence = list(dict.fromkeys(
            evidence_id
            for text_unit_id in text_unit_ids
            for evidence_id in _as_list(
                text_unit_context.get(text_unit_id, {}).get("block_ids")
            )
        ))
        evidence_ids = list(dict.fromkeys([
            *_as_list(payload.get("evidence_ids")), *contextual_evidence
        ]))
        source_pages = sorted(set([
            *[int(item) for item in payload.get("source_pages", [])],
            *[
                int(text_unit_context.get(item, {}).get("page_start", 0) or 0)
                for item in text_unit_ids
                if text_unit_context.get(item, {}).get("page_start")
            ],
        ]))
        modalities = list(dict.fromkeys([
            *_as_list(payload.get("modalities")),
            *[
                str(text_unit_context.get(item, {}).get("modality", ""))
                for item in text_unit_ids
                if text_unit_context.get(item, {}).get("modality")
            ],
        ]))
        qualifiers = _as_list(payload.get("qualifiers"))
        edge_id = _stable_graph_id(
            "relationship", source, target, relation_normalized
        )
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
                "relation_normalized": relation_normalized,
                "evidence_text": description,
                "strength": float(row.get("weight", 1.0) or 1.0),
                "text_unit_id": text_unit_id,
                "evidence_id": (
                    evidence_ids[0] if evidence_ids else text_unit_id
                ),
                "source_modality": (
                    modalities[0] if modalities else "multimodal_knowledge"
                ),
                "source_page": int(context.get("page_start", 0) or 0),
                "qualifier_text": "；".join(qualifiers),
                "confidence": float(payload.get("confidence", 1.0) or 0.0),
            })
        edge = {
            "id": edge_id,
            "source": source,
            "target": target,
            "type": relation,
            "relation": relation,
            "relation_normalized": relation_normalized,
            "description": description,
            "mention_ids": mention_ids,
            "text_unit_ids": text_unit_ids,
            "evidence_ids": evidence_ids,
            "evidence_count": len(evidence_ids),
            "source_pages": source_pages,
            "modalities": modalities,
            "confidence": float(payload.get("confidence", 1.0) or 0.0),
            "weight": float(row.get("weight", 1.0) or 1.0),
            "qualifier_texts": qualifiers,
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

    attribute_facts: list[dict[str, Any]] = []
    for _, row in attributes_df.iterrows():
        subject_name = _canonical_entity_name(str(row.get("subject", "")))
        subject = title_to_node.get(_entity_key(subject_name))
        evidence_text = str(row.get("evidence_text", "")).strip()
        if not subject or not evidence_text:
            continue
        text_unit_id = str(row.get("text_unit_id", ""))
        context = text_unit_context.get(text_unit_id, {})
        evidence_ids = list(dict.fromkeys([
            *_as_list(row.get("evidence_id")),
            *_as_list(context.get("block_ids")),
        ]))
        source_page = int(
            row.get("source_page", 0) or context.get("page_start", 0) or 0
        )
        relation = str(row.get("relation_original", "")).strip() or "具有属性"
        relation_normalized = str(
            row.get("relation_normalized", relation)
        ).strip() or relation
        value = str(row.get("value", "")).strip()
        fact_id = _stable_graph_id(
            "attribute", subject, relation_normalized, value, text_unit_id
        )
        attribute_facts.append({
            "id": fact_id,
            "subject": subject,
            "subject_name": node_lookup[subject]["name"],
            "relation": relation,
            "relation_normalized": relation_normalized,
            "value": value,
            "value_type": str(row.get("value_type", "text")),
            "qualifiers": _as_list(row.get("qualifiers")),
            "evidence_text": evidence_text,
            "evidence_ids": evidence_ids,
            "text_unit_id": text_unit_id,
            "source_page": source_page,
            "modality": str(
                row.get("modality", context.get("modality", "text"))
            ),
            "confidence": float(row.get("confidence", 0.0) or 0.0),
        })
        node_lookup[subject]["evidence_count"] = max(
            int(node_lookup[subject].get("evidence_count", 0)), len(evidence_ids)
        )

    used_node_ids = {
        node_id
        for edge in edges
        for node_id in (str(edge["source"]), str(edge["target"]))
    }
    attribute_node_ids = {
        str(fact["subject"]) for fact in attribute_facts
    }
    used_node_ids.update(attribute_node_ids)
    nodes = [node for node in nodes if node["id"] in used_node_ids]
    node_lookup = {node["id"]: node for node in nodes}
    for node in nodes:
        node["standalone"] = (
            node["id"] in attribute_node_ids
            and int(node.get("relationship_count", 0)) == 0
        )

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
    block_evidence_by_id: dict[str, dict[str, Any]] = {}
    for unit in unit_values:
        for source in unit.text_evidence:
            if source.get("id"):
                block_evidence_by_id[str(source["id"])] = dict(source)
        for element in unit.knowledge_elements:
            if element.get("id"):
                block_evidence_by_id[str(element["id"])] = {
                    "id": str(element["id"]),
                    "page": int(element.get("page", unit.page_start) or unit.page_start),
                    "modality": str(element.get("type", "multimodal")),
                    "text": str(element.get("raw_text", "")),
                    "bbox": list(element.get("bbox", [])),
                    "polygon": list(element.get("polygon", [])),
                    "confidence": float(element.get("confidence", 0.0) or 0.0),
                    "processor": str(element.get("processor", "")),
                    "ocr_block_id": element.get("ocr_block_id"),
                    "table_cells": list(element.get("table_cells", [])),
                    "evidence_metadata": dict(element.get("evidence_metadata", {})),
                }
    evidence = [
        {
            **text_unit,
            "knowledge_elements": (
                unit_map.get(str(text_unit.get("knowledge_unit_id", ""))).knowledge_elements
                if str(text_unit.get("knowledge_unit_id", "")) in unit_map
                else []
            ),
            "block_evidence": [
                block_evidence_by_id[block_id]
                for block_id in _as_list(text_unit.get("block_ids"))
                if block_id in block_evidence_by_id
            ],
        }
        for text_unit in text_units
    ]
    graph = {
        "schema_version": MICROSOFT_GRAPHRAG_SCHEMA_VERSION,
        "nodes": nodes,
        "edges": edges,
        "relationship_mentions": mentions,
        "relationship_links": links,
        "attribute_facts": attribute_facts,
        "text_units": text_units,
        "evidence": evidence,
        "communities": communities,
        "community_reports": reports,
        "symbol_definitions": [],
        "stats": {
            "entities": len(nodes),
            "relationships": len(edges),
            "relationship_mentions": len(mentions),
            "attribute_facts": len(attribute_facts),
            "text_units": len(text_units),
            "communities": len(communities),
            "atomic_statements": len(statement_map),
            "knowledge_units": len(unit_values),
            "knowledge_units_without_statements": sum(
                not unit.statements for unit in unit_values
            ),
            "knowledge_elements": sum(
                bool(element.get("included_in_graph"))
                for unit in unit_values for element in unit.knowledge_elements
            ),
            "knowledge_elements_with_statements": len({
                statement.evidence_id
                for _, statement in statement_map.values()
                if statement.evidence_id
                and any(
                    statement.evidence_id == str(element.get("id", ""))
                    for unit in unit_values for element in unit.knowledge_elements
                    if element.get("included_in_graph")
                )
            }),
            "expected_modalities": sorted({
                str(element.get("type", ""))
                for unit in unit_values for element in unit.knowledge_elements
                if element.get("included_in_graph") and element.get("type")
            }),
            "statement_modalities": {
                modality: sum(
                    statement.modality == modality
                    for _, statement in statement_map.values()
                )
                for modality in sorted({
                    statement.modality for _, statement in statement_map.values()
                })
            },
            "extraction_method": "microsoft_graphrag",
            "extraction_model": "qwen3.7-flash",
            "embedding_model": "Qwen3-Embedding-0.6B",
        },
    }
    repair_community_relationship_mapping(graph)
    return graph


def audit_microsoft_graphrag(graph: dict[str, Any]) -> dict[str, Any]:
    node_ids = {str(node.get("id", "")) for node in graph.get("nodes", [])}
    edges = graph.get("edges", [])
    edge_ids = {str(edge.get("id", "")) for edge in edges}
    attributes = graph.get("attribute_facts", [])
    text_units = graph.get("text_units", [])
    untyped = sum(
        not str(node.get("entity_type", "")).strip() for node in graph.get("nodes", [])
    )
    invalid_names = sum(
        _invalid_entity(str(node.get("name", "")), str(node.get("entity_type", "")))
        and not (
            node.get("generated_by") == "graph_consolidation"
            and node.get("entity_type") in {"教材结构", "知识属性"}
            and bool(str(node.get("name", "")).strip())
        )
        for node in graph.get("nodes", [])
    )
    dangling = sum(
        str(edge.get("source", "")) not in node_ids
        or str(edge.get("target", "")) not in node_ids
        for edge in edges
    )
    dangling_attributes = sum(
        str(fact.get("subject", "")) not in node_ids for fact in attributes
    )
    unsupported_relationships = sum(
        not edge.get("evidence_ids") or not str(edge.get("description", "")).strip()
        for edge in edges
    )
    unsupported_attributes = sum(
        not fact.get("evidence_ids") or not str(fact.get("evidence_text", "")).strip()
        for fact in attributes
    )
    missing_pages = sum(
        not node.get("source_pages") for node in graph.get("nodes", [])
    )
    used_node_ids = {
        str(node_id)
        for edge in edges
        for node_id in (edge.get("source", ""), edge.get("target", ""))
    }
    used_node_ids.update(str(fact.get("subject", "")) for fact in attributes)
    isolated = len(node_ids - used_node_ids)
    fact_text_unit_ids = {
        str(item)
        for edge in edges
        for item in _as_list(edge.get("text_unit_ids"))
    }
    fact_text_unit_ids.update(
        str(fact.get("text_unit_id", "")) for fact in attributes
        if fact.get("text_unit_id")
    )
    text_unit_ids = {str(item.get("id", "")) for item in text_units}
    fact_coverage = (
        len(fact_text_unit_ids & text_unit_ids) / len(text_unit_ids)
        if text_unit_ids else 0.0
    )
    unclassified_community_edges = sum(
        not edge.get("community_ids")
        and not edge.get("boundary_community_ids")
        for edge in edges
    )
    community_context_coverage = (
        sum(
            bool(edge.get("community_ids") or edge.get("boundary_community_ids"))
            for edge in edges
        ) / len(edges)
        if edges else 1.0
    )
    community_by_id = {
        str(item.get("id", "")): item
        for item in graph.get("communities", [])
        if isinstance(item, dict) and item.get("id")
    }

    def invalid_community_mapping(edge: dict[str, Any]) -> bool:
        edge_id = str(edge.get("id", ""))
        source = str(edge.get("source", ""))
        target = str(edge.get("target", ""))
        for community_id in _as_list(edge.get("community_ids")):
            community = community_by_id.get(str(community_id))
            if community is None:
                return True
            members = {str(item) for item in _as_list(community.get("entity_ids"))}
            relationships = {
                str(item) for item in _as_list(community.get("relationship_ids"))
            }
            if source not in members or target not in members:
                return True
            if edge_id not in relationships:
                return True
        for community_id in _as_list(edge.get("boundary_community_ids")):
            community = community_by_id.get(str(community_id))
            if community is None:
                return True
            members = {str(item) for item in _as_list(community.get("entity_ids"))}
            boundaries = {
                str(item)
                for item in _as_list(community.get("boundary_relationship_ids"))
            }
            if source not in members and target not in members:
                return True
            if edge_id not in boundaries:
                return True
        return False

    invalid_community_mappings = sum(
        invalid_community_mapping(edge) for edge in edges
    )
    dangling_community_members = sum(
        sum(str(item) not in node_ids for item in _as_list(community.get("entity_ids")))
        + sum(
            str(item) not in edge_ids
            for item in _as_list(community.get("relationship_ids"))
        )
        for community in community_by_id.values()
    )
    no_facts = int(not edges and not attributes)
    units_without_facts = int(
        graph.get("stats", {}).get("knowledge_units_without_statements", 0) or 0
    )
    statement_modalities = set(
        graph.get("stats", {}).get("statement_modalities", {})
    )
    expected_modalities = set(
        graph.get("stats", {}).get("expected_modalities", [])
    )
    missing_modalities = sorted(expected_modalities - statement_modalities)
    critical = sum((
        untyped,
        invalid_names,
        dangling,
        dangling_attributes,
        unsupported_relationships,
        unsupported_attributes,
        isolated,
        unclassified_community_edges,
        invalid_community_mappings,
        dangling_community_members,
        no_facts,
        units_without_facts,
        len(missing_modalities),
    ))
    issues = []
    for code, count in (
        ("untyped_entities", untyped),
        ("invalid_entity_names", invalid_names),
        ("dangling_relationships", dangling),
        ("dangling_attribute_facts", dangling_attributes),
        ("relationships_without_evidence", unsupported_relationships),
        ("attributes_without_evidence", unsupported_attributes),
        ("isolated_entities", isolated),
        ("unclassified_community_relationships", unclassified_community_edges),
        ("invalid_community_relationship_mappings", invalid_community_mappings),
        ("dangling_community_members", dangling_community_members),
        ("graph_without_facts", no_facts),
        ("knowledge_units_without_facts", units_without_facts),
        ("knowledge_modalities_without_facts", len(missing_modalities)),
    ):
        if count:
            issues.append({"severity": "critical", "code": code, "count": count})
    warnings: list[tuple[str, int]] = []
    if missing_pages:
        warnings.append(("entities_without_source_pages", missing_pages))
    uncovered_text_units = len(text_unit_ids - fact_text_unit_ids)
    if text_unit_ids and fact_coverage < 0.75:
        warnings.append(("low_atomic_fact_coverage", uncovered_text_units))
    expected_elements = int(graph.get("stats", {}).get("knowledge_elements", 0) or 0)
    covered_elements = int(
        graph.get("stats", {}).get("knowledge_elements_with_statements", 0) or 0
    )
    element_coverage = covered_elements / expected_elements if expected_elements else 1.0
    if expected_elements and element_coverage < 0.5:
        warnings.append(("low_multimodal_element_fact_coverage", expected_elements - covered_elements))
    for code, count in warnings:
        issues.append({"severity": "warning", "code": code, "count": count})
    return {
        "schema_version": "2.0-microsoft-graphrag-quality",
        "status": "passed" if critical == 0 else "failed",
        "critical_issues": critical,
        "warning_issues": sum(count for _, count in warnings),
        "metrics": {
            "entities": len(node_ids),
            "relationships": len(edges),
            "attribute_facts": len(attributes),
            "facts": len(edges) + len(attributes),
            "text_units": len(text_units),
            "text_unit_fact_coverage": round(fact_coverage, 4),
            "isolated_entities": isolated,
            "entities_without_source_pages": missing_pages,
            "cross_community_relationships": sum(
                bool(edge.get("cross_community")) for edge in edges
            ),
            "community_context_relationship_coverage": round(
                community_context_coverage, 4
            ),
            "invalid_community_relationship_mappings": invalid_community_mappings,
            "communities": len(graph.get("communities", [])),
            "official_communities": sum(
                item.get("algorithm")
                != "microsoft-graphrag-connected-component-fallback"
                for item in graph.get("communities", [])
                if isinstance(item, dict)
            ),
            "fallback_communities": sum(
                item.get("algorithm")
                == "microsoft-graphrag-connected-component-fallback"
                for item in graph.get("communities", [])
                if isinstance(item, dict)
            ),
            "knowledge_units_without_facts": units_without_facts,
            "missing_knowledge_modalities": missing_modalities,
            "multimodal_element_fact_coverage": round(element_coverage, 4),
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
                "textbook_statements": _statement_map(unit_values),
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
    from backend.app.rag.graph_consolidation import consolidate_semantic_graph

    graph, consolidation_audit = consolidate_semantic_graph(
        graph, embedding_model_path
    )
    audit = audit_microsoft_graphrag(graph)
    audit["consolidation"] = consolidation_audit
    (output_dir / "semantic_knowledge_graph.json").write_text(
        json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "semantic_quality_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "semantic_consolidation_audit.json").write_text(
        json.dumps(consolidation_audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if audit["status"] != "passed" or consolidation_audit["status"] != "passed":
        raise RuntimeError(
            "Microsoft GraphRAG 质量门禁失败："
            f"{audit['critical_issues'] + consolidation_audit['critical_issues']} "
            "个关键问题"
        )
    return graph, audit
