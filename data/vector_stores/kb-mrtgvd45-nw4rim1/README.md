# 电子电路 Schema 4 知识库快照

本目录是 CircuitMind 当前使用的“电子电路”知识库快照，可直接由运行时加载。

- 知识库 ID：`kb-mrtgvd45-nw4rim1`
- Schema：`4.0-hierarchical-summary-entity-graph`
- 正文 Chunk：2,036
- 章节节点：202
- 实体节点：1,137
- 概念关系：1,268
- 完整图谱：1,339 个节点、3,319 条边
- 文本与实体向量：`Qwen3-Embedding-0.6B`，1,024 维、归一化
- OCR：`PaddleOCR-VL-1.6` API
- 图像理解与图谱语义模型：`qwen3.7-flash`
- 习题章节：已从运行时证据、实体、关系和属性事实中滤除
- 质量审计：通过，悬空关系、无效证据和孤立实体均为 0

## 已包含

- `index_meta.json`：运行时元数据和质量审计
- `chunks.jsonl`、`vectors.faiss`：正文检索数据
- `semantic_knowledge_graph.json`：Schema 4 权威图谱
- `entity_vectors.faiss`、`entity_embeddings.npy`：实体向量
- `attribute_facts.jsonl`：公式、参数、表格和图片总结事实
- `evidence_store.jsonl`：Paddle 块级证据
- `artifacts/`：当前证据引用所需的图片、表格和公式裁剪
- Paddle OCR、PDF-Extract-Kit 与分阶段构建缓存，用于免 OCR 续建

## 未包含

- `backups/` 中的习题过滤前快照
- 旧视觉模型归档目录
- 构建日志、失败日志和临时文件
- 任何 API 密钥或服务端凭据

克隆后需执行 `git lfs pull` 获取向量、JSONL、大型图谱 JSON 和证据图片。原始教材 PDF 位于 `RAG_Resources/knowledge_bases/kb-mrtgvd45-nw4rim1/`，同样由 Git LFS 管理。
