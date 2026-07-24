# CircuitMind 多智能体电路课程教学平台

这是一个本地优先的电路课程教学 MVP。当前已完成学生学习工作台与教师作业工作台；当前版本按单教师、单学生流程运行，班级与学生管理暂不在范围内。

已跑通的链路：

- 学生端对话答题及图片/文档附件识别使用页面当前配置的模型；私有思考字段不会返回前端。
- LangGraph 编排的大模型路由 Agent、答疑 Agent、检索 Agent、出题 Agent、学习规划 Agent 和 SymPy 验算 Agent；学习规划会结合知识库资料生成可执行路线。
- 图片出题先提取“电路拓扑、已知量、特殊条件、待求量”蓝图；连续“再出一道”会沿用最近生成题，同类题不调用知识库检索并必须通过同构校验。
- 教材清洗、章节/段落语义切分、章/节/原始页码元数据、384 维向量化和 populated FAISS/Qdrant 索引；Excel/JSON 题库与课程知识库严格隔离。
- 向量语义检索 + BM25 关键词检索 + 规则重排。
- FastAPI、CORS、统一异常处理、日志、POST SSE 真正 token 流式输出、上传与后台重建知识库。
- Redis 最近 N 轮会话记忆；Redis 不可用时自动切换本地持久化记忆，服务重启后仍可执行出题去重。
- 学生交互栏支持题目图片和 PDF/Word/Excel/Markdown 等附件；PDF 页面和 Office 内嵌图片会交给当前配置且支持视觉输入的模型识别公式、电路图与题干。
- React + TypeScript + Ant Design + Zustand + KaTeX 学生端，含 LaTeX 定界符容错预处理。
- 右上角可切换本地 Ollama、DeepSeek、通义千问或自定义兼容 API；配置会保存在当前浏览器，也可通过后端环境变量提供。
- 左侧“最近学习”读取持久化会话列表，支持点击恢复历史对话；刷新页面后会自动恢复当前会话。
- 学生端知识图谱默认展示聚合后的“教材—页面—知识点—电路图—去重元件”语义关系；公式、文本片段和网络节点保留在底层图中作为检索证据，不直接铺到画布上。
- 答疑、AI 出题和外部题库内容均可加入持久化错题本；支持来源追踪、分类、批注、知识图谱对齐、章节/前置知识定位和确定性薄弱点学习规划。实现、接口与迁移约定见 [错题本集成说明](docs/MISTAKE_BOOK.md)。
- 教师可上传试卷、课后习题、学习指导书、图片或扫描版习题册；`qwen3-vl-flash` 联合 PDF-Extract-Kit 过滤目录、知识讲解等非题目内容，按题提取题号、共同题干、分层小问、选项、所属插图、参考答案与评分点，并重排为可打印的作业内容和参考答案。
- 学生端“我的作业”支持多张作答照片提交；`qwen3-vl-flash` 识别手写内容并评分，`qwen3-vl-8b-instruct` 独立复核漏题、错读、步骤分与总分，疑点会标记为教师复查。

## 当前数据成果

默认知识库先按 MVP 范围索引《模拟电子技术基础》第一章：

- 教材范围：PDF 第 25–94 页，共 69 个有效文本页。
- 示例题库文件仍作为独立资源保留，但不会进入课程知识库、知识图谱或检索。
- 向量库：83 个纯教材 Chunk，向量维度 384，题库 Chunk 为 0，状态 `populated`。
- 元数据：每个教材 Chunk 保留来源、章、节、PDF 页码和知识点标签。

主要产物位于：

```text
RAG_Resources/
  模拟电子技术基础-童诗白.pdf
  电路课程示例题库.xlsx  # 独立题库，不参与知识库构建
data/vector_stores/default/
  cleaned_documents/
  chunks.jsonl
  question_bank.json      # 隔离审计占位，不含题目
  vectors.faiss
  index_meta.json
```

## 架构

```mermaid
flowchart LR
  S[学生 Web] -->|POST SSE| API[FastAPI]
  API --> M[Redis 会话记忆]
  API --> R{路由 Agent}
  R -->|答疑| A[答疑 Agent]
  R -->|出题| Q[出题 Agent]
  R -->|学习规划| LP[学习规划 Agent]
  A --> W[Query 改写]
  Q --> K[原题或最近题蓝图提取]
  W --> H[向量 + BM25 + Rerank]
  H --> V[(FAISS + Chunk 元数据)]
  LP --> H
  A --> L[当前配置的回答模型]
  K --> L
  Q --> P[知识点校验 + 会话去重 + SymPy 验算]
  API --> U[上传与后台知识库重建]
```

## 直接启动

若使用本地模型，可启动 Ollama 并确认存在可用模型：

```powershell
ollama list
```

Ollama 未启动不会阻止程序启动；可以先在页面配置云端 API，后续启动 Ollama 后重新打开模型弹窗即可选择刚发现的本地模型。

模型权重不会提交到 GitHub。首次运行先下载项目使用的嵌入模型：

```powershell
conda activate llm
python scripts/download_embedding_model.py
```

脚本只下载 `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` 推理所需文件到
`models/paraphrase-multilingual-MiniLM-L12-v2`，不会读取或写入任何 API Key。

项目包含默认向量库。推荐在项目根目录使用启动脚本：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start.ps1
```

脚本会先在 `frontend` 中执行锁定依赖安装（`npm ci --no-audit --no-fund`）和 `npm run build`，确保 FastAPI 提供的前端产物与当前源码一致，然后使用 `llm` 环境启动后端。构建或依赖安装失败时，脚本会停止而不会继续提供旧页面。

打开 `http://127.0.0.1:8000/student`；教师作业工作台位于 `http://127.0.0.1:8000/teacher`。生产构建由 FastAPI 直接提供；开发前端可在 `frontend` 中运行 `npm run dev`，Vite 会代理 `/api` 到 8000 端口。

## 作业布置与双模型批改

教师端上传附件后会在后台执行以下流程：

1. PDF 页面渲染或图片标准化，PDF-Extract-Kit 给出版面区域候选。
2. `qwen3-vl-flash` 逐页过滤非题目内容，返回题型、题目标识、共同题干/分层小问/选项/插图/答案 bbox、参考答案和评分点；选择题缺项时会自动二次校对，跨页题干合并时会去除重复复述。
3. 系统保留大题层级、题号、题干换行、选项分栏和题图顺序，以结构化数据重新排版；题图单独裁取后插回对应题目，不再生成“截图 + 白块遮答案”的题面。
4. 教师预览后发布，学生端只取得不含 `answer`、`rubric` 和原始附件地址的数据。
5. 学生拍照提交后，`qwen3-vl-flash` 识别并逐题评分，再由 `qwen3-vl-8b-instruct` 独立复核；复核不通过的提交进入“待教师复查”。

需要在 `.env` 配置 `QWEN_API_KEY`。相关模型与限制可通过 `QWEN_HOMEWORK_EXTRACTION_MODEL`、`QWEN_HOMEWORK_GRADING_MODEL`、`QWEN_HOMEWORK_REVIEW_MODEL`、`MAX_HOMEWORK_UPLOAD_MB` 和 `MAX_HOMEWORK_ANSWER_IMAGES` 调整。作业及提交保存在 `data/homework/`。

## 模型切换与 API 配置

点击学生端右上角的模型名称可选择本地 Ollama、DeepSeek、通义千问或自定义 OpenAI 兼容 API。当前选择同时用于文本答题和附件理解，最终答案始终由该模型生成。

- 检索阶段可以临时调用文本或多模态 Embedding、BM25、知识图谱和重排器；这些专用模型只产生检索依据，不会替换当前回答模型。
- `qwen3-vl-embedding` 仅用于知识库多模态向量化，不能作为聊天模型，因此在模型列表中保持禁用。
- 上传或重建知识库仍固定使用 `qwen3-vl-flash` 做视觉/OCR 分析，并按需调用 `qwen3-vl-embedding`；完成后不会改变页面的回答模型配置。

页面输入的模型配置和 API Key 会写入当前浏览器的 `localStorage`，不会写入项目文件；配置弹窗提供清除入口。公用电脑不建议保存云端密钥。也可以在 `.env` 配置对应服务的 API Key 和 Base URL。使用云端模型时，题目、最近对话、检索上下文及附件视觉内容会发送到所选 API。

## 环境重建

所有 Python 操作均在现有 `llm` 环境中进行：

```powershell
conda activate llm
python -m pip install -r requirements.txt
python scripts/download_embedding_model.py
python scripts/build_knowledge_base.py --chapter-limit 1
cd frontend
npm install
npm run build
```

如果需要索引教材全部章节：

```powershell
conda activate llm
python scripts/build_knowledge_base.py --full
```

## Redis 会话记忆

若本机已有 Redis，服务会自动连接 `redis://127.0.0.1:6379/0`。也可以使用项目中的可选配置：

```powershell
docker compose up -d redis
```

没有 Redis 时会话会保存到本机 `data/session_memory`，服务重启后仍可恢复最近对话；`/api/health` 会显示 `local-persistent`。

## 新增教材或题库

学生端右侧点击“添加教材 / 新建知识库”即可：

1. 先选择首份 PDF、Word、Markdown 或文本资料。
2. 输入新知识库的英文标识，再点击“确认建立知识库”。
3. 后端使用 `qwen3-vl-flash` 执行语义清洗、版面解析、电路图理解、Chunking、Embedding 和索引重载；API Key 不写入知识库产物。
4. `/api/kb/status` 返回 `building`、`ready` 或 `error`；Windows 下的短暂进度文件占用会自动重试，不会中断实际构建。

默认知识库、当前目标知识库和追加资料仍在同一弹窗的“管理已有知识库”区域维护。已有资料无需重新上传：点击“使用 qwen3-vl-flash 重新构建已有资料”即可启动 v2 多模态重建。

Excel/JSON 题库不会进入 RAG 知识库，也不会参与检索或图谱构建。出题 Agent 只依据学生原题和会话历史生成同构变式。

学生交互栏的回形针按钮可上传题目图片或文档附件。图片、PDF 页面及 Office 内嵌图片会由当前配置且支持视觉输入的模型识别题干、公式、参数、连接关系和知识点，再由同一配置模型完成答疑或同类出题。每轮最多发送 `MAX_CHAT_DOCUMENT_IMAGES`（默认 6）张由文档产生的视觉页面或内嵌图片。

## 分层多模态图文知识库（v2.1）

新版建库同时产出以下可审计数据：

- `<教材名>.page_ocr.jsonl`：扫描版 PDF 的逐页 OCR、章/节与知识点缓存；文档哈希或视觉模型变化后自动失效。
- `cleaning_audit.json`：`qwen3-vl-flash`/规则对每页的保留或丢弃决定及原因，原 PDF 永不物理修改。
- `multimodal_elements.jsonl`：文本、公式、表格、图片、电路图的页码、bbox、阅读顺序、原图路径和内容哈希。
- `artifacts/`：从 PDF 提取的原始图片。
- `knowledge_graph.json`：教材—原始页码—Chunk—课程概念—电路元件—网络关系；配置 Neo4j 后会同步到图数据库。
- `chapter_knowledge_points.json`：按教材章节归档的知识点、证据数量与来源页；学生端可从知识图谱下方进入章节窗口查看。
- `pipeline_audit.json`：清洗、解析、模态处理、融合、检索和应用六层状态与数量审计。
- `qdrant/`：Linux/macOS 未配置 `QDRANT_URL` 时可使用 Qdrant 嵌入式持久化；同时保留 `vectors.faiss` 兼容回退。

完整处理顺序为：原生文本提取或 `qwen3-vl-flash` 扫描页 OCR（恢复章/节与页面知识点）→ 大模型语义清洗（过滤版权/广告/噪声与独立习题页，并安全保留正文技术内容和带讲解的例题）→ PDF-Extract-Kit Layout/MFD 结构解析 → 文本 NLP → 独立公式筛选、原生 PDF 字形/坐标恢复 LaTeX（扫描件使用 `qwen3-vl-flash` 回退）→ `qwen3-vl-flash` 电路图识别 → 元件/网络/Netlist/描述融合 → 文本与多模态向量 → Qdrant/FAISS + BM25 + Neo4j/本地图 → 融合重排。行内数学符号保留在正文，不会各自生成公式 Chunk；构建结束会强制校验扫描页占位符比例、知识点数量、Chunk/向量数量、图边完整性、bbox 与题库隔离，校验失败的暂存索引不会被激活。

PDF-Extract-Kit 使用本地 GPU 与官方权重，Qwen3-VL 使用百炼 API：

1. 在 `.env` 设置 `PDF_EXTRACT_KIT_DIR=third_party/PDF-Extract-Kit`；小样联调可用 `PDF_EXTRACT_KIT_PAGE_LIMIT=3`限制页数。
2. 设置 `QWEN_CIRCUIT_VISION_MODEL=qwen3-vl-flash` 和 `QWEN_MULTIMODAL_EMBEDDING_MODEL=qwen3-vl-embedding`；电路图由 Flash 识别，图片与文本向量默认为 1024 维。
3. 设置 `RERANK_MODEL_PATH` 可额外启用 CrossEncoder 重排；未配置时仍使用向量、BM25、图关系和图片相似度融合。

Qdrant 和 Neo4j 可用 Docker 启动：

```powershell
docker compose up -d qdrant redis
```

本机 Neo4j Windows Server 可用 `powershell -File scripts/neo4j_server.ps1 -Action start` 管理；也可选择 `docker compose up -d neo4j`。随后设置 `QDRANT_URL=http://127.0.0.1:6333`、`NEO4J_URI=bolt://127.0.0.1:7687`、`NEO4J_HTTP_URL=http://127.0.0.1:7474` 和 Neo4j 密码。Windows 检索侧通过 Qdrant/Neo4j REST API 查询，避免其 Python 客户端与 Torch/FAISS 本地运行库冲突。

## 核心 API

### `POST /api/chat`

```json
{
  "session_id": "student-demo",
  "message": "PN结为什么具有单向导电性？",
  "mode": "auto",
  "knowledge_base": "default",
  "attachment_ids": [],
  "model_provider": "ollama",
  "model": "qwen3.5:2b",
  "api_key": "",
  "base_url": "http://127.0.0.1:11434"
}
```

返回 SSE 事件：`connected`、`status`、`delta`、`meta`、`done`；错误为 `error`。`connected`、`meta` 和会话历史会记录本次实际使用的回答模型；模型思维链不会传输。

### `POST /api/attachments`

`multipart/form-data` 字段为 `file` 与 `session_id`。接口返回附件 ID，将其放入 `/api/chat` 的 `attachment_ids` 即可让 Agent 读取；支持 PNG/JPEG/WebP/BMP、PDF、DOCX、XLSX、TXT、Markdown 和 JSON。

### `POST /api/upload`

`multipart/form-data` 字段：

- `file`：上传文件。
- `knowledge_base`：默认 `default`。
- `rebuild`：默认 `true`。
- `model_provider`、`model`、`api_key`、`base_url`：保留用于兼容现有客户端；建库视觉处理固定使用 `qwen3-vl-flash`，只有请求本身为 Qwen 且提供了浏览器 API Key 时才复用该 Qwen 凭据，否则使用后端 Qwen 配置。

### 其他

- `GET /api/health`
- `GET /api/models`
- `GET /api/sessions`
- `GET /api/sessions/{session_id}`
- `DELETE /api/sessions/{session_id}`（同时删除该会话的附件）
- `GET /api/kb/status`
- `GET /api/kb/{knowledge_base}/graph`
- `POST /api/kb/rebuild`（使用 `qwen3-vl-flash` 重建已有资料）
- `GET /api/mistakes?student_id=...`
- `POST /api/mistakes`（提取知识点、推断可信来源并对齐课程图谱后归档）
- `PATCH /api/mistakes/{mistake_id}?student_id=...`
- `DELETE /api/mistakes/{mistake_id}?student_id=...`
- `GET /api/mistakes/analysis?student_id=...`
- `GET|POST /api/mistakes/categories`
- `PATCH|DELETE /api/mistakes/categories/{category_id}`
- `POST /api/mistakes/{mistake_id}/annotations`
- `PATCH|DELETE /api/mistakes/{mistake_id}/annotations/{annotation_id}`
- `GET /api/homeworks?role=teacher|student&student_id=...`
- `POST /api/homeworks`（上传 PDF/图片并后台拆题）
- `POST /api/homeworks/{homework_id}/publish`
- `POST /api/homeworks/{homework_id}/reprocess`
- `POST /api/homeworks/{homework_id}/submissions`（上传学生作答图片并后台双模型批改）
- `GET /api/teacher/status`

## 测试与诊断

```powershell
conda activate llm
python -m pytest -q
python scripts/retrieval_smoke_test.py
python scripts/ollama_smoke_test.py
```

后端日志写入 `logs/backend.log`。
