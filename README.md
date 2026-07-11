# CircuitMind 多智能体电路课程教学平台

这是一个本地优先的电路课程教学 MVP。当前已完成学生端；教师端保留页面、路由和 `/api/teacher/status` 接口，后续可直接扩展。

已跑通的链路：

- 默认使用本地 Ollama `qwen3.5:2b`；可切换其他已安装 Ollama 模型，或接入 DeepSeek、通义千问和自定义 OpenAI 兼容 API。私有思考字段不会返回前端。
- LangGraph 编排的路由 Agent、答疑 Agent、检索 Agent、出题 Agent 和 SymPy 验算 Agent；检索 Agent 仅服务答疑链路。
- 图片出题先提取“电路拓扑、已知量、特殊条件、待求量”蓝图；连续“再出一道”会沿用最近生成题，同类题不调用知识库检索并必须通过同构校验。
- 教材清洗、章节/段落语义切分、章/节/页码元数据、结构化题库、384 维向量化和 populated FAISS 索引。
- 向量语义检索 + BM25 关键词检索 + 规则重排。
- FastAPI、CORS、统一异常处理、日志、POST SSE 真正 token 流式输出、上传与后台重建知识库。
- Redis 最近 N 轮会话记忆；Redis 不可用时自动切换本地持久化记忆，服务重启后仍可执行出题去重。
- 学生交互栏支持题目图片和 PDF/Word/Excel/Markdown 等附件；`qwen3.5:2b` 直接完成图片题干识别。
- React + TypeScript + Ant Design + Zustand + KaTeX 学生端，含 LaTeX 定界符容错预处理。
- 右上角模型选择器动态读取本机 Ollama 模型；所选模型与云端 API 配置会保存在当前浏览器，也可通过后端环境变量配置。
- 左侧“最近学习”读取持久化会话列表，支持点击恢复历史对话；刷新页面后会自动恢复当前会话。

## 当前数据成果

默认知识库先按 MVP 范围索引《模拟电子技术基础》第一章：

- 教材范围：PDF 第 25–94 页，共 69 个有效文本页。
- 示例题库：12 道题，包含题号、题目文本、知识点标签、标准答案、易错点、难度、题型、解题步骤。
- 向量库：95 个 Chunk，向量维度 384，状态 `populated`。
- 元数据：每个教材 Chunk 保留来源、章、节、PDF 页码和知识点标签。

主要产物位于：

```text
RAG_Resources/
  模拟电子技术基础-童诗白.pdf
  电路课程示例题库.xlsx
data/vector_stores/default/
  cleaned_documents/
  chunks.jsonl
  question_bank.json
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
  A --> W[Query 改写]
  Q --> K[原题或最近题蓝图提取]
  W --> H[向量 + BM25 + Rerank]
  H --> V[(FAISS + Chunk 元数据)]
  A --> L[Ollama qwen3.5:2b]
  K --> L
  Q --> P[知识点校验 + 会话去重 + SymPy 验算]
  API --> U[上传与后台知识库重建]
```

## 直接启动

确保 Ollama 已运行且存在 `qwen3.5:2b`：

```powershell
ollama list
```

模型权重不会提交到 GitHub。首次运行先下载项目使用的嵌入模型：

```powershell
conda activate llm
python scripts/download_embedding_model.py
```

脚本只下载 `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` 推理所需文件到
`models/paraphrase-multilingual-MiniLM-L12-v2`，不会读取或写入任何 API Key。

项目已经包含构建好的前端和默认向量库，因此只需在项目根目录运行：

```powershell
conda activate llm
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

或：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start.ps1
```

打开 `http://127.0.0.1:8000/student`。生产构建由 FastAPI 直接提供；开发前端可在 `frontend` 中运行 `npm run dev`，Vite 会代理 `/api` 到 8000 端口。

## 模型切换与 API 配置

点击学生端右上角的模型名称即可选择：

- 本地 Ollama：自动显示 `ollama list` 中已经安装的模型。
- DeepSeek API：默认提供 `deepseek-v4-flash` 和 `deepseek-v4-pro`。
- 通义千问 API：默认提供 `qwen-plus`、`qwen-max` 和 `qwen-turbo`，Base URL 可按百炼工作空间修改。
- 自定义 API：填写任意兼容 OpenAI Chat Completions 的模型名称、API Key 和 Base URL。

页面输入的模型配置和 API Key 会写入当前浏览器的 `localStorage`，不会写入项目文件；配置弹窗提供清除已保存密钥的入口。公用电脑不建议保存云端密钥。也可以在 `.env` 配置 `DEEPSEEK_API_KEY`、`QWEN_API_KEY` 及对应 Base URL。使用云端模型时，题目、最近对话和检索上下文会发送到所选服务；附件图片仍先由本地视觉节点提取结构化题目蓝图。

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

1. 选择默认知识库，或输入英文标识创建独立知识库。
2. 上传 PDF、Word、Markdown、文本、Excel 或 JSON。
3. 后端在后台执行清洗、Chunking、Embedding 和索引重载。
4. `/api/kb/status` 返回 `building`、`ready` 或 `error`。

Excel 题库至少需要以下列：`题号`、`题目文本`、`知识点标签`、`标准答案`、`易错点`。其余支持列为 `难度`、`题型`、`解题步骤`。

学生交互栏的回形针按钮可上传题目图片或文档附件。图片会由本地 `qwen3.5:2b` 视觉能力识别题干、参数、连接关系和知识点，再进入答疑或同类出题工作流；附件和识别过程均保留在本机。

