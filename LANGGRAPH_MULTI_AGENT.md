# CircuitMind LangGraph 多智能体工作流

本文档对应当前学生端的实际 LangGraph 编排。总图先完成附件理解和大模型意图路由，再进入答疑、同类出题或学习规划子图。

```mermaid
flowchart TB
    Student["学生 Web 页面"] --> API["POST /api/chat<br/>SSE 流式接口"]

    API --> Memory["会话记忆<br/>Redis / 本地持久化"]
    API --> AttachmentStore["附件存储<br/>图片 / PDF / Word / Excel"]

    subgraph Orchestrator["LangGraph 总编排图"]
        direction TB
        AttachmentReader["附件理解 Agent<br/>题干、拓扑、已知量、待求量"]
        IntentRouter{"意图路由 Agent"}
        AnswerEntry["答疑 Agent 子图"]
        QuizEntry["出题 Agent 子图"]
        PlanEntry["学习规划 Agent 子图"]

        AttachmentReader --> IntentRouter
        IntentRouter -->|"答疑"| AnswerEntry
        IntentRouter -->|"同类出题"| QuizEntry
        IntentRouter -->|"学习规划"| PlanEntry
    end

    AttachmentStore --> AttachmentReader
    Memory --> AttachmentReader
    VisionModel["Ollama qwen3.5:2b<br/>视觉理解 + 思考模式"] --> AttachmentReader

    subgraph AnswerGraph["答疑 Agent 工作流"]
        direction LR
        Rewrite["Query 改写 Agent<br/>口语问题专业化"]
        AnswerRetrieve["混合检索 Agent<br/>向量 + BM25 + Rerank"]
        Compose["Prompt 组装 Agent<br/>资料、历史、LaTeX 约束"]
        AnswerLLM["答疑生成 Agent<br/>分步解答 + SSE"]

        Rewrite --> AnswerRetrieve --> Compose --> AnswerLLM
    end

    AnswerEntry --> Rewrite

    subgraph QuizGraph["同类出题 Agent 工作流"]
        direction TB
        Extract["原题分析 Agent<br/>知识点 + 题型 + 结构蓝图"]
        Generate["出题 Agent<br/>保持拓扑与设问同构"]
        Verify{"验算与校验 Agent<br/>结构 / 去重 / SymPy"}
        Repair["修正 Agent<br/>生成同构可验证变式"]
        VerifyRepair{"二次校验"}
        Render["结果渲染<br/>题目、思路、答案、易错点"]

        Extract --> Generate --> Verify
        Verify -->|"通过"| Render
        Verify -->|"未通过"| Repair --> VerifyRepair --> Render
    end

    QuizEntry --> Extract

    subgraph PlanGraph["学习规划 Agent 工作流"]
        direction LR
        PlanAnalyze["目标与薄弱点分析"]
        PlanRetrieve["前置知识与资料检索"]
        PlanGenerate["阶段路线 + 7天起步清单"]
        PlanAnalyze --> PlanRetrieve --> PlanGenerate
    end

    PlanEntry --> PlanAnalyze

    subgraph RAG["课程知识库"]
        direction TB
        CleanDocs["清洗后的教材与结构化题库"]
        Chunks["带章、节、页码元数据的 Chunk"]
        VectorDB["FAISS 向量索引"]
        BM25["BM25 关键词索引"]

        CleanDocs --> Chunks
        Chunks --> VectorDB
        Chunks --> BM25
    end

    VectorDB --> AnswerRetrieve
    BM25 --> AnswerRetrieve
    VectorDB --> PlanRetrieve
    BM25 --> PlanRetrieve

    ModelGateway["模型网关<br/>Ollama / DeepSeek / 通义千问 / 自定义 API"] --> AnswerLLM
    ModelGateway --> Generate
    ModelGateway --> PlanAnalyze
    ModelGateway --> PlanGenerate
    SymPy["Python / SymPy<br/>数值合理性验算"] --> Verify
    SymPy --> VerifyRepair

    AnswerLLM --> Stream["SSE: status / delta / meta / done"]
    Render --> Stream
    PlanGenerate --> Stream
    Stream --> Student

    classDef agent fill:#e6f4f1,stroke:#0f766e,color:#173f3c,stroke-width:1.5px;
    classDef decision fill:#fff6df,stroke:#c58a18,color:#5e4412,stroke-width:1.5px;
    classDef storage fill:#edf2ff,stroke:#526fa8,color:#273d68,stroke-width:1.2px;
    classDef model fill:#f5ecff,stroke:#8057a6,color:#4b2c68,stroke-width:1.2px;

    class AttachmentReader,AnswerEntry,QuizEntry,PlanEntry,Rewrite,AnswerRetrieve,Compose,AnswerLLM,Extract,Generate,Repair,Render,PlanAnalyze,PlanRetrieve,PlanGenerate agent;
    class IntentRouter,Verify,VerifyRepair decision;
    class Memory,AttachmentStore,CleanDocs,Chunks,VectorDB,BM25 storage;
    class VisionModel,ModelGateway,SymPy model;
```

## 关键状态流转

LangGraph 状态中主要保存以下信息：

- `message`、`history`、`knowledge_base`：学生输入、最近对话和当前知识库。
- `attachment_context`、`attachment_blueprint`：附件识别文本以及电路拓扑、已知量、待求量蓝图。
- `intent`：路由结果，取值为 `answer`、`quiz` 或 `plan`。
- `rewritten_query`、`hits`：专业化检索问题和混合检索结果。
- `knowledge_point`、`quiz_type`、`quiz_family`：出题知识点、数值/概念题型和同构题家族。
- `draft`、`verification`：生成题草稿以及结构、去重和 SymPy 校验结果。
- `plan_profile`：学习目标、知识点、当前水平、时间范围与约束。
- `response`、`sources`：最终回复和可追溯资料来源。
