import { useEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  App as AntApp,
  Button,
  Input,
  Modal,
  Popconfirm,
  Segmented,
  Select,
  Tag,
  Tooltip,
  Upload,
  type UploadProps,
} from 'antd'
import {
  ArrowUp,
  BookOpen,
  Bot,
  BrainCircuit,
  BookmarkPlus,
  Check,
  ChevronDown,
  ChevronRight,
  CircleStop,
  Cloud,
  Clock3,
  Cpu,
  Database,
  FileText,
  GraduationCap,
  HelpCircle,
  Layers3,
  Network,
  LoaderCircle,
  Menu,
  MessageSquareText,
  Plus,
  Paperclip,
  Search,
  KeyRound,
  ServerCog,
  ShieldCheck,
  Trash2,
  UploadCloud,
  UserRound,
  WandSparkles,
  X,
  Zap,
} from 'lucide-react'
import MathMarkdown from '../components/MathMarkdown'
import {
  addMistake,
  deleteMistake,
  deleteSession,
  fetchKnowledgeGraph,
  fetchKnowledgeBases,
  fetchMistakes,
  fetchModels,
  fetchSession,
  fetchSessions,
  KBStatus,
  KnowledgeGraph,
  ModelCatalog,
  ModelConfig,
  ModelProviderId,
  MistakeItem,
  SessionSummary,
  SourceInfo,
  uploadKnowledgeFile,
  rebuildKnowledgeBase,
} from '../lib/api'
import { ChatMode, useChatStore } from '../store/chatStore'

const { TextArea } = Input
type WorkspaceView = 'chat' | 'graph' | 'mistakes'

const providerLabels: Record<ModelProviderId, string> = {
  ollama: '本地',
  deepseek: 'DeepSeek',
  qwen: '通义千问',
  custom: '自定义 API',
}

const fallbackModelCatalog: ModelCatalog = {
  default: { provider: 'ollama', model: 'qwen3.5:2b' },
  providers: [
    {
      id: 'ollama',
      label: '本地 Ollama',
      description: '使用本机已安装模型，数据不离开本机',
      models: ['qwen3.5:2b'],
      default_model: 'qwen3.5:2b',
      base_url: 'http://127.0.0.1:11434',
      requires_api_key: false,
      configured: true,
    },
    {
      id: 'deepseek',
      label: 'DeepSeek API',
      description: 'DeepSeek 官方 OpenAI 兼容接口',
      models: ['deepseek-v4-flash', 'deepseek-v4-pro'],
      default_model: 'deepseek-v4-flash',
      base_url: 'https://api.deepseek.com',
      requires_api_key: true,
      configured: false,
    },
    {
      id: 'qwen',
      label: '通义千问 API',
      description: '阿里云百炼文本与多模态 OpenAI 兼容接口',
      models: ['Qwen3.7-Plus', 'Qwen3.7-Max', 'qwen-vl-max', 'qwen3-vl-8b-instruct', 'qwen3-vl-plus', 'qwen3-vl-flash', 'qwen3-vl-embedding'],
      default_model: 'qwen3-vl-plus',
      base_url: 'https://dashscope.aliyuncs.com/compatible-mode/v1',
      requires_api_key: true,
      configured: false,
    },
    {
      id: 'custom',
      label: '自定义 API',
      description: '连接其他 OpenAI Chat Completions 兼容服务',
      models: [],
      default_model: '',
      base_url: '',
      requires_api_key: true,
      configured: false,
    },
  ],
}

const quickPrompts = [
  {
    icon: <Zap size={19} />,
    eyebrow: '概念答疑',
    title: 'PN 结为什么具有单向导电性？',
    hint: '从势垒与载流子运动解释',
    mode: 'answer' as ChatMode,
  },
  {
    icon: <BrainCircuit size={19} />,
    eyebrow: '分步计算',
    title: '二极管导通后该如何建立等效电路？',
    hint: '结合恒压降模型进行分析',
    mode: 'answer' as ChatMode,
  },
  {
    icon: <WandSparkles size={19} />,
    eyebrow: '同类出题',
    title: '根据二极管伏安特性出一道基础题',
    hint: '生成新参数并用 SymPy 验算',
    mode: 'quiz' as ChatMode,
  },
]

function LogoMark() {
  return (
    <span className="logo-mark" aria-hidden="true">
      <span className="logo-node logo-node-a" />
      <span className="logo-node logo-node-b" />
      <span className="logo-node logo-node-c" />
    </span>
  )
}

function sessionTime(value: string) {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return '历史会话'
  const today = new Date()
  if (date.toDateString() === today.toDateString()) {
    return date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
  }
  return date.toLocaleDateString('zh-CN', { month: '2-digit', day: '2-digit' })
}

function Sidebar({
  open,
  onClose,
  sessions,
  activeSessionId,
  onSelectSession,
  onDeleteSession,
  onNewSession,
  activeView,
  onView,
}: {
  open: boolean
  onClose: () => void
  sessions: SessionSummary[]
  activeSessionId: string
  onSelectSession: (sessionId: string) => void
  onDeleteSession: (sessionId: string, title: string) => void
  onNewSession: () => void
  activeView: WorkspaceView
  onView: (view: WorkspaceView) => void
}) {
  const modelProvider = useChatStore((state) => state.modelConfig.provider)
  return (
    <>
      {open && <button className="sidebar-backdrop" onClick={onClose} aria-label="关闭导航" />}
      <aside className={`sidebar ${open ? 'is-open' : ''}`}>
        <div className="brand-row">
          <LogoMark />
          <div>
            <strong>CircuitMind</strong>
            <span>多智能体电路助教</span>
          </div>
          <button className="mobile-close" onClick={onClose} aria-label="关闭导航">
            <X size={18} />
          </button>
        </div>

        <Button className="new-chat-button" icon={<Plus size={16} />} onClick={onNewSession} block>
          开始新对话
        </Button>

        <nav className="main-nav" aria-label="学生端主导航">
          <div className="nav-label">学习空间</div>
          <button className={`nav-item ${activeView === 'chat' ? 'active' : ''}`} onClick={() => onView('chat')}>
            <MessageSquareText size={17} />
            <span>智能学习台</span>
            <span className="nav-live-dot" />
          </button>
          <button className={`nav-item ${activeView === 'graph' ? 'active' : ''}`} onClick={() => onView('graph')}>
            <BookOpen size={17} />
            <span>知识图谱</span>
          </button>
          <button className={`nav-item ${activeView === 'mistakes' ? 'active' : ''}`} onClick={() => onView('mistakes')}>
            <Layers3 size={17} />
            <span>错题本</span>
          </button>
        </nav>

        <div className="recent-section">
          <div className="nav-label">最近学习</div>
          <div className="recent-list">
            {sessions.length ? sessions.map((session) => (
              <div
                key={session.session_id}
                className={`recent-row ${session.session_id === activeSessionId ? 'active' : ''}`}
              >
                <button
                  className="recent-item"
                  onClick={() => onSelectSession(session.session_id)}
                  title={session.title}
                >
                  <span className="recent-icon"><Clock3 size={14} /></span>
                  <span>
                    <strong>{session.title}</strong>
                    <small>{sessionTime(session.updated_at)} · {Math.max(1, Math.ceil(session.message_count / 2))} 轮</small>
                  </span>
                </button>
                <Popconfirm
                  title="删除这条历史对话？"
                  description="对话记录和该会话上传的附件将一并删除。"
                  okText="删除"
                  cancelText="取消"
                  okButtonProps={{ danger: true }}
                  onConfirm={() => onDeleteSession(session.session_id, session.title)}
                >
                  <button
                    type="button"
                    className="recent-delete"
                    aria-label={`删除历史对话 ${session.title}`}
                    title="删除历史对话"
                  >
                    <Trash2 size={14} />
                  </button>
                </Popconfirm>
              </div>
            )) : (
              <div className="recent-empty">完成一次提问后，会话会显示在这里</div>
            )}
          </div>
        </div>

        <div className="sidebar-bottom">
          <Link to="/teacher" className="teacher-link">
            <GraduationCap size={17} />
            <span>切换到教师端</span>
            <ChevronRight size={15} />
          </Link>
          <div className="profile-row">
            <span className="profile-avatar"><UserRound size={17} /></span>
            <span>
              <strong>电路学习者</strong>
              <small>学生端 · {modelProvider === 'ollama' ? '本地模型' : '云端模型'}</small>
            </span>
          </div>
        </div>
      </aside>
    </>
  )
}

function Welcome({ onAsk }: { onAsk: (prompt: string, mode: ChatMode) => void }) {
  return (
    <div className="welcome-wrap">
      <div className="welcome-hero">
        <div className="hero-circuit" aria-hidden="true">
          <span className="circuit-line line-a" />
          <span className="circuit-line line-b" />
          <span className="circuit-node node-a" />
          <span className="circuit-node node-b" />
          <span className="circuit-chip"><BrainCircuit size={28} /></span>
        </div>
        <div className="hero-copy">
          <h1>你好，今天想弄懂哪一道电路题？</h1>
        </div>
      </div>
      <div className="quick-grid">
        {quickPrompts.map((item) => (
          <button key={item.title} className="quick-card" onClick={() => onAsk(item.title, item.mode)}>
            <span className="quick-card-icon">{item.icon}</span>
            <span className="quick-card-copy">
              <small>{item.eyebrow}</small>
              <strong>{item.title}</strong>
              <span>{item.hint}</span>
            </span>
            <ArrowUp className="quick-arrow" size={16} />
          </button>
        ))}
      </div>
      <div className="ability-row">
        <span><Search size={15} /> 混合检索</span>
        <span><Bot size={15} /> 模型推理解答</span>
        <span><Check size={15} /> 答案自动验算</span>
        <span><FileText size={15} /> 来源可追溯</span>
      </div>
    </div>
  )
}

function SourceCard({ source, index }: { source: SourceInfo; index: number }) {
  const page = source.page_start
    ? source.page_start === source.page_end
      ? `第 ${source.page_start} 页`
      : `第 ${source.page_start}–${source.page_end} 页`
    : '结构化题库'
  return (
    <article className="source-card">
      <div className="source-card-top">
        <span className={`source-type ${source.doc_type === 'question' ? 'question' : ''}`}>
          {source.doc_type === 'question' ? <WandSparkles size={13} /> : <FileText size={13} />}
          资料 {index + 1}
        </span>
        <span className="source-score">{Math.round(source.score * 100)}%</span>
      </div>
      <strong>{source.section || source.chapter || source.source}</strong>
      <p>{source.source}</p>
      {source.excerpt && <p className="source-excerpt">{source.excerpt}</p>}
      {source.knowledge_tags?.length ? (
        <div className="source-tags">{source.knowledge_tags.slice(0, 4).map((tag) => <span key={tag}>{tag}</span>)}</div>
      ) : null}
      <div className="source-score-grid" aria-label="检索评分组成">
        {[
          ['向量', source.vector_score],
          ['关键词', source.bm25_score],
          ['图谱', source.graph_score],
        ].map(([label, score]) => (
          <div key={String(label)}>
            <span>{label}</span>
            <i><b style={{ width: `${Math.max(0, Math.min(100, Number(score || 0) * 100))}%` }} /></i>
          </div>
        ))}
      </div>
      <div className="source-meta"><span>{page}</span><span>已重排</span></div>
    </article>
  )
}

function KnowledgePanel({ statuses, onCreate }: { statuses: KBStatus[]; onCreate: () => void }) {
  const activeSources = useChatStore((state) => state.activeSources)
  const knowledgeBase = useChatStore((state) => state.knowledgeBase)
  const modelProvider = useChatStore((state) => state.modelConfig.provider)
  const mode = useChatStore((state) => state.mode)
  const messages = useChatStore((state) => state.messages)
  const current = statuses.find((item) => item.id === knowledgeBase)
  const latestAssistant = [...messages].reverse().find((item) => item.role === 'assistant')
  const quizContext = mode === 'quiz' || latestAssistant?.agent === '出题 Agent'
  return (
    <aside className="knowledge-panel">
      <div className="panel-heading">
        <div>
          <span className="panel-kicker">{quizContext ? 'SOURCE PROBLEM' : 'RAG CONTEXT'}</span>
          <h2>{quizContext ? '原题依据' : '检索依据'}</h2>
        </div>
        <Tooltip title={quizContext ? '出题 Agent 仅使用原题和会话历史，不调用知识库检索' : '检索结果已通过向量、BM25 与重排综合评分'}>
          <HelpCircle size={17} />
        </Tooltip>
      </div>

      <div className="kb-summary-card">
        <span className="kb-icon">{quizContext ? <BrainCircuit size={18} /> : <Database size={18} />}</span>
        <div>
          <strong>{quizContext ? '原题驱动出题' : knowledgeBase === 'default' ? '默认课程知识库' : knowledgeBase}</strong>
          <span>{quizContext ? '会话上下文 · 不检索知识库' : `${current?.chunks || 0} 个文本块 · ${current?.documents || 0} 份资料`}</span>
        </div>
        <span className={`kb-state ${quizContext ? 'ready' : current?.state || 'missing'}`}>
          {quizContext ? '已锁定' : current?.state === 'building' ? '构建中' : current?.state === 'ready' ? '就绪' : '待构建'}
        </span>
      </div>

      <div className="source-list">
        {quizContext ? (
          <div className="source-empty quiz-reference-empty">
            <span><WandSparkles size={22} /></span>
            <strong>保持原题结构</strong>
            <p>首次出题读取当前原题；“再出一道”会沿用最近题目的拓扑、已知量和分项设问。</p>
          </div>
        ) : activeSources.length ? (
          activeSources.slice(0, 5).map((source, index) => (
            <SourceCard key={`${source.id}-${index}`} source={source} index={index} />
          ))
        ) : (
          <div className="source-empty">
            <span><Search size={22} /></span>
            <strong>等待你的问题</strong>
            <p>提问后，这里会展示命中的教材章节、页码与相关度。</p>
          </div>
        )}
      </div>

      <div className="panel-bottom">
        <button className="manage-kb-button" onClick={onCreate}>
          <UploadCloud size={16} />
          <span>添加教材 / 新建知识库</span>
          <ChevronRight size={15} />
        </button>
        <div className="privacy-note">
          <span className={`privacy-dot ${modelProvider === 'ollama' ? '' : 'cloud'}`} />
          {modelProvider === 'ollama' ? '资料与模型推理均保留在本机' : '提问内容将发送至所选模型 API'}
        </div>
      </div>
    </aside>
  )
}

function ChatComposer({ onSend }: { onSend: (value: string) => void }) {
  const [value, setValue] = useState('')
  const fileInputRef = useRef<HTMLInputElement>(null)
  const mode = useChatStore((state) => state.mode)
  const setMode = useChatStore((state) => state.setMode)
  const streaming = useChatStore((state) => state.streaming)
  const stop = useChatStore((state) => state.stop)
  const pendingAttachments = useChatStore((state) => state.pendingAttachments)
  const addAttachments = useChatStore((state) => state.addAttachments)
  const removeAttachment = useChatStore((state) => state.removeAttachment)
  const hasReadyAttachment = pendingAttachments.some((item) => item.status === 'ready')
  const hasUnfinishedAttachment = pendingAttachments.some((item) => item.status !== 'ready')

  const submit = () => {
    if ((!value.trim() && !hasReadyAttachment) || streaming || hasUnfinishedAttachment) return
    onSend(value)
    setValue('')
  }

  const selectFiles = (files: File[]) => {
    if (!files.length) return
    void addAttachments(files)
  }

  return (
    <div className="composer-shell">
      <div className="composer-card">
        <div className="composer-topline">
          <Segmented<ChatMode>
            size="small"
            value={mode}
            onChange={setMode}
            options={[
              { label: '智能路由', value: 'auto' },
              { label: 'AI 答疑', value: 'answer' },
              { label: '同类出题', value: 'quiz' },
              { label: '学习规划', value: 'plan' },
            ]}
          />
          <span className="composer-tip">Shift + Enter 换行</span>
        </div>
        {pendingAttachments.length > 0 && (
          <div className="pending-attachments" aria-label="待发送附件">
            {pendingAttachments.map((item) => (
              <div key={item.localId} className={`pending-attachment ${item.status}`}>
                <span className="pending-file-icon">
                  {item.status === 'uploading' ? <LoaderCircle size={15} /> : item.kind === 'image' ? <FileText size={15} /> : <Paperclip size={15} />}
                </span>
                <span className="pending-file-copy">
                  <strong>{item.name}</strong>
                  <small>{item.status === 'uploading' ? '正在上传…' : item.status === 'error' ? item.error : `${Math.max(1, Math.round(item.size / 1024))} KB · 已就绪`}</small>
                </span>
                <button type="button" onClick={() => removeAttachment(item.localId)} aria-label={`移除附件 ${item.name}`}>
                  <X size={13} />
                </button>
              </div>
            ))}
          </div>
        )}
        <div className="composer-input-row">
          <input
            ref={fileInputRef}
            className="sr-only-file"
            type="file"
            multiple
            accept=".png,.jpg,.jpeg,.webp,.bmp,.pdf,.docx,.txt,.md,.xlsx,.json"
            onChange={(event) => {
              selectFiles(Array.from(event.target.files || []))
              event.target.value = ''
            }}
          />
          <Tooltip title="添加题目图片或附件">
            <Button
              className="attach-button"
              shape="circle"
              onClick={() => fileInputRef.current?.click()}
              disabled={streaming || pendingAttachments.length >= 5}
              icon={<Paperclip size={17} />}
              aria-label="添加题目图片或附件"
            />
          </Tooltip>
          <TextArea
            value={value}
            onChange={(event) => setValue(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault()
                submit()
              }
            }}
            onPaste={(event) => {
              const files = Array.from(event.clipboardData.files || [])
              if (files.length) {
                event.preventDefault()
                selectFiles(files)
              }
            }}
            autoSize={{ minRows: 1, maxRows: 5 }}
            placeholder={mode === 'quiz' ? '粘贴原题，或描述想练习的知识点…' : mode === 'plan' ? '描述学习目标、薄弱点和可用时间…' : '输入电路问题，支持 LaTeX 公式…'}
            variant="borderless"
            aria-label="输入电路问题"
          />
          {streaming ? (
            <Tooltip title="停止生成">
              <Button className="send-button stop" shape="circle" onClick={stop} icon={<CircleStop size={18} />} />
            </Tooltip>
          ) : (
            <Tooltip title="发送">
              <Button
                type="primary"
                className="send-button"
                shape="circle"
                onClick={submit}
                disabled={(!value.trim() && !hasReadyAttachment) || hasUnfinishedAttachment}
                icon={<ArrowUp size={18} />}
              />
            </Tooltip>
          )}
        </div>
      </div>
      <p className="composer-footnote">AI 可能犯错，重要计算请结合教材与实验结果复核。</p>
    </div>
  )
}

function normalizeQuizTitle(content: string) {
  return content.replace(
    /^(#{1,3}\s*同类型新题)(?:\s*[·•・—-]\s*[^\r\n]+)?\s*$/m,
    '$1',
  )
}

function Conversation({ onAddMistake }: { onAddMistake: (content: string, agent: string) => void }) {
  const messages = useChatStore((state) => state.messages)
  const streaming = useChatStore((state) => state.streaming)
  const stage = useChatStore((state) => state.stage)
  const stageAgent = useChatStore((state) => state.stageAgent)
  const endRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages, stage])

  return (
    <div className="conversation">
      {messages.map((message, index) => (
        <div key={message.id} className={`message-row ${message.role}`}>
          {message.role === 'assistant' && (
            <span className="assistant-avatar"><LogoMark /></span>
          )}
          <div className={`message-bubble ${message.failed ? 'failed' : ''}`}>
            {message.role === 'assistant' && (
              <div className="message-agent">
                <span>{message.agent || (streaming && index === messages.length - 1 ? stageAgent || '多智能体助教' : '多智能体助教')}</span>
                <Tag bordered={false} title={`${providerLabels[message.provider || 'ollama']} · ${message.model || ''}`}>
                  {providerLabels[message.provider || 'ollama']} · {message.model || 'qwen3.5:2b'}
                </Tag>
              </div>
            )}
            {message.attachments?.length ? (
              <div className="message-attachments">
                {message.attachments.map((attachment) =>
                  attachment.kind === 'image' ? (
                    <a key={attachment.id} href={attachment.url} target="_blank" rel="noreferrer" className="message-image-attachment">
                      <img src={attachment.url} alt={attachment.name} />
                      <span>{attachment.name}</span>
                    </a>
                  ) : (
                    <a key={attachment.id} href={attachment.url} target="_blank" rel="noreferrer" className="message-file-attachment">
                      <FileText size={16} />
                      <span>{attachment.name}</span>
                    </a>
                  ),
                )}
              </div>
            ) : null}
            {message.content ? (
              message.role === 'assistant'
                ? <MathMarkdown content={normalizeQuizTitle(message.content)} />
                : <p>{message.content}</p>
            ) : (
              <div className="thinking-placeholder">
                <span className="thinking-dots"><i /><i /><i /></span>
                <span>{stage || '正在准备…'}</span>
              </div>
            )}
            {message.content && !(streaming && index === messages.length - 1) && (
              <div className="message-tools">
                <button
                  type="button"
                  onClick={() => onAddMistake(message.content, message.role === 'assistant' ? message.agent || '答疑 Agent' : '学生原题')}
                >
                  <BookmarkPlus size={14} /> 加入错题本
                </button>
              </div>
            )}
          </div>
        </div>
      ))}
      {streaming && messages.at(-1)?.content && stage && (
        <div className="stage-pill"><span className="thinking-dots"><i /><i /><i /></span>{stageAgent} · {stage}</div>
      )}
      <div ref={endRef} />
    </div>
  )
}

function KnowledgeGraphView({ graph, loading }: { graph?: KnowledgeGraph; loading: boolean }) {
  const [selectedId, setSelectedId] = useState('')
  const visual = useMemo(() => {
    if (!graph) return { nodes: [], edges: [] }
    const degree = new Map<string, number>()
    graph.edges.forEach((edge) => {
      degree.set(edge.source, (degree.get(edge.source) || 0) + 1)
      degree.set(edge.target, (degree.get(edge.target) || 0) + 1)
    })
    const concepts = graph.nodes
      .filter((node) => node.type === 'concept')
      .sort((a, b) => (degree.get(b.id) || 0) - (degree.get(a.id) || 0))
      .slice(0, 24)
    const conceptIds = new Set(concepts.map((node) => node.id))
    const relatedIds = new Set<string>()
    graph.edges.forEach((edge) => {
      if (conceptIds.has(edge.source)) relatedIds.add(edge.target)
      if (conceptIds.has(edge.target)) relatedIds.add(edge.source)
    })
    const supports = graph.nodes.filter((node) => relatedIds.has(node.id) && node.type !== 'concept').slice(0, 32)
    const positioned = [...concepts, ...supports].map((node, index) => {
      const isConcept = node.type === 'concept'
      const groupIndex = isConcept ? index : index - concepts.length
      const groupSize = isConcept ? Math.max(1, concepts.length) : Math.max(1, supports.length)
      const angle = (Math.PI * 2 * groupIndex) / groupSize - Math.PI / 2
      const radius = isConcept ? 150 : 235
      return {
        ...node,
        x: 450 + Math.cos(angle) * radius,
        y: 270 + Math.sin(angle) * radius,
        degree: degree.get(node.id) || 0,
      }
    })
    const ids = new Set(positioned.map((node) => node.id))
    return {
      nodes: positioned,
      edges: graph.edges.filter((edge) => ids.has(edge.source) && ids.has(edge.target)),
    }
  }, [graph])
  const positions = new Map(visual.nodes.map((node) => [node.id, node]))
  const selected = graph?.nodes.find((node) => node.id === selectedId)
  const neighbors = selectedId && graph
    ? graph.edges.filter((edge) => edge.source === selectedId || edge.target === selectedId).length
    : 0

  if (loading) return <div className="workspace-empty"><LoaderCircle className="spin" /><strong>正在整理知识图谱…</strong></div>
  if (!graph?.nodes.length) return <div className="workspace-empty"><Network /><strong>当前知识库还没有图谱数据</strong><p>重建知识库后会自动提取知识点与资料关系。</p></div>
  return (
    <section className="feature-view graph-view">
      <div className="feature-heading">
        <div><span>KNOWLEDGE MAP</span><h1>课程知识图谱</h1><p>点击节点查看知识点与教材片段之间的关系。</p></div>
        <div className="feature-stats"><strong>{graph.stats.concepts}</strong><span>知识点</span><strong>{graph.stats.edges}</strong><span>关系</span></div>
      </div>
      <div className="graph-layout">
        <div className="graph-canvas">
          <svg viewBox="0 0 900 540" role="img" aria-label="课程知识关系图">
            <g className="graph-edges">
              {visual.edges.map((edge, index) => {
                const from = positions.get(edge.source)
                const to = positions.get(edge.target)
                return from && to ? <line key={`${edge.source}-${edge.target}-${index}`} x1={from.x} y1={from.y} x2={to.x} y2={to.y} /> : null
              })}
            </g>
            <g>
              {visual.nodes.map((node) => (
                <g
                  key={node.id}
                  className={`graph-node ${node.type} ${selectedId === node.id ? 'selected' : ''}`}
                  transform={`translate(${node.x} ${node.y})`}
                  onClick={() => setSelectedId(node.id)}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter' || event.key === ' ') setSelectedId(node.id)
                  }}
                  role="button"
                  tabIndex={0}
                >
                  <circle r={node.type === 'concept' ? Math.min(24, 13 + node.degree) : 8} />
                  <text y={node.type === 'concept' ? 36 : 22}>{node.name?.slice(0, 12) || '资料片段'}</text>
                </g>
              ))}
            </g>
          </svg>
          <div className="graph-legend"><span><i className="concept" />知识点</span><span><i />教材/题库片段</span></div>
        </div>
        <aside className="graph-detail">
          {selected ? <><span>{selected.type === 'concept' ? '知识点' : '资料节点'}</span><h2>{selected.name || '未命名节点'}</h2><p>连接 {neighbors} 个相关节点。可回到智能学习台围绕该知识点提问，或加入学习规划。</p><code>{selected.id}</code></> : <><Network size={28} /><h2>探索知识关系</h2><p>图中较大的绿色节点是核心知识点，外围节点对应教材章节、题目或电路结构。</p></>}
        </aside>
      </div>
    </section>
  )
}

function MistakeBookView({
  mistakes,
  onDelete,
  onPlan,
}: {
  mistakes: MistakeItem[]
  onDelete: (id: string) => void
  onPlan: () => void
}) {
  const pointCounts = useMemo(() => {
    const counts = new Map<string, number>()
    mistakes.flatMap((item) => item.knowledge_points).forEach((point) => counts.set(point, (counts.get(point) || 0) + 1))
    return [...counts.entries()].sort((a, b) => b[1] - a[1])
  }, [mistakes])
  return (
    <section className="feature-view mistakes-view">
      <div className="feature-heading">
        <div><span>MISTAKE REVIEW</span><h1>错题本</h1><p>归档时已自动提取知识点，用于查漏补缺和巩固规划。</p></div>
        <Button type="primary" icon={<BrainCircuit size={16} />} onClick={onPlan} disabled={!mistakes.length}>生成知识补全规划</Button>
      </div>
      {pointCounts.length > 0 && <div className="weakness-strip"><strong>高频薄弱点</strong>{pointCounts.slice(0, 8).map(([point, count]) => <Tag key={point}>{point} · {count}</Tag>)}</div>}
      {mistakes.length ? (
        <div className="mistake-grid">
          {mistakes.map((item) => (
            <article className="mistake-card" key={item.id}>
              <div className="mistake-card-head"><span>{item.agent}</span><small>{new Date(item.created_at).toLocaleDateString('zh-CN')}</small></div>
              <h2>{item.summary}</h2>
              <div className="mistake-points">{item.knowledge_points.map((point) => <Tag key={point}>{point}</Tag>)}</div>
              <div className="mistake-content"><MathMarkdown content={item.content} /></div>
              <Popconfirm title="从错题本删除？" okText="删除" cancelText="取消" onConfirm={() => onDelete(item.id)}>
                <button className="mistake-delete"><Trash2 size={14} /> 删除</button>
              </Popconfirm>
            </article>
          ))}
        </div>
      ) : <div className="workspace-empty"><Layers3 size={30} /><strong>错题本还是空的</strong><p>在答疑或出题结果旁点击“加入错题本”，系统会自动识别知识点。</p></div>}
    </section>
  )
}

function ModelSettingsModal({
  open,
  onClose,
  catalog,
}: {
  open: boolean
  onClose: () => void
  catalog: ModelCatalog
}) {
  const active = useChatStore((state) => state.modelConfig)
  const setModelConfig = useChatStore((state) => state.setModelConfig)
  const [draft, setDraft] = useState<ModelConfig>(active)
  const { message: toast } = AntApp.useApp()

  useEffect(() => {
    if (open) setDraft(active)
  }, [open, active])

  const provider = catalog.providers.find((item) => item.id === draft.provider)
    || fallbackModelCatalog.providers[0]

  const chooseProvider = (id: ModelProviderId) => {
    const next = catalog.providers.find((item) => item.id === id)
      || fallbackModelCatalog.providers.find((item) => item.id === id)!
    setDraft({
      provider: id,
      model: next.default_model || '',
      apiKey: '',
      baseUrl: next.base_url,
    })
  }

  const applyModel = () => {
    if (!draft.model.trim()) {
      toast.warning('请填写模型名称')
      return
    }
    if (draft.provider !== 'ollama' && !draft.baseUrl.trim()) {
      toast.warning('请填写 API Base URL')
      return
    }
    if (provider.requires_api_key && !provider.configured && !draft.apiKey.trim()) {
      toast.warning('请填写 API Key，或在后端环境变量中配置')
      return
    }
    setModelConfig({ ...draft, model: draft.model.trim(), baseUrl: draft.baseUrl.trim() })
    onClose()
    toast.success(`已切换到 ${draft.model.trim()}`)
  }

  const clearSavedApiKey = () => {
    const cleared = { ...active, apiKey: '' }
    setModelConfig(cleared)
    setDraft((value) => ({ ...value, apiKey: '' }))
    toast.success('已清除当前浏览器保存的 API Key')
  }

  const providerIcon = (id: ModelProviderId) => {
    if (id === 'ollama') return <Cpu size={18} />
    if (id === 'custom') return <ServerCog size={18} />
    return <Cloud size={18} />
  }

  return (
    <Modal
      open={open}
      onCancel={onClose}
      footer={null}
      title={null}
      width={650}
      className="model-modal"
    >
      <div className="modal-heading model-modal-heading">
        <span className="modal-icon"><ServerCog size={22} /></span>
        <div>
          <h2>选择与配置模型</h2>
          <p>Ollama 可稍后启动；未连接时仍可配置并使用云端 OpenAI 兼容模型。</p>
        </div>
      </div>

      <div className="provider-grid" role="radiogroup" aria-label="模型提供商">
        {catalog.providers.map((item) => (
          <button
            type="button"
            role="radio"
            aria-checked={draft.provider === item.id}
            key={item.id}
            className={`provider-card ${draft.provider === item.id ? 'active' : ''}`}
            onClick={() => chooseProvider(item.id)}
          >
            <span className="provider-icon">{providerIcon(item.id)}</span>
            <span>
              <strong>{item.label}</strong>
              <small>{item.description}</small>
            </span>
            {draft.provider === item.id && <Check size={15} className="provider-check" />}
          </button>
        ))}
      </div>

      <div className="model-config-panel">
        <div className="model-field">
          <label>模型名称</label>
          {draft.provider !== 'custom' ? (
            <Select
              value={draft.model}
              options={provider.models.map((model) => ({ value: model, label: model }))}
              onChange={(model) => setDraft((value) => ({ ...value, model }))}
              style={{ width: '100%' }}
              showSearch
              aria-label="选择模型"
            />
          ) : (
            <Input
              value={draft.model}
              onChange={(event) => setDraft((value) => ({ ...value, model: event.target.value }))}
              placeholder="输入模型名称"
              prefix={<Bot size={15} />}
            />
          )}
          {draft.provider === 'ollama' && provider.status_message && <small className="model-status-hint">{provider.status_message}</small>}
        </div>

        {draft.provider !== 'ollama' && (
          <>
            <div className="model-field">
              <label>API Key</label>
              <Input.Password
                value={draft.apiKey}
                onChange={(event) => setDraft((value) => ({ ...value, apiKey: event.target.value }))}
                placeholder={provider.configured ? '后端已配置；留空即可使用' : '保存后在当前浏览器中保留'}
                prefix={<KeyRound size={15} />}
                autoComplete="off"
              />
              {active.provider === draft.provider && active.apiKey && (
                <button type="button" className="clear-api-key" onClick={clearSavedApiKey}>
                  清除已保存的 API Key
                </button>
              )}
            </div>
            <div className="model-field">
              <label>API Base URL</label>
              <Input
                value={draft.baseUrl}
                onChange={(event) => setDraft((value) => ({ ...value, baseUrl: event.target.value }))}
                placeholder="https://example.com/v1"
                prefix={<Cloud size={15} />}
              />
            </div>
          </>
        )}

        <div className={`model-security-note ${draft.provider === 'ollama' ? 'local' : 'cloud'}`}>
          <ShieldCheck size={16} />
          <span>
            {draft.provider === 'ollama'
              ? '模型在本机运行；题目、检索上下文和回答不会发送到第三方模型服务。'
              : '使用云端模型时，题目、最近对话及检索上下文会发送到所选 API；配置和 API Key 会保存在此浏览器的本地存储中，不写入项目文件。'}
          </span>
        </div>
      </div>

      <div className="model-modal-actions">
        <Button onClick={onClose}>取消</Button>
        <Button type="primary" onClick={applyModel}>应用模型</Button>
      </div>
    </Modal>
  )
}

function StudentPageContent() {
  const [activeView, setActiveView] = useState<WorkspaceView>('chat')
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [kbModalOpen, setKbModalOpen] = useState(false)
  const [modelModalOpen, setModelModalOpen] = useState(false)
  const [newKbName, setNewKbName] = useState('')
  const [statuses, setStatuses] = useState<KBStatus[]>([])
  const [sessions, setSessions] = useState<SessionSummary[]>([])
  const [modelCatalog, setModelCatalog] = useState<ModelCatalog>(fallbackModelCatalog)
  const [knowledgeGraph, setKnowledgeGraph] = useState<KnowledgeGraph>()
  const [graphLoading, setGraphLoading] = useState(false)
  const [mistakes, setMistakes] = useState<MistakeItem[]>([])
  const studentId = useChatStore((state) => state.studentId)
  const messages = useChatStore((state) => state.messages)
  const sessionId = useChatStore((state) => state.sessionId)
  const mode = useChatStore((state) => state.mode)
  const setMode = useChatStore((state) => state.setMode)
  const send = useChatStore((state) => state.send)
  const knowledgeBase = useChatStore((state) => state.knowledgeBase)
  const setKnowledgeBase = useChatStore((state) => state.setKnowledgeBase)
  const modelConfig = useChatStore((state) => state.modelConfig)
  const setModelConfig = useChatStore((state) => state.setModelConfig)
  const loadSession = useChatStore((state) => state.loadSession)
  const clear = useChatStore((state) => state.clear)
  const { message: toast } = AntApp.useApp()

  const refreshStatuses = async () => {
    try {
      setStatuses(await fetchKnowledgeBases())
    } catch {
      setStatuses([{ id: 'default', state: 'missing', documents: 0, chunks: 0, message: '后端未连接' }])
    }
  }

  const refreshSessions = async () => {
    try {
      setSessions(await fetchSessions())
    } catch {
      setSessions([])
    }
  }

  const refreshMistakes = async () => {
    try {
      setMistakes(await fetchMistakes(studentId))
    } catch {
      setMistakes([])
    }
  }

  const refreshModels = async (allowAutoSwitch = false) => {
    try {
      const catalog = await fetchModels()
      setModelCatalog(catalog)
      const local = catalog.providers.find((item) => item.id === 'ollama')
      const preferred = catalog.providers.find((item) => item.id === catalog.default.provider)
      if (allowAutoSwitch && modelConfig.provider === 'ollama' && !local?.configured && preferred?.configured && preferred.id !== 'ollama') {
        setModelConfig({ provider: preferred.id, model: catalog.default.model, apiKey: '', baseUrl: preferred.base_url })
        toast.info(`Ollama 未启动，已使用已配置的 ${preferred.label}`)
      }
    } catch {
      setModelCatalog(fallbackModelCatalog)
    }
  }

  useEffect(() => {
    void refreshStatuses()
    void fetchSession(sessionId).then((stored) => {
      if (stored.length) loadSession(sessionId, stored)
    }).catch(() => undefined)
    void refreshSessions()
    void refreshMistakes()
    void refreshModels(true)
    const timer = window.setInterval(() => {
      void refreshStatuses()
      void refreshSessions()
      void refreshModels()
    }, 5000)
    return () => window.clearInterval(timer)
  }, [])

  useEffect(() => {
    if (modelModalOpen) void refreshModels()
  }, [modelModalOpen])

  useEffect(() => {
    if (activeView !== 'graph') return
    setGraphLoading(true)
    void fetchKnowledgeGraph(knowledgeBase)
      .then(setKnowledgeGraph)
      .catch(() => setKnowledgeGraph(undefined))
      .finally(() => setGraphLoading(false))
  }, [activeView, knowledgeBase])

  const kbOptions = useMemo(() => {
    const base = statuses.map((item) => ({
      value: item.id,
      label: item.id === 'default' ? '默认课程知识库' : item.id,
    }))
    if (!base.some((item) => item.value === knowledgeBase)) {
      base.push({ value: knowledgeBase, label: knowledgeBase })
    }
    return base
  }, [statuses, knowledgeBase])

  const ask = (prompt: string, preferredMode?: ChatMode) => {
    if (preferredMode) setMode(preferredMode)
    void send(prompt).then(() => refreshSessions())
  }

  const saveMistake = async (content: string, agent: string) => {
    try {
      const item = await addMistake(studentId, sessionId, content, agent, modelConfig)
      setMistakes((current) => [item, ...current.filter((existing) => existing.id !== item.id)])
      toast.success(`已加入错题本，并识别知识点：${item.knowledge_points.join('、')}`)
    } catch (error) {
      toast.error(error instanceof Error ? error.message : '加入错题本失败')
    }
  }

  const removeMistake = async (id: string) => {
    try {
      await deleteMistake(studentId, id)
      setMistakes((current) => current.filter((item) => item.id !== id))
      toast.success('已从错题本删除')
    } catch (error) {
      toast.error(error instanceof Error ? error.message : '删除错题失败')
    }
  }

  const planFromMistakes = () => {
    const points = [...new Set(mistakes.flatMap((item) => item.knowledge_points))]
    const summaries = mistakes.slice(0, 12).map((item, index) => `${index + 1}. ${item.summary}（${item.knowledge_points.join('、')}）`)
    setActiveView('chat')
    setMode('plan')
    ask(
      `请依据我的错题本制定知识补全与巩固学习规划。\n薄弱知识点：${points.join('、')}\n错题摘要：\n${summaries.join('\n')}`,
      'plan',
    )
  }

  const selectHistorySession = async (selectedSessionId: string) => {
    try {
      const stored = await fetchSession(selectedSessionId)
      loadSession(selectedSessionId, stored)
      setActiveView('chat')
      setSidebarOpen(false)
    } catch (error) {
      toast.error(error instanceof Error ? error.message : '历史会话恢复失败')
    }
  }

  const startNewSession = () => {
    clear()
    setActiveView('chat')
    setSidebarOpen(false)
  }

  const deleteHistorySession = async (deletedSessionId: string, title: string) => {
    try {
      await deleteSession(deletedSessionId)
      setSessions((current) => current.filter((item) => item.session_id !== deletedSessionId))
      if (deletedSessionId === sessionId) {
        clear()
        setSidebarOpen(false)
      }
      toast.success(`已删除“${title}”`)
    } catch (error) {
      toast.error(error instanceof Error ? error.message : '历史会话删除失败')
    }
  }

  const uploadRequest: NonNullable<UploadProps['customRequest']> = async (options) => {
    try {
      const result = await uploadKnowledgeFile(options.file as File, knowledgeBase, modelConfig)
      options.onSuccess?.(result)
      toast.success(result.message)
      void refreshStatuses()
    } catch (error) {
      const detail = error instanceof Error ? error.message : '上传失败'
      options.onError?.(new Error(detail))
      toast.error(detail)
    }
  }

  const rebuildCurrentKnowledgeBase = async () => {
    try {
      const result = await rebuildKnowledgeBase(knowledgeBase, modelConfig)
      toast.success(result.message)
      void refreshStatuses()
    } catch (error) {
      toast.error(error instanceof Error ? error.message : '知识库重建失败')
    }
  }

  const createKnowledgeBase = () => {
    const normalized = newKbName.trim().replace(/\s+/g, '-')
    if (!/^[A-Za-z0-9_-]{1,48}$/.test(normalized)) {
      toast.warning('名称仅支持字母、数字、连字符和下划线')
      return
    }
    setKnowledgeBase(normalized)
    setNewKbName('')
    toast.success(`已切换到新知识库 ${normalized}，请上传第一份资料`)
  }

  return (
    <div className="student-app">
      <Sidebar
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        sessions={sessions}
        activeSessionId={sessionId}
        onSelectSession={(selectedSessionId) => void selectHistorySession(selectedSessionId)}
        onDeleteSession={(deletedSessionId, title) => void deleteHistorySession(deletedSessionId, title)}
        onNewSession={startNewSession}
        activeView={activeView}
        onView={(view) => { setActiveView(view); setSidebarOpen(false) }}
      />
      <main className="main-workspace">
        <header className="topbar">
          <div className="topbar-left">
            <button className="menu-button" onClick={() => setSidebarOpen(true)} aria-label="打开导航"><Menu size={19} /></button>
            <div>
              <span className="breadcrumb">学生工作台 /</span>
              <strong>{activeView === 'graph' ? '知识图谱' : activeView === 'mistakes' ? '错题本' : mode === 'quiz' ? '同类题生成' : mode === 'answer' ? '课程答疑' : mode === 'plan' ? '学习规划' : '智能学习'}</strong>
            </div>
          </div>
          <div className="topbar-actions">
            <Select
              className="kb-select"
              value={knowledgeBase}
              options={kbOptions}
              onChange={setKnowledgeBase}
              suffixIcon={<Database size={14} />}
              aria-label="选择知识库"
            />
            <button
              type="button"
              className="model-badge model-picker-button"
              onClick={() => setModelModalOpen(true)}
              aria-label="选择和配置模型"
            >
              <span className={`online-dot ${modelConfig.provider === 'ollama' ? '' : 'cloud'}`} />
              <span>{modelConfig.model}</span>
              <small>{providerLabels[modelConfig.provider]}</small>
              <ChevronDown size={13} />
            </button>
          </div>
        </header>

        {activeView === 'chat' ? (
          <section className="learning-grid">
            <div className="chat-column">
              <div className="chat-scroll">
                {messages.length === 0 ? <Welcome onAsk={ask} /> : <Conversation onAddMistake={(content, agent) => void saveMistake(content, agent)} />}
              </div>
              <ChatComposer onSend={(value) => ask(value)} />
            </div>
            <KnowledgePanel statuses={statuses} onCreate={() => setKbModalOpen(true)} />
          </section>
        ) : activeView === 'graph' ? (
          <KnowledgeGraphView graph={knowledgeGraph} loading={graphLoading} />
        ) : (
          <MistakeBookView mistakes={mistakes} onDelete={(id) => void removeMistake(id)} onPlan={planFromMistakes} />
        )}
      </main>

      <ModelSettingsModal
        open={modelModalOpen}
        onClose={() => setModelModalOpen(false)}
        catalog={modelCatalog}
      />

      <Modal
        open={kbModalOpen}
        onCancel={() => setKbModalOpen(false)}
        footer={null}
        title={null}
        width={560}
        className="kb-modal"
      >
        <div className="modal-heading">
          <span className="modal-icon"><Database size={22} /></span>
          <div>
            <h2>扩充课程知识库</h2>
            <p>上传教材、讲义或题库后，系统会自动清洗、分块、嵌入并重建索引。</p>
          </div>
        </div>
        <div className="modal-section">
          <label>当前目标知识库</label>
          <Select value={knowledgeBase} options={kbOptions} onChange={setKnowledgeBase} style={{ width: '100%' }} />
        </div>
        <div className="new-kb-row">
          <Input
            value={newKbName}
            onChange={(event) => setNewKbName(event.target.value)}
            placeholder="新知识库英文标识，如 analog-circuits"
            prefix={<Plus size={15} />}
          />
          <Button onClick={createKnowledgeBase}>新建并切换</Button>
        </div>
        <Upload.Dragger
          multiple={false}
          accept=".pdf,.md,.txt,.docx,.xlsx,.json"
          customRequest={uploadRequest}
          showUploadList
          className="kb-dragger"
        >
          <p className="ant-upload-drag-icon"><UploadCloud size={28} /></p>
          <p className="ant-upload-text">拖入教材或题库，或点击选择文件</p>
          <p className="ant-upload-hint">支持 PDF、Word、Markdown、Excel、JSON，单文件最大 80 MB</p>
        </Upload.Dragger>
        <Button
          block
          icon={<Database size={16} />}
          onClick={() => void rebuildCurrentKnowledgeBase()}
          disabled={statuses.find((item) => item.id === knowledgeBase)?.state === 'building'}
        >
          使用当前模型重新构建已有资料
        </Button>
        <div className="modal-note">
          <Check size={15} /> 新知识库构建期间可继续使用其他已就绪知识库
        </div>
      </Modal>
    </div>
  )
}

export default function StudentPage() {
  return (
    <AntApp>
      <StudentPageContent />
    </AntApp>
  )
}
