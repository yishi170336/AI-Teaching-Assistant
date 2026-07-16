import { useEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  App as AntApp,
  Button,
  Input,
  Modal,
  Popconfirm,
  Progress,
  Segmented,
  Select,
  Slider,
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
  ExternalLink,
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
  AttachmentInfo,
  cancelKnowledgeBaseBuild,
  deleteKnowledgeBase,
  deleteMistake,
  deleteSession,
  fetchKnowledgeGraph,
  fetchKnowledgeBases,
  fetchMistakes,
  fetchModels,
  fetchSession,
  fetchSessions,
  KBStatus,
  knowledgeBaseSourceUrl,
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
import { CHAT_MODEL, CHAT_MODEL_PROVIDER, ChatMessage, ChatMode, useChatStore } from '../store/chatStore'

const { TextArea } = Input
type WorkspaceView = 'chat' | 'graph' | 'mistakes'
type MistakeDraft = {
  question: string
  answer: string
  agent: string
  attachments: AttachmentInfo[]
}

const providerLabels: Record<ModelProviderId, string> = {
  ollama: '本地',
  deepseek: 'DeepSeek',
  qwen: '通义千问',
  custom: '自定义 API',
}

const fallbackModelCatalog: ModelCatalog = {
  default: { provider: CHAT_MODEL_PROVIDER, model: CHAT_MODEL },
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
      models: ['qwen3.7-plus', 'qwen3.7-max', 'qwen-vl-max', 'qwen3-vl-plus', 'qwen3-vl-flash'],
      model_options: [
        { value: 'qwen3.7-plus', label: 'Qwen3.7-Plus' },
        { value: 'qwen3.7-max', label: 'Qwen3.7-Max' },
        { value: 'qwen-vl-max', label: 'qwen-vl-max' },
        { value: 'qwen3-vl-8b-instruct', label: 'qwen3-vl-8b-instruct', disabled: true, description: '当前账号未开放' },
        { value: 'qwen3-vl-plus', label: 'qwen3-vl-plus' },
        { value: 'qwen3-vl-flash', label: 'qwen3-vl-flash' },
        { value: 'qwen3-vl-embedding', label: 'qwen3-vl-embedding', disabled: true, description: '仅用于向量化' },
      ],
      default_model: 'qwen3-vl-flash',
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
          <Link to="/practice" className="nav-item practice-entry-link">
            <FileText size={17} />
            <span>刷题训练</span>
          </Link>
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

function SourceCard({
  source,
  index,
  isCited,
  fallbackKnowledgeBase,
}: {
  source: SourceInfo
  index: number
  isCited: boolean
  fallbackKnowledgeBase: string
}) {
  const page = source.page_start
    ? source.page_start === source.page_end
      ? `第 ${source.page_start} 页`
      : `第 ${source.page_start}–${source.page_end} 页`
    : '结构化题库'
  const openSource = () => {
    const knowledgeBase = source.knowledge_base || fallbackKnowledgeBase
    if (!knowledgeBase) return
    window.open(
      knowledgeBaseSourceUrl(knowledgeBase, source.source, source.page_start),
      '_blank',
      'noopener,noreferrer',
    )
  }
  return (
    <article
      className="source-card source-card-openable"
      role="link"
      tabIndex={0}
      aria-label={`查看完整资料 ${source.source}`}
      onClick={openSource}
      onKeyDown={(event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault()
          openSource()
        }
      }}
    >
      <div className="source-card-top">
        <div className="source-labels">
          <span className={`source-type ${source.doc_type === 'question' ? 'question' : ''}`}>
            {source.doc_type === 'question' ? <WandSparkles size={13} /> : <FileText size={13} />}
            资料 {index + 1}
          </span>
          {isCited && <span className="source-cited-badge">已引用</span>}
        </div>
        <span className="source-score">
          {source.historical ? '历史记录' : `${Math.round(source.score * 100)}%`}
        </span>
      </div>
      <strong>{source.section || source.chapter || source.source}</strong>
      <p>{source.source}</p>
      {source.excerpt && <p className="source-excerpt">{source.excerpt}</p>}
      {source.knowledge_tags?.length ? (
        <div className="source-tags">{source.knowledge_tags.slice(0, 4).map((tag) => <span key={tag}>{tag}</span>)}</div>
      ) : null}
      {!source.historical && (
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
      )}
      <div className="source-meta">
        <span>{page}</span>
        <span className="source-open-label"><ExternalLink size={12} /> 查看全文</span>
      </div>
    </article>
  )
}

function KnowledgePanel({ statuses, onCreate }: { statuses: KBStatus[]; onCreate: () => void }) {
  const activeSources = useChatStore((state) => state.activeSources)
  const activeCitedSources = useChatStore((state) => state.activeCitedSources)
  const activeMessageId = useChatStore((state) => state.activeMessageId)
  const knowledgeBase = useChatStore((state) => state.knowledgeBase)
  const defaultKnowledgeBase = useChatStore((state) => state.defaultKnowledgeBase)
  const modelProvider = useChatStore((state) => state.modelConfig.provider)
  const messages = useChatStore((state) => state.messages)
  const activeAssistant = messages.find((item) => item.id === activeMessageId)
    || [...messages].reverse().find((item) => item.role === 'assistant')
  const activeKnowledgeBase = activeAssistant?.knowledgeBase
    || activeSources[0]?.knowledge_base
    || knowledgeBase
  const current = statuses.find((item) => item.id === activeKnowledgeBase)
  const quizContext = activeAssistant?.agent === '出题 Agent'
  const citedSourceIds = new Set(activeCitedSources.map((source) => source.id))
  const citedIndices = new Set(
    activeCitedSources
      .map((source) => source.citation_index)
      .filter((index): index is number => typeof index === 'number'),
  )
  return (
    <aside className="knowledge-panel">
      <div className="panel-heading">
        <div>
          <span className="panel-kicker">{quizContext ? 'SOURCE PROBLEM' : 'RAG CANDIDATES'}</span>
          <h2>{quizContext ? '原题依据' : '召回候选资料'}</h2>
        </div>
        <Tooltip title={quizContext ? '出题 Agent 仅使用原题和会话历史，不调用知识库检索' : '这里展示本轮检索召回的候选资料；真正进入答案引用的资料会标记“已引用”'}>
          <HelpCircle size={17} />
        </Tooltip>
      </div>

      <div className="kb-summary-card">
        <span className="kb-icon">{quizContext ? <BrainCircuit size={18} /> : <Database size={18} />}</span>
        <div>
          <strong>{quizContext ? '原题驱动出题' : activeKnowledgeBase === defaultKnowledgeBase ? '默认课程知识库' : activeKnowledgeBase}</strong>
          <span>{quizContext ? '会话上下文 · 不检索知识库' : `${current?.chunks || 0} 个文本块 · ${current?.documents || 0} 份资料`}</span>
        </div>
        <span className={`kb-state ${quizContext ? 'ready' : current?.state || 'missing'}`}>
          {quizContext
            ? '已锁定'
            : current?.state === 'building'
              ? '构建中'
              : current?.state === 'cancelling'
                ? '取消中'
                : current?.state === 'cancelled'
                  ? '已取消'
                  : current?.validation?.status === 'passed'
                    ? '已校验'
                    : current?.state === 'ready' ? '就绪' : '待构建'}
        </span>
      </div>

      {!quizContext && activeSources.length > 0 && (
        <div className={`source-usage-summary ${activeCitedSources.length ? '' : 'uncited'}`}>
          <span>本轮召回 {activeSources.length} 条</span>
          <strong>答案引用 {activeCitedSources.length} 条</strong>
        </div>
      )}

      <div className="source-list">
        {quizContext ? (
          <div className="source-empty quiz-reference-empty">
            <span><WandSparkles size={22} /></span>
            <strong>保持原题结构</strong>
            <p>首次出题读取当前原题；“再出一道”会沿用最近题目的拓扑、已知量和分项设问。</p>
          </div>
        ) : activeSources.length ? (
          activeSources.map((source, index) => (
            <SourceCard
              key={`${source.id}-${index}`}
              source={source}
              index={index}
              isCited={citedSourceIds.has(source.id) || citedIndices.has(index + 1)}
              fallbackKnowledgeBase={activeKnowledgeBase}
            />
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
          {modelProvider === 'ollama' ? '资料与模型推理均保留在本机' : '提问内容将发送至通义千问 API'}
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

function mistakeAttachmentsForMessage(messages: ChatMessage[], index: number): AttachmentInfo[] {
  const message = messages[index]
  if (message.attachments?.length) return message.attachments
  if (message.role !== 'assistant') return []
  for (let previousIndex = index - 1; previousIndex >= 0; previousIndex -= 1) {
    const previous = messages[previousIndex]
    if (previous.role !== 'user') continue
    if (previous.attachments?.length) return previous.attachments
    if (!/(上述|该电路|此电路|这个电路|该图|此图|上图|图中|刚才)/.test(previous.content)) return []
  }
  return []
}

function mistakeDraftForAssistant(messages: ChatMessage[], index: number): MistakeDraft | null {
  const message = messages[index]
  if (message.role !== 'assistant' || !message.content) return null
  const agent = message.agent || ''
  if (agent !== '答疑 Agent' && agent !== '出题 Agent') return null
  const previousUser = [...messages.slice(0, index)].reverse().find((item) => item.role === 'user')
  if (!previousUser?.content) return null
  let question = previousUser.content
  let answer = message.content
  if (agent === '出题 Agent') {
    const questionMatch = message.content.match(
      /(?:^|\n)###\s*题目\s*\n+([\s\S]*?)(?=\n+---|\n+###\s*(?:解题步骤|标准答案|易错点)|$)/,
    )
    const answerStart = message.content.search(/(?:^|\n)###\s*(?:解题步骤|标准答案)/)
    if (questionMatch?.[1]?.trim()) question = questionMatch[1].trim()
    if (answerStart >= 0) answer = message.content.slice(answerStart).trim()
  }
  return {
    question,
    answer,
    agent,
    attachments: mistakeAttachmentsForMessage(messages, index),
  }
}

function Conversation({ onAddMistake }: { onAddMistake: (draft: MistakeDraft) => void }) {
  const messages = useChatStore((state) => state.messages)
  const streaming = useChatStore((state) => state.streaming)
  const stage = useChatStore((state) => state.stage)
  const stageAgent = useChatStore((state) => state.stageAgent)
  const activeMessageId = useChatStore((state) => state.activeMessageId)
  const activateMessage = useChatStore((state) => state.activateMessage)
  const endRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages, stage])

  useEffect(() => {
    const container = endRef.current?.closest('.chat-scroll') as HTMLElement | null
    if (!container) return
    let frame = 0
    const updateActiveMessage = () => {
      window.cancelAnimationFrame(frame)
      frame = window.requestAnimationFrame(() => {
        const rows = Array.from(
          container.querySelectorAll<HTMLElement>('.message-row.assistant[data-message-id]'),
        )
        if (!rows.length) return
        const containerRect = container.getBoundingClientRect()
        const focusY = containerRect.top + containerRect.height * 0.42
        const closest = rows.reduce((best, row) => {
          const rect = row.getBoundingClientRect()
          const distance = Math.abs(rect.top + Math.min(rect.height / 2, 140) - focusY)
          return distance < best.distance ? { row, distance } : best
        }, { row: rows[0], distance: Number.POSITIVE_INFINITY })
        const messageId = closest.row.dataset.messageId
        if (messageId) activateMessage(messageId)
      })
    }
    container.addEventListener('scroll', updateActiveMessage, { passive: true })
    updateActiveMessage()
    return () => {
      container.removeEventListener('scroll', updateActiveMessage)
      window.cancelAnimationFrame(frame)
    }
  }, [messages.length, activateMessage])

  return (
    <div className="conversation">
      {messages.map((message, index) => (
        <div
          key={message.id}
          className={`message-row ${message.role} ${message.id === activeMessageId ? 'active-evidence-message' : ''}`}
          data-message-id={message.id}
          onClick={() => message.role === 'assistant' && activateMessage(message.id)}
        >
          {message.role === 'assistant' && (
            <span className="assistant-avatar"><LogoMark /></span>
          )}
          <div className={`message-bubble ${message.failed ? 'failed' : ''}`}>
            {message.role === 'assistant' && (
              <div className="message-agent">
                <span>{message.agent || (streaming && index === messages.length - 1 ? stageAgent || '多智能体助教' : '多智能体助教')}</span>
                <Tag bordered={false} title={`${providerLabels[message.provider || CHAT_MODEL_PROVIDER]} · ${message.model || ''}`}>
                  {providerLabels[message.provider || CHAT_MODEL_PROVIDER]} · {message.model || CHAT_MODEL}
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
            {message.content
              && message.role === 'assistant'
              && !message.failed
              && (message.agent === '答疑 Agent' || message.agent === '出题 Agent')
              && !(streaming && index === messages.length - 1) && (
              <div className="message-tools">
                <button
                  type="button"
                  onClick={() => {
                    const draft = mistakeDraftForAssistant(messages, index)
                    if (draft) onAddMistake(draft)
                  }}
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
  const [rangeMode, setRangeMode] = useState<'core' | 'extended' | 'all' | 'custom'>('core')
  const [limits, setLimits] = useState({ concepts: 14, pages: 8, structures: 14 })
  const totals = useMemo(() => ({
    concepts: graph?.nodes.filter((node) => node.type === 'concept').length || 0,
    pages: graph?.nodes.filter((node) => node.type === 'page').length || 0,
    structures: graph?.nodes.filter((node) => node.type === 'circuit' || node.type === 'component').length || 0,
  }), [graph])

  const presetLimits = (mode: 'core' | 'extended' | 'all') => ({
    concepts: mode === 'all' ? totals.concepts : Math.min(totals.concepts, mode === 'core' ? 14 : 28),
    pages: mode === 'all' ? totals.pages : Math.min(totals.pages, mode === 'core' ? 8 : 20),
    structures: mode === 'all' ? totals.structures : Math.min(totals.structures, mode === 'core' ? 14 : 28),
  })

  useEffect(() => {
    setRangeMode('core')
    setLimits({
      concepts: Math.min(totals.concepts, 14),
      pages: Math.min(totals.pages, 8),
      structures: Math.min(totals.structures, 14),
    })
    setSelectedId('')
  }, [graph?.knowledge_base])

  const chooseRangeMode = (value: string | number) => {
    const mode = String(value) as 'core' | 'extended' | 'all' | 'custom'
    setRangeMode(mode)
    if (mode !== 'custom') setLimits(presetLimits(mode))
  }

  const changeRange = (key: keyof typeof limits, value: number) => {
    setRangeMode('custom')
    setLimits((current) => ({ ...current, [key]: value }))
  }

  const visual = useMemo(() => {
    if (!graph) return { nodes: [], edges: [] }
    const degree = new Map<string, number>()
    graph.edges.forEach((edge) => {
      degree.set(edge.source, (degree.get(edge.source) || 0) + 1)
      degree.set(edge.target, (degree.get(edge.target) || 0) + 1)
    })
    const documents = graph.nodes.filter((node) => node.type === 'document').slice(0, 3)
    const concepts = graph.nodes
      .filter((node) => node.type === 'concept')
      .sort((a, b) => (b.evidence_count || degree.get(b.id) || 0) - (a.evidence_count || degree.get(a.id) || 0))
      .slice(0, limits.concepts)
    const selectedConcepts = new Set(concepts.map((node) => node.id))
    const selectedCoverage = new Map<string, number>()
    graph.edges.forEach((edge) => {
      if (edge.type === 'COVERS' && selectedConcepts.has(edge.target)) {
        selectedCoverage.set(edge.source, (selectedCoverage.get(edge.source) || 0) + (edge.evidence_count || 1))
      }
    })
    const pages = graph.nodes
      .filter((node) => node.type === 'page')
      .sort((a, b) => (
        (selectedCoverage.get(b.id) || 0) - (selectedCoverage.get(a.id) || 0)
        || (degree.get(b.id) || 0) - (degree.get(a.id) || 0)
        || (a.page || 0) - (b.page || 0)
      ))
      .slice(0, limits.pages)
    const structures = graph.nodes
      .filter((node) => node.type === 'circuit' || node.type === 'component')
      .sort((a, b) => (degree.get(b.id) || 0) - (degree.get(a.id) || 0))
      .slice(0, limits.structures)
    const groups = [
      { nodes: documents, radius: 0, ringGap: 34, capacity: 1 },
      { nodes: concepts, radius: 125, ringGap: 38, capacity: 18 },
      { nodes: pages, radius: 260, ringGap: 34, capacity: 24 },
      { nodes: structures, radius: 345, ringGap: 30, capacity: 28 },
    ]
    const positioned = groups.flatMap((group) => group.nodes.map((node, index) => {
      const ringIndex = Math.floor(index / group.capacity)
      const ringStart = ringIndex * group.capacity
      const ringSize = Math.min(group.capacity, group.nodes.length - ringStart)
      const ringPosition = index - ringStart
      const angle = (Math.PI * 2 * ringPosition) / Math.max(1, ringSize) - Math.PI / 2
      const radius = group.radius + ringIndex * group.ringGap
      return {
        ...node,
        x: 400 + Math.cos(angle) * radius,
        y: 380 + Math.sin(angle) * radius,
        degree: degree.get(node.id) || 0,
      }
    }))
    const ids = new Set(positioned.map((node) => node.id))
    return {
      nodes: positioned,
      edges: graph.edges.filter((edge) => ids.has(edge.source) && ids.has(edge.target)),
    }
  }, [graph, limits])
  const positions = new Map(visual.nodes.map((node) => [node.id, node]))
  const selected = graph?.nodes.find((node) => node.id === selectedId)
  const neighbors = selectedId && graph
    ? graph.edges.filter((edge) => edge.source === selectedId || edge.target === selectedId).length
    : 0
  const typeLabel: Record<string, string> = {
    document: '教材',
    page: '教材页面',
    concept: '知识点',
    circuit: '电路图',
    component: '电路元件',
  }
  const selectedPages = selected?.pages?.length
    ? selected.pages
    : selected?.page
      ? [selected.page]
      : []

  if (loading) return <div className="workspace-empty"><LoaderCircle className="spin" /><strong>正在整理知识图谱…</strong></div>
  if (!graph?.nodes.length) return <div className="workspace-empty"><Network /><strong>当前知识库还没有图谱数据</strong><p>重建知识库后会自动提取知识点与资料关系。</p></div>
  return (
    <section className="feature-view graph-view">
      <div className="feature-heading">
        <div><span>KNOWLEDGE MAP</span><h1>课程知识图谱</h1><p>展示教材、页面、知识点与电路结构；公式和文本片段作为证据收纳在节点详情中。</p></div>
        <div className="feature-stats"><strong>{graph.stats.concepts}</strong><span>知识点</span><strong>{graph.stats.pages || 0}</strong><span>页面</span><strong>{graph.stats.edges}</strong><span>关系</span></div>
      </div>
      <div className="graph-scope-panel">
        <div className="graph-scope-heading">
          <div><strong>图谱显示范围</strong><span>当前显示 {visual.nodes.length} 个节点、{visual.edges.length} 条关系</span></div>
          <Segmented
            value={rangeMode}
            onChange={chooseRangeMode}
            options={[
              { label: '核心', value: 'core' },
              { label: '扩展', value: 'extended' },
              { label: '全部', value: 'all' },
              { label: '自定义', value: 'custom' },
            ]}
          />
        </div>
        <div className="graph-range-grid">
          <label>
            <span>知识点 <b>{limits.concepts}/{totals.concepts}</b></span>
            <Slider min={0} max={Math.max(1, totals.concepts)} value={limits.concepts} disabled={!totals.concepts} onChange={(value) => changeRange('concepts', value)} />
          </label>
          <label>
            <span>教材页面 <b>{limits.pages}/{totals.pages}</b></span>
            <Slider min={0} max={Math.max(1, totals.pages)} value={limits.pages} disabled={!totals.pages} onChange={(value) => changeRange('pages', value)} />
          </label>
          <label>
            <span>电路与元件 <b>{limits.structures}/{totals.structures}</b></span>
            <Slider min={0} max={Math.max(1, totals.structures)} value={limits.structures} disabled={!totals.structures} onChange={(value) => changeRange('structures', value)} />
          </label>
        </div>
      </div>
      <div className="graph-layout">
        <div className="graph-canvas">
          <svg viewBox="0 0 800 760" role="img" aria-label="课程知识关系图">
            <g className="graph-edges">
              {visual.edges.map((edge, index) => {
                const from = positions.get(edge.source)
                const to = positions.get(edge.target)
                return from && to ? <line className={`relation-${edge.type.toLowerCase()}`} key={`${edge.source}-${edge.target}-${index}`} x1={from.x} y1={from.y} x2={to.x} y2={to.y} /> : null
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
                  <circle r={node.type === 'document' ? 25 : node.type === 'page' ? 16 : node.type === 'concept' ? Math.min(22, 12 + node.degree) : node.type === 'circuit' ? 12 : 9} />
                  {(node.type !== 'page' || limits.pages <= 12 || selectedId === node.id) && (
                    <text y={node.type === 'document' ? 40 : node.type === 'page' || node.type === 'concept' ? 32 : 23}>{node.name?.replace(/^电路图\s*[·•]\s*/, '').slice(0, 18) || typeLabel[node.type] || '资料'}</text>
                  )}
                </g>
              ))}
            </g>
          </svg>
          <div className="graph-legend"><span><i className="document" />教材</span><span><i className="page" />页面</span><span><i className="concept" />知识点</span><span><i className="circuit" />电路图</span><span><i className="component" />元件</span></div>
        </div>
        <aside className="graph-detail">
          {selected ? <><span>{typeLabel[selected.type] || '知识节点'}</span><h2>{selected.name || '未命名节点'}</h2><p>连接 {neighbors} 个语义节点{selected.evidence_count ? `，由 ${selected.evidence_count} 条教材证据支持` : ''}。公式与正文片段不会单独铺在图中，但仍参与检索和答案引用。</p>{selectedPages.length > 0 && <div className="graph-page-list">来源页码：{selectedPages.map((page) => `第 ${page} 页`).join('、')}</div>}</> : <><Network size={28} /><h2>探索知识关系</h2><p>教材位于中心，绿色知识点构成语义核心，蓝色页面与外围电路结构作为可追溯证据。</p></>}
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
  const [selectedMistake, setSelectedMistake] = useState<MistakeItem | null>(null)
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
              {item.attachments?.length ? (
                <div className="mistake-attachments">
                  {item.attachments.map((attachment) => (
                    <a key={attachment.id} href={attachment.url} target="_blank" rel="noreferrer">
                      {attachment.kind === 'image'
                        ? <img src={attachment.url} alt={attachment.name} />
                        : <span><FileText size={16} />{attachment.name}</span>}
                    </a>
                  ))}
                </div>
              ) : null}
              <div className="mistake-content"><MathMarkdown content={item.question || item.content} /></div>
              <div className="mistake-actions">
                <button className="mistake-open" onClick={() => setSelectedMistake(item)}>
                  <BookOpen size={14} /> 查看题目与答案
                </button>
                <Popconfirm title="从错题本删除？" okText="删除" cancelText="取消" onConfirm={() => onDelete(item.id)}>
                  <button className="mistake-delete"><Trash2 size={14} /> 删除</button>
                </Popconfirm>
              </div>
            </article>
          ))}
        </div>
      ) : <div className="workspace-empty"><Layers3 size={30} /><strong>错题本还是空的</strong><p>在答疑或出题结果旁点击“加入错题本”，系统会自动识别知识点。</p></div>}
      <Modal
        open={Boolean(selectedMistake)}
        title={selectedMistake?.summary || '错题详情'}
        width={860}
        onCancel={() => setSelectedMistake(null)}
        footer={<Button onClick={() => setSelectedMistake(null)}>关闭</Button>}
      >
        {selectedMistake && (
          <div className="mistake-detail">
            <section>
              <span className="mistake-detail-label">题目</span>
              {selectedMistake.attachments?.length ? (
                <div className="mistake-attachments mistake-detail-attachments">
                  {selectedMistake.attachments.map((attachment) => (
                    <a key={attachment.id} href={attachment.url} target="_blank" rel="noreferrer">
                      {attachment.kind === 'image'
                        ? <img src={attachment.url} alt={attachment.name} />
                        : <span><FileText size={16} />{attachment.name}</span>}
                    </a>
                  ))}
                </div>
              ) : null}
              <div className="mistake-detail-content"><MathMarkdown content={selectedMistake.question || selectedMistake.content} /></div>
            </section>
            <section>
              <span className="mistake-detail-label">答案</span>
              <div className="mistake-detail-content answer">
                {selectedMistake.answer
                  ? <MathMarkdown content={selectedMistake.answer} />
                  : <p>该历史错题没有可恢复的答案。</p>}
              </div>
            </section>
          </div>
        )}
      </Modal>
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
  const selectableModels = provider.model_options || provider.models.map((model) => ({
    value: model,
    label: model,
    disabled: false,
    description: '',
  }))

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
    const selectedOption = provider.model_options?.find((option) => option.value === draft.model)
    if (selectedOption?.disabled) {
      toast.warning(selectedOption.description || '该模型不能用于当前对话')
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
          <p>答题与附件识别使用当前模型；检索可临时调用专用模型，最终仍由当前模型回答。</p>
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
              options={selectableModels.map((option) => ({
                value: option.value,
                label: option.description ? `${option.label} · ${option.description}` : option.label,
                disabled: option.disabled,
              }))}
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
  const [newKbFile, setNewKbFile] = useState<File | null>(null)
  const [newKbName, setNewKbName] = useState('')
  const [creatingKnowledgeBase, setCreatingKnowledgeBase] = useState(false)
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
  const defaultKnowledgeBase = useChatStore((state) => state.defaultKnowledgeBase)
  const setKnowledgeBase = useChatStore((state) => state.setKnowledgeBase)
  const setDefaultKnowledgeBase = useChatStore((state) => state.setDefaultKnowledgeBase)
  const syncKnowledgeBases = useChatStore((state) => state.syncKnowledgeBases)
  const modelConfig = useChatStore((state) => state.modelConfig)
  const setModelConfig = useChatStore((state) => state.setModelConfig)
  const loadSession = useChatStore((state) => state.loadSession)
  const clear = useChatStore((state) => state.clear)
  const { message: toast } = AntApp.useApp()
  const previousBuildStates = useRef<Record<string, KBStatus['state']>>({})
  const activeBuilds = statuses.filter((item) => item.state === 'building' || item.state === 'cancelling')
  const hasActiveBuilds = activeBuilds.length > 0
  const currentKbStatus = statuses.find((item) => item.id === knowledgeBase)
  const deleteKbDisabledReason = !currentKbStatus
    ? '该知识库尚未创建'
    : currentKbStatus.state === 'building' || currentKbStatus.state === 'cancelling'
      ? '请先取消正在进行的构建任务'
      : ''

  const refreshStatuses = async () => {
    try {
      const nextStatuses = await fetchKnowledgeBases()
      setStatuses(nextStatuses)
      syncKnowledgeBases(nextStatuses)
    } catch {
      // Keep the last successful list and selection during a transient backend outage.
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

  const upsertBuildStatus = (state?: KBStatus) => {
    if (!state?.id) return
    setStatuses((current) => (
      current.some((item) => item.id === state.id)
        ? current.map((item) => item.id === state.id ? { ...item, ...state } : item)
        : [...current, state]
    ))
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
    if (!hasActiveBuilds) return
    const timer = window.setInterval(() => void refreshStatuses(), 1200)
    return () => window.clearInterval(timer)
  }, [hasActiveBuilds])

  useEffect(() => {
    const previous = previousBuildStates.current
    statuses.forEach((item) => {
      const prior = previous[item.id]
      if (prior === 'building' || prior === 'cancelling') {
        if (item.state === 'ready') toast.success(`知识库 ${item.id} 构建完成`)
        if (item.state === 'cancelled') toast.info(`知识库 ${item.id} 构建已取消，缓存已清理`)
        if (item.state === 'error') toast.error(`知识库 ${item.id} 构建失败：${item.message}`)
      }
    })
    previousBuildStates.current = Object.fromEntries(
      statuses.map((item) => [item.id, item.state]),
    )
  }, [statuses])

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
      label: item.id === defaultKnowledgeBase ? `${item.id}（默认课程）` : item.id,
    }))
    if (knowledgeBase && !base.some((item) => item.value === knowledgeBase)) {
      base.push({
        value: knowledgeBase,
        label: knowledgeBase === defaultKnowledgeBase ? `${knowledgeBase}（默认课程）` : knowledgeBase,
      })
    }
    return base
  }, [statuses, knowledgeBase, defaultKnowledgeBase])

  const defaultKbOptions = useMemo(() => statuses.map((item) => ({
    value: item.id,
    label: item.id === defaultKnowledgeBase ? `${item.id}（当前默认）` : item.id,
    disabled: item.state !== 'ready' && !item.available,
  })), [statuses, defaultKnowledgeBase])

  const chooseDefaultKnowledgeBase = (id: string) => {
    const target = statuses.find((item) => item.id === id)
    if (!target || (target.state !== 'ready' && !target.available)) {
      toast.warning('只有存在可用索引的知识库可以设为默认课程知识库')
      return
    }
    setDefaultKnowledgeBase(id)
    toast.success(`已将 ${id} 设为默认课程知识库`)
  }

  const ask = (prompt: string, preferredMode?: ChatMode) => {
    if (preferredMode) setMode(preferredMode)
    void send(prompt).then(() => refreshSessions())
  }

  const saveMistake = async ({ question, answer, agent, attachments }: MistakeDraft) => {
    try {
      const item = await addMistake(studentId, sessionId, question, answer, agent, attachments, modelConfig)
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
      upsertBuildStatus(result.build)
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
      upsertBuildStatus(result.build)
      toast.success(result.message)
      void refreshStatuses()
    } catch (error) {
      toast.error(error instanceof Error ? error.message : '知识库重建失败')
    }
  }

  const cancelBuild = async (id: string) => {
    try {
      const result = await cancelKnowledgeBaseBuild(id)
      setStatuses((current) => current.map((item) => (
        item.id === id ? { ...item, ...result.state } : item
      )))
      toast.info(result.message)
    } catch (error) {
      toast.error(error instanceof Error ? error.message : '取消构建失败')
    }
  }

  const removeKnowledgeBase = async () => {
    try {
      const deleted = knowledgeBase
      const replacement = statuses.find((item) => (
        item.id !== deleted
        && (item.state === 'ready' || item.available)
      ))
      const nextKnowledgeBase = replacement?.id || ''
      const result = await deleteKnowledgeBase(deleted)
      if (deleted === defaultKnowledgeBase) {
        setDefaultKnowledgeBase(nextKnowledgeBase)
      } else {
        setKnowledgeBase(
          statuses.some((item) => item.id === defaultKnowledgeBase && item.id !== deleted)
            ? defaultKnowledgeBase
            : nextKnowledgeBase,
        )
      }
      setKnowledgeGraph(undefined)
      await refreshStatuses()
      toast.success(result.message)
    } catch (error) {
      toast.error(error instanceof Error ? error.message : '删除知识库失败')
    }
  }

  const createKnowledgeBase = async () => {
    if (!newKbFile) {
      toast.warning('请先选择需要上传的教材文件')
      return
    }
    const normalized = newKbName.trim().replace(/\s+/g, '-')
    if (!/^[A-Za-z0-9_-]{1,48}$/.test(normalized)) {
      toast.warning('名称仅支持字母、数字、连字符和下划线')
      return
    }
    if (statuses.some((item) => item.id === normalized)) {
      toast.warning(`知识库 ${normalized} 已存在，请在下方选择后追加资料`)
      return
    }
    setCreatingKnowledgeBase(true)
    try {
      const result = await uploadKnowledgeFile(newKbFile, normalized, modelConfig)
      setKnowledgeBase(normalized)
      upsertBuildStatus(result.build)
      setNewKbFile(null)
      setNewKbName('')
      toast.success(`知识库 ${normalized} 已创建，正在构建索引`)
      void refreshStatuses()
    } catch (error) {
      toast.error(error instanceof Error ? error.message : '知识库创建失败')
    } finally {
      setCreatingKnowledgeBase(false)
    }
  }

  const closeKnowledgeBaseModal = () => {
    if (creatingKnowledgeBase) return
    setKbModalOpen(false)
    setNewKbFile(null)
    setNewKbName('')
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
              aria-label={`配置 ${modelConfig.model}`}
            >
              <span className={`online-dot ${modelConfig.provider === 'ollama' ? '' : 'cloud'}`} />
              <span>{modelConfig.model}</span>
              <small>{providerLabels[modelConfig.provider]}</small>
              <ChevronDown size={13} />
            </button>
          </div>
        </header>

        {activeBuilds.length > 0 && (
          <section className="build-task-stack" aria-label="知识库构建任务">
            {activeBuilds.map((item) => (
              <div className="build-task-banner" key={item.id}>
                <span className="build-task-icon"><LoaderCircle className="spin" size={18} /></span>
                <div className="build-task-copy">
                  <strong>{item.state === 'cancelling' ? `正在取消 ${item.id}` : `正在构建知识库 ${item.id}`}</strong>
                  <span>{item.message}</span>
                  <Progress percent={item.progress || 0} size="small" showInfo={false} />
                </div>
                <div className="build-task-progress">{item.progress || 0}%</div>
                <Button
                  danger
                  size="small"
                  disabled={!item.cancellable}
                  onClick={() => void cancelBuild(item.id)}
                >
                  {item.state === 'cancelling' ? '清理中' : '取消构建'}
                </Button>
              </div>
            ))}
          </section>
        )}

        {activeView === 'chat' ? (
          <section className="learning-grid">
            <div className="chat-column">
              <div className="chat-scroll">
                {messages.length === 0 ? <Welcome onAsk={ask} /> : <Conversation onAddMistake={(draft) => void saveMistake(draft)} />}
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
        onCancel={closeKnowledgeBaseModal}
        maskClosable={!creatingKnowledgeBase}
        closable={!creatingKnowledgeBase}
        footer={null}
        title={null}
        width={560}
        className="kb-modal"
      >
        <div className="modal-heading">
          <span className="modal-icon"><Database size={22} /></span>
          <div>
            <h2>添加教材 / 新建知识库</h2>
            <p>先选择首份资料，再命名并确认创建；系统随后自动构建索引。</p>
          </div>
        </div>
        <div className="modal-section new-kb-builder">
          <div className="modal-section-title">
            <strong>新建知识库</strong>
            <span>按顺序完成以下步骤</span>
          </div>
          <label><span className="step-number">1</span> 选择首份资料</label>
          <Upload.Dragger
            multiple={false}
            maxCount={1}
            accept=".pdf,.md,.txt,.docx"
            beforeUpload={(file) => {
              setNewKbFile(file)
              return Upload.LIST_IGNORE
            }}
            showUploadList={false}
            className="kb-dragger kb-create-dragger"
            disabled={creatingKnowledgeBase}
          >
            <p className="ant-upload-drag-icon"><UploadCloud size={28} /></p>
            <p className="ant-upload-text">
              {newKbFile ? '重新选择首份资料' : '拖入教材，或点击选择文件'}
            </p>
            <p className="ant-upload-hint">支持 PDF、Word、Markdown 和文本</p>
          </Upload.Dragger>
          {newKbFile && (
            <div className="kb-selected-file" aria-label={`已选择 ${newKbFile.name}`}>
              <span className="kb-selected-file-icon"><FileText size={17} /></span>
              <span>
                <strong>{newKbFile.name}</strong>
                <small>{Math.max(1, Math.round(newKbFile.size / 1024))} KB · 等待创建</small>
              </span>
              <button
                type="button"
                onClick={() => setNewKbFile(null)}
                disabled={creatingKnowledgeBase}
                aria-label={`移除 ${newKbFile.name}`}
              >
                <X size={15} />
              </button>
            </div>
          )}
          <label className="new-kb-name-label"><span className="step-number">2</span> 输入新知识库名称</label>
          <Input
            value={newKbName}
            onChange={(event) => setNewKbName(event.target.value)}
            placeholder="英文标识，如 analog-circuits"
            prefix={<Plus size={15} />}
            disabled={!newKbFile || creatingKnowledgeBase}
            onPressEnter={() => void createKnowledgeBase()}
          />
          <p className="modal-field-help">支持字母、数字、连字符和下划线，创建后不可直接改名。</p>
          <Button
            type="primary"
            block
            loading={creatingKnowledgeBase}
            disabled={!newKbFile || !newKbName.trim()}
            icon={<Database size={16} />}
            onClick={() => void createKnowledgeBase()}
          >
            <span className="step-number button-step-number">3</span>
            确认建立知识库
          </Button>
        </div>
        <div className="kb-modal-divider"><span>管理已有知识库</span></div>
        <div className="modal-section">
          <label>默认课程知识库</label>
          <Select
            value={defaultKnowledgeBase || undefined}
            options={defaultKbOptions}
            onChange={chooseDefaultKnowledgeBase}
            placeholder="选择默认课程知识库"
            aria-label="选择默认课程知识库"
            style={{ width: '100%' }}
          />
          <p className="modal-field-help">重新打开学生端或开始新会话时优先使用；该设置保存在当前浏览器中。</p>
        </div>
        <div className="modal-section">
          <label>当前目标知识库</label>
          <div className="kb-target-row">
            <Select
              value={knowledgeBase || undefined}
              options={kbOptions}
              onChange={setKnowledgeBase}
              placeholder="选择或新建目标知识库"
              aria-label="选择当前目标知识库"
              style={{ width: '100%' }}
            />
            <Tooltip title={deleteKbDisabledReason || `删除知识库 ${knowledgeBase}`}>
              <span>
                <Popconfirm
                  title={`确认删除知识库 ${knowledgeBase}？`}
                  description="索引、知识图谱及已上传资料都会被永久删除。"
                  okText="确认删除"
                  cancelText="取消"
                  okButtonProps={{ danger: true }}
                  disabled={Boolean(deleteKbDisabledReason)}
                  onConfirm={() => void removeKnowledgeBase()}
                >
                  <Button
                    danger
                    icon={<Trash2 size={15} />}
                    disabled={Boolean(deleteKbDisabledReason)}
                    aria-label={`删除知识库 ${knowledgeBase}`}
                  >
                    删除
                  </Button>
                </Popconfirm>
              </span>
            </Tooltip>
          </div>
          <p className="modal-field-help">上传与重建仅作用于这里选择的知识库，不会改变上面的默认设置。</p>
        </div>
        <div className="modal-section existing-kb-upload">
          <label>向当前知识库追加资料</label>
          <Upload.Dragger
            multiple={false}
            accept=".pdf,.md,.txt,.docx"
            customRequest={uploadRequest}
            showUploadList
            className="kb-dragger"
            disabled={!currentKbStatus || currentKbStatus.state === 'building' || currentKbStatus.state === 'cancelling'}
          >
            <p className="ant-upload-drag-icon"><UploadCloud size={28} /></p>
            <p className="ant-upload-text">拖入新资料，或点击选择文件</p>
            <p className="ant-upload-hint">文件将追加到 {knowledgeBase || '当前知识库'} 并触发重建</p>
          </Upload.Dragger>
        </div>
        {(currentKbStatus?.state === 'building' || currentKbStatus?.state === 'cancelling') && (
          <div className="modal-build-progress" aria-label={`${knowledgeBase} 构建进度`}>
            <div>
              <strong>{currentKbStatus.state === 'cancelling' ? '正在取消并清理缓存' : currentKbStatus.message}</strong>
              <span>{currentKbStatus.progress || 0}%</span>
            </div>
            <Progress
              percent={currentKbStatus.progress || 0}
              status={currentKbStatus.state === 'cancelling' ? 'exception' : 'active'}
              showInfo={false}
            />
            <Button
              danger
              block
              disabled={!currentKbStatus.cancellable}
              onClick={() => void cancelBuild(knowledgeBase)}
            >
              {currentKbStatus.state === 'cancelling' ? '正在清理未完成缓存…' : '取消本次构建'}
            </Button>
          </div>
        )}
        <Button
          block
          icon={<Database size={16} />}
          onClick={() => void rebuildCurrentKnowledgeBase()}
          disabled={!currentKbStatus || currentKbStatus.state === 'building' || currentKbStatus.state === 'cancelling'}
        >
          使用 qwen3-vl-flash 重新构建已有资料
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
