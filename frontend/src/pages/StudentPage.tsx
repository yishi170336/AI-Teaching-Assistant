import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
} from 'react'
import {
  forceCollide,
  forceLink,
  forceManyBody,
  forceSimulation,
  forceX,
  forceY,
  type SimulationLinkDatum,
  type SimulationNodeDatum,
} from 'd3-force'
import { Link } from 'react-router-dom'
import {
  App as AntApp,
  Button,
  Empty,
  Image as AntImage,
  Input,
  InputNumber,
  Modal,
  Pagination,
  Popconfirm,
  Progress,
  Segmented,
  Select,
  Slider,
  Spin,
  Tag,
  Tooltip,
  Upload,
  type UploadFile,
  type UploadProps,
} from 'antd'
import {
  AlertTriangle,
  ArrowUp,
  BookOpen,
  BookMarked,
  Bot,
  BrainCircuit,
  BookmarkPlus,
  CalendarCheck2,
  CalendarDays,
  Check,
  CheckCircle2,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Circle,
  CircleStop,
  Cloud,
  Clock3,
  Cpu,
  Database,
  Download,
  FileText,
  Eye,
  FileCheck2,
  GraduationCap,
  HelpCircle,
  Layers3,
  LibraryBig,
  Network,
  LoaderCircle,
  ListTodo,
  Menu,
  MessageSquareText,
  Plus,
  Pencil,
  Paperclip,
  Presentation,
  RefreshCw,
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
  ZoomIn,
  ZoomOut,
  RotateCcw,
  ScanLine,
} from 'lucide-react'
import MathMarkdown, { InlineMath } from '../components/MathMarkdown'
import HomeworkView from './HomeworkView'
import AnswerReportsView from './AnswerReportsView'
import {
  addMistakeAnnotation,
  addScheduleItem,
  AttachmentInfo,
  cancelQuestionBank,
  cancelKnowledgeExplanation,
  cancelKnowledgeBaseBuild,
  ChapterKnowledgeSummary,
  ConversationFocus,
  confirmMistakeCandidate,
  createQuestionReferenceMistakeCandidate,
  createQuestionBank,
  createKnowledgeExplanation,
  createMistakeCandidate,
  createMistakeCategory,
  deleteKnowledgeExplanation,
  deleteKnowledgeBase,
  deleteMistake,
  deleteMistakeAnnotation,
  deleteMistakeCategory,
  deleteScheduleItem,
  deleteQuestionBank,
  deleteQuestionBankQuestion,
  deleteSession,
  fetchKnowledgeGraph,
  fetchKnowledgeGraphEvidence,
  fetchKnowledgeBases,
  fetchKnowledgeExplanation,
  fetchKnowledgeExplanations,
  fetchMistakeNotebook,
  fetchSchedule,
  fetchModels,
  fetchQuestionBank,
  fetchQuestionBanks,
  getRecommendedQuestionAnswer,
  fetchSession,
  fetchSessions,
  HomeworkQuestion,
  KBStatus,
  KnowledgeExplanation,
  KnowledgeGraph,
  KnowledgeGraphEdge,
  KnowledgeGraphEvidence,
  KnowledgeGraphNode,
  generateLearningPlanPpt,
  ModelCatalog,
  ModelConfig,
  ModelProviderId,
  OCRModelConfig,
  MistakeItem,
  PhotoRecognition,
  PracticeExercise,
  PracticeGrading,
  QuestionBank,
  QuestionReference,
  QuestionRecommendation,
  MistakeAnalysis,
  MistakeCandidateDraft,
  MistakeCategory,
  MistakePhotoRetention,
  MistakeReason,
  MistakeSource,
  renameMistakeCategory,
  reprocessQuestionBank,
  ScheduleCategory,
  ScheduleItem,
  ScheduleItemDraft,
  SessionSummary,
  SourceInfo,
  setScheduleItemCompleted,
  updateMistake,
  updateMistakeAnnotation,
  uploadChatAttachment,
  uploadKnowledgeFile,
  rebuildKnowledgeBase,
} from '../lib/api'
import { groupQuestionBankQuestions } from '../lib/questionBankGrouping'
import { CHAT_MODEL, CHAT_MODEL_PROVIDER, ChatMessage, ChatMode, useChatStore } from '../store/chatStore'

const { TextArea } = Input
type WorkspaceView = 'chat' | 'graph' | 'question-bank' | 'homework' | 'mistakes' | 'reports' | 'schedule'

const mistakeSourceLabels: Record<MistakeSource, string> = {
  question_bank: '题库',
  ai_generated: 'AI 生成',
  user_uploaded: '用户上传',
}

const mistakeReasonLabels: Record<MistakeReason, string> = {
  wrong: '做错了',
  unknown: '不会做',
  concept_gap: '概念不清',
  calculation_error: '计算失误',
  bookmark: '想收藏',
}

const photoRetentionLabels: Record<MistakePhotoRetention, string> = {
  original_and_processed: '保存原图和清晰化副本',
  processed_only: '只保存清晰化副本',
  text_only: '只保存题目文字',
}

const knowledgeBaseDisplayName = (status: KBStatus | undefined, fallbackId = '') => (
  status?.display_name?.trim()
  || (fallbackId === 'default' ? '默认课程知识库' : fallbackId)
  || '当前知识库'
)

const createKnowledgeBaseInternalId = () => (
  `kb-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 9)}`
)

const providerLabels: Record<ModelProviderId, string> = {
  ollama: '本地',
  deepseek: 'DeepSeek',
  qwen: '通义千问',
  custom: '自定义 API',
}

const fallbackModelCatalog: ModelCatalog = {
  default: { provider: CHAT_MODEL_PROVIDER, model: CHAT_MODEL },
  ocr: {
    default_provider: 'local',
    model: 'PaddleOCR-VL-1.6',
    api_job_url: 'https://paddleocr.aistudio-app.com/api/v2/ocr/jobs',
    api_configured: false,
    providers: [
      {
        id: 'local',
        label: '本地 PaddleOCR-VL 1.6',
        description: '使用本机 GPU/CPU，教材页面不上传',
        configured: true,
      },
      {
        id: 'api',
        label: 'PaddleOCR-VL 1.6 API',
        description: '逐页调用 Paddle AI Studio 作业 API',
        configured: false,
      },
    ],
  },
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
      models: ['qwen3.7-flash', 'qwen3.7-plus', 'qwen3.7-max'],
      model_options: [
        { value: 'qwen3.7-flash', label: 'Qwen3.7-Flash' },
        { value: 'qwen3.7-plus', label: 'Qwen3.7-Plus' },
        { value: 'qwen3.7-max', label: 'Qwen3.7-Max' },
      ],
      text_model_options: [
        { value: 'qwen3.7-flash', label: 'Qwen3.7-Flash' },
        { value: 'qwen3.7-plus', label: 'Qwen3.7-Plus' },
        { value: 'qwen3.7-max', label: 'Qwen3.7-Max' },
      ],
      default_model: 'qwen3.7-flash',
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
          <button className={`nav-item ${activeView === 'question-bank' ? 'active' : ''}`} onClick={() => onView('question-bank')}>
            <LibraryBig size={17} />
            <span>题库</span>
          </button>
          <button className={`nav-item ${activeView === 'homework' ? 'active' : ''}`} onClick={() => onView('homework')}>
            <FileCheck2 size={17} />
            <span>我的作业</span>
          </button>
          <button className={`nav-item ${activeView === 'mistakes' ? 'active' : ''}`} onClick={() => onView('mistakes')}>
            <Layers3 size={17} />
            <span>错题本</span>
          </button>
          <button className={`nav-item ${activeView === 'reports' ? 'active' : ''}`} onClick={() => onView('reports')}>
            <FileText size={17} />
            <span>答案分析报告</span>
          </button>
          <button className={`nav-item ${activeView === 'schedule' ? 'active' : ''}`} onClick={() => onView('schedule')}>
            <CalendarDays size={17} />
            <span>学习日历</span>
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
                  title={normalizeQuestionNumberText(session.title)}
                >
                  <span className="recent-icon"><Clock3 size={14} /></span>
                  <span>
                    <strong>{normalizeQuestionNumberText(session.title)}</strong>
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

const scheduleCategoryLabels: Record<ScheduleCategory, string> = {
  exam: '考试',
  study: '学习',
  activity: '活动',
  other: '其他',
}

function localDateKey(date = new Date()) {
  const year = date.getFullYear()
  const month = String(date.getMonth() + 1).padStart(2, '0')
  const day = String(date.getDate()).padStart(2, '0')
  return `${year}-${month}-${day}`
}

function scheduleCategoryIcon(category: ScheduleCategory, size = 15) {
  if (category === 'exam') return <GraduationCap size={size} />
  if (category === 'activity') return <CalendarCheck2 size={size} />
  if (category === 'other') return <ListTodo size={size} />
  return <BookOpen size={size} />
}

function TodayAgenda({
  items,
  onOpen,
  onToggle,
}: {
  items: ScheduleItem[]
  onOpen: () => void
  onToggle: (item: ScheduleItem) => void
}) {
  const today = new Date()
  const pending = items.filter((item) => !item.completed).length
  return (
    <section className="today-agenda" aria-label="今日安排">
      <div className="today-date-block">
        <small>{today.toLocaleDateString('zh-CN', { month: 'short' })}</small>
        <strong>{today.getDate()}</strong>
        <span>{today.toLocaleDateString('zh-CN', { weekday: 'short' })}</span>
      </div>
      <div className="today-agenda-main">
        <div className="today-agenda-head">
          <div>
            <span className="today-agenda-icon"><CalendarCheck2 size={17} /></span>
            <strong>今日安排</strong>
            {items.length > 0 && <small>{pending ? `${pending} 项待完成` : '今日计划已完成'}</small>}
          </div>
          <button type="button" onClick={onOpen}>查看日历 <ChevronRight size={14} /></button>
        </div>
        {items.length ? (
          <div className="today-agenda-list">
            {items.slice(0, 3).map((item) => (
              <button
                type="button"
                className={`today-agenda-item category-${item.category} ${item.completed ? 'completed' : ''}`}
                key={item.id}
                onClick={() => onToggle(item)}
                aria-label={`${item.completed ? '标记为未完成' : '标记为已完成'}：${item.title}`}
              >
                {item.completed ? <CheckCircle2 size={16} /> : <Circle size={16} />}
                <span>
                  <strong>{item.title}</strong>
                  <small>{item.time || '全天'} · {scheduleCategoryLabels[item.category]}</small>
                </span>
              </button>
            ))}
            {items.length > 3 && <button className="today-agenda-more" type="button" onClick={onOpen}>还有 {items.length - 3} 项安排</button>}
          </div>
        ) : (
          <button type="button" className="today-agenda-empty" onClick={onOpen}>
            <span><Plus size={15} /></span>
            <span><strong>今天还没有安排</strong><small>为考试、复习或活动留个位置</small></span>
          </button>
        )}
      </div>
    </section>
  )
}

function Welcome({
  onAsk,
  todaySchedule,
  onOpenSchedule,
  onToggleSchedule,
}: {
  onAsk: (prompt: string, mode: ChatMode) => void
  todaySchedule: ScheduleItem[]
  onOpenSchedule: () => void
  onToggleSchedule: (item: ScheduleItem) => void
}) {
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
          <h1>多智能体电路助教平台</h1>
        </div>
      </div>
      <TodayAgenda items={todaySchedule} onOpen={onOpenSchedule} onToggle={onToggleSchedule} />
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
    </div>
  )
}

function retrievalPageLabel(source: SourceInfo) {
  if (!source.page_start) return '结构化知识'
  return source.page_start === source.page_end
    ? `第 ${source.page_start} 页`
    : `第 ${source.page_start}–${source.page_end} 页`
}

function StructuredRetrievalCard({
  source,
  label,
  cited,
}: {
  source: SourceInfo
  label: string
  cited: boolean
}) {
  return (
    <article className="retrieval-result-card">
      <header>
        <span><FileText size={12} />{label}</span>
        <div>{cited && <b>已引用</b>}<strong>{Math.round(source.score * 100)}%</strong></div>
      </header>
      <div className="retrieval-result-content">
        <MathMarkdown content={source.excerpt || '暂无内容。'} />
      </div>
      <footer><InlineMath content={source.section || source.chapter || '未标注章节'} /><span>{retrievalPageLabel(source)}</span></footer>
    </article>
  )
}

function RetrievalGraphNodeCard({
  node,
  graph,
  score,
}: {
  node: KnowledgeGraphNode
  graph: KnowledgeGraph
  score: number
}) {
  const [expanded, setExpanded] = useState(false)
  const graphNodeMap = useMemo(
    () => new Map(graph.nodes.map((item) => [item.id, item])),
    [graph.nodes],
  )
  const relationships = useMemo(() => graph.edges
    .filter((edge) => edge.type === 'concept_relation' && (edge.source === node.id || edge.target === node.id))
    .sort((left, right) => (
      (right.strength || 0) - (left.strength || 0)
      || (right.confidence || 0) - (left.confidence || 0)
    )), [graph.edges, node.id])
  const description = node.description || node.raw_description || '暂无描述。'
  return (
    <article className={`retrieval-entity-card ${expanded ? 'expanded' : ''}`}>
      <button type="button" aria-expanded={expanded} onClick={() => setExpanded((value) => !value)}>
        <span style={{ background: neo4jEntityColor(node.entity_type) }}><Network size={13} /></span>
        <div><strong><InlineMath content={graphNodeName(node)} /></strong><small>{node.entity_type || '知识实体'} · {relationships.length} 条关系</small></div>
        <b>{Math.round(score * 100)}%</b>
        <ChevronDown size={14} />
      </button>
      {expanded && <div className="retrieval-entity-detail"><h4>描述</h4><MathMarkdown content={description} /><h4>关联关系</h4>{relationships.length > 0 ? <div className="retrieval-relationship-list">{relationships.slice(0, 12).map((edge, index) => {
        const outgoing = edge.source === node.id
        const neighborId = outgoing ? edge.target : edge.source
        const neighbor = graphNodeMap.get(neighborId)
        return <article key={`${edge.source}-${edge.target}-${edge.relation}-${index}`}><div><span>{edge.relation || '关联'}</span><strong>{outgoing ? '→' : '←'} <InlineMath content={graphNodeName(neighbor) || neighborId} /></strong></div>{edge.description && <MathMarkdown content={edge.description} />}</article>
      })}</div> : <p>暂无关联关系。</p>}</div>}
    </article>
  )
}

function PaddleEvidenceCard({ evidence }: { evidence: KnowledgeGraphEvidence }) {
  const content = evidence.text || evidence.caption || evidence.description || '暂无可显示的块级内容。'
  const modality = ({
    text: '正文', formula: '公式', table: '表格', image: '图片', circuit: '课程图片',
  } as Record<string, string>)[evidence.modality] || evidence.modality || '证据'
  const page = evidence.page || evidence.page_start
  return (
    <article className="retrieval-result-card paddle-evidence-card">
      <header><span><ScanLine size={12} />Paddle {modality}</span><strong>{page ? `第 ${page} 页` : ''}</strong></header>
      <div className="retrieval-result-content"><MathMarkdown content={content} /></div>
      <footer><span>{evidence.source || '课程教材'}</span><small>{evidence.id}</small></footer>
    </article>
  )
}

function KnowledgePanel({ statuses, onCreate }: { statuses: KBStatus[]; onCreate: () => void }) {
  const activeSources = useChatStore((state) => state.activeSources)
  const activeCitedSources = useChatStore((state) => state.activeCitedSources)
  const activeMessageId = useChatStore((state) => state.activeMessageId)
  const knowledgeBase = useChatStore((state) => state.knowledgeBase)
  const messages = useChatStore((state) => state.messages)
  const activeAssistant = messages.find((item) => item.id === activeMessageId)
    || [...messages].reverse().find((item) => item.role === 'assistant')
  const activeKnowledgeBase = activeAssistant?.knowledgeBase
    || activeSources[0]?.knowledge_base
    || knowledgeBase
  const current = statuses.find((item) => item.id === activeKnowledgeBase)
  const noGroundedEvidence = activeAssistant?.evidenceMode === 'general_only'
  const [retrievalGraph, setRetrievalGraph] = useState<KnowledgeGraph>()
  const [retrievalGraphLoading, setRetrievalGraphLoading] = useState(false)
  const [paddleEvidence, setPaddleEvidence] = useState<KnowledgeGraphEvidence[]>([])
  const [paddleEvidenceLoading, setPaddleEvidenceLoading] = useState(false)
  const citedSourceIds = new Set(activeCitedSources.map((source) => source.id))
  const citedIndices = new Set(
    activeCitedSources
      .map((source) => source.citation_index)
      .filter((index): index is number => typeof index === 'number'),
  )
  const isSourceCited = (source: SourceInfo) => {
    const index = activeSources.findIndex((item) => item.id === source.id)
    return citedSourceIds.has(source.id) || citedIndices.has(index + 1)
  }
  const atomicStatements = activeSources.filter((source) => source.element_type === 'atomic_statement')
  const formulaKnowledge = activeSources.filter((source) => source.element_type === 'formula_knowledge')
  const relatedEntityIds = useMemo(() => {
    const prioritized = [
      ...activeSources.filter((source) => source.element_type === 'graph_entity'),
      ...activeSources.filter((source) => source.element_type !== 'graph_entity'),
    ]
    return [...new Set(prioritized.flatMap((source) => source.matched_entity_ids || []))].slice(0, 8)
  }, [activeSources])
  const relatedEntityKey = relatedEntityIds.join('\u0000')
  const evidenceIds = useMemo(() => {
    const prioritized = [...activeSources].sort((left, right) => {
      const priority = (source: SourceInfo) => (
        source.element_type === 'formula_knowledge' ? 2
          : source.element_type === 'atomic_statement' ? 1
            : 0
      )
      return priority(right) - priority(left)
    })
    return [...new Set(prioritized.flatMap((source) => (source.evidence_ids || []).slice(0, 2)))].slice(0, 10)
  }, [activeSources])
  const evidenceKey = evidenceIds.join('\u0000')

  useEffect(() => {
    let active = true
    if (!activeKnowledgeBase || !relatedEntityIds.length) {
      setRetrievalGraph(undefined)
      setRetrievalGraphLoading(false)
      return () => { active = false }
    }
    if (retrievalGraph?.knowledge_base === activeKnowledgeBase) return () => { active = false }
    setRetrievalGraphLoading(true)
    void fetchKnowledgeGraph(activeKnowledgeBase)
      .then((result) => { if (active) setRetrievalGraph(result) })
      .catch(() => { if (active) setRetrievalGraph(undefined) })
      .finally(() => { if (active) setRetrievalGraphLoading(false) })
    return () => { active = false }
  }, [activeKnowledgeBase, relatedEntityKey, retrievalGraph?.knowledge_base])

  useEffect(() => {
    let active = true
    setPaddleEvidence([])
    if (!activeKnowledgeBase || !evidenceIds.length) {
      setPaddleEvidenceLoading(false)
      return () => { active = false }
    }
    setPaddleEvidenceLoading(true)
    void fetchKnowledgeGraphEvidence(activeKnowledgeBase, evidenceIds)
      .then((items) => { if (active) setPaddleEvidence(items) })
      .catch(() => { if (active) setPaddleEvidence([]) })
      .finally(() => { if (active) setPaddleEvidenceLoading(false) })
    return () => { active = false }
  }, [activeKnowledgeBase, evidenceKey])

  const graphNodeMap = useMemo(
    () => new Map((retrievalGraph?.nodes || []).map((node) => [node.id, node])),
    [retrievalGraph],
  )
  const entityScores = useMemo(() => {
    const scores = new Map<string, number>()
    activeSources.forEach((source) => (source.matched_entity_ids || []).forEach((entityId) => {
      scores.set(entityId, Math.max(scores.get(entityId) || 0, source.score || 0))
    }))
    return scores
  }, [activeSources])
  const relatedEntities = relatedEntityIds
    .map((entityId) => graphNodeMap.get(entityId))
    .filter((node): node is KnowledgeGraphNode => node?.type === 'entity')
  const visibleResultCount = relatedEntities.length
    + atomicStatements.length
    + formulaKnowledge.length
    + paddleEvidence.length
  return (
    <aside className="knowledge-panel">
      <div className="panel-heading">
        <div>
          <span className="panel-kicker">SCHEMA 4 RETRIEVAL</span>
          <h2>检索结果</h2>
        </div>
        <Tooltip title="展示本轮命中的图谱节点、原子陈述、公式知识整理及其原始 Paddle 块级证据。">
          <HelpCircle size={17} />
        </Tooltip>
      </div>
      {activeSources.length > 0 && (
        <div className={`source-usage-summary ${activeCitedSources.length ? '' : 'uncited'}`}>
          <span>{knowledgeBaseDisplayName(current, activeKnowledgeBase)} · 召回 {activeSources.length} 条</span>
          <strong>{visibleResultCount} 项结果</strong>
        </div>
      )}
      <div className="source-list retrieval-result-list">
        {activeSources.length ? (
          <>
            {(relatedEntities.length > 0 || retrievalGraphLoading) && <section className="retrieval-result-group"><h3><Network size={13} />相关图谱节点 <span>{relatedEntities.length}</span></h3>{retrievalGraphLoading && !relatedEntities.length ? <div className="retrieval-loading"><LoaderCircle className="spin" size={14} /> 正在读取图谱节点…</div> : relatedEntities.map((node) => <RetrievalGraphNodeCard key={node.id} node={node} graph={retrievalGraph as KnowledgeGraph} score={entityScores.get(node.id) || 0} />)}</section>}
            {atomicStatements.length > 0 && <section className="retrieval-result-group"><h3><ListTodo size={13} />原子陈述 <span>{atomicStatements.length}</span></h3>{atomicStatements.map((source) => <StructuredRetrievalCard key={source.id} source={source} label="原子陈述" cited={isSourceCited(source)} />)}</section>}
            {formulaKnowledge.length > 0 && <section className="retrieval-result-group"><h3><Cpu size={13} />公式知识整理 <span>{formulaKnowledge.length}</span></h3>{formulaKnowledge.map((source) => <StructuredRetrievalCard key={source.id} source={source} label="公式知识" cited={isSourceCited(source)} />)}</section>}
            {(paddleEvidence.length > 0 || paddleEvidenceLoading) && <section className="retrieval-result-group"><h3><ScanLine size={13} />原始 Paddle 证据 <span>{paddleEvidence.length}</span></h3>{paddleEvidenceLoading && !paddleEvidence.length ? <div className="retrieval-loading"><LoaderCircle className="spin" size={14} /> 正在读取块级证据…</div> : paddleEvidence.map((item) => <PaddleEvidenceCard key={item.id} evidence={item} />)}</section>}
          </>
        ) : noGroundedEvidence ? (
          <div className="source-empty">
            <span><Search size={22} /></span>
            <strong>未找到结构化检索结果</strong>
            <p>本轮没有命中可验证的图谱节点、知识陈述、公式知识或 Paddle 证据。</p>
          </div>
        ) : (
          <div className="source-empty">
            <span><Search size={22} /></span>
            <strong>等待你的问题</strong>
            <p>提问后，这里会展示命中的图谱节点、原子陈述、公式知识和原始证据。</p>
          </div>
        )}
      </div>

      <div className="panel-bottom">
        <button className="manage-kb-button" onClick={onCreate}>
          <UploadCloud size={16} />
          <span>添加教材 / 新建知识库</span>
          <ChevronRight size={15} />
        </button>
      </div>
    </aside>
  )
}

type ComposerMode = ChatMode | 'image_answer'

const photoSuffixPattern = /\.(png|jpe?g|webp|bmp)$/i

function ChatComposer({
  onSend,
  onPdfUpload,
  externalBusy = false,
  onExternalStop,
  explanationImageModel = 'qwen-image-2.0',
}: {
  onSend: (value: string) => void
  onPdfUpload: (file: File) => void
  externalBusy?: boolean
  onExternalStop?: () => void
  explanationImageModel?: string
}) {
  const { message: toast } = AntApp.useApp()
  const [value, setValue] = useState('')
  const fileInputRef = useRef<HTMLInputElement>(null)
  const mode = useChatStore((state) => state.mode)
  const setMode = useChatStore((state) => state.setMode)
  const scene = useChatStore((state) => state.scene)
  const setScene = useChatStore((state) => state.setScene)
  const activePractice = useChatStore((state) => state.activePractice)
  const activeFocus = useChatStore((state) => state.activeFocus)
  const setActiveFocus = useChatStore((state) => state.setActiveFocus)
  const messages = useChatStore((state) => state.messages)
  const parentFocus = activeFocus?.parent_focus_id
    ? [...messages].reverse().find((item) => item.focus?.id === activeFocus.parent_focus_id)?.focus
    : undefined
  const streaming = useChatStore((state) => state.streaming)
  const stop = useChatStore((state) => state.stop)
  const pendingAttachments = useChatStore((state) => state.pendingAttachments)
  const addAttachments = useChatStore((state) => state.addAttachments)
  const removeAttachment = useChatStore((state) => state.removeAttachment)
  const hasReadyAttachment = pendingAttachments.some((item) => item.status === 'ready')
  const hasReadyImage = pendingAttachments.some((item) => item.status === 'ready' && item.kind === 'image')
  const hasUnfinishedAttachment = pendingAttachments.some((item) => item.status !== 'ready')
  const composerMode: ComposerMode = scene === 'image_answer' ? 'image_answer' : mode
  const explanationBusy = mode === 'explain' && externalBusy
  const busy = streaming || explanationBusy
  const canSubmit = mode === 'explain'
    ? Boolean(value.trim())
    : scene === 'image_answer'
    ? hasReadyImage && !hasUnfinishedAttachment
    : scene === 'quiz_grade'
      ? (Boolean(value.trim()) || hasReadyImage) && !hasUnfinishedAttachment
    : (Boolean(value.trim()) || hasReadyAttachment) && !hasUnfinishedAttachment

  const submit = () => {
    if (!canSubmit || busy) return
    onSend(value)
    setValue('')
  }
  const changeComposerMode = (nextMode: ComposerMode) => {
    if (nextMode === 'image_answer') {
      setMode('answer')
      setScene('image_answer')
    } else {
      setScene('chat')
      setMode(nextMode)
    }
  }

  const selectFiles = (files: File[]) => {
    if (!files.length) return
    if (scene === 'image_answer') {
      const pdfs = files.filter((file) => /\.pdf$/i.test(file.name))
      if (pdfs.length) {
        if (files.length > 1) toast.info('整份 PDF 每次处理一个；其余文件本次未加入')
        onPdfUpload(pdfs[0])
        return
      }
    }
    const accepted = scene === 'image_answer' || scene === 'quiz_grade'
      ? files.filter((file) => (!file.type || file.type.startsWith('image/')) && photoSuffixPattern.test(file.name))
      : files
    if (accepted.length !== files.length) toast.warning('图片提交仅支持 PNG、JPEG、WebP 和 BMP')
    if (accepted.length) void addAttachments(accepted)
  }

  return (
    <div className="composer-shell">
      <div className="composer-card">
        <div className="composer-topline">
          <Segmented<ComposerMode>
            className="composer-mode-desktop"
            size="small"
            value={composerMode}
            onChange={changeComposerMode}
            options={[
              { label: '智能路由', value: 'auto' },
              { label: 'AI 答疑', value: 'answer' },
              { label: '拍照答题', value: 'image_answer' },
              { label: '同类出题', value: 'quiz' },
              { label: '题库荐题', value: 'recommend' },
              { label: '知识讲解', value: 'explain' },
              { label: '学习规划', value: 'plan' },
            ]}
          />
          <div className="composer-mode-mobile">
            <Segmented<ComposerMode>
              size="small"
              value={(['answer', 'image_answer', 'recommend'] as ComposerMode[]).includes(composerMode) ? composerMode : 'answer'}
              onChange={changeComposerMode}
              options={[
                { label: '答疑', value: 'answer' },
                { label: '拍照', value: 'image_answer' },
                { label: '题库荐题', value: 'recommend' },
              ]}
            />
            <Select
              size="small"
              value={(['auto', 'quiz', 'explain', 'plan'] as ComposerMode[]).includes(composerMode) ? composerMode : undefined}
              placeholder="更多"
              onChange={(nextMode) => changeComposerMode(nextMode as ComposerMode)}
              options={[
                { label: '智能路由', value: 'auto' },
                { label: '同类出题', value: 'quiz' },
                { label: '知识讲解', value: 'explain' },
                { label: '学习规划', value: 'plan' },
              ]}
            />
          </div>
          <span className="composer-tip">{mode === 'explain' ? `${knowledgeImageModelLabel(explanationImageModel)} · 逐页生成` : 'Shift + Enter 换行'}</span>
        </div>
        {activeFocus && mode !== 'explain' && mode !== 'plan' && (
          <div className="conversation-focus-bar">
            <span className="conversation-focus-icon"><BrainCircuit size={15} /></span>
            <div className="conversation-focus-copy">
              <small>当前对话焦点 · 切换功能仍会保留</small>
              <strong><InlineMath content={activeFocus.label} /></strong>
              {activeFocus.summary ? <span><InlineMath content={activeFocus.summary} /></span> : null}
            </div>
            <div className="conversation-focus-actions">
              {parentFocus ? (
                <Button type="text" size="small" onClick={() => setActiveFocus(parentFocus)}>
                  返回上一题
                </Button>
              ) : null}
              <Button type="text" size="small" onClick={() => setActiveFocus(undefined)}>
                换个主题
              </Button>
            </div>
          </div>
        )}
        {pendingAttachments.length > 0 && mode !== 'explain' && (
          <div className="pending-attachments" aria-label="待发送附件">
            {pendingAttachments.map((item) => (
              <div key={item.localId} className={`pending-attachment ${item.status}`}>
                <span className="pending-file-icon">
                  {item.status === 'ready' && item.kind === 'image' && item.attachment
                    ? <img src={item.attachment.url} alt="" />
                    : item.status === 'uploading' ? <LoaderCircle size={15} /> : item.kind === 'image' ? <FileText size={15} /> : <Paperclip size={15} />}
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
        {scene === 'quiz_grade' && activePractice && (
          <div className="practice-submit-banner">
            <span className="practice-submit-icon"><FileCheck2 size={18} /></span>
            <div>
              <strong>正在作答：{activePractice.knowledge_point || '同类练习题'}</strong>
              <small>可输入答案，也可上传手写过程；AI 将按步骤、数值和单位逐项批改。</small>
            </div>
            <Button size="small" onClick={() => setScene('chat')}>取消作答</Button>
          </div>
        )}
        {(scene === 'image_answer' || scene === 'quiz_grade') && (
          <Upload.Dragger
            className={scene === 'quiz_grade' ? 'photo-upload-dragger practice-upload-dragger' : 'photo-upload-dragger'}
            multiple
            showUploadList={false}
            accept={scene === 'image_answer' ? '.png,.jpg,.jpeg,.webp,.bmp,.pdf' : '.png,.jpg,.jpeg,.webp,.bmp'}
            disabled={busy || pendingAttachments.length >= 5}
            beforeUpload={(file) => {
              selectFiles([file])
              return Upload.LIST_IGNORE
            }}
          >
            <div className="photo-upload-content">
              <UploadCloud size={22} />
              <strong>{scene === 'quiz_grade' ? '上传你的手写作答' : '上传题目图片或整份 PDF'}</strong>
              <span>{scene === 'quiz_grade' ? '按作答顺序上传，支持多页；也可以只在下方输入答案' : '图片最多 5 张；PDF 每次 1 份，后台拆题后逐题答疑'}</span>
            </div>
          </Upload.Dragger>
        )}
        {scene === 'image_answer' && !hasReadyImage && (
          <div className="photo-submit-hint">请先拍照、选择题目图片，或上传整份 PDF。</div>
        )}
        <div className="composer-input-row">
          {scene === 'chat' && mode !== 'explain' && <input
            ref={fileInputRef}
            className="sr-only-file"
            type="file"
            multiple
            accept=".png,.jpg,.jpeg,.webp,.bmp,.pdf,.docx,.txt,.md,.xlsx,.json"
            onChange={(event) => {
              selectFiles(Array.from(event.target.files || []))
              event.target.value = ''
            }}
          />}
          {scene === 'chat' && mode !== 'explain' && <Tooltip title="添加题目图片或附件">
            <Button
              className="attach-button"
              shape="circle"
              onClick={() => fileInputRef.current?.click()}
              disabled={busy || pendingAttachments.length >= 5}
              icon={<Paperclip size={17} />}
              aria-label="添加题目图片或附件"
            />
          </Tooltip>}
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
              if (mode === 'explain') return
              const files = Array.from(event.clipboardData.files || [])
              if (files.length) {
                event.preventDefault()
                selectFiles(files)
              }
            }}
            autoSize={{ minRows: 1, maxRows: 5 }}
            placeholder={scene === 'quiz_grade' ? '输入你的答案或补充说明，例如“图片中第 2 步我取向右为正”…' : scene === 'image_answer' ? '可选：补充题目要求、指定解法或你看清的文字…' : mode === 'recommend' ? activeFocus ? '说明想从题库找怎样的相似题；Agent 会以当前题为参照阅读候选题干…' : '描述想练的知识点、题型和难度，例如“基础二极管限幅计算题”…' : mode === 'quiz' ? activeFocus ? '说明希望保留或改变哪些条件，Agent 会基于当前题生成变式…' : '粘贴原题，上传原题图片，或描述想练习的知识点…' : mode === 'explain' ? '输入想讲清楚的问题，例如“为什么傅里叶变换能把时域信号分解成频率成分？”…' : mode === 'plan' ? '描述学习目标、薄弱点和可用时间…' : activeFocus ? '继续追问当前题，或说明要解释、批改、生成变式还是检索相似题…' : '输入电路问题，支持 LaTeX 公式…'}
            variant="borderless"
            aria-label="输入电路问题"
          />
          {busy ? (
            <Tooltip title={explanationBusy ? '取消讲解页生成' : '停止生成'}>
              <Button className="send-button stop" shape="circle" onClick={explanationBusy ? onExternalStop : stop} icon={<CircleStop size={18} />} />
            </Tooltip>
          ) : (
            <Tooltip title="发送">
              <Button
                type="primary"
                className="send-button"
                shape="circle"
                onClick={submit}
                disabled={!canSubmit}
                icon={<ArrowUp size={18} />}
              />
            </Tooltip>
          )}
        </div>
      </div>
      <p className="composer-footnote">{mode === 'explain' ? '讲解页由 AI 生成，请核对关键公式、数值与专业术语。' : 'AI 可能犯错，重要计算请结合教材与实验结果复核。'}</p>
    </div>
  )
}

function formatQuestionNumber(value?: string | number | null) {
  const number = String(value ?? '').trim()
  if (!number) return '题号未标注'
  if (/^第\s*.+\s*题$/.test(number) || /^(?:例|习题|题)/.test(number)) return number
  return `第 ${number} 题`
}

function normalizeQuestionNumberText(value: string) {
  return value.replace(/第\s*(例\s*[\d.]+)\s*题/g, '$1')
}

function normalizeQuizTitle(content: string) {
  return content.replace(
    /^(#{1,3}\s*同类型新题)(?:\s*[·•・—-]\s*[^\r\n]+)?\s*$/m,
    '$1',
  )
}

function stripLeadingListOrdinals(value: string) {
  let normalized = value.trim()
  const ordinal = /^(?:[（(]\s*\d+\s*[）)]|\d+\s*[.、．)])\s*/
  for (let index = 0; index < 3; index += 1) {
    const stripped = normalized.replace(ordinal, '').trim()
    if (stripped === normalized) break
    normalized = stripped
  }
  return normalized
}

function normalizeQuizContent(content: string) {
  return normalizeQuizTitle(content).replace(
    /^(\s*)(\d+)([.)、．])(\s+)(?:\(\s*\2\s*\)|（\s*\2\s*）|\2\s*[.)、．])\s*/gm,
    '$1$2$3$4',
  )
}

function PracticeCard({
  practice,
  grading,
  disabled,
  onStart,
  onGenerateSimilar,
}: {
  practice: PracticeExercise
  grading?: PracticeGrading
  disabled: boolean
  onStart: (practice: PracticeExercise) => void
  onGenerateSimilar: () => void
}) {
  const [showAnswer, setShowAnswer] = useState(false)
  const solutionSteps = (practice.solution_steps?.length
    ? practice.solution_steps
    : practice.solution ? [practice.solution] : [])
    .map(stripLeadingListOrdinals)
    .filter(Boolean)
  const answerItems = (practice.answer_items?.length
    ? practice.answer_items
    : practice.answer ? [practice.answer] : [])
    .map(stripLeadingListOrdinals)
    .filter(Boolean)
  const commonMistakes = (practice.common_mistakes || [])
    .map(stripLeadingListOrdinals)
    .filter(Boolean)

  return (
    <section className={`practice-card ${grading ? 'graded' : ''}`} onClick={(event) => event.stopPropagation()}>
      <div className="practice-card-head">
        <div>
          <span className="practice-card-eyebrow">{grading ? '本题练习结果' : '先练后看'}</span>
          <strong>{grading ? `${grading.score} / ${grading.max_score} 分` : '答案已为你隐藏'}</strong>
        </div>
        <div className="practice-card-tags">
          {practice.difficulty && <Tag bordered={false}>{practice.difficulty}</Tag>}
          {practice.knowledge_point && <Tag bordered={false}><InlineMath content={practice.knowledge_point} /></Tag>}
          {Boolean(practice.verification?.passed) && <Tag bordered={false} color="success">已校验</Tag>}
        </div>
      </div>
      <p className="practice-card-tip">
        {grading
          ? '可展开标准答案对照订正，或基于当前题目继续生成一道同构变式。'
          : '建议先独立完成并提交。支持直接输入，也支持上传最多 5 张手写过程图片。'}
      </p>
      {practice.circuit_diagram?.attachments.length ? (
        <div className="practice-circuit-reference">
          <div className="practice-circuit-reference-head">
            <span><BrainCircuit size={15} /></span>
            <div>
              <strong>
                {practice.circuit_diagram.source === 'question_bank'
                  ? '题库原题图 · 拓扑参考'
                  : '上传原题图 · 拓扑参考'}
              </strong>
              <small><InlineMath content={practice.circuit_diagram.notice} /></small>
            </div>
          </div>
          <div className={`practice-circuit-images count-${Math.min(3, practice.circuit_diagram.attachments.length)}`}>
            {practice.circuit_diagram.attachments.map((attachment) => (
              <a key={attachment.id} href={attachment.url} target="_blank" rel="noreferrer">
                <img src={attachment.url} alt={`${attachment.name}，原题电路拓扑参考图`} />
              </a>
            ))}
          </div>
          {practice.circuit_diagram.topology && (
            <p><strong>结构识别：</strong><InlineMath content={practice.circuit_diagram.topology} /></p>
          )}
        </div>
      ) : null}
      <div className="practice-card-actions">
        {!grading && (
          <Button type="primary" disabled={disabled} icon={<FileCheck2 size={15} />} onClick={() => onStart(practice)}>
            提交我的答案
          </Button>
        )}
        <Button icon={<BookOpen size={15} />} onClick={() => setShowAnswer((current) => !current)}>
          {showAnswer ? '收起答案' : '查看 AI 答案'}
        </Button>
        <Button disabled={disabled} icon={<RotateCcw size={15} />} onClick={onGenerateSimilar}>
          再出一道
        </Button>
      </div>
      {showAnswer && (
        <div className="practice-answer-panel">
          <div>
            <span>参考步骤</span>
            {solutionSteps.length
              ? solutionSteps.map((step, index) => <MathMarkdown key={`${index}-${step}`} content={`${index + 1}. ${step}`} />)
              : <p>本题暂无分步解析。</p>}
          </div>
          <div>
            <span>标准答案</span>
            {answerItems.length
              ? answerItems.map((answer, index) => <MathMarkdown key={`${index}-${answer}`} content={`${index + 1}. ${answer}`} />)
              : <p>本题暂无标准答案。</p>}
          </div>
          {commonMistakes.length > 0 && (
            <div>
              <span>易错提醒</span>
              <ul>{commonMistakes.map((item) => <li key={item}><InlineMath content={item} /></li>)}</ul>
            </div>
          )}
        </div>
      )}
    </section>
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

function mistakeDraftForAssistant(messages: ChatMessage[], index: number): MistakeCandidateDraft | null {
  const message = messages[index]
  if (message.role !== 'assistant' || !message.content) return null
  const agent = message.agent || ''
  // 仅 AI 出题、同类出题（出题 Agent）和拍照答题（答疑 Agent + 含识别结果）可加入错题本
  if (agent === '出题 Agent') {
    // allow
  } else if (agent === '答疑 Agent' && message.recognition) {
    // allow
  } else {
    return null
  }
  const previousUser = [...messages.slice(0, index)].reverse().find((item) => item.role === 'user')
  if (!previousUser?.content) return null
  let question = previousUser.content
  let answer = message.content
  let title = ''
  const boundQuestion = message.questionRef?.kind === 'question_bank'
    ? message.questionRef
    : undefined
  if (!message.practice && message.questionSummary?.prompt) {
    question = message.questionSummary.prompt
    title = `${message.questionSummary.bank_title} ${formatQuestionNumber(message.questionSummary.number)}`
  }
  if (message.practice) {
    question = message.practice.question
    answer = [
      '### 解题步骤',
      ...message.practice.solution_steps.map((item, itemIndex) => `${itemIndex + 1}. ${stripLeadingListOrdinals(item)}`),
      '',
      '### 标准答案',
      ...message.practice.answer_items.map((item, itemIndex) => `${itemIndex + 1}. ${stripLeadingListOrdinals(item)}`),
    ].join('\n')
  }
  else if (agent === '出题 Agent') {
    const questionMatch = message.content.match(
      /(?:^|\n)###\s*题目\s*\n+([\s\S]*?)(?=\n+---|\n+###\s*(?:解题步骤|标准答案|易错点)|$)/,
    )
    const answerStart = message.content.search(/(?:^|\n)###\s*(?:解题步骤|标准答案)/)
    if (questionMatch?.[1]?.trim()) question = questionMatch[1].trim()
    if (answerStart >= 0) answer = message.content.slice(answerStart).trim()
  }
  const attachments = mistakeAttachmentsForMessage(messages, index)
  const isGenerated = agent === '出题 Agent'
  const isPhoto = Boolean(message.recognition) || attachments.some((attachment) => attachment.kind === 'image')
  return {
    question,
    answer,
    agent,
    attachments,
    questionBankId: boundQuestion
      ? `QB:${boundQuestion.question_bank_id}:${boundQuestion.question_id}`
      : '',
    source: boundQuestion ? 'question_bank' : isGenerated ? 'ai_generated' : 'user_uploaded',
    sourceRef: boundQuestion ? {
      kind: 'question_bank',
      question_bank_id: boundQuestion.question_bank_id,
      question_id: boundQuestion.question_id,
    } : {
      kind: isGenerated ? 'ai_practice' : isPhoto ? 'photo' : 'chat',
      practice_id: isGenerated ? message.id : '',
      question_id: previousUser.id,
    },
    attempt: message.grading ? {
      student_answer: message.grading.extracted_answer,
      score: message.grading.score,
      max_score: message.grading.max_score,
      is_correct: message.grading.is_correct,
      grading_feedback: message.grading.summary,
    } : {},
    solution: {
      answer,
      explanation: answer,
      model: message.model,
      verification: message.practice?.verification || {},
    },
    recognition: message.recognition,
    suggestedReason: message.grading && message.grading.score < message.grading.max_score
      ? 'wrong'
      : isGenerated ? 'bookmark' : 'unknown',
    title: title || question.slice(0, 80),
  }
}

function RecognitionConfirmationCard({
  recognition,
  disabled,
  onConfirm,
}: {
  recognition: PhotoRecognition
  disabled: boolean
  onConfirm: (content: string) => void
}) {
  const setMode = useChatStore((state) => state.setMode)
  const setScene = useChatStore((state) => state.setScene)
  const [transcription, setTranscription] = useState(recognition.transcription)
  const [knowns, setKnowns] = useState(recognition.knowns.join('；'))
  const [unknowns, setUnknowns] = useState(recognition.unknowns.join('；'))
  const [uncertain, setUncertain] = useState(recognition.uncertain_regions.join('；'))

  const confirm = () => {
    if (!transcription.trim() || !unknowns.trim()) return
    onConfirm([
      '上述题目经我确认，以下修订内容优先于图片识别：',
      `题干：${transcription.trim()}`,
      `已知量：${knowns.trim() || '无额外已知量'}`,
      `待求量：${unknowns.trim()}`,
      recognition.constraints.length ? `特殊条件：${recognition.constraints.join('；')}` : '',
      uncertain.trim() ? `仍不确定项：${uncertain.trim()}` : '不确定项：无',
    ].filter(Boolean).join('\n'))
  }

  return (
    <section className="recognition-confirmation-card" onClick={(event) => event.stopPropagation()}>
      <div className="recognition-card-head">
        <span><FileCheck2 size={18} /></span>
        <div>
          <strong>请确认图片识别结果</strong>
          <small>识别置信度 {Math.round(recognition.confidence * 100)}%，你的修订将拥有最高优先级</small>
        </div>
      </div>
      <label>
        <span>题干</span>
        <TextArea value={transcription} onChange={(event) => setTranscription(event.target.value)} autoSize={{ minRows: 3, maxRows: 8 }} />
      </label>
      <div className="recognition-fields">
        <label><span>已知量</span><Input value={knowns} onChange={(event) => setKnowns(event.target.value)} /></label>
        <label><span>待求量</span><Input value={unknowns} onChange={(event) => setUnknowns(event.target.value)} /></label>
      </div>
      <label>
        <span>不确定项（确认无误可清空）</span>
        <Input value={uncertain} onChange={(event) => setUncertain(event.target.value)} />
      </label>
      <div className="recognition-hints">
        {recognition.knowledge_points.map((point) => <Tag key={point}><InlineMath content={point} /></Tag>)}
        {recognition.topology && <span>拓扑：<InlineMath content={recognition.topology} /></span>}
      </div>
      <div className="recognition-actions">
        <Button
          onClick={() => {
            setMode('answer')
            setScene('image_answer')
            document.querySelector('.photo-upload-dragger')?.scrollIntoView({ behavior: 'smooth', block: 'center' })
          }}
        >重新上传</Button>
        <Button type="primary" disabled={disabled || !transcription.trim() || !unknowns.trim()} onClick={confirm}>
          修正后继续
        </Button>
      </div>
    </section>
  )
}

function MistakeConfirmModal({
  drafts,
  categories,
  saving,
  onCancel,
  onConfirm,
}: {
  drafts: MistakeCandidateDraft[]
  categories: MistakeCategory[]
  saving: boolean
  onCancel: () => void
  onConfirm: (decision: {
    reason: MistakeReason
    categoryId: string
    title: string
    photoRetention: MistakePhotoRetention
  }) => void
}) {
  const first = drafts[0]
  const [reason, setReason] = useState<MistakeReason>('wrong')
  const [categoryId, setCategoryId] = useState('uncategorized')
  const [title, setTitle] = useState('')
  const [photoRetention, setPhotoRetention] = useState<MistakePhotoRetention>('original_and_processed')
  const hasPhoto = drafts.some((draft) => (
    draft.sourceRef.kind === 'photo'
    || draft.attachments.some((attachment) => attachment.kind === 'image')
    || Boolean(draft.attachmentUrls?.length)
  ))

  useEffect(() => {
    setReason(first?.suggestedReason || 'wrong')
    setCategoryId('uncategorized')
    setTitle(drafts.length === 1 ? first?.title || first?.question.slice(0, 80) || '' : '')
    setPhotoRetention('original_and_processed')
  }, [first, drafts.length])

  return (
    <Modal
      open={drafts.length > 0}
      title={drafts.length > 1 ? `确认加入 ${drafts.length} 道错题` : '是否加入错题本？'}
      onCancel={onCancel}
      mask={{ closable: !saving }}
      closable={!saving}
      footer={[
        <Button key="cancel" disabled={saving} onClick={onCancel}>暂不加入</Button>,
        <Button
          key="confirm"
          type="primary"
          loading={saving}
          onClick={() => onConfirm({ reason, categoryId, title: title.trim(), photoRetention })}
        >
          由我确认加入
        </Button>,
      ]}
      width={760}
      className="mistake-confirm-modal"
      destroyOnHidden
    >
      <div className="mistake-confirm-intro">
        <ShieldCheck size={18} />
        <div>
          <strong>系统只提供整理建议，是否入库由你决定</strong>
          <span>确认后才会计入薄弱知识分析；选择“暂不加入”不会创建错题记录。</span>
        </div>
      </div>
      <div className="mistake-candidate-list">
        {drafts.slice(0, 8).map((draft, index) => (
          <article key={`${draft.sourceRef.kind}-${draft.sourceRef.question_id || draft.sourceRef.practice_id || index}`}>
            <div>
              <Tag>{mistakeSourceLabels[draft.source]}</Tag>
              {draft.attempt?.score != null && draft.attempt.max_score != null && (
                <Tag color={draft.attempt.is_correct ? 'success' : 'warning'}>
                  {draft.attempt.score} / {draft.attempt.max_score} 分
                </Tag>
              )}
            </div>
            <strong><InlineMath content={draft.title || draft.question.slice(0, 100)} /></strong>
            <MathMarkdown content={draft.question.slice(0, 220)} />
            {(draft.attachments.some((attachment) => attachment.kind === 'image') || Boolean(draft.attachmentUrls?.length)) && (
              <div className="mistake-candidate-images">
                {draft.attachments.filter((attachment) => attachment.kind === 'image').slice(0, 3).map((attachment) => (
                  <img src={attachment.url} alt={attachment.name} key={attachment.id} />
                ))}
                {(draft.attachmentUrls || []).slice(0, Math.max(0, 3 - draft.attachments.filter((attachment) => attachment.kind === 'image').length)).map((url, imageIndex) => (
                  <img src={url} alt={`待归档题图 ${imageIndex + 1}`} key={url} />
                ))}
              </div>
            )}
          </article>
        ))}
      </div>
      <div className="mistake-confirm-fields">
        {drafts.length === 1 && (
          <label>
            <span>错题名称</span>
            <Input value={title} maxLength={120} onChange={(event) => setTitle(event.target.value)} />
          </label>
        )}
        <label>
          <span>加入原因</span>
          <Select
            value={reason}
            onChange={setReason}
            options={(Object.keys(mistakeReasonLabels) as MistakeReason[]).map((value) => ({
              value,
              label: mistakeReasonLabels[value],
            }))}
          />
        </label>
        <label>
          <span>错题分类</span>
          <Select
            value={categoryId}
            onChange={setCategoryId}
            options={categories.map((category) => ({ value: category.id, label: category.name }))}
          />
        </label>
        {hasPhoto && (
          <label>
            <span>图片保存方式</span>
            <Select
              value={photoRetention}
              onChange={setPhotoRetention}
              options={(Object.keys(photoRetentionLabels) as MistakePhotoRetention[]).map((value) => ({
                value,
                label: photoRetentionLabels[value],
              }))}
            />
            <small>原图永不被生成式修改；清晰化仅包含方向纠正、对比度增强和尺寸约束。</small>
          </label>
        )}
      </div>
    </Modal>
  )
}

function RecommendationCard({
  recommendation,
  disabled,
  onSimilar,
  onBookmark,
  onAnother,
  onStart,
  onHint,
}: {
  recommendation: QuestionRecommendation
  disabled: boolean
  onSimilar: (reference: QuestionReference) => void
  onBookmark: (reference: QuestionReference) => void
  onAnother: (reference: QuestionReference) => void
  onStart: (recommendation: QuestionRecommendation) => void
  onHint: (reference: QuestionReference, level: 'direction' | 'formula') => void
}) {
  const studentId = useChatStore((state) => state.studentId)
  const [answer, setAnswer] = useState<Awaited<ReturnType<typeof getRecommendedQuestionAnswer>>>()
  const [loadingAnswer, setLoadingAnswer] = useState(false)
  const [answerError, setAnswerError] = useState('')
  const question = recommendation.question
  const difficultyLabels = { basic: '基础', intermediate: '进阶', advanced: '挑战' }

  const revealAnswer = async () => {
    if (answer || loadingAnswer) return
    setLoadingAnswer(true)
    setAnswerError('')
    try {
      setAnswer(await getRecommendedQuestionAnswer(recommendation.question_ref, studentId))
    } catch (error) {
      setAnswerError(error instanceof Error ? error.message : '参考答案读取失败')
    } finally {
      setLoadingAnswer(false)
    }
  }
  const typeLabels: Record<string, string> = {
    calculation: '计算题', choice: '选择题', true_false: '判断题',
    design: '设计题', short_answer: '简答题', other: '综合题',
  }
  return (
    <section className="recommendation-card" onClick={(event) => event.stopPropagation()}>
      <div className="recommendation-question">
        <div className="recommendation-kicker">
          <Tag color="blue">原书题目</Tag>
          <span>{recommendation.source.book_title}</span>
          <span>{formatQuestionNumber(recommendation.source.number || question.sequence)}</span>
        </div>
        <MathMarkdown content={question.prompt} />
        {question.subquestions?.map((item) => (
          <div className="recommendation-subquestion" key={item.label}>
            <strong>({item.label})</strong><MathMarkdown content={item.text} />
          </div>
        ))}
        {question.options?.length ? (
          <div className="recommendation-options">
            {question.options.map((item) => (
              <div key={item.label}><strong>{item.label}.</strong> <MathMarkdown content={item.text} /></div>
            ))}
          </div>
        ) : null}
        {question.figures?.length ? (
          <div className="recommendation-figures">
            {question.figures.map((figure) => (
              <div className="recommendation-figure" key={figure.file}>
                <AntImage src={figure.url} alt={figure.caption || '题图'} />
              </div>
            ))}
          </div>
        ) : null}
        <div className="recommendation-tags">
          {recommendation.profile.knowledge_points.map((item) => <Tag key={item}><InlineMath content={item} /></Tag>)}
          <Tag>{difficultyLabels[recommendation.profile.difficulty]}</Tag>
          <Tag>{typeLabels[recommendation.profile.question_type] || recommendation.profile.question_type}</Tag>
        </div>
        <div className="recommendation-answer">
          {!answer ? (
            <>
              <Button loading={loadingAnswer} onClick={() => void revealAnswer()}>
                展开参考答案
              </Button>
              {answerError && (
                <div className="recommendation-answer-error">
                  <AlertTriangle size={14} /><span>{answerError}</span>
                  <Button size="small" onClick={() => void revealAnswer()}>重试</Button>
                </div>
              )}
            </>
          ) : (
            <details open>
              <summary>参考答案（建议完成后核对）</summary>
              <MathMarkdown content={answer.answer || '原书未提供文字答案'} />
              {answer.answer_subquestions?.map((item) => (
                <div className="recommendation-answer-part" key={item.label}>
                  <strong>({item.label})</strong>
                  <MathMarkdown content={item.text} />
                </div>
              ))}
              {answer.answer_figures?.length ? (
                <div className="recommendation-answer-figures">
                  {answer.answer_figures.map((figure) => (
                    <figure key={figure.file}>
                      <AntImage src={figure.url} alt={figure.caption || '答案图'} />
                      {figure.caption ? <figcaption>{figure.caption}</figcaption> : null}
                    </figure>
                  ))}
                </div>
              ) : null}
            </details>
          )}
        </div>
      </div>
      <aside className="recommendation-reason">
        <div className="recommendation-reason-heading">
          <div className={`recommendation-match ${recommendation.match_status}`}>
            {recommendation.match_status === 'exact' ? '完全匹配' : recommendation.match_status === 'relaxed' ? '已放宽部分条件' : recommendation.match_status === 'partial' ? '部分匹配' : '最接近候选'}
          </div>
          <span>{recommendation.selection_method === 'agent_rerank' ? 'Agent 阅读题干后选择' : '检索排序推荐'}</span>
          {recommendation.reference_available ? <span>参考答案可核验</span> : null}
        </div>
        {recommendation.agent_analysis?.intent_summary && (
          <div className="recommendation-intent">
            <small>Agent 理解的训练目标</small>
            <MathMarkdown content={recommendation.agent_analysis.intent_summary} />
          </div>
        )}
        <strong className="recommendation-reason-copy"><InlineMath content={recommendation.reason} /></strong>
        <ul className="recommendation-evidence">
          {recommendation.evidence.map((item) => <li key={item}><InlineMath content={item} /></li>)}
        </ul>
        <div className="recommendation-meta-grid">
          <div><small>章节</small><p>{recommendation.source.chapter || '待复核'}</p></div>
          <div><small>训练技能</small><p><InlineMath content={recommendation.profile.skills.join('、') || '综合训练'} /></p></div>
          {recommendation.agent_analysis?.reasoning_focus && (
            <div className="recommendation-reasoning-focus">
              <small>关键推理</small><MathMarkdown content={recommendation.agent_analysis.reasoning_focus} />
            </div>
          )}
        </div>
        <div className="recommendation-actions">
          <Button disabled={disabled} type="primary" onClick={() => onStart(recommendation)}>开始作答</Button>
          <Button disabled={disabled} onClick={() => onHint(recommendation.question_ref, 'direction')}>给点提示</Button>
          <Button disabled={disabled} onClick={() => onHint(recommendation.question_ref, 'formula')}>关键公式</Button>
          <Button disabled={disabled} icon={<RotateCcw size={14} />} onClick={() => onAnother(recommendation.question_ref)}>再来一道</Button>
          <Button disabled={disabled} icon={<WandSparkles size={14} />} onClick={() => onSimilar(recommendation.question_ref)}>同类出题</Button>
          <Button disabled={disabled} icon={<BookmarkPlus size={14} />} onClick={() => onBookmark(recommendation.question_ref)}>收藏错题</Button>
        </div>
      </aside>
    </section>
  )
}

function ReferenceMistakeConfirmModal({
  candidate,
  categories,
  saving,
  onCancel,
  onConfirm,
}: {
  candidate?: { id: string; summary: string; question: string; knowledge_points: string[] }
  categories: MistakeCategory[]
  saving: boolean
  onCancel: () => void
  onConfirm: (decision: { reason: MistakeReason; categoryId: string; title: string }) => void
}) {
  const [reason, setReason] = useState<MistakeReason>('bookmark')
  const [categoryId, setCategoryId] = useState('uncategorized')
  const [title, setTitle] = useState('')
  useEffect(() => {
    setReason('bookmark')
    setCategoryId('uncategorized')
    setTitle(candidate?.summary || '')
  }, [candidate?.id])
  return (
    <Modal
      open={Boolean(candidate)}
      title="确认加入错题本"
      onCancel={onCancel}
      closable={!saving}
      footer={[
        <Button key="cancel" disabled={saving} onClick={onCancel}>暂不加入</Button>,
        <Button key="ok" type="primary" loading={saving} onClick={() => onConfirm({ reason, categoryId, title })}>确认加入</Button>,
      ]}
    >
      {candidate?.question ? <MathMarkdown content={candidate.question.slice(0, 260)} /> : null}
      <div className="mistake-confirm-fields">
        <label><span>错题名称</span><Input value={title} onChange={(event) => setTitle(event.target.value)} /></label>
        <label>
          <span>加入原因</span>
          <Select value={reason} onChange={setReason} options={(Object.keys(mistakeReasonLabels) as MistakeReason[]).map((value) => ({ value, label: mistakeReasonLabels[value] }))} />
        </label>
        <label>
          <span>错题分类</span>
          <Select value={categoryId} onChange={setCategoryId} options={[
            { value: 'uncategorized', label: '未分类' },
            ...categories.map((item) => ({ value: item.id, label: item.name })),
          ]} />
        </label>
      </div>
      <div>{candidate?.knowledge_points.map((item) => <Tag key={item}>{item}</Tag>)}</div>
    </Modal>
  )
}

function Conversation({
  onAddMistake,
  onConfirmPhoto,
  onStartPractice,
  onGenerateSimilar,
  onRecommendationSimilar,
  onRecommendationBookmark,
  onRecommendationAnother,
  onRecommendationStart,
  onRecommendationHint,
}: {
  onAddMistake: (draft: MistakeCandidateDraft) => void
  onConfirmPhoto: (content: string) => void
  onStartPractice: (practice: PracticeExercise, focus?: ConversationFocus) => void
  onGenerateSimilar: (focus?: ConversationFocus) => void
  onRecommendationSimilar: (reference: QuestionReference) => void
  onRecommendationBookmark: (reference: QuestionReference) => void
  onRecommendationAnother: (reference: QuestionReference, continuationTaskId?: string, focus?: ConversationFocus) => void
  onRecommendationStart: (recommendation: QuestionRecommendation) => void
  onRecommendationHint: (reference: QuestionReference, level: 'direction' | 'formula') => void
}) {
  const { message: toast } = AntApp.useApp()
  const messages = useChatStore((state) => state.messages)
  const sessionId = useChatStore((state) => state.sessionId)
  const streaming = useChatStore((state) => state.streaming)
  const stage = useChatStore((state) => state.stage)
  const stageAgent = useChatStore((state) => state.stageAgent)
  const activeMessageId = useChatStore((state) => state.activeMessageId)
  const activateMessage = useChatStore((state) => state.activateMessage)
  const send = useChatStore((state) => state.send)
  const setMode = useChatStore((state) => state.setMode)
  const setScene = useChatStore((state) => state.setScene)
  const endRef = useRef<HTMLDivElement>(null)
  const [generatingPptId, setGeneratingPptId] = useState('')
  const [generatedPptIds, setGeneratedPptIds] = useState<Set<string>>(() => new Set())
  const [annotationSelection, setAnnotationSelection] = useState<{
    messageId: string
    text: string
    x: number
    y: number
  } | null>(null)
  const [annotationOpen, setAnnotationOpen] = useState(false)
  const [annotationQuestion, setAnnotationQuestion] = useState('这里没看懂，请结合原题解释。')

  const captureAnnotationSelection = () => {
    if (streaming) return
    const selection = window.getSelection()
    if (!selection || selection.isCollapsed || !selection.rangeCount) {
      setAnnotationSelection(null)
      return
    }
    const text = selection.toString().replace(/\s+/g, ' ').trim()
    if (text.length < 2) {
      setAnnotationSelection(null)
      return
    }
    if (text.length > 1600) {
      toast.info('一次最多批注 1600 个字符，请缩小选择范围')
      setAnnotationSelection(null)
      return
    }
    const elementForNode = (node: Node | null) => (
      node instanceof HTMLElement ? node : node?.parentElement || null
    )
    const startRoot = elementForNode(selection.anchorNode)?.closest<HTMLElement>('[data-annotation-message-id]')
    const endRoot = elementForNode(selection.focusNode)?.closest<HTMLElement>('[data-annotation-message-id]')
    if (!startRoot || startRoot !== endRoot) {
      setAnnotationSelection(null)
      return
    }
    const messageId = startRoot.dataset.annotationMessageId || ''
    const sourceMessage = messages.find((item) => item.id === messageId)
    if (!messageId || sourceMessage?.role !== 'assistant' || sourceMessage.failed) {
      setAnnotationSelection(null)
      return
    }
    const rect = selection.getRangeAt(0).getBoundingClientRect()
    setAnnotationSelection({
      messageId,
      text,
      x: Math.min(window.innerWidth - 132, Math.max(12, rect.right + 8)),
      y: Math.min(window.innerHeight - 48, Math.max(12, rect.bottom + 8)),
    })
  }

  const submitAnnotationFollowup = () => {
    if (!annotationSelection || !annotationQuestion.trim() || streaming) return
    const sourceMessage = messages.find((item) => item.id === annotationSelection.messageId)
    if (!sourceMessage) return
    activateMessage(sourceMessage.id, true)
    setMode('answer')
    setScene('chat')
    const quoted = annotationSelection.text
      .split('\n')
      .map((line) => `> ${line}`)
      .join('\n')
    const payload = [
      '【学习批注追问】',
      `标记来源：${sourceMessage.agent || 'AI 助教'}`,
      `来源消息：${sourceMessage.id}`,
      '',
      '标记内容：',
      quoted,
      '',
      `我的问题：${annotationQuestion.trim()}`,
    ].join('\n')
    void send(payload, {
      focusId: sourceMessage.focus?.id,
      questionRef: sourceMessage.questionRef || sourceMessage.focus?.question_ref,
    })
    window.getSelection()?.removeAllRanges()
    setAnnotationOpen(false)
    setAnnotationSelection(null)
    setAnnotationQuestion('这里没看懂，请结合原题解释。')
  }

  const downloadLearningPlanPpt = async (message: ChatMessage, index: number) => {
    setGeneratingPptId(message.id)
    try {
      const topic = [...messages.slice(0, index)]
        .reverse()
        .find((item) => item.role === 'user')?.content || ''
      const result = await generateLearningPlanPpt(sessionId, message.content, topic)
      const url = URL.createObjectURL(result.blob)
      const anchor = document.createElement('a')
      anchor.href = url
      anchor.download = result.filename
      document.body.appendChild(anchor)
      anchor.click()
      anchor.remove()
      window.setTimeout(() => URL.revokeObjectURL(url), 1000)
      setGeneratedPptIds((current) => new Set(current).add(message.id))
      toast.success(result.slideCount ? `PPT 已生成，共 ${result.slideCount} 页` : 'PPT 已生成并开始下载')
    } catch (error) {
      toast.error(error instanceof Error ? error.message : '学习规划 PPT 生成失败')
    } finally {
      setGeneratingPptId('')
    }
  }

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

  useEffect(() => {
    const container = endRef.current?.closest('.chat-scroll') as HTMLElement | null
    if (!container || annotationOpen) return
    const dismissSelectionAction = () => setAnnotationSelection(null)
    container.addEventListener('scroll', dismissSelectionAction, { passive: true })
    return () => container.removeEventListener('scroll', dismissSelectionAction)
  }, [annotationOpen])

  return (
    <div className="conversation">
      {messages.map((message, index) => (
        <div
          key={message.id}
          className={`message-row ${message.role} ${message.id === activeMessageId ? 'active-evidence-message' : ''}`}
          data-message-id={message.id}
          onClick={() => message.role === 'assistant' && activateMessage(message.id, true)}
        >
          {message.role === 'assistant' && (
            <span className="assistant-avatar"><LogoMark /></span>
          )}
          <div className={`message-bubble ${message.failed ? 'failed' : ''}`}>
            {message.role === 'assistant' && (
              <div className="message-agent">
                <span>{message.agent || (streaming && index === messages.length - 1 ? stageAgent || '多智能体助教' : '多智能体助教')}</span>
                {message.model && (
                  <Tag bordered={false} title={`${providerLabels[message.provider || CHAT_MODEL_PROVIDER]} · ${message.model}`}>
                    {providerLabels[message.provider || CHAT_MODEL_PROVIDER]} · {message.model}
                  </Tag>
                )}
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
            {message.questionSummary ? (
              <div className="message-question-source">
                <BookMarked size={16} />
                <div>
                  <small>来源题目 · {message.questionSummary.bank_title}</small>
                  <strong>{formatQuestionNumber(message.questionSummary.number)}</strong>
                  <MathMarkdown content={message.questionSummary.prompt} />
                  {message.questionSummary.figures?.length ? (
                    <div className="message-question-figure-group">
                      <small>题图</small>
                      <div className="message-question-figures">
                        {message.questionSummary.figures.map((figure) => (
                          <figure key={figure.file || figure.url}>
                            <AntImage src={figure.url} alt={figure.caption || '题目电路图'} />
                            {figure.caption ? <figcaption>{figure.caption}</figcaption> : null}
                          </figure>
                        ))}
                      </div>
                    </div>
                  ) : null}
                </div>
              </div>
            ) : null}
            {message.content ? (
              message.role === 'assistant'
                ? (
                  <div
                    className="assistant-answer-content"
                    data-annotation-message-id={message.id}
                    onMouseUp={() => window.setTimeout(captureAnnotationSelection, 0)}
                    onTouchEnd={() => window.setTimeout(captureAnnotationSelection, 0)}
                  >
                    <MathMarkdown content={normalizeQuizContent(message.content)} />
                  </div>
                )
                : <div className="user-message-content"><MathMarkdown content={message.content} /></div>
            ) : message.role === 'assistant'
              && streaming
              && index === messages.length - 1
              && !['cancelled', 'failed', 'error'].includes(message.status || '') ? (
              <div className="thinking-placeholder">
                <span className="thinking-dots"><i /><i /><i /></span>
                <span>{stage || '正在准备…'}</span>
              </div>
            ) : message.role === 'assistant' ? (
              <div className={`message-terminal-state ${message.status || 'empty'}`}>
                {message.status === 'cancelled'
                  ? '生成已取消'
                  : message.status === 'failed' || message.status === 'error'
                    ? '生成失败'
                    : '本次回答未生成内容'}
              </div>
            ) : null}
            {message.role === 'assistant'
              && message.needsConfirmation
              && message.recognition
              && !messages.slice(index + 1).some((later) => later.role === 'user' && later.content.startsWith('上述题目经我确认'))
              && (
              <RecognitionConfirmationCard
                recognition={message.recognition}
                disabled={streaming}
                onConfirm={onConfirmPhoto}
              />
            )}
            {message.role === 'assistant'
              && message.recommendation
              && !message.failed
              && !(streaming && index === messages.length - 1)
              && (
                <RecommendationCard
                  recommendation={message.recommendation}
                  disabled={streaming}
                  onSimilar={onRecommendationSimilar}
                  onBookmark={onRecommendationBookmark}
                  onAnother={(reference) => onRecommendationAnother(reference, message.resolvedContext?.turn_id, message.focus)}
                  onStart={onRecommendationStart}
                  onHint={onRecommendationHint}
                />
              )}
            {message.role === 'assistant'
              && message.practice
              && !message.failed
              && !(streaming && index === messages.length - 1)
              && (
                <PracticeCard
                  practice={message.practice}
                  grading={message.grading}
                  disabled={streaming}
                  onStart={(practice) => onStartPractice(practice, message.focus)}
                  onGenerateSimilar={() => onGenerateSimilar(message.focus)}
                />
              )}
            {message.content
              && message.role === 'assistant'
              && !message.failed
              && (message.agent === '出题 Agent' || (message.agent === '答疑 Agent' && message.recognition))
              && !(streaming && index === messages.length - 1) && (
              <div className="message-tools">
                {message.agent === '答疑 Agent' && (
                  <button type="button" disabled={streaming} onClick={() => onGenerateSimilar(message.focus)}>
                    <WandSparkles size={14} /> 生成同类题
                  </button>
                )}
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
            {message.content
              && message.role === 'assistant'
              && !message.failed
              && message.agent === '学习规划 Agent'
              && !(streaming && index === messages.length - 1) && (
              <section className={`learning-plan-ppt-card ${generatedPptIds.has(message.id) ? 'ready' : ''}`}>
                <span className="learning-plan-ppt-icon"><Presentation size={22} /></span>
                <div className="learning-plan-ppt-copy">
                  <span className="learning-plan-ppt-eyebrow">PLAN TO DECK</span>
                  <strong>{generatedPptIds.has(message.id) ? '学习规划 PPT 已就绪' : '生成学习规划 PPT'}</strong>
                  <p>将本次目标、阶段任务与验收标准整理成高质量可编辑演示文稿。</p>
                  <div className="learning-plan-ppt-tags">
                    <span>16:9 宽屏</span><span>清晰路线图</span><span>可编辑内容</span>
                  </div>
                </div>
                <button
                  type="button"
                  className="learning-plan-ppt-button"
                  disabled={Boolean(generatingPptId)}
                  onClick={(event) => {
                    event.stopPropagation()
                    void downloadLearningPlanPpt(message, index)
                  }}
                >
                  {generatingPptId === message.id
                    ? <><LoaderCircle className="spin" size={15} /> 正在生成</>
                    : <><Download size={15} /> {generatedPptIds.has(message.id) ? '再次下载' : '生成并下载'}</>}
                </button>
              </section>
            )}
          </div>
        </div>
      ))}
      {streaming && messages.at(-1)?.content && stage && (
        <div className="stage-pill"><span className="thinking-dots"><i /><i /><i /></span>{stageAgent} · {stage}</div>
      )}
      {annotationSelection && !annotationOpen && (
        <button
          type="button"
          className="learning-annotation-trigger"
          style={{ left: annotationSelection.x, top: annotationSelection.y }}
          onMouseDown={(event) => event.preventDefault()}
          onClick={(event) => {
            event.stopPropagation()
            setAnnotationOpen(true)
          }}
        >
          <MessageSquareText size={14} />
          向 AI 追问
        </button>
      )}
      <Modal
        className="learning-annotation-modal"
        title="针对标记内容追问"
        open={annotationOpen}
        okText="发送追问"
        cancelText="取消"
        centered
        destroyOnClose={false}
        okButtonProps={{ disabled: streaming || !annotationQuestion.trim() }}
        onOk={submitAnnotationFollowup}
        onCancel={() => {
          window.getSelection()?.removeAllRanges()
          setAnnotationOpen(false)
          setAnnotationSelection(null)
        }}
      >
        <p className="learning-annotation-hint">AI 会结合原题、题图和这段回答，只解释你标记的部分。</p>
        <blockquote className="learning-annotation-quote">
          {annotationSelection?.text}
        </blockquote>
        <div className="learning-annotation-quick-actions" aria-label="常用追问">
          {['解释这句话', '这一步怎么来的', '为什么这样判断', '只给我一个提示'].map((question) => (
            <button
              type="button"
              key={question}
              onClick={() => setAnnotationQuestion(question)}
            >
              {question}
            </button>
          ))}
        </div>
        <TextArea
          value={annotationQuestion}
          onChange={(event) => setAnnotationQuestion(event.target.value)}
          onPressEnter={(event) => {
            if (!event.shiftKey) {
              event.preventDefault()
              submitAnnotationFollowup()
            }
          }}
          placeholder="例如：这里为什么能使用虚短？"
          autoFocus
          autoSize={{ minRows: 3, maxRows: 6 }}
          maxLength={500}
          showCount
        />
      </Modal>
      <div ref={endRef} />
    </div>
  )
}

const GRAPH_VIEWBOX_WIDTH = 800
const GRAPH_VIEWBOX_HEIGHT = 760
const GRAPH_ZOOM_MIN = 0.6
const GRAPH_ZOOM_MAX = 3

const clampGraphZoom = (value: number) => (
  Math.min(GRAPH_ZOOM_MAX, Math.max(GRAPH_ZOOM_MIN, value))
)

const graphNodeName = (node?: KnowledgeGraphNode) => (
  node?.display_name?.trim() || node?.name?.trim() || ''
)

const NEO4J_VIEWBOX_WIDTH = 1200
const NEO4J_VIEWBOX_HEIGHT = 760
const NEO4J_ZOOM_MIN = 0.4
const NEO4J_ZOOM_MAX = 8

const clampNeo4jZoom = (value: number) => (
  Math.min(NEO4J_ZOOM_MAX, Math.max(NEO4J_ZOOM_MIN, value))
)

const neo4jEntityColors: Record<string, string> = {
  '课程概念': '#68BDF6',
  '物理定律与原理': '#FFD86E',
  '电路与系统': '#FF756E',
  '器件与元件': '#DE9BF9',
  '参数与物理量': '#6DCE9E',
  '物理过程与效应': '#FB95AF',
  '方法与模型': '#A5ABB6',
  '材料与结构': '#F79767',
}

const neo4jEntityColor = (entityType?: string) => (
  neo4jEntityColors[String(entityType || '')] || '#A5ABB6'
)

const neo4jNodeColor = (node: KnowledgeGraphNode) => {
  if (node.type !== 'section') return neo4jEntityColor(node.entity_type)
  if ((node.level || 0) === 0) return '#4C8EDA'
  if ((node.level || 0) === 1) return '#68BDF6'
  return '#A5ABB6'
}

const neo4jEdgeTypeLabel = (type?: string) => ({
  concept_relation: '概念关系',
  entity_section: '实体归属',
  parent_child: '章节层级',
}[String(type || '').toLocaleLowerCase()] || '关系')

const neo4jNodeRadius = (node: KnowledgeGraphNode, total: number) => {
  const densityScale = total < 120 ? 2 : total < 480 ? 1.45 : 1
  if (node.type === 'section') {
    if ((node.level || 0) === 0) return 8 * densityScale
    if ((node.level || 0) === 1) return 6.5 * densityScale
    return 3.5 * densityScale
  }
  return (node.is_core ? 3.8 : 2.6) * densityScale
}

const stableGraphHash = (value: string) => {
  let hash = 2166136261
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index)
    hash = Math.imul(hash, 16777619)
  }
  return hash >>> 0
}

type Neo4jForceNode = SimulationNodeDatum & {
  id: string
  graphNode: KnowledgeGraphNode
  group: string
  x: number
  y: number
}

type Neo4jForceLink = SimulationLinkDatum<Neo4jForceNode> & {
  source: string | Neo4jForceNode
  target: string | Neo4jForceNode
  graphEdge: KnowledgeGraphEdge
}

const buildNeo4jForceLayout = (
  nodes: KnowledgeGraphNode[],
  edges: KnowledgeGraphEdge[],
  completeGraph: boolean,
  sectionMap: Map<string, KnowledgeGraphNode>,
) => {
  const degrees = new Map<string, number>()
  const adjacency = new Map(nodes.map((node) => [node.id, new Set<string>()]))
  edges.forEach((edge) => {
    if (!adjacency.has(edge.source) || !adjacency.has(edge.target)) return
    adjacency.get(edge.source)?.add(edge.target)
    adjacency.get(edge.target)?.add(edge.source)
    degrees.set(edge.source, (degrees.get(edge.source) || 0) + 1)
    degrees.set(edge.target, (degrees.get(edge.target) || 0) + 1)
  })

  const componentByNode = new Map<string, string>()
  const components: string[][] = []
  nodes.forEach((node) => {
    if (componentByNode.has(node.id)) return
    const component: string[] = []
    const pending = [node.id]
    componentByNode.set(node.id, '')
    while (pending.length) {
      const current = pending.pop()
      if (!current) continue
      component.push(current)
      adjacency.get(current)?.forEach((neighbor) => {
        if (componentByNode.has(neighbor)) return
        componentByNode.set(neighbor, '')
        pending.push(neighbor)
      })
    }
    components.push(component)
  })
  components.sort((left, right) => right.length - left.length || left[0].localeCompare(right[0]))
  components.forEach((component, index) => {
    const id = `component:${index}`
    component.forEach((nodeId) => componentByNode.set(nodeId, id))
  })

  const topSectionId = (node: KnowledgeGraphNode) => {
    let section = node.type === 'section'
      ? node
      : sectionMap.get(node.section_id || node.section_ids?.[0] || '')
    const visited = new Set<string>()
    while (section?.parent && !visited.has(section.id)) {
      if ((section.level || 0) <= 1) break
      visited.add(section.id)
      section = sectionMap.get(section.parent)
    }
    return section?.id || 'section:unassigned'
  }

  const groupByNode = new Map<string, string>()
  nodes.forEach((node) => {
    groupByNode.set(node.id, completeGraph ? topSectionId(node) : (componentByNode.get(node.id) || node.id))
  })
  const groupSizes = new Map<string, number>()
  groupByNode.forEach((group) => groupSizes.set(group, (groupSizes.get(group) || 0) + 1))
  const groups = [...groupSizes.keys()].sort((left, right) => (
    (groupSizes.get(right) || 0) - (groupSizes.get(left) || 0) || left.localeCompare(right)
  ))
  const groupCenters = new Map<string, { x: number; y: number }>()
  groups.forEach((group, index) => {
    if (index === 0) {
      groupCenters.set(group, { x: 0, y: 0 })
      return
    }
    const angle = index * 2.399963229728653
    const rawRadius = (completeGraph ? 115 : 82) + Math.sqrt(index) * (completeGraph ? 74 : 42)
    const radiusLimit = completeGraph ? 390 : 440
    const radius = radiusLimit * Math.tanh(rawRadius / radiusLimit)
    groupCenters.set(group, {
      x: Math.cos(angle) * radius * 1.45,
      y: Math.sin(angle) * radius,
    })
  })

  const membersSeen = new Map<string, number>()
  const forceNodes: Neo4jForceNode[] = nodes.map((graphNode) => {
    const group = groupByNode.get(graphNode.id) || graphNode.id
    const localIndex = membersSeen.get(group) || 0
    membersSeen.set(group, localIndex + 1)
    const center = groupCenters.get(group) || { x: 0, y: 0 }
    const hash = stableGraphHash(graphNode.id)
    const angle = (hash % 6283) / 1000 + localIndex * 2.399963229728653
    const radius = 4 + Math.sqrt(localIndex) * (completeGraph ? 5.1 : 4.5)
    return {
      id: graphNode.id,
      graphNode,
      group,
      x: center.x + Math.cos(angle) * radius,
      y: center.y + Math.sin(angle) * radius,
    }
  })
  const forceLinks: Neo4jForceLink[] = edges.map((graphEdge) => ({
    source: graphEdge.source,
    target: graphEdge.target,
    graphEdge,
  }))
  const simulation = forceSimulation<Neo4jForceNode>(forceNodes)
    .alpha(1)
    .alphaDecay(0.026)
    .velocityDecay(0.36)
    .force('link', forceLink<Neo4jForceNode, Neo4jForceLink>(forceLinks)
      .id((node) => node.id)
      .distance((link) => link.graphEdge.type === 'concept_relation' ? 12 : 8)
      .strength((link) => link.graphEdge.type === 'parent_child' ? 0.9 : 0.55))
    .force('charge', forceManyBody<Neo4jForceNode>()
      .strength(completeGraph ? -8 : -10)
      .distanceMax(completeGraph ? 86 : 72))
    .force('collision', forceCollide<Neo4jForceNode>()
      .radius((node) => neo4jNodeRadius(node.graphNode, nodes.length) + 1.5)
      .strength(0.9))
    .force('group-x', forceX<Neo4jForceNode>((node) => groupCenters.get(node.group)?.x || 0)
      .strength(completeGraph ? 0.025 : 0.045))
    .force('group-y', forceY<Neo4jForceNode>((node) => groupCenters.get(node.group)?.y || 0)
      .strength(completeGraph ? 0.025 : 0.045))
    .stop()
  for (let index = 0; index < 160; index += 1) simulation.tick()
  forceNodes.forEach((node) => {
    const center = groupCenters.get(node.group) || { x: 0, y: 0 }
    const size = groupSizes.get(node.group) || 1
    const localScale = 1 + Math.min(
      completeGraph ? 0.75 : 1.6,
      Math.sqrt(size) / (completeGraph ? 34 : 18),
    )
    node.x = center.x + (node.x - center.x) * localScale
    node.y = center.y + (node.y - center.y) * localScale
  })

  const xs = forceNodes.map((node) => node.x)
  const ys = forceNodes.map((node) => node.y)
  const minX = Math.min(...xs, 0)
  const maxX = Math.max(...xs, 1)
  const minY = Math.min(...ys, 0)
  const maxY = Math.max(...ys, 1)
  const scale = Math.min(
    (NEO4J_VIEWBOX_WIDTH - 66) / Math.max(1, maxX - minX),
    (NEO4J_VIEWBOX_HEIGHT - 66) / Math.max(1, maxY - minY),
  )
  const offsetX = (NEO4J_VIEWBOX_WIDTH - (maxX - minX) * scale) / 2 - minX * scale
  const offsetY = (NEO4J_VIEWBOX_HEIGHT - (maxY - minY) * scale) / 2 - minY * scale
  const positions = new Map(forceNodes.map((node) => [node.id, {
    x: node.x * scale + offsetX,
    y: node.y * scale + offsetY,
  }]))
  return {
    positions,
    degrees,
    componentCount: components.length,
    groupCount: groups.length,
  }
}

function HierarchicalKnowledgeGraphView({
  graph,
  displayName,
}: {
  graph: KnowledgeGraph
  displayName: string
}) {
  const [view, setView] = useState<'sections' | 'entities'>('entities')
  const [coreOnly, setCoreOnly] = useState(false)
  const [selectedId, setSelectedId] = useState('')
  const [selectedEdgeKey, setSelectedEdgeKey] = useState('')
  const [entityQuery, setEntityQuery] = useState('')
  const [entityType, setEntityType] = useState('all')
  const [graphZoom, setGraphZoom] = useState(1)
  const [graphCenter, setGraphCenter] = useState({ x: NEO4J_VIEWBOX_WIDTH / 2, y: NEO4J_VIEWBOX_HEIGHT / 2 })
  const [graphDragging, setGraphDragging] = useState(false)
  const graphSvgRef = useRef<SVGSVGElement | null>(null)
  const graphDragRef = useRef<{
    pointerId: number
    clientX: number
    clientY: number
    centerX: number
    centerY: number
    moved: boolean
  } | null>(null)
  const suppressGraphClickRef = useRef(false)
  const [expandedChapterId, setExpandedChapterId] = useState('')
  const [evidence, setEvidence] = useState<KnowledgeGraphEvidence[]>([])
  const [evidenceLoading, setEvidenceLoading] = useState(false)
  const sections = useMemo(
    () => graph.nodes.filter((node) => node.type === 'section'),
    [graph],
  )
  const allEntities = useMemo(
    () => graph.nodes.filter((node) => node.type === 'entity'),
    [graph],
  )
  const sectionMap = useMemo(
    () => new Map(sections.map((node) => [node.id, node])),
    [sections],
  )
  const entityMap = useMemo(
    () => new Map(allEntities.map((node) => [node.id, node])),
    [allEntities],
  )
  const graphNodeMap = useMemo(
    () => new Map(graph.nodes.map((node) => [node.id, node])),
    [graph.nodes],
  )
  const conceptEdges = useMemo(
    () => graph.edges.filter((edge) => edge.type === 'concept_relation'),
    [graph],
  )
  const entityTypeCounts = useMemo(() => {
    const counts = new Map<string, number>()
    allEntities.forEach((node) => {
      const label = node.entity_type || '其他实体'
      counts.set(label, (counts.get(label) || 0) + 1)
    })
    return [...counts.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0], 'zh-CN'))
  }, [allEntities])
  const graphNodes = useMemo(() => {
    if (view === 'sections') return graph.nodes
    return allEntities.filter((node) => {
      if (coreOnly && !node.is_core) return false
      if (entityType !== 'all' && node.entity_type !== entityType) return false
      return true
    })
  }, [allEntities, coreOnly, entityType, graph.nodes, view])
  const graphNodeIds = useMemo(
    () => new Set(graphNodes.map((node) => node.id)),
    [graphNodes],
  )
  const graphEdges = useMemo(
    () => (view === 'sections' ? graph.edges : conceptEdges)
      .filter((edge) => graphNodeIds.has(edge.source) && graphNodeIds.has(edge.target)),
    [conceptEdges, graph.edges, graphNodeIds, view],
  )
  const layout = useMemo(
    () => buildNeo4jForceLayout(graphNodes, graphEdges, view === 'sections', sectionMap),
    [graphEdges, graphNodes, sectionMap, view],
  )
  const normalizedEntityQuery = entityQuery.trim().toLocaleLowerCase('zh-CN')
  const matchedNodeIds = useMemo(() => {
    if (!normalizedEntityQuery) return new Set<string>()
    return new Set(graphNodes.filter((node) => (
      node.type === 'entity'
      && [graphNodeName(node), node.name, node.entity_type, ...(node.aliases || [])]
        .some((value) => String(value || '').toLocaleLowerCase('zh-CN').includes(normalizedEntityQuery))
    )).map((node) => node.id))
  }, [graphNodes, normalizedEntityQuery])
  const labelNodeIds = useMemo(() => {
    const result = new Set<string>(matchedNodeIds)
    if (selectedId) result.add(selectedId)
    if (view === 'sections' && graphZoom >= 1.35) {
      graphNodes.filter((node) => node.type === 'section' && (node.level || 0) <= 1)
        .forEach((node) => result.add(node.id))
    }
    const labelLimit = graphZoom >= 2.2 ? 150 : graphZoom >= 1.65 ? 72 : graphZoom >= 1.25 ? 28 : 0
    if (labelLimit) {
      ;[...graphNodes]
        .sort((left, right) => (
          (layout.degrees.get(right.id) || 0) - (layout.degrees.get(left.id) || 0)
          || Number(Boolean(right.is_core)) - Number(Boolean(left.is_core))
          || graphNodeName(left).localeCompare(graphNodeName(right), 'zh-CN')
        ))
        .slice(0, labelLimit)
        .forEach((node) => result.add(node.id))
    }
    return result
  }, [graphNodes, graphZoom, layout.degrees, matchedNodeIds, selectedId, view])
  const selected = graph.nodes.find((node) => node.id === selectedId)
  const edgeKey = (edge: KnowledgeGraphEdge) => `${edge.source}\u0000${edge.relation || edge.type}\u0000${edge.target}`
  const selectedEdge = graphEdges.find((edge) => edgeKey(edge) === selectedEdgeKey)
  const graphViewport = {
    width: NEO4J_VIEWBOX_WIDTH / graphZoom,
    height: NEO4J_VIEWBOX_HEIGHT / graphZoom,
    x: graphCenter.x - NEO4J_VIEWBOX_WIDTH / graphZoom / 2,
    y: graphCenter.y - NEO4J_VIEWBOX_HEIGHT / graphZoom / 2,
  }
  const graphViewBox = `${graphViewport.x} ${graphViewport.y} ${graphViewport.width} ${graphViewport.height}`
  const changeNeo4jZoom = (requestedZoom: number) => setGraphZoom(clampNeo4jZoom(requestedZoom))
  const resetNeo4jViewport = () => {
    setGraphZoom(1)
    setGraphCenter({ x: NEO4J_VIEWBOX_WIDTH / 2, y: NEO4J_VIEWBOX_HEIGHT / 2 })
  }
  const handleNeo4jPointerDown = (event: ReactPointerEvent<SVGSVGElement>) => {
    if (event.button !== 0) return
    const target = event.target
    if (target instanceof Element && target.closest('.neo4j-nodes > g, .neo4j-edges > g')) return
    graphDragRef.current = {
      pointerId: event.pointerId,
      clientX: event.clientX,
      clientY: event.clientY,
      centerX: graphCenter.x,
      centerY: graphCenter.y,
      moved: false,
    }
    event.currentTarget.setPointerCapture(event.pointerId)
    setGraphDragging(true)
  }
  const handleNeo4jPointerMove = (event: ReactPointerEvent<SVGSVGElement>) => {
    const drag = graphDragRef.current
    if (!drag || drag.pointerId !== event.pointerId) return
    event.preventDefault()
    const rect = event.currentTarget.getBoundingClientRect()
    if (!rect.width || !rect.height) return
    const deltaX = event.clientX - drag.clientX
    const deltaY = event.clientY - drag.clientY
    if (Math.abs(deltaX) + Math.abs(deltaY) > 4) drag.moved = true
    setGraphCenter({
      x: drag.centerX - deltaX * graphViewport.width / rect.width,
      y: drag.centerY - deltaY * graphViewport.height / rect.height,
    })
  }
  const finishNeo4jPointer = (event: ReactPointerEvent<SVGSVGElement>) => {
    const drag = graphDragRef.current
    if (!drag || drag.pointerId !== event.pointerId) return
    if (drag.moved) {
      suppressGraphClickRef.current = true
      window.setTimeout(() => { suppressGraphClickRef.current = false }, 0)
    }
    graphDragRef.current = null
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId)
    }
    setGraphDragging(false)
  }

  useEffect(() => {
    const roots = sections.filter((node) => !node.parent)
    const initialEntity = [...allEntities].sort((a, b) => (
      Number(Boolean(b.is_core)) - Number(Boolean(a.is_core))
      || (b.core_score || 0) - (a.core_score || 0)
    ))[0]
    setSelectedId(initialEntity?.id || roots[0]?.id || sections[0]?.id || '')
    setSelectedEdgeKey('')
    resetNeo4jViewport()
  }, [graph.knowledge_base])

  const evidenceIds = selectedEdge?.evidence_ids
    || (selected?.type === 'section' ? selected.evidence_ids : [])
    || []
  useEffect(() => {
    let active = true
    setEvidence([])
    if (!evidenceIds.length) {
      setEvidenceLoading(false)
      return () => { active = false }
    }
    setEvidenceLoading(true)
    fetchKnowledgeGraphEvidence(graph.knowledge_base, evidenceIds)
      .then((items) => { if (active) setEvidence(items) })
      .catch(() => { if (active) setEvidence([]) })
      .finally(() => { if (active) setEvidenceLoading(false) })
    return () => { active = false }
  }, [graph.knowledge_base, selectedId, selectedEdgeKey])

  const selectedSectionEntities = selected?.type === 'section'
    ? (selected.entities || []).map((id) => entityMap.get(id)).filter(Boolean) as KnowledgeGraphNode[]
    : []
  const selectedNeighbors = selected?.type === 'entity'
    ? conceptEdges.filter((edge) => edge.source === selected.id || edge.target === selected.id).length
    : 0
  const selectedEntityRelationships = selected?.type === 'entity'
    ? conceptEdges
      .filter((edge) => edge.source === selected.id || edge.target === selected.id)
      .map((edge) => {
        const outgoing = edge.source === selected.id
        const neighborId = outgoing ? edge.target : edge.source
        return {
          edge,
          outgoing,
          neighbor: graphNodeMap.get(neighborId),
          neighborId,
        }
      })
      .sort((left, right) => (
        (right.edge.strength || 0) - (left.edge.strength || 0)
        || (right.edge.confidence || 0) - (left.edge.confidence || 0)
        || String(left.edge.relation || '').localeCompare(String(right.edge.relation || ''), 'zh-CN')
      ))
    : []
  const selectedEntityDescription = selected?.type === 'entity'
    ? selected.description || selected.raw_description || '暂无描述。'
    : ''
  const chapterOverviews = useMemo(() => {
    const chapterMetadata = new Map((graph.chapters || []).map((item) => [item.id, item as ChapterKnowledgeSummary & {
      summary?: string
      knowledge_points?: string[]
      title?: string
    }]))
    const collectSectionIds = (root: KnowledgeGraphNode) => {
      const collected = new Set<string>()
      const visitSection = (section: KnowledgeGraphNode) => {
        if (collected.has(section.id)) return
        collected.add(section.id)
        ;(section.children || []).forEach((id) => {
          const child = sectionMap.get(id)
          if (child) visitSection(child)
        })
      }
      visitSection(root)
      return collected
    }
    return sections
      .filter((section) => section.level === 1 && (/^\d+$/.test(section.number || '') || String(section.number || '').startsWith('appendix-')))
      .map((section) => {
        const sectionIds = collectSectionIds(section)
        const entityIds = new Set<string>()
        const coreIds = new Set<string>()
        sectionIds.forEach((id) => {
          const item = sectionMap.get(id)
          ;(item?.entities || []).forEach((entityId) => entityIds.add(entityId))
          ;(item?.core_entity_ids || []).forEach((entityId) => coreIds.add(entityId))
        })
        const metadata = chapterMetadata.get(section.id)
        ;(metadata?.knowledge_points || []).forEach((id) => coreIds.add(id))
        const entities = [...entityIds]
          .map((id) => entityMap.get(id))
          .filter(Boolean) as KnowledgeGraphNode[]
        entities.sort((a, b) => (
          Number(coreIds.has(b.id)) - Number(coreIds.has(a.id))
          || (b.core_score || 0) - (a.core_score || 0)
          || graphNodeName(a).localeCompare(graphNodeName(b), 'zh-CN')
        ))
        return {
          section,
          entities,
          coreCount: entities.filter((entity) => coreIds.has(entity.id) || entity.is_core).length,
          summary: metadata?.summary || section.summary || '',
        }
      })
  }, [entityMap, graph.chapters, sectionMap, sections])
  const entityNodeCount = graph.stats.entities || allEntities.length
  const entityEdgeCount = graph.stats.semantic_relations || conceptEdges.length
  const completeNodeCount = graph.stats.nodes
  const completeEdgeCount = graph.stats.edges

  return (
    <section className="feature-view neo4j-graph-view">
      <header className="neo4j-page-heading">
        <div className="neo4j-page-brand"><span className="neo4j-brand-mark"><Network size={22} /></span><div><span>KNOWLEDGE GRAPH</span><h1>{displayName}</h1></div></div>
      </header>
      <div className="neo4j-workbench">
        <div className="neo4j-workbench-body">
          <aside className="neo4j-database-panel">
            <div className="neo4j-panel-title"><Database size={17} /><div><strong>课程知识图谱</strong><span>已连接</span></div></div>
            <section><h2>节点 <small>{graph.stats.nodes}</small></h2><button type="button" className="active"><i style={{ background: '#68BDF6' }} />知识实体 <b>{graph.stats.entities || allEntities.length}</b></button><button type="button"><i style={{ background: '#A5ABB6' }} />章节 <b>{graph.stats.sections || sections.length}</b></button></section>
            <section className="neo4j-label-list"><h2>节点类型</h2><button type="button" className={entityType === 'all' ? 'active' : ''} aria-pressed={entityType === 'all'} onClick={() => { setEntityType('all'); setView('entities'); resetNeo4jViewport() }}><i style={{ background: '#4C8EDA' }} />全部类型<b>{allEntities.length}</b></button>{entityTypeCounts.map(([label, count]) => <button type="button" className={entityType === label ? 'active' : ''} aria-pressed={entityType === label} key={label} onClick={() => { setEntityType(entityType === label ? 'all' : label); setView('entities'); resetNeo4jViewport() }}><i style={{ background: neo4jEntityColor(label) }} />{label}<b>{count}</b></button>)}</section>
            <section><h2>关系</h2><button type="button"><span className="neo4j-rel-chip">概念关系</span><b>{conceptEdges.length}</b></button><button type="button"><span className="neo4j-rel-chip">实体归属</span><b>{graph.edges.filter((edge) => edge.type === 'entity_section').length}</b></button><button type="button"><span className="neo4j-rel-chip">章节层级</span><b>{graph.edges.filter((edge) => edge.type === 'parent_child').length}</b></button></section>
          </aside>
          <main className="neo4j-result-panel">
            <div className="neo4j-result-toolbar">
              <Segmented
                value={view}
                onChange={(value) => { setView(String(value) as 'sections' | 'entities'); setSelectedEdgeKey(''); resetNeo4jViewport() }}
                options={[
                  { label: `实体关系 · ${entityNodeCount} / ${entityEdgeCount}`, value: 'entities' },
                  { label: `完整图谱 · ${completeNodeCount} / ${completeEdgeCount}`, value: 'sections' },
                ]}
              />
              {view === 'entities' && <><Input allowClear value={entityQuery} onChange={(event) => setEntityQuery(event.target.value)} prefix={<Search size={14} />} placeholder="搜索实体或别名" /><Select value={entityType} onChange={(value) => { setEntityType(value); resetNeo4jViewport() }} options={[{ value: 'all', label: '全部类型' }, ...entityTypeCounts.map(([label, count]) => ({ value: label, label: `${label} (${count})` }))]} /><Segmented value={coreOnly ? 'core' : 'all'} onChange={(value) => { setCoreOnly(value === 'core'); resetNeo4jViewport() }} options={[{ label: '核心', value: 'core' }, { label: '全部', value: 'all' }]} /></>}
            </div>
            <div className="neo4j-graph-stage" data-mode={view}>
              <div className="neo4j-canvas-toolbar"><button type="button" aria-label="放大图谱" disabled={graphZoom >= NEO4J_ZOOM_MAX} onClick={() => changeNeo4jZoom(graphZoom * 1.35)}><ZoomIn size={18} /></button><button type="button" aria-label="缩小图谱" disabled={graphZoom <= NEO4J_ZOOM_MIN} onClick={() => changeNeo4jZoom(graphZoom / 1.35)}><ZoomOut size={18} /></button><button type="button" aria-label="重置图谱位置与缩放" onClick={resetNeo4jViewport}><RotateCcw size={17} /></button><span>{Math.round(graphZoom * 100)}%</span></div>
              {graphNodes.length ? <svg ref={graphSvgRef} className={graphDragging ? 'is-dragging' : ''} viewBox={graphViewBox} role="img" aria-label={`${displayName}${view === 'sections' ? '完整' : '实体关系'}知识图谱`} onPointerDown={handleNeo4jPointerDown} onPointerMove={handleNeo4jPointerMove} onPointerUp={finishNeo4jPointer} onPointerCancel={finishNeo4jPointer}>
                <defs><marker id="neo4j-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" /></marker></defs>
                <g>
                  <g className="neo4j-edges">{graphEdges.map((edge, index) => {
                    const from = layout.positions.get(edge.source)
                    const to = layout.positions.get(edge.target)
                    if (!from || !to) return null
                    const key = edgeKey(edge)
                    const isSelected = selectedEdgeKey === key
                    return <g key={`${key}\u0000${index}`} className={`${edge.type} ${isSelected ? 'selected' : ''}`} role="button" onClick={(event) => { event.stopPropagation(); if (suppressGraphClickRef.current) return; setSelectedEdgeKey(key); setSelectedId('') }}><line className="hit" x1={from.x} y1={from.y} x2={to.x} y2={to.y} /><line className="line" x1={from.x} y1={from.y} x2={to.x} y2={to.y} markerEnd={isSelected ? 'url(#neo4j-arrow)' : undefined} />{isSelected && <text x={(from.x + to.x) / 2} y={(from.y + to.y) / 2 - 5}>{(edge.relation || edge.type || '关联').slice(0, 12)}</text>}</g>
                  })}</g>
                  <g className="neo4j-nodes">{graphNodes.map((node) => {
                    const position = layout.positions.get(node.id) || { x: NEO4J_VIEWBOX_WIDTH / 2, y: NEO4J_VIEWBOX_HEIGHT / 2 }
                    const radius = neo4jNodeRadius(node, graphNodes.length)
                    const queryMatch = matchedNodeIds.has(node.id)
                    const queryDim = Boolean(normalizedEntityQuery) && node.type === 'entity' && !queryMatch
                    return <g key={node.id} className={`${node.type === 'section' ? 'section' : 'entity'} ${node.is_core ? 'core' : ''} ${selectedId === node.id ? 'selected' : ''} ${queryMatch ? 'query-match' : ''} ${queryDim ? 'query-dim' : ''}`} transform={`translate(${position.x} ${position.y})`} role="button" tabIndex={0} aria-pressed={selectedId === node.id} onClick={(event) => { event.stopPropagation(); if (suppressGraphClickRef.current) return; setSelectedId((current) => current === node.id ? '' : node.id); setSelectedEdgeKey('') }} onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); setSelectedId((current) => current === node.id ? '' : node.id); setSelectedEdgeKey('') } }}><title>{graphNodeName(node)}{node.type === 'entity' && node.entity_type ? ` · ${node.entity_type}` : ''}</title><circle className="hit" r={Math.max(6, radius + 3.5)} /><circle className="node" r={radius} style={{ fill: neo4jNodeColor(node) }} />{labelNodeIds.has(node.id) && <text y={radius + 11}>{graphNodeName(node).slice(0, node.type === 'section' ? 18 : 14)}</text>}</g>
                  })}</g>
                </g>
              </svg> : <div className="neo4j-empty"><Search size={26} /><strong>没有符合筛选条件的实体</strong><span>尝试切换实体类型或显示全部实体。</span></div>}
              <div className="neo4j-stage-status"><span>{view === 'sections' ? '完整图谱模式' : '实体关系模式'}</span><span>节点（{graphNodes.length}）</span><span>关系（{graphEdges.length}）</span><small>{normalizedEntityQuery ? `搜索匹配 ${matchedNodeIds.size} 个实体 · ` : ''}已加载当前范围的全部节点和关系，可拖动画布查看</small></div>
            </div>
          </main>
          <aside className="neo4j-inspector">
            <div className="neo4j-inspector-heading"><strong>属性</strong><span>{selectedEdge ? '关系' : selected ? '节点' : '选择项'}</span></div>
            {selectedEdge ? <div className="neo4j-property-content"><span className="neo4j-object-type">{neo4jEdgeTypeLabel(selectedEdge.type)}</span><h2>{selectedEdge.relation || '关联'}</h2><dl><dt>起点</dt><dd><InlineMath content={graphNodeName(graphNodeMap.get(selectedEdge.source)) || selectedEdge.source} /></dd><dt>终点</dt><dd><InlineMath content={graphNodeName(graphNodeMap.get(selectedEdge.target)) || selectedEdge.target} /></dd><dt>强度</dt><dd>{selectedEdge.strength?.toFixed(1) || '—'}</dd><dt>置信度</dt><dd>{selectedEdge.confidence ? `${Math.round(selectedEdge.confidence * 100)}%` : '—'}</dd></dl>{selectedEdge.description && <MathMarkdown content={selectedEdge.description} />}</div> : selected?.type === 'section' ? <div className="neo4j-property-content"><span className="neo4j-object-type section">章节 · 第 {selected.level || 0} 级</span><h2><InlineMath content={selected.title || selected.name} /></h2><p>{selected.summary || '该节点仅用于补全目录结构，没有可摘要的直接正文。'}</p>{selectedSectionEntities.length > 0 && <div className="hierarchical-chip-list">{selectedSectionEntities.slice(0, 24).map((entity) => <button type="button" key={entity.id} className={entity.is_core ? 'core' : ''} onClick={() => { setView('entities'); setSelectedId(entity.id) }}>{entity.name}</button>)}</div>}</div> : selected?.type === 'entity' ? <div className="neo4j-property-content neo4j-entity-property-content"><h2><InlineMath content={graphNodeName(selected)} /></h2><section><h3>关系数量</h3><strong className="neo4j-relationship-count">{selectedNeighbors}</strong></section><section><h3>描述</h3><MathMarkdown content={selectedEntityDescription} /></section><section><h3>关联关系</h3>{selectedEntityRelationships.length > 0 ? <div className="neo4j-related-list">{selectedEntityRelationships.map(({ edge, outgoing, neighbor, neighborId }, index) => <article key={`${edgeKey(edge)}\u0000${index}`}><div><span>{edge.relation || '关联'}</span><strong>{outgoing ? '→' : '←'} <InlineMath content={graphNodeName(neighbor) || neighborId} /></strong></div>{edge.description && <MathMarkdown content={edge.description} />}</article>)}</div> : <p>暂无关联关系。</p>}</section></div> : <div className="neo4j-inspector-empty"><Network size={30} /><strong>选择图谱对象</strong><span>点击节点、关系或章节，查看属性和块级证据。</span></div>}
            {(selectedEdge || selected?.type === 'section') && <><div className="graph-evidence-title">证据块</div>{evidenceLoading && <div className="graph-evidence-state"><LoaderCircle className="spin" size={15} /> 正在读取证据…</div>}{!evidenceLoading && evidence.length > 0 && <div className="graph-evidence-list">{evidence.slice(0, 10).map((item) => <article key={item.id}><span>{item.modality} {item.page ? `· 第 ${item.page} 页` : ''}</span><p>{item.text || item.caption || item.description}</p><small>{item.id}</small></article>)}</div>}{!evidenceLoading && !evidence.length && <div className="graph-evidence-state">当前选择没有可展示的证据块。</div>}</>}
          </aside>
        </div>
      </div>
      <section className="neo4j-chapter-overview" aria-labelledby="neo4j-chapter-title">
        <header><div><span>CHAPTER KNOWLEDGE INDEX</span><h2 id="neo4j-chapter-title">各章节知识点汇总</h2><p>每章汇总其全部子章节中的实体、核心知识点、页码范围与章节摘要。</p></div><div><strong>{chapterOverviews.length}</strong><span>课程章节</span><strong>{allEntities.length}</strong><span>唯一实体</span></div></header>
        <div className="neo4j-chapter-grid">{chapterOverviews.map(({ section, entities, coreCount, summary }, index) => {
          const isExpanded = expandedChapterId === section.id
          const displayedEntities = entities.slice(0, isExpanded ? 36 : 8)
          return <article className={isExpanded ? 'expanded' : ''} key={section.id}><div className="neo4j-chapter-top"><span>{String(index + 1).padStart(2, '0')}</span><small>第 {section.page_start || '—'}–{section.page_end || section.page_start || '—'} 页</small></div><h3><InlineMath content={section.title || section.name} /></h3>{summary && <p>{summary}</p>}<div className="neo4j-chapter-stats"><span><strong>{entities.length}</strong> 个知识实体</span><span><strong>{coreCount}</strong> 个核心知识点</span></div><div className="neo4j-chapter-points">{displayedEntities.map((entity) => <button type="button" key={entity.id} className={entity.is_core ? 'core' : ''} onClick={() => { setView('entities'); setSelectedId(entity.id); setSelectedEdgeKey(''); document.querySelector('.neo4j-workbench')?.scrollIntoView({ behavior: 'smooth', block: 'start' }) }}><i style={{ background: neo4jEntityColor(entity.entity_type) }} /><InlineMath content={graphNodeName(entity)} /></button>)}{!entities.length && <span className="empty">暂无可汇总实体</span>}</div>{entities.length > 8 && <button type="button" className="neo4j-chapter-expand" onClick={() => setExpandedChapterId(isExpanded ? '' : section.id)}>{isExpanded ? '收起知识点' : `展开全部 ${entities.length} 个知识点`}<ChevronDown size={14} /></button>}</article>
        })}</div>
      </section>
    </section>
  )
}

function KnowledgeGraphView({
  graph,
  loading,
  knowledgeBase,
  displayName,
}: {
  graph?: KnowledgeGraph
  loading: boolean
  knowledgeBase: string
  displayName: string
}) {
  if (loading) return <div className="workspace-empty"><LoaderCircle className="spin" /><strong>正在整理知识图谱…</strong></div>
  if (graph && graph.knowledge_base !== knowledgeBase) return <div className="workspace-empty"><ShieldCheck /><strong>正在切换知识库图谱…</strong><p>旧知识库数据已隐藏，等待当前选择加载完成。</p></div>
  if (!graph?.nodes.length) return <div className="workspace-empty"><Network /><strong>当前知识库还没有图谱数据</strong><p>重建知识库后会从教材正文与电路图中抽取知识实体和原始关系。</p></div>
  if (graph.schema_version?.startsWith('4.')) return <HierarchicalKnowledgeGraphView graph={graph} displayName={displayName} />
  return <LegacyKnowledgeGraphView graph={graph} loading={false} />
}

function LegacyKnowledgeGraphView({ graph, loading }: { graph?: KnowledgeGraph; loading: boolean }) {
  const [selectedId, setSelectedId] = useState('')
  const [selectedEdgeKey, setSelectedEdgeKey] = useState('')
  const [selectedEvidence, setSelectedEvidence] = useState<KnowledgeGraphEvidence[]>([])
  const [evidenceLoading, setEvidenceLoading] = useState(false)
  const [evidenceError, setEvidenceError] = useState('')
  const [selectedChapterId, setSelectedChapterId] = useState('')
  const [chapterWindowOpen, setChapterWindowOpen] = useState(false)
  const [chapterConceptQuery, setChapterConceptQuery] = useState('')
  const [rangeMode, setRangeMode] = useState<'core' | 'extended' | 'all' | 'custom'>('core')
  const [limits, setLimits] = useState({ concepts: 14, pages: 8, structures: 14 })
  const [graphZoom, setGraphZoom] = useState(1)
  const [graphCenter, setGraphCenter] = useState({ x: GRAPH_VIEWBOX_WIDTH / 2, y: GRAPH_VIEWBOX_HEIGHT / 2 })
  const [graphDragging, setGraphDragging] = useState(false)
  const graphSvgRef = useRef<SVGSVGElement | null>(null)
  const graphDragRef = useRef<{
    pointerId: number
    clientX: number
    clientY: number
    centerX: number
    centerY: number
    moved: boolean
  } | null>(null)
  const suppressGraphClickRef = useRef(false)
  const semanticGraph = Boolean(graph?.schema_version?.startsWith('3.'))
  const totals = useMemo(() => ({
    concepts: graph?.nodes.filter((node) => node.type === 'concept' || node.type === 'entity').length || 0,
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
    setSelectedEdgeKey('')
    setSelectedEvidence([])
    setEvidenceError('')
    setSelectedChapterId('')
    setChapterWindowOpen(false)
    setChapterConceptQuery('')
    setGraphZoom(1)
    setGraphCenter({ x: GRAPH_VIEWBOX_WIDTH / 2, y: GRAPH_VIEWBOX_HEIGHT / 2 })
  }, [graph?.knowledge_base])

  const graphViewport = {
    width: GRAPH_VIEWBOX_WIDTH / graphZoom,
    height: GRAPH_VIEWBOX_HEIGHT / graphZoom,
    x: graphCenter.x - GRAPH_VIEWBOX_WIDTH / graphZoom / 2,
    y: graphCenter.y - GRAPH_VIEWBOX_HEIGHT / graphZoom / 2,
  }
  const graphViewBox = `${graphViewport.x} ${graphViewport.y} ${graphViewport.width} ${graphViewport.height}`

  const changeGraphZoom = (requestedZoom: number, anchor?: { ratioX: number; ratioY: number; x: number; y: number }) => {
    const nextZoom = clampGraphZoom(requestedZoom)
    if (Math.abs(nextZoom - graphZoom) < 0.001) return
    if (anchor) {
      const nextWidth = GRAPH_VIEWBOX_WIDTH / nextZoom
      const nextHeight = GRAPH_VIEWBOX_HEIGHT / nextZoom
      setGraphCenter({
        x: anchor.x + (0.5 - anchor.ratioX) * nextWidth,
        y: anchor.y + (0.5 - anchor.ratioY) * nextHeight,
      })
    }
    setGraphZoom(nextZoom)
  }

  const resetGraphViewport = () => {
    setGraphZoom(1)
    setGraphCenter({ x: GRAPH_VIEWBOX_WIDTH / 2, y: GRAPH_VIEWBOX_HEIGHT / 2 })
  }

  useEffect(() => {
    const svg = graphSvgRef.current
    const canvas = svg?.parentElement
    if (!svg || !canvas) return
    const handleWheel = (event: WheelEvent) => {
      event.preventDefault()
      event.stopPropagation()
      const rect = svg.getBoundingClientRect()
      if (!rect.width || !rect.height) return
      const ratioX = Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width))
      const ratioY = Math.min(1, Math.max(0, (event.clientY - rect.top) / rect.height))
      changeGraphZoom(graphZoom * Math.exp(-event.deltaY * 0.0015), {
        ratioX,
        ratioY,
        x: graphViewport.x + ratioX * graphViewport.width,
        y: graphViewport.y + ratioY * graphViewport.height,
      })
    }
    canvas.addEventListener('wheel', handleWheel, { passive: false })
    return () => canvas.removeEventListener('wheel', handleWheel)
  }, [graphZoom, graphCenter.x, graphCenter.y])

  const handleGraphPointerDown = (event: ReactPointerEvent<SVGSVGElement>) => {
    if (event.button !== 0) return
    graphDragRef.current = {
      pointerId: event.pointerId,
      clientX: event.clientX,
      clientY: event.clientY,
      centerX: graphCenter.x,
      centerY: graphCenter.y,
      moved: false,
    }
    event.currentTarget.setPointerCapture(event.pointerId)
    setGraphDragging(true)
  }

  const handleGraphPointerMove = (event: ReactPointerEvent<SVGSVGElement>) => {
    const drag = graphDragRef.current
    if (!drag || drag.pointerId !== event.pointerId) return
    event.preventDefault()
    event.stopPropagation()
    const rect = event.currentTarget.getBoundingClientRect()
    if (!rect.width || !rect.height) return
    const deltaX = event.clientX - drag.clientX
    const deltaY = event.clientY - drag.clientY
    if (Math.abs(deltaX) + Math.abs(deltaY) > 4) drag.moved = true
    setGraphCenter({
      x: drag.centerX - deltaX * graphViewport.width / rect.width,
      y: drag.centerY - deltaY * graphViewport.height / rect.height,
    })
  }

  const finishGraphPointer = (event: ReactPointerEvent<SVGSVGElement>) => {
    const drag = graphDragRef.current
    if (!drag || drag.pointerId !== event.pointerId) return
    if (drag.moved) {
      suppressGraphClickRef.current = true
      window.setTimeout(() => { suppressGraphClickRef.current = false }, 0)
    }
    graphDragRef.current = null
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId)
    }
    setGraphDragging(false)
  }

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
    const documents = semanticGraph ? [] : graph.nodes.filter((node) => node.type === 'document').slice(0, 3)
    const concepts = graph.nodes
      .filter((node) => node.type === 'concept' || node.type === 'entity')
      .sort((a, b) => (
        Number((degree.get(b.id) || 0) > 0) - Number((degree.get(a.id) || 0) > 0)
        || (degree.get(b.id) || 0) - (degree.get(a.id) || 0)
        || (b.evidence_count || 0) - (a.evidence_count || 0)
        || graphNodeName(a).localeCompare(graphNodeName(b), 'zh-CN')
      ))
      .slice(0, limits.concepts)
    const selectedConcepts = new Set(concepts.map((node) => node.id))
    const selectedCoverage = new Map<string, number>()
    graph.edges.forEach((edge) => {
      if (edge.type === 'COVERS' && selectedConcepts.has(edge.target)) {
        selectedCoverage.set(edge.source, (selectedCoverage.get(edge.source) || 0) + (edge.evidence_count || 1))
      }
    })
    const pages = (semanticGraph ? [] : graph.nodes)
      .filter((node) => node.type === 'page')
      .sort((a, b) => (
        (selectedCoverage.get(b.id) || 0) - (selectedCoverage.get(a.id) || 0)
        || (degree.get(b.id) || 0) - (degree.get(a.id) || 0)
        || (a.page || 0) - (b.page || 0)
      ))
      .slice(0, limits.pages)
    const structures = (semanticGraph ? [] : graph.nodes)
      .filter((node) => node.type === 'circuit' || node.type === 'component')
      .sort((a, b) => (degree.get(b.id) || 0) - (degree.get(a.id) || 0))
      .slice(0, limits.structures)
    const groups = semanticGraph
      ? [
          { nodes: concepts.slice(0, 1), radius: 0, ringGap: 34, capacity: 1 },
          { nodes: concepts.slice(1), radius: 135, ringGap: 55, capacity: 32 },
        ]
      : [
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
  }, [graph, limits, semanticGraph])
  const positions = new Map(visual.nodes.map((node) => [node.id, node]))
  const selected = graph?.nodes.find((node) => node.id === selectedId)
  const edgeKey = (source: string, relation: string, target: string) => `${source}\u0000${relation}\u0000${target}`
  const selectedEdge = visual.edges.find((edge) => (
    edgeKey(edge.source, edge.relation || edge.type || '', edge.target) === selectedEdgeKey
  ))
  const selectedEdgeSource = graph?.nodes.find((node) => node.id === selectedEdge?.source)
  const selectedEdgeTarget = graph?.nodes.find((node) => node.id === selectedEdge?.target)
  const neighbors = selectedId && graph
    ? graph.edges.filter((edge) => edge.source === selectedId || edge.target === selectedId).length
    : 0
  const typeLabel: Record<string, string> = {
    document: '教材',
    page: '教材页面',
    concept: '知识点',
    circuit: '电路图',
    component: '电路元件',
    entity: '知识实体',
  }
  const selectedPages = selected?.pages?.length
    ? selected.pages
    : selected?.page
      ? [selected.page]
      : []
  const chapters = graph?.chapters || []
  const selectedChapter = chapters.find((chapter) => chapter.id === selectedChapterId)
    || chapters[0]
  const visibleChapterConcepts = selectedChapter?.concepts.filter((concept) => (
    !chapterConceptQuery.trim()
    || concept.name.toLowerCase().includes(chapterConceptQuery.trim().toLowerCase())
  )) || []
  const openChapterWindow = (chapter: ChapterKnowledgeSummary) => {
    setSelectedChapterId(chapter.id)
    setChapterConceptQuery('')
    setChapterWindowOpen(true)
  }
  const chapterPageRange = (chapter: ChapterKnowledgeSummary) => {
    if (!chapter.page_start) return '暂无页码'
    return chapter.page_end && chapter.page_end !== chapter.page_start
      ? `第 ${chapter.page_start}–${chapter.page_end} 页`
      : `第 ${chapter.page_start} 页`
  }

  useEffect(() => {
    let active = true
    setSelectedEvidence([])
    setEvidenceError('')
    const evidenceIds = selectedEdge?.evidence_ids || []
    if (!graph || !selectedEdgeKey || !evidenceIds.length) {
      setEvidenceLoading(false)
      return () => { active = false }
    }
    setEvidenceLoading(true)
    fetchKnowledgeGraphEvidence(graph.knowledge_base, evidenceIds)
      .then((items) => {
        if (active) setSelectedEvidence(items)
      })
      .catch((error) => {
        if (active) setEvidenceError(error instanceof Error ? error.message : '关系证据读取失败')
      })
      .finally(() => {
        if (active) setEvidenceLoading(false)
      })
    return () => { active = false }
  }, [graph?.knowledge_base, selectedEdgeKey])

  if (loading) return <div className="workspace-empty"><LoaderCircle className="spin" /><strong>正在整理知识图谱…</strong></div>
  if (!graph?.nodes.length) return <div className="workspace-empty"><Network /><strong>当前知识库还没有图谱数据</strong><p>重建知识库后会从教材正文与电路图中抽取知识实体和原始关系。</p></div>
  return (
    <section className="feature-view graph-view">
      <div className="feature-heading">
        <div><span>KNOWLEDGE MAP</span><h1>课程知识图谱</h1><p>{semanticGraph ? '只展示从教材正文与电路图中抽取的知识实体和原始关系；页码、文本片段与具体电路图作为证据单独保存。' : '展示旧版知识图谱数据；重建知识库后将升级为实体—原始关系语义图谱。'}</p></div>
        <div className="feature-stats"><strong>{graph.stats.entities ?? graph.stats.concepts}</strong><span>知识实体</span><strong>{graph.stats.semantic_relations ?? graph.stats.edges}</strong><span>语义关系</span><strong>{graph.stats.communities || 0}</strong><span>社区</span></div>
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
        <div className={`graph-range-grid ${semanticGraph ? 'semantic' : ''}`}>
          <label>
            <span>{semanticGraph ? '知识实体' : '知识点'} <b>{limits.concepts}/{totals.concepts}</b></span>
            <Slider min={0} max={Math.max(1, totals.concepts)} value={limits.concepts} disabled={!totals.concepts} onChange={(value) => changeRange('concepts', value)} />
          </label>
          {!semanticGraph && <label>
            <span>教材页面 <b>{limits.pages}/{totals.pages}</b></span>
            <Slider min={0} max={Math.max(1, totals.pages)} value={limits.pages} disabled={!totals.pages} onChange={(value) => changeRange('pages', value)} />
          </label>}
          {!semanticGraph && <label>
            <span>电路与元件 <b>{limits.structures}/{totals.structures}</b></span>
            <Slider min={0} max={Math.max(1, totals.structures)} value={limits.structures} disabled={!totals.structures} onChange={(value) => changeRange('structures', value)} />
          </label>}
        </div>
      </div>
      <div className="graph-layout">
        <div className={`graph-canvas ${graphDragging ? 'is-dragging' : ''}`}>
          <svg
            ref={graphSvgRef}
            viewBox={graphViewBox}
            role="img"
            aria-label={`课程知识关系图，当前缩放 ${Math.round(graphZoom * 100)}%`}
            onPointerDown={handleGraphPointerDown}
            onPointerMove={handleGraphPointerMove}
            onPointerUp={finishGraphPointer}
            onPointerCancel={finishGraphPointer}
          >
            <defs>
              <marker id="graph-arrowhead" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="strokeWidth">
                <path d="M 0 0 L 8 4 L 0 8 z" />
              </marker>
            </defs>
            <g className="graph-edges">
              {visual.edges.map((edge, index) => {
                const from = positions.get(edge.source)
                const to = positions.get(edge.target)
                if (!from || !to) return null
                const relation = edge.relation || edge.type || '关联'
                const key = edgeKey(edge.source, relation, edge.target)
                const middleX = (from.x + to.x) / 2
                const middleY = (from.y + to.y) / 2
                return (
                  <g
                    className={`graph-edge ${selectedEdgeKey === key ? 'selected' : ''}`}
                    key={`${key}-${index}`}
                    onClick={(event) => {
                      event.stopPropagation()
                      if (!suppressGraphClickRef.current) {
                        setSelectedEdgeKey(key)
                        setSelectedId('')
                      }
                    }}
                    role="button"
                    tabIndex={0}
                    onKeyDown={(event) => {
                      if (event.key === 'Enter' || event.key === ' ') {
                        setSelectedEdgeKey(key)
                        setSelectedId('')
                      }
                    }}
                  >
                    <line className="graph-edge-hit" x1={from.x} y1={from.y} x2={to.x} y2={to.y} />
                    <line className="graph-edge-line" markerEnd="url(#graph-arrowhead)" x1={from.x} y1={from.y} x2={to.x} y2={to.y} />
                    {semanticGraph && <text className="graph-edge-label" x={middleX} y={middleY - 5}>{relation.slice(0, 16)}</text>}
                  </g>
                )
              })}
            </g>
            <g>
              {visual.nodes.map((node) => (
                <g
                  key={node.id}
                  className={`graph-node ${node.type} ${selectedId === node.id ? 'selected' : ''}`}
                  transform={`translate(${node.x} ${node.y})`}
                  onClick={() => {
                    if (!suppressGraphClickRef.current) {
                      setSelectedId(node.id)
                      setSelectedEdgeKey('')
                    }
                  }}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter' || event.key === ' ') {
                      setSelectedId(node.id)
                      setSelectedEdgeKey('')
                    }
                  }}
                  role="button"
                  tabIndex={0}
                >
                  <circle r={node.type === 'document' ? 25 : node.type === 'page' ? 16 : node.type === 'concept' || node.type === 'entity' ? Math.min(22, 12 + node.degree) : node.type === 'circuit' ? 12 : 9} />
                  {(node.type !== 'page' || limits.pages <= 12 || selectedId === node.id) && (
                    <text y={node.type === 'document' ? 40 : node.type === 'page' || node.type === 'concept' || node.type === 'entity' ? 32 : 23}>{graphNodeName(node).replace(/^电路图\s*[·•]\s*/, '').slice(0, 18) || typeLabel[node.type] || '资料'}</text>
                  )}
                </g>
              ))}
            </g>
          </svg>
          <div className="graph-zoom-controls" role="group" aria-label="知识图谱缩放控制">
            <button type="button" onClick={() => changeGraphZoom(graphZoom / 1.25)} disabled={graphZoom <= GRAPH_ZOOM_MIN} aria-label="缩小知识图谱" title="缩小">
              <ZoomOut size={16} />
            </button>
            <button type="button" className="graph-zoom-value" onClick={resetGraphViewport} aria-label="恢复知识图谱默认大小" title="恢复默认视图">
              {Math.round(graphZoom * 100)}%
            </button>
            <button type="button" onClick={() => changeGraphZoom(graphZoom * 1.25)} disabled={graphZoom >= GRAPH_ZOOM_MAX} aria-label="放大知识图谱" title="放大">
              <ZoomIn size={16} />
            </button>
            <button type="button" onClick={resetGraphViewport} aria-label="重置知识图谱视图" title="重置视图">
              <RotateCcw size={15} />
            </button>
          </div>
          <div className="graph-zoom-hint">滚轮缩放 · 拖拽移动</div>
          <div className="graph-legend">{semanticGraph ? <><span><i className="entity" />知识实体</span><span><i className="relation" />原始关系</span></> : <><span><i className="document" />教材</span><span><i className="page" />页面</span><span><i className="concept" />知识点</span><span><i className="circuit" />电路图</span><span><i className="component" />元件</span></>}</div>
        </div>
        <aside className="graph-detail">
          {selectedEdge ? <>
            <span>原始关系</span>
            <h2><InlineMath content={selectedEdge.relation || selectedEdge.type} /></h2>
            <p className="graph-relation-statement"><InlineMath content={graphNodeName(selectedEdgeSource) || selectedEdge.source} /> <strong>{selectedEdge.relation || selectedEdge.type}</strong> <InlineMath content={graphNodeName(selectedEdgeTarget) || selectedEdge.target} /></p>
            <div className="graph-evidence-title">关系证据 {selectedEdge.evidence_count ? `· ${selectedEdge.evidence_count} 条` : ''}</div>
            {evidenceLoading && <div className="graph-evidence-state"><LoaderCircle className="spin" size={15} /> 正在读取证据…</div>}
            {evidenceError && <div className="graph-evidence-state error">{evidenceError}</div>}
            {!evidenceLoading && !evidenceError && selectedEvidence.length > 0 && <div className="graph-evidence-list">
              {selectedEvidence.map((item) => (
                <article key={item.id}>
                  <span>{item.modality === 'circuit' ? '电路图识别' : '教材正文'}{item.page || item.page_start ? ` · 第 ${item.page || item.page_start} 页` : ''}</span>
                  <p>{item.text || item.caption || item.description || '该关系来自已保存的原始证据。'}</p>
                  <small>{item.source}</small>
                </article>
              ))}
            </div>}
            {!evidenceLoading && !evidenceError && !selectedEvidence.length && <div className="graph-evidence-state">当前关系没有可展示的证据片段。</div>}
          </> : selected ? <><span>{typeLabel[selected.type] || '知识节点'}</span><h2><InlineMath content={graphNodeName(selected) || '未命名节点'} /></h2>{selected.display_name && selected.display_name !== selected.name && <p>教材原符号：<InlineMath content={selected.name} /></p>}{selected.description && <p>{selected.description}</p>}<p>连接 {neighbors} 个知识实体{selected.evidence_count ? `，由 ${selected.evidence_count} 条原始证据支持` : ''}。页码与具体电路图只用于追溯来源，不作为知识图谱节点。</p>{selectedPages.length > 0 && <div className="graph-page-list">来源页码：{selectedPages.map((page) => `第 ${page} 页`).join('、')}</div>}</> : <><Network size={28} /><h2>探索知识关系</h2><p>{semanticGraph ? '点击实体查看说明，点击带文字的箭头查看原始关系及其教材或电路图证据。' : '这是旧版图谱；重建知识库后可查看实体之间的原始语义关系。'}</p></>}
        </aside>
      </div>
      {chapters.length > 0 && (
        <section className="chapter-directory" aria-labelledby="chapter-directory-title">
          <div className="chapter-directory-heading">
            <div>
              <span>CHAPTER INDEX</span>
              <h2 id="chapter-directory-title">按章节查看知识点</h2>
              <p>构建完成后自动按教材章节归档，进入任一章节即可查看知识点与教材证据页。</p>
            </div>
            <button type="button" onClick={() => openChapterWindow(chapters[0])}>
              <BookOpen size={15} /> 查看全部章节 <ChevronRight size={14} />
            </button>
          </div>
          <div className="chapter-card-grid">
            {chapters.map((chapter, index) => (
              <article className="chapter-card" key={chapter.id}>
                <div className="chapter-card-topline">
                  <span>{String(index + 1).padStart(2, '0')}</span>
                  <small>{chapterPageRange(chapter)}</small>
                </div>
                <h3><InlineMath content={chapter.name} /></h3>
                <div className="chapter-card-stats">
                  <span><strong>{chapter.concept_count}</strong> 个知识点</span>
                  <span><strong>{chapter.section_count}</strong> 个小节</span>
                </div>
                <div className="chapter-card-preview">
                  {chapter.concepts.slice(0, 5).map((concept) => (
                    <span key={concept.id}><InlineMath content={concept.name} /></span>
                  ))}
                  {chapter.concept_count > 5 && <span>+{chapter.concept_count - 5}</span>}
                </div>
                <button type="button" className="chapter-card-enter" onClick={() => openChapterWindow(chapter)}>
                  进入查看 <ChevronRight size={14} />
                </button>
              </article>
            ))}
          </div>
        </section>
      )}
      <Modal
        open={chapterWindowOpen && Boolean(selectedChapter)}
        onCancel={() => setChapterWindowOpen(false)}
        footer={null}
        width={960}
        title={null}
        className="chapter-knowledge-modal"
      >
        {selectedChapter && (
          <div className="chapter-window">
            <aside className="chapter-window-nav">
              <div className="chapter-window-brand">
                <GraduationCap size={22} />
                <div><span>CHAPTER MAP</span><strong>章节知识目录</strong></div>
              </div>
              <nav aria-label="章节知识点">
                {chapters.map((chapter, index) => (
                  <button
                    type="button"
                    className={chapter.id === selectedChapter.id ? 'active' : ''}
                    key={chapter.id}
                    onClick={() => {
                      setSelectedChapterId(chapter.id)
                      setChapterConceptQuery('')
                    }}
                  >
                    <span>{String(index + 1).padStart(2, '0')}</span>
                    <div><strong><InlineMath content={chapter.name} /></strong><small>{chapter.concept_count} 个知识点</small></div>
                    <ChevronRight size={14} />
                  </button>
                ))}
              </nav>
            </aside>
            <main className="chapter-window-main">
              <div className="chapter-window-heading">
                <div>
                  <span>第 {selectedChapter.order} 组 · {chapterPageRange(selectedChapter)}</span>
                  <h2><InlineMath content={selectedChapter.name} /></h2>
                  <p>{selectedChapter.sources.join('、')} · {selectedChapter.section_count} 个小节 · {selectedChapter.concept_count} 个知识点</p>
                </div>
                <div className="chapter-concept-total"><strong>{selectedChapter.concept_count}</strong><span>KNOWLEDGE POINTS</span></div>
              </div>
              <Input
                allowClear
                value={chapterConceptQuery}
                onChange={(event) => setChapterConceptQuery(event.target.value)}
                prefix={<Search size={15} />}
                placeholder="在本章知识点中搜索"
                className="chapter-concept-search"
              />
              <div className="chapter-concept-grid">
                {visibleChapterConcepts.map((concept, index) => (
                  <article className="chapter-concept-card" key={concept.id}>
                    <span>{String(index + 1).padStart(2, '0')}</span>
                    <div>
                      <h3><InlineMath content={concept.name} /></h3>
                      <p>{concept.evidence_count} 条教材证据{concept.pages.length ? ` · 第 ${concept.pages.join('、')} 页` : ''}</p>
                    </div>
                  </article>
                ))}
                {!visibleChapterConcepts.length && (
                  <div className="chapter-concept-empty"><Search size={22} /><span>没有匹配的知识点</span></div>
                )}
              </div>
            </main>
          </div>
        )}
      </Modal>
    </section>
  )
}

const studentQuestionBankStatus = {
  processing: { label: '提取中', color: 'processing', icon: <LoaderCircle className="spin" size={13} /> },
  ready: { label: '可使用', color: 'success', icon: <CheckCircle2 size={13} /> },
  error: { label: '提取失败', color: 'error', icon: <AlertTriangle size={13} /> },
  cancelled: { label: '已取消', color: 'default', icon: <CircleStop size={13} /> },
} as const

function questionBankTime(value: string) {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return '刚刚'
  return date.toLocaleString('zh-CN', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  })
}

function questionTypeName(value: string) {
  const labels: Record<string, string> = {
    choice: '选择题',
    fill_blank: '填空题',
    true_false: '判断题',
    short_answer: '简答题',
    calculation: '计算题',
    design: '设计题',
    other: '综合题',
  }
  return labels[value] || '题目'
}

const questionTypeOptionsForStudent = [
  'choice', 'fill_blank', 'true_false', 'short_answer', 'calculation', 'design', 'other',
].map((value) => ({ value, label: questionTypeName(value) }))

function QuestionKnowledgeAlignment({ question }: { question: HomeworkQuestion }) {
  const tags = question.knowledge_tags?.length
    ? question.knowledge_tags
    : (question.knowledge_points || []).map((point) => ({
      tag_id: point,
      tag_name: point,
      tag_source: 'custom' as const,
      knowledge_node_id: null,
      match_type: 'unmatched' as const,
      confidence: 0,
      is_exact: false,
      needs_confirmation: true,
    }))
  return (
    <section className="question-knowledge-alignment">
      <header>
        <span>KNOWLEDGE ALIGNMENT</span>
        <small>
          {question.location?.chapter || '暂未确定'}
          {question.location?.section && question.location.section !== '暂未确定' ? ` · ${question.location.section}` : ''}
        </small>
      </header>
      {tags.length ? (
        <div className="question-knowledge-tags">
          {tags.map((tag, index) => {
            const confidence = Math.round(tag.confidence * 100)
            const matchLabel = tag.match_type === 'exact'
              ? '图谱精确匹配'
              : tag.match_type === 'approximate'
                ? '图谱近似匹配'
                : '暂未匹配图谱'
            return (
              <Tooltip key={`${tag.tag_id}-${index}`} title={`${matchLabel} · 匹配程度 ${confidence}%`}>
                <span className={`question-knowledge-tag ${tag.match_type}`}>
                  <Tag color={tag.match_type === 'exact' ? 'green' : tag.match_type === 'approximate' ? 'gold' : 'default'}>
                    <InlineMath content={tag.tag_name} />
                  </Tag>
                  <b>{confidence}%</b>
                </span>
              </Tooltip>
            )
          })}
        </div>
      ) : (
        <p className="question-knowledge-empty">知识点正在补充中</p>
      )}
      {question.prerequisites?.length ? (
        <p className="question-prerequisites">
          前置知识：<InlineMath content={question.prerequisites.map((item) => item.name).join('、')} />
        </p>
      ) : null}
    </section>
  )
}

function QuestionBankFigureGallery({
  figures,
  questionNumber,
  answer = false,
}: {
  figures: NonNullable<HomeworkQuestion['answer_figures']>
  questionNumber: HomeworkQuestion['number'] | HomeworkQuestion['sequence']
  answer?: boolean
}) {
  return (
    <AntImage.PreviewGroup>
      <div className={`student-bank-figures${answer ? ' answer' : ''}`}>
        {figures.map((figure) => (
          <figure key={figure.file}>
            <AntImage
              src={figure.url}
              alt={figure.caption || `第 ${questionNumber} 题${answer ? '答案图' : '题图'}`}
              preview={{
                mask: (
                  <span className="student-bank-figure-preview-mask">
                    <ZoomIn size={15} />
                    点击放大
                  </span>
                ),
              }}
            />
            {figure.caption ? <figcaption>{figure.caption}</figcaption> : null}
          </figure>
        ))}
      </div>
    </AntImage.PreviewGroup>
  )
}

function QuestionBankQuestionCard({
  question,
  paper = false,
  deleting,
  onDelete,
  onAsk,
}: {
  question: HomeworkQuestion
  paper?: boolean
  deleting: boolean
  onDelete: () => void
  onAsk: () => void
}) {
  const [answerOpen, setAnswerOpen] = useState(false)
  const sourceKindLabel = question.source_kind === 'example'
    ? '例题'
    : question.source_kind === 'exercise'
      ? '习题'
      : '题目'
  const displayedNumber = question.number || question.sequence
  const displayedTitle = question.source_kind === 'example' && String(displayedNumber).startsWith('例')
    ? displayedNumber
    : `第 ${displayedNumber} 题`
  const metadata = paper
    ? `${sourceKindLabel} · ${questionTypeName(question.question_type)}`
    : `${sourceKindLabel} · ${question.section_title || '题目'} · ${questionTypeName(question.question_type)}`
  return (
    <article className="student-bank-question">
      <header className="student-bank-question-head">
        <span className="student-bank-question-number">{question.number || question.sequence}</span>
        <div>
          <small>{metadata}</small>
          <strong>{displayedTitle}</strong>
        </div>
        <div className="student-bank-question-actions">
          <Tag color={question.answer_readiness?.status === 'needs_confirmation' ? 'warning' : 'success'}>
            {question.answer_readiness?.status === 'needs_confirmation' ? '需确认' : '可直接答疑'}
          </Tag>
          <Button type="primary" icon={<MessageSquareText size={14} />} onClick={onAsk}>AI 答疑</Button>
          <Popconfirm
            title="从题库中删除这道题？"
            description="删除后无法恢复，已经生成的作业不受影响。"
            okText="删除"
            cancelText="取消"
            okButtonProps={{ danger: true }}
            onConfirm={onDelete}
          >
            <Button danger type="text" loading={deleting} icon={<Trash2 size={14} />}>删除</Button>
          </Popconfirm>
        </div>
      </header>
      <div className="student-bank-question-body">
        {question.answer_readiness?.reasons?.length ? (
          <div className="student-bank-card-error">
            {question.answer_readiness.reasons.join('；')}
          </div>
        ) : null}
        <MathMarkdown content={question.prompt || '未识别到题干'} />
        {question.subquestions?.length ? (
          <div className="student-bank-subquestions">
            {question.subquestions.map((part) => (
              <div key={part.label}><b>（{part.label}）</b><MathMarkdown content={part.text} /></div>
            ))}
          </div>
        ) : null}
        {question.options?.length ? (
          <div className={`student-bank-options columns-${question.option_columns || 1}`}>
            {question.options.map((option) => (
              <div key={option.label}><b>{option.label}.</b><MathMarkdown content={option.text} /></div>
            ))}
          </div>
        ) : null}
        {question.figures?.length ? (
          <QuestionBankFigureGallery
            figures={question.figures}
            questionNumber={displayedNumber}
          />
        ) : null}
        <QuestionKnowledgeAlignment question={question} />
        {(question.answer || question.answer_subquestions?.length || question.answer_figures?.length) ? (
          <details
            className="student-bank-answer"
            onToggle={(event) => setAnswerOpen(event.currentTarget.open)}
          >
            <summary>查看参考答案</summary>
            {answerOpen ? (
              <>
                {question.answer ? <MathMarkdown content={question.answer} /> : null}
                {question.answer_subquestions?.map((part) => (
                  <div className="student-bank-answer-part" key={part.label}>
                    <b>（{part.label}）</b>
                    <MathMarkdown content={part.text} />
                  </div>
                ))}
                {question.answer_figures?.length ? (
                  <QuestionBankFigureGallery
                    figures={question.answer_figures}
                    questionNumber={displayedNumber}
                    answer
                  />
                ) : null}
              </>
            ) : null}
          </details>
        ) : null}
      </div>
    </article>
  )
}

const QUESTION_BANK_PAGE_SIZE = 20

function PhotoPdfQuestionPickerModal({
  bankId,
  studentId,
  onClose,
  onAsk,
}: {
  bankId: string
  studentId: string
  onClose: () => void
  onAsk: (bank: QuestionBank, question: HomeworkQuestion) => void
}) {
  const [bank, setBank] = useState<QuestionBank>()
  const [questions, setQuestions] = useState<HomeworkQuestion[]>([])
  const [loading, setLoading] = useState(false)
  const [query, setQuery] = useState('')
  const [questionType, setQuestionType] = useState('')
  const [readiness, setReadiness] = useState('')
  const [page, setPage] = useState(1)
  const pageSize = 10

  useEffect(() => {
    if (!bankId) {
      setBank(undefined)
      setQuestions([])
      return
    }
    let cancelled = false
    const load = async () => {
      setLoading(true)
      try {
        const first = await fetchQuestionBank(bankId, 0, 100, studentId)
        if (cancelled) return
        setBank(first)
        if (first.status === 'ready' && first.question_count > first.questions.length) {
          const offsets = Array.from(
            { length: Math.ceil(first.question_count / 100) - 1 },
            (_, index) => (index + 1) * 100,
          )
          const pages = await Promise.all(
            offsets.map((offset) => fetchQuestionBank(bankId, offset, 100, studentId)),
          )
          if (!cancelled) setQuestions([...(first.questions || []), ...pages.flatMap((item) => item.questions || [])])
        } else {
          setQuestions(first.questions || [])
        }
      } finally {
        if (!cancelled) setLoading(false)
      }
    }
    void load()
    const timer = window.setInterval(() => {
      if (bank?.status !== 'ready') void load()
    }, 1800)
    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [bankId, studentId, bank?.status])

  const filtered = questions.filter((question) => {
    const text = `${question.number} ${question.section_title} ${question.prompt}`.toLowerCase()
    return (!query.trim() || text.includes(query.trim().toLowerCase()))
      && (!questionType || question.question_type === questionType)
      && (!readiness || question.answer_readiness?.status === readiness)
  })
  const visible = filtered.slice((page - 1) * pageSize, page * pageSize)
  const questionLayout = groupQuestionBankQuestions(
    questions,
    bank?.source_origin,
    `${bank?.title || ''} ${bank?.source_name || ''}`,
    bank?.document_kind,
  )

  return (
    <Modal
      open={Boolean(bankId)}
      title="PDF 已保存，选择一道题开始答疑"
      onCancel={onClose}
      width={980}
      footer={<Button onClick={onClose}>稍后处理</Button>}
      className="photo-pdf-picker-modal"
      destroyOnHidden
    >
      {!bank || bank.status === 'processing' ? (
        <div className="photo-pdf-processing">
          <LoaderCircle className="spin" size={25} />
          <strong>{bank?.processing_message || '正在创建题库并准备拆题'}</strong>
          <Progress percent={bank?.processing_progress || 1} status="active" />
          <span>可以关闭弹窗稍后处理，PDF 会继续在后台拆题。</span>
        </div>
      ) : bank.status === 'error' ? (
        <div className="student-bank-card-error">{bank.processing_error || 'PDF 拆题失败，请到题库页面重试。'}</div>
      ) : (
        <>
          <div className="photo-pdf-filter-bar">
            <Input allowClear value={query} onChange={(event) => { setQuery(event.target.value); setPage(1) }} placeholder="搜索题号、章节或题干" prefix={<Search size={14} />} />
            <Select allowClear value={questionType || undefined} onChange={(value) => { setQuestionType(value || ''); setPage(1) }} placeholder="题型" options={questionTypeOptionsForStudent} />
            <Select allowClear value={readiness || undefined} onChange={(value) => { setReadiness(value || ''); setPage(1) }} placeholder="识别状态" options={[
              { value: 'ready', label: '可直接答疑' },
              { value: 'needs_confirmation', label: '需要确认' },
            ]} />
          </div>
          {bank.processing_warnings?.length ? (
            <div className="student-bank-card-error">{bank.processing_warnings.slice(0, 3).join('；')}</div>
          ) : null}
          <div className="photo-pdf-question-list">
            {visible.map((question) => (
              <article key={question.id}>
                <div>
                  <Tag>{question.number || question.sequence}</Tag>
                  <Tag>{questionTypeName(question.question_type)}</Tag>
                  <Tag color={question.answer_readiness?.status === 'needs_confirmation' ? 'warning' : 'success'}>
                    {question.answer_readiness?.status === 'needs_confirmation' ? '需确认' : '可直接答疑'}
                  </Tag>
                </div>
                <strong>{questionLayout.paper ? questionTypeName(question.question_type) : question.section_title || '未定位章节'}</strong>
                <MathMarkdown content={question.prompt || '未识别到题干'} />
                <Button type="primary" onClick={() => onAsk(bank, question)}>AI 答疑</Button>
              </article>
            ))}
          </div>
          <Pagination current={page} pageSize={pageSize} total={filtered.length} showSizeChanger={false} onChange={setPage} />
          {loading && <div className="photo-pdf-loading"><Spin size="small" /> 正在刷新题目列表</div>}
        </>
      )}
    </Modal>
  )
}

function QuestionBankView({
  knowledgeBase,
  studentId,
  initialBankId = '',
  onAskQuestion,
}: {
  knowledgeBase: string
  studentId: string
  initialBankId?: string
  onAskQuestion: (bank: QuestionBank, question: HomeworkQuestion) => void
}) {
  const { message: toast } = AntApp.useApp()
  const [banks, setBanks] = useState<QuestionBank[]>([])
  const [loading, setLoading] = useState(true)
  const [selectedBankId, setSelectedBankId] = useState('')
  const [selectedBankDetail, setSelectedBankDetail] = useState<QuestionBank | null>(null)
  const [questionPage, setQuestionPage] = useState(1)
  const [detailLoading, setDetailLoading] = useState(false)
  const [uploadOpen, setUploadOpen] = useState(false)
  const [fileList, setFileList] = useState<UploadFile[]>([])
  const [title, setTitle] = useState('')
  const [uploading, setUploading] = useState(false)
  const [actionId, setActionId] = useState('')
  const [deletingQuestionId, setDeletingQuestionId] = useState('')
  const detailRequestId = useRef(0)
  const openedInitialBankId = useRef('')
  const selectedBankSummary = banks.find((bank) => bank.id === selectedBankId) || null
  const selectedBank = selectedBankDetail?.id === selectedBankId
    ? { ...(selectedBankSummary || selectedBankDetail), ...selectedBankDetail }
    : selectedBankSummary
  const questionLayout = groupQuestionBankQuestions(
    selectedBank?.questions || [],
    selectedBank?.source_origin,
    `${selectedBank?.title || ''} ${selectedBank?.source_name || ''}`,
    selectedBank?.document_kind,
  )

  const loadBanks = useCallback(async (withSpinner = false) => {
    if (withSpinner) setLoading(true)
    try {
      const nextBanks = await fetchQuestionBanks({ includeQuestions: false, studentId })
      setBanks(nextBanks)
      return nextBanks
    } catch (error) {
      toast.error(error instanceof Error ? error.message : '题库读取失败')
      return []
    } finally {
      if (withSpinner) setLoading(false)
    }
  }, [studentId, toast])

  const loadBankPage = useCallback(async (
    bankId: string,
    page: number,
    withSpinner = true,
  ) => {
    const requestId = ++detailRequestId.current
    if (withSpinner) setDetailLoading(true)
    try {
      const detail = await fetchQuestionBank(
        bankId,
        (page - 1) * QUESTION_BANK_PAGE_SIZE,
        QUESTION_BANK_PAGE_SIZE,
        studentId,
      )
      if (requestId === detailRequestId.current) {
        setSelectedBankDetail(detail)
      }
      return detail
    } catch (error) {
      toast.error(error instanceof Error ? error.message : '题库详情读取失败')
      return null
    } finally {
      if (withSpinner && requestId === detailRequestId.current) {
        setDetailLoading(false)
      }
    }
  }, [studentId, toast])

  useEffect(() => {
    void loadBanks(true)
  }, [loadBanks])

  const hasProcessingBank = banks.some((bank) => bank.status === 'processing')
  useEffect(() => {
    if (!hasProcessingBank) return
    const timer = window.setInterval(() => {
      void loadBanks()
      if (
        selectedBankId
        && banks.find((bank) => bank.id === selectedBankId)?.status === 'processing'
      ) {
        void loadBankPage(selectedBankId, questionPage, false)
      }
    }, 2800)
    return () => window.clearInterval(timer)
  }, [banks, hasProcessingBank, loadBankPage, loadBanks, questionPage, selectedBankId])

  const openBank = (bankId: string) => {
    setSelectedBankId(bankId)
    setSelectedBankDetail(null)
    setQuestionPage(1)
    void loadBankPage(bankId, 1)
  }

  useEffect(() => {
    if (!initialBankId || openedInitialBankId.current === initialBankId) return
    openedInitialBankId.current = initialBankId
    setSelectedBankId(initialBankId)
    setSelectedBankDetail(null)
    setQuestionPage(1)
    void loadBankPage(initialBankId, 1)
  }, [initialBankId, loadBankPage])

  const closeBank = () => {
    detailRequestId.current += 1
    setSelectedBankId('')
    setSelectedBankDetail(null)
    setQuestionPage(1)
  }

  const changeQuestionPage = (page: number) => {
    if (!selectedBankId || page === questionPage) return
    setQuestionPage(page)
    void loadBankPage(selectedBankId, page)
  }

  const closeUpload = () => {
    if (uploading) return
    setUploadOpen(false)
    setFileList([])
    setTitle('')
  }

  const uploadBank = async () => {
    const file = fileList[0]?.originFileObj
    if (!file) {
      toast.warning('请先选择 PDF、图片或扫描版习题册')
      return
    }
    setUploading(true)
    try {
      const bank = await createQuestionBank(file, title, knowledgeBase, { studentId })
      toast.success('题库附件已保存，正在提取题目与知识点')
      setUploadOpen(false)
      setFileList([])
      setTitle('')
      setSelectedBankId(bank.id)
      setSelectedBankDetail(null)
      setQuestionPage(1)
      await loadBanks()
      await loadBankPage(bank.id, 1)
    } catch (error) {
      toast.error(error instanceof Error ? error.message : '题库上传失败')
    } finally {
      setUploading(false)
    }
  }

  const runBankAction = async (bankId: string, action: 'retry' | 'cancel' | 'delete') => {
    setActionId(bankId)
    try {
      if (action === 'retry') {
        await reprocessQuestionBank(bankId, studentId)
        toast.success('已重新开始识别题库')
        setQuestionPage(1)
        await loadBankPage(bankId, 1)
      } else if (action === 'cancel') {
        await cancelQuestionBank(bankId, studentId)
        toast.success('已取消建立题库，附件仍保留，可稍后重新识别')
        await loadBankPage(bankId, questionPage, false)
      } else {
        await deleteQuestionBank(bankId, studentId)
        closeBank()
        toast.success('题库已删除')
      }
      await loadBanks()
    } catch (error) {
      toast.error(error instanceof Error ? error.message : '题库操作失败')
    } finally {
      setActionId('')
    }
  }

  const removeQuestion = async (bankId: string, questionId: string) => {
    setDeletingQuestionId(questionId)
    try {
      await deleteQuestionBankQuestion(bankId, questionId, studentId)
      toast.success('题目已从题库删除')
      const nextTotal = Math.max(0, (selectedBank?.question_count || 1) - 1)
      const nextPage = Math.min(
        questionPage,
        Math.max(1, Math.ceil(nextTotal / QUESTION_BANK_PAGE_SIZE)),
      )
      setQuestionPage(nextPage)
      await Promise.all([
        loadBanks(),
        loadBankPage(bankId, nextPage),
      ])
    } catch (error) {
      toast.error(error instanceof Error ? error.message : '题目删除失败')
    } finally {
      setDeletingQuestionId('')
    }
  }

  return (
    <section className="feature-view student-question-bank-view">
      <div className="feature-heading">
        <div>
          <span>QUESTION BANK</span>
          <h1>题库</h1>
          <p>上传习题册并自动提取题目、答案和知识点；每道题都会与当前课程知识图谱对齐。</p>
        </div>
        <div className="student-bank-heading-actions">
          <Button icon={<RefreshCw size={14} />} loading={loading} onClick={() => void loadBanks(true)}>刷新</Button>
          <Button type="primary" icon={<UploadCloud size={15} />} onClick={() => setUploadOpen(true)}>上传题库</Button>
        </div>
      </div>

      <section className="student-bank-library">
        <header>
          <div><LibraryBig size={20} /><span><strong>{banks.length}</strong> 本题库</span></div>
          <small>{banks.reduce((total, bank) => total + bank.question_count, 0)} 道题 · 上传时按所选知识库匹配</small>
        </header>
        {loading && !banks.length ? (
          <div className="workspace-empty"><LoaderCircle className="spin" /><strong>正在读取题库…</strong></div>
        ) : banks.length ? (
          <div className="question-bank-grid">
            {banks.map((bank) => {
              const status = studentQuestionBankStatus[bank.status]
              return (
                <article className="question-bank-card" key={bank.id} onClick={() => openBank(bank.id)}>
                  <div className="question-bank-card-top">
                    <span><BookMarked size={21} /></span>
                    <Tag color={status.color} icon={status.icon}>{status.label}</Tag>
                  </div>
                  <h3>{bank.title}</h3>
                  <p>{bank.source_name}</p>
                  {bank.status === 'processing' ? <Progress percent={bank.processing_progress || 1} showInfo={false} status="active" /> : null}
                  {bank.processing_error ? <div className="student-bank-card-error">{bank.processing_error}</div> : null}
                  <div className="question-bank-card-data">
                    <span><strong>{bank.question_count}</strong> 道题</span>
                    <span><strong>{bank.page_count || '—'}</strong> 页</span>
                  </div>
                  {bank.status === 'processing' ? (
                    <div className="question-bank-card-actions" onClick={(event) => event.stopPropagation()}>
                      <Button size="small" danger icon={<CircleStop size={14} />} loading={actionId === bank.id} onClick={() => void runBankAction(bank.id, 'cancel')}>取消建立</Button>
                    </div>
                  ) : null}
                  <footer><span>{questionBankTime(bank.updated_at)} 更新 · 查看知识对齐</span><ChevronRight size={16} /></footer>
                </article>
              )
            })}
          </div>
        ) : (
          <button className="question-bank-empty" type="button" onClick={() => setUploadOpen(true)}>
            <span><LibraryBig size={27} /></span>
            <strong>建立第一本题库</strong>
            <small>支持 PDF、图片和扫描版习题册</small>
          </button>
        )}
      </section>

      <Modal
        open={uploadOpen}
        onCancel={closeUpload}
        onOk={() => void uploadBank()}
        okText="保存并开始提取"
        cancelText="取消"
        confirmLoading={uploading}
        width={650}
        className="student-bank-upload-modal"
        title={null}
      >
        <div className="student-bank-modal-heading">
          <span><LibraryBig size={23} /></span>
          <div><small>NEW QUESTION BANK</small><h2>上传一本题库</h2><p>系统将提取题干、题图、答案，并匹配课程知识图谱。</p></div>
        </div>
        <div className="student-bank-upload-form">
          <label>
            <span>题库名称</span>
            <Input value={title} onChange={(event) => setTitle(event.target.value)} placeholder="留空时使用附件名称" maxLength={120} />
          </label>
          <div className="student-bank-kb-note"><Database size={14} /> 对齐知识库：{knowledgeBaseDisplayName(undefined, knowledgeBase)}</div>
          <Upload.Dragger
            accept=".pdf,.png,.jpg,.jpeg,.webp,.bmp"
            maxCount={1}
            fileList={fileList}
            beforeUpload={() => false}
            onChange={({ fileList: next }) => setFileList(next.slice(-1))}
          >
            <p className="ant-upload-drag-icon"><LibraryBig size={34} /></p>
            <p className="ant-upload-text">拖入学习指导书、习题册或扫描图片</p>
            <p className="ant-upload-hint">自动提取题号、题目、题图、答案与知识点</p>
          </Upload.Dragger>
        </div>
      </Modal>

      <Modal
        open={Boolean(selectedBankId)}
        onCancel={closeBank}
        footer={null}
        width={1120}
        className="student-bank-detail-modal"
        title={null}
        destroyOnHidden
      >
        {selectedBank ? (
          <div className="student-bank-detail">
            <header className="student-bank-detail-header">
              <div>
                <Tag color={studentQuestionBankStatus[selectedBank.status].color}>{studentQuestionBankStatus[selectedBank.status].label}</Tag>
                <h2>{selectedBank.title}</h2>
                <p>{selectedBank.source_name} · {selectedBank.question_count} 道题 · {selectedBank.page_count || 0} 页</p>
                <small><Network size={12} /> 知识图谱：{knowledgeBaseDisplayName(undefined, selectedBank.knowledge_base)}</small>
              </div>
              <div>
                {selectedBank.source_url ? <Button href={selectedBank.source_url} target="_blank" icon={<Eye size={15} />}>原始附件</Button> : null}
                {selectedBank.status === 'processing' ? (
                  <Button danger icon={<CircleStop size={15} />} loading={actionId === selectedBank.id} onClick={() => void runBankAction(selectedBank.id, 'cancel')}>取消建立</Button>
                ) : null}
                {selectedBank.status === 'error' || selectedBank.status === 'cancelled' ? (
                  <Button type="primary" icon={<RefreshCw size={15} />} loading={actionId === selectedBank.id} onClick={() => void runBankAction(selectedBank.id, 'retry')}>重新识别</Button>
                ) : null}
                <Popconfirm
                  title="删除整本题库？"
                  description="题库文件和题目将永久删除。"
                  okText="删除"
                  cancelText="取消"
                  okButtonProps={{ danger: true }}
                  onConfirm={() => void runBankAction(selectedBank.id, 'delete')}
                >
                  <Button danger icon={<Trash2 size={15} />}>删除题库</Button>
                </Popconfirm>
              </div>
            </header>
            {selectedBank.status === 'processing' ? (
              <div className="homework-processing-panel">
                <LoaderCircle className="spin" size={30} />
                <div><strong>{selectedBank.processing_message || '正在逐页提取题库内容'}</strong><span>进度 {selectedBank.processing_progress || 0}% · 完成后将自动显示知识点匹配程度。</span></div>
              </div>
            ) : null}
            {selectedBank.status === 'ready' ? (
              <div className="student-bank-alignment-hint">
                <Network size={15} />
                <span>知识对齐显示在每道题的题图下方；百分比表示题目知识点与课程知识图谱的匹配程度。</span>
              </div>
            ) : null}
            {selectedBank.processing_error ? (
              <div className="homework-detail-error"><AlertTriangle size={18} /><div><strong>题库识别未完成</strong><span>{selectedBank.processing_error}</span></div></div>
            ) : null}
            {detailLoading && !selectedBankDetail ? (
              <div className="student-bank-detail-loading">
                <LoaderCircle className="spin" size={24} />
                <strong>正在加载第 {questionPage} 页题目…</strong>
              </div>
            ) : selectedBank.status === 'ready' && selectedBank.questions.length ? (
              <>
                <div className={`student-bank-question-list ${detailLoading ? 'is-loading' : ''}`}>
                  {questionLayout.grouped ? questionLayout.groups.map((group) => (
                    <section className="student-bank-chapter-group" key={group.key}>
                      <header className="student-bank-chapter-heading">
                        <div><BookOpen size={16} /><strong>{group.title}</strong></div>
                        <span>{group.questions.length} 道题</span>
                      </header>
                      <div className="student-bank-chapter-questions">
                        {group.questions.map((question) => (
                          <QuestionBankQuestionCard
                            key={question.id}
                            question={question}
                            paper={questionLayout.paper}
                            deleting={deletingQuestionId === question.id}
                            onDelete={() => void removeQuestion(selectedBank.id, question.id)}
                            onAsk={() => onAskQuestion(selectedBank, question)}
                          />
                        ))}
                      </div>
                    </section>
                  )) : selectedBank.questions.map((question) => (
                    <QuestionBankQuestionCard
                      key={question.id}
                      question={question}
                      paper={questionLayout.paper}
                      deleting={deletingQuestionId === question.id}
                      onDelete={() => void removeQuestion(selectedBank.id, question.id)}
                      onAsk={() => onAskQuestion(selectedBank, question)}
                    />
                  ))}
                </div>
                {selectedBank.question_count > QUESTION_BANK_PAGE_SIZE ? (
                  <div className="student-bank-pagination">
                    <Pagination
                      current={questionPage}
                      pageSize={QUESTION_BANK_PAGE_SIZE}
                      total={selectedBank.question_count}
                      showSizeChanger={false}
                      showTotal={(total, range) => `第 ${range[0]}-${range[1]} 题，共 ${total} 题`}
                      onChange={changeQuestionPage}
                    />
                  </div>
                ) : null}
              </>
            ) : selectedBank.status === 'ready' && !detailLoading ? (
              <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="暂未提取到可用题目" />
            ) : null}
          </div>
        ) : null}
      </Modal>
    </section>
  )
}

function MistakeBookView({
  mistakes,
  categories,
  analysis,
  studentId,
  onDelete,
  onPlan,
  onChanged,
}: {
  mistakes: MistakeItem[]
  categories: MistakeCategory[]
  analysis?: MistakeAnalysis
  studentId: string
  onDelete: (id: string) => void
  onPlan: () => void
  onChanged: () => Promise<void>
}) {
  const { message: toast } = AntApp.useApp()
  const [selectedId, setSelectedId] = useState('')
  const [sourceFilter, setSourceFilter] = useState<'all' | MistakeSource>('all')
  const [categoryFilter, setCategoryFilter] = useState('all')
  const [newCategory, setNewCategory] = useState('')
  const [renamingCategoryId, setRenamingCategoryId] = useState('')
  const [categoryName, setCategoryName] = useState('')
  const [titleDraft, setTitleDraft] = useState('')
  const [annotationDraft, setAnnotationDraft] = useState('')
  const [annotationRequestId, setAnnotationRequestId] = useState('')
  const [editingAnnotationId, setEditingAnnotationId] = useState('')
  const [editingAnnotationContent, setEditingAnnotationContent] = useState('')
  const [savingAction, setSavingAction] = useState('')
  const selectedMistake = mistakes.find((item) => item.id === selectedId) || null
  const visibleMistakes = useMemo(() => mistakes.filter((item) => (
    (sourceFilter === 'all' || item.source === sourceFilter)
    && (categoryFilter === 'all' || item.category_id === categoryFilter)
  )), [mistakes, sourceFilter, categoryFilter])

  const runAction = async (key: string, action: () => Promise<void>, success: string) => {
    if (savingAction) return
    setSavingAction(key)
    try {
      await action()
      await onChanged()
      toast.success(success)
    } catch (error) {
      toast.error(error instanceof Error ? error.message : '操作失败，请稍后重试')
    } finally {
      setSavingAction('')
    }
  }

  const createCategory = async () => {
    const name = newCategory.trim()
    if (!name) return
    await runAction('category-create', async () => {
      await createMistakeCategory(studentId, name)
      setNewCategory('')
    }, '错题分类已创建')
  }

  const renameCategory = async () => {
    const name = categoryName.trim()
    if (!renamingCategoryId || !name) return
    await runAction('category-rename', async () => {
      await renameMistakeCategory(studentId, renamingCategoryId, name)
      setRenamingCategoryId('')
      setCategoryName('')
    }, '错题分类已重命名')
  }

  const saveAnnotation = async () => {
    if (!selectedMistake || !annotationDraft.trim()) return
    const requestId = annotationRequestId || `annotation-${Date.now().toString(36)}`
    setAnnotationRequestId(requestId)
    await runAction('annotation-create', async () => {
      await addMistakeAnnotation(studentId, selectedMistake.id, annotationDraft.trim(), requestId)
      setAnnotationDraft('')
      setAnnotationRequestId('')
    }, '批注已保存')
  }

  return (
    <section className="feature-view mistakes-view">
      <div className="feature-heading">
        <div><span>MISTAKE REVIEW</span><h1>错题本</h1><p>按来源和分类复盘错题，结合课程图谱定位章节与前置知识。</p></div>
        <Button type="primary" icon={<BrainCircuit size={16} />} onClick={onPlan} disabled={!mistakes.length}>生成知识补全规划</Button>
      </div>
      {analysis && analysis.total_mistakes > 0 && (
        <section className="mistake-analysis-panel">
          <div className="mistake-analysis-head">
            <div><span>STRUCTURED REVIEW PLAN</span><h2>总体学习规划</h2></div>
            <p className={analysis.data_sufficient ? '' : 'limited'}>{analysis.notice}</p>
          </div>
          <div className="mistake-analysis-grid">
            {analysis.recommended_order.slice(0, 4).map((area, index) => (
              <article key={area.knowledge_point}>
                <span>优先级 {index + 1}</span>
                <strong><InlineMath content={area.knowledge_point} /></strong>
                <Tag color={area.severity === '重度薄弱' ? 'red' : area.severity === '中度薄弱' ? 'orange' : 'blue'}>{area.severity}</Tag>
                <p>{area.chapter}{area.section !== '暂未确定' ? ` · ${area.section}` : ''}</p>
                <small>{area.mistake_count} 道关联错题{area.prerequisites.length ? ` · 先复习 ${area.prerequisites.map((item) => item.name).join('、')}` : ''}</small>
              </article>
            ))}
          </div>
        </section>
      )}
      <div className="mistake-toolbar">
        <Select
          value={sourceFilter}
          onChange={setSourceFilter}
          options={[
            { value: 'all', label: '全部来源' },
            ...Object.entries(mistakeSourceLabels).map(([value, label]) => ({ value, label })),
          ]}
        />
        <Select
          value={categoryFilter}
          onChange={setCategoryFilter}
          options={[{ value: 'all', label: '全部分类' }, ...categories.map((item) => ({ value: item.id, label: item.name }))]}
        />
        <Input
          value={newCategory}
          onChange={(event) => setNewCategory(event.target.value)}
          onPressEnter={() => void createCategory()}
          placeholder="新分类名称"
          maxLength={40}
        />
        <Button icon={<Plus size={14} />} loading={savingAction === 'category-create'} onClick={() => void createCategory()}>新建分类</Button>
      </div>
      {categories.some((item) => item.id !== 'uncategorized') && (
        <div className="mistake-category-manager">
          {categories.filter((item) => item.id !== 'uncategorized').map((category) => (
            <div key={category.id}>
              {renamingCategoryId === category.id ? (
                <Input
                  size="small"
                  autoFocus
                  value={categoryName}
                  onChange={(event) => setCategoryName(event.target.value)}
                  onPressEnter={() => void renameCategory()}
                  maxLength={40}
                  suffix={<button aria-label="保存分类名称" onClick={() => void renameCategory()}><Check size={13} /></button>}
                />
              ) : (
                <><span>{category.name}</span><button aria-label="重命名分类" onClick={() => { setRenamingCategoryId(category.id); setCategoryName(category.name) }}><Pencil size={12} /></button></>
              )}
              <Popconfirm title="删除分类后，错题将移入未分类。" okText="删除" cancelText="取消" onConfirm={() => void runAction('category-delete', async () => {
                await deleteMistakeCategory(studentId, category.id)
                if (categoryFilter === category.id) setCategoryFilter('all')
              }, '错题分类已删除')}>
                <button aria-label="删除分类"><Trash2 size={12} /></button>
              </Popconfirm>
            </div>
          ))}
        </div>
      )}
      {visibleMistakes.length ? (
        <div className="mistake-grid">
          {visibleMistakes.map((item) => (
            <article className="mistake-card" key={item.id}>
              <div className="mistake-card-head">
                <span>
                  {mistakeSourceLabels[item.source] || '用户上传'} · {item.agent}
                  {item.decision?.reason ? ` · ${mistakeReasonLabels[item.decision.reason]}` : ''}
                </span>
                <small>{new Date(item.updated_at || item.created_at).toLocaleDateString('zh-CN')}</small>
              </div>
              <div className="mistake-title"><MathMarkdown content={item.title || item.summary} /></div>
              <div className="mistake-points">
                {(item.knowledge_tags?.length ? item.knowledge_tags : item.knowledge_points.map((point) => ({ tag_id: point, tag_name: point, match_type: 'unmatched' as const, confidence: 0 }))).map((tag) => (
                  <Tooltip key={tag.tag_id} title={`${tag.match_type === 'exact' ? '图谱精确匹配' : tag.match_type === 'approximate' ? '图谱近似匹配' : '独立标签'} · 置信度 ${Math.round(tag.confidence * 100)}%`}>
                    <Tag color={tag.match_type === 'exact' ? 'green' : tag.match_type === 'approximate' ? 'gold' : 'default'}><InlineMath content={tag.tag_name} /></Tag>
                  </Tooltip>
                ))}
              </div>
              <p className="mistake-location">{item.location?.chapter || '暂未确定'} · {item.location?.section || '暂未确定'}</p>
              {item.attempt?.score != null && item.attempt.max_score != null && (
                <p className="mistake-location">本次作答：{item.attempt.score} / {item.attempt.max_score} 分</p>
              )}
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
                <Select
                  size="small"
                  value={item.category_id || 'uncategorized'}
                  options={categories.map((category) => ({ value: category.id, label: category.name }))}
                  onChange={(categoryId) => void runAction(`move-${item.id}`, async () => { await updateMistake(studentId, item.id, { category_id: categoryId }) }, '错题分类已更新')}
                />
                <button className="mistake-open" onClick={() => { setSelectedId(item.id); setTitleDraft(item.title || item.summary) }}>
                  <BookOpen size={14} /> 查看题目与答案
                </button>
                <Popconfirm title="从错题本删除？" okText="删除" cancelText="取消" onConfirm={() => onDelete(item.id)}>
                  <button className="mistake-delete"><Trash2 size={14} /> 删除</button>
                </Popconfirm>
              </div>
            </article>
          ))}
        </div>
      ) : <div className="workspace-empty"><Layers3 size={30} /><strong>{mistakes.length ? '当前筛选没有错题' : '错题本还是空的'}</strong><p>{mistakes.length ? '可切换来源或分类查看其他错题。' : '在答疑或出题结果旁点击“加入错题本”，系统会自动识别知识点。'}</p></div>}
      <Modal
        open={Boolean(selectedMistake)}
        title={selectedMistake
          ? <div className="mistake-modal-title"><MathMarkdown content={selectedMistake.title || selectedMistake.summary} /></div>
          : '错题详情'}
        width={860}
        onCancel={() => setSelectedId('')}
        footer={<Button onClick={() => setSelectedId('')}>关闭</Button>}
      >
        {selectedMistake && (
          <div className="mistake-detail">
            <section className="mistake-detail-overview">
              <span className="mistake-detail-label">归档信息</span>
              <div className="mistake-title-editor">
                <Input value={titleDraft} maxLength={120} onChange={(event) => setTitleDraft(event.target.value)} />
                <Button loading={savingAction === 'title'} onClick={() => void runAction('title', async () => { await updateMistake(studentId, selectedMistake.id, { title: titleDraft.trim() }) }, '错题名称已更新')}>保存名称</Button>
              </div>
              <p>来源：{mistakeSourceLabels[selectedMistake.source] || '用户上传'}</p>
              <p>加入原因：{selectedMistake.decision?.reason ? mistakeReasonLabels[selectedMistake.decision.reason] : '历史记录未标注'}</p>
              {selectedMistake.attempt?.score != null && selectedMistake.attempt.max_score != null && (
                <p>本次作答：{selectedMistake.attempt.score} / {selectedMistake.attempt.max_score} 分</p>
              )}
              <p>章节：{selectedMistake.location?.chapter || '暂未确定'}；小节：{selectedMistake.location?.section || '暂未确定'}</p>
              <p>建议先复习：{selectedMistake.prerequisites?.length
                ? selectedMistake.prerequisites.map((item) => `${item.name}（${item.source === 'knowledge_graph' ? '图谱关系' : '章节顺序推断'}）`).join('、')
                : '暂时无法判断'}</p>
            </section>
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
            {selectedMistake.messages?.length > 2 && (
              <section>
                <span className="mistake-detail-label">完整归档对话</span>
                <div className="mistake-dialogue">
                  {selectedMistake.messages.map((message, index) => (
                    <article key={`${message.role}-${index}`} className={message.role}>
                      <small>{message.role === 'user' ? '学生' : message.agent || 'AI 助教'}</small>
                      <MathMarkdown content={message.content} />
                    </article>
                  ))}
                </div>
              </section>
            )}
            <section>
              <span className="mistake-detail-label">我的批注</span>
              <div className="mistake-annotation-list">
                {selectedMistake.annotations?.map((annotation) => (
                  <article key={annotation.id}>
                    {editingAnnotationId === annotation.id ? (
                      <TextArea
                        value={editingAnnotationContent}
                        maxLength={4000}
                        showCount
                        autoSize={{ minRows: 3, maxRows: 8 }}
                        onChange={(event) => setEditingAnnotationContent(event.target.value)}
                      />
                    ) : <MathMarkdown content={annotation.content} />}
                    <div>
                      {editingAnnotationId === annotation.id ? (
                        <>
                          <Button
                            size="small"
                            loading={savingAction === 'annotation-update'}
                            onClick={() => void runAction('annotation-update', async () => {
                              await updateMistakeAnnotation(
                                studentId,
                                selectedMistake.id,
                                annotation.id,
                                editingAnnotationContent.trim(),
                              )
                              setEditingAnnotationId('')
                            }, '批注已更新')}
                          >保存</Button>
                          <Button size="small" onClick={() => setEditingAnnotationId('')}>取消</Button>
                        </>
                      ) : (
                        <Button size="small" onClick={() => { setEditingAnnotationId(annotation.id); setEditingAnnotationContent(annotation.content) }}>编辑</Button>
                      )}
                      <Popconfirm
                        title="删除这条批注？"
                        okText="删除"
                        cancelText="取消"
                        onConfirm={() => void runAction('annotation-delete', async () => {
                          await deleteMistakeAnnotation(studentId, selectedMistake.id, annotation.id)
                        }, '批注已删除')}
                      >
                        <Button size="small" danger>删除</Button>
                      </Popconfirm>
                    </div>
                  </article>
                ))}
                {!selectedMistake.annotations?.length && (
                  <p className="mistake-annotation-empty">还没有批注，可记录错误原因、正确思路或复习提醒。</p>
                )}
              </div>
              <TextArea
                value={annotationDraft}
                maxLength={4000}
                showCount
                autoSize={{ minRows: 3, maxRows: 8 }}
                onChange={(event) => { setAnnotationDraft(event.target.value); setAnnotationRequestId('') }}
                placeholder="记录错误原因、解题反思、正确思路、易错点或后续复习提示…"
              />
              <Button
                type="primary"
                loading={savingAction === 'annotation-create'}
                disabled={!annotationDraft.trim()}
                onClick={() => void saveAnnotation()}
              >保存批注</Button>
            </section>
          </div>
        )}
      </Modal>
    </section>
  )
}

function ScheduleView({
  items,
  onAdd,
  onToggle,
  onDelete,
}: {
  items: ScheduleItem[]
  onAdd: (draft: ScheduleItemDraft) => Promise<boolean>
  onToggle: (item: ScheduleItem) => void
  onDelete: (id: string) => void
}) {
  const todayKey = localDateKey()
  const today = new Date()
  const [visibleMonth, setVisibleMonth] = useState(() => new Date(today.getFullYear(), today.getMonth(), 1))
  const [selectedDate, setSelectedDate] = useState(todayKey)
  const [addOpen, setAddOpen] = useState(false)
  const [adding, setAdding] = useState(false)
  const [draft, setDraft] = useState<ScheduleItemDraft>({
    title: '',
    date: todayKey,
    time: '',
    category: 'study',
    note: '',
  })
  const itemsByDate = useMemo(() => {
    const grouped = new Map<string, ScheduleItem[]>()
    items.forEach((item) => grouped.set(item.date, [...(grouped.get(item.date) || []), item]))
    return grouped
  }, [items])
  const calendarDays = useMemo(() => {
    const year = visibleMonth.getFullYear()
    const month = visibleMonth.getMonth()
    const mondayOffset = (new Date(year, month, 1).getDay() + 6) % 7
    return Array.from({ length: 42 }, (_, index) => new Date(year, month, index - mondayOffset + 1))
  }, [visibleMonth])
  const selectedItems = itemsByDate.get(selectedDate) || []
  const monthPrefix = `${visibleMonth.getFullYear()}-${String(visibleMonth.getMonth() + 1).padStart(2, '0')}`
  const monthItems = items.filter((item) => item.date.startsWith(monthPrefix))
  const monthPending = monthItems.filter((item) => !item.completed).length
  const selectedDateObject = (() => {
    const [year, month, day] = selectedDate.split('-').map(Number)
    return new Date(year, month - 1, day)
  })()

  const openAdd = (date = selectedDate) => {
    setDraft({ title: '', date, time: '', category: 'study', note: '' })
    setAddOpen(true)
  }

  const submitDraft = async () => {
    if (!draft.title.trim() || !draft.date) return
    setAdding(true)
    const added = await onAdd({ ...draft, title: draft.title.trim(), note: draft.note.trim() })
    setAdding(false)
    if (added) {
      const [year, month] = draft.date.split('-').map(Number)
      setSelectedDate(draft.date)
      setVisibleMonth(new Date(year, month - 1, 1))
      setAddOpen(false)
    }
  }

  const changeMonth = (offset: number) => {
    const next = new Date(visibleMonth.getFullYear(), visibleMonth.getMonth() + offset, 1)
    setVisibleMonth(next)
  }

  const backToToday = () => {
    const now = new Date()
    setVisibleMonth(new Date(now.getFullYear(), now.getMonth(), 1))
    setSelectedDate(localDateKey(now))
  }

  return (
    <section className="feature-view schedule-view">
      <div className="feature-heading schedule-feature-heading">
        <div><span>STUDY PLANNER</span><h1>学习日历</h1></div>
        <Button type="primary" icon={<Plus size={16} />} onClick={() => openAdd()}>添加安排</Button>
      </div>
      <div className="schedule-overview">
        <div className="schedule-overview-copy">
          <span className="schedule-overview-icon"><CalendarCheck2 size={24} /></span>
          <div><strong>本月计划</strong></div>
        </div>
        <div className="schedule-overview-progress">
          <strong>{monthItems.length ? Math.round(((monthItems.length - monthPending) / monthItems.length) * 100) : 0}<small>%</small></strong>
          <span>完成进度</span>
        </div>
      </div>
      <div className="schedule-layout">
        <div className="calendar-card">
          <header className="calendar-toolbar">
            <div>
              <button type="button" onClick={() => changeMonth(-1)} aria-label="上个月"><ChevronLeft size={18} /></button>
              <strong>{visibleMonth.getFullYear()}年 {visibleMonth.getMonth() + 1}月</strong>
              <button type="button" onClick={() => changeMonth(1)} aria-label="下个月"><ChevronRight size={18} /></button>
            </div>
            <button className="calendar-today-button" type="button" onClick={backToToday}>回到今天</button>
          </header>
          <div className="calendar-weekdays" aria-hidden="true">
            {['一', '二', '三', '四', '五', '六', '日'].map((day) => <span key={day}>周{day}</span>)}
          </div>
          <div className="calendar-grid">
            {calendarDays.map((date) => {
              const key = localDateKey(date)
              const dayItems = itemsByDate.get(key) || []
              const inMonth = date.getMonth() === visibleMonth.getMonth()
              return (
                <button
                  type="button"
                  key={key}
                  className={`calendar-day ${inMonth ? '' : 'outside'} ${key === todayKey ? 'today' : ''} ${key === selectedDate ? 'selected' : ''}`}
                  onClick={() => setSelectedDate(key)}
                  onDoubleClick={() => openAdd(key)}
                  aria-label={`${date.toLocaleDateString('zh-CN')}，${dayItems.length} 项安排`}
                >
                  <span className="calendar-day-number">{date.getDate()}</span>
                  {dayItems.length > 0 && (
                    <span className="calendar-day-events">
                      <span className="calendar-event-dots">
                        {dayItems.slice(0, 3).map((item) => <i className={`category-${item.category}`} key={item.id} />)}
                      </span>
                      <small>{dayItems[0].title}</small>
                      {dayItems.length > 1 && <b>+{dayItems.length - 1}</b>}
                    </span>
                  )}
                </button>
              )
            })}
          </div>
        </div>
        <aside className="day-schedule-panel">
          <header>
            <div>
              <span>{selectedDate === todayKey ? '今天' : selectedDateObject.toLocaleDateString('zh-CN', { weekday: 'long' })}</span>
              <h2>{selectedDateObject.toLocaleDateString('zh-CN', { month: 'long', day: 'numeric' })}</h2>
            </div>
            <button type="button" onClick={() => openAdd()} aria-label="为所选日期添加安排"><Plus size={18} /></button>
          </header>
          {selectedItems.length ? (
            <div className="day-schedule-list">
              {selectedItems.map((item) => (
                <article className={`day-schedule-item category-${item.category} ${item.completed ? 'completed' : ''}`} key={item.id}>
                  <button type="button" className="schedule-check" onClick={() => onToggle(item)} aria-label={`${item.completed ? '恢复' : '完成'} ${item.title}`}>
                    {item.completed ? <CheckCircle2 size={18} /> : <Circle size={18} />}
                  </button>
                  <span className="day-schedule-category">{scheduleCategoryIcon(item.category, 14)}</span>
                  <div>
                    <div><small>{item.time || '全天'} · {scheduleCategoryLabels[item.category]}</small></div>
                    <strong><InlineMath content={item.title} /></strong>
                    {item.note && <MathMarkdown content={item.note} />}
                  </div>
                  <Popconfirm title="删除这项安排？" okText="删除" cancelText="取消" onConfirm={() => onDelete(item.id)}>
                    <button type="button" className="schedule-delete" aria-label={`删除 ${item.title}`}><Trash2 size={14} /></button>
                  </Popconfirm>
                </article>
              ))}
            </div>
          ) : (
            <button type="button" className="day-schedule-empty" onClick={() => openAdd()}>
              <span><CalendarDays size={24} /></span>
              <strong>这一天还是空白</strong>
              <small>点击添加一项值得期待的安排</small>
            </button>
          )}
          <div className="schedule-legend">
            {(Object.keys(scheduleCategoryLabels) as ScheduleCategory[]).map((category) => <span key={category}><i className={`category-${category}`} />{scheduleCategoryLabels[category]}</span>)}
          </div>
        </aside>
      </div>
      <Modal
        open={addOpen}
        onCancel={() => setAddOpen(false)}
        footer={null}
        title={null}
        width={500}
        className="schedule-modal"
      >
        <div className="modal-heading schedule-modal-heading">
          <span className="modal-icon"><CalendarCheck2 size={22} /></span>
          <div><h2>添加新的安排</h2><p>记下重要的事，为当天留出时间。</p></div>
        </div>
        <div className="schedule-form">
          <label>
            <span>安排名称</span>
            <Input value={draft.title} maxLength={120} autoFocus placeholder="例如：模拟电路期中考试" onChange={(event) => setDraft((value) => ({ ...value, title: event.target.value }))} onPressEnter={() => void submitDraft()} />
          </label>
          <div className="schedule-form-row">
            <label><span>日期</span><Input type="date" value={draft.date} onChange={(event) => setDraft((value) => ({ ...value, date: event.target.value }))} /></label>
            <label><span>时间（可选）</span><Input type="time" value={draft.time} onChange={(event) => setDraft((value) => ({ ...value, time: event.target.value }))} /></label>
          </div>
          <label>
            <span>类型</span>
            <Select
              value={draft.category}
              onChange={(category) => setDraft((value) => ({ ...value, category }))}
              options={(Object.keys(scheduleCategoryLabels) as ScheduleCategory[]).map((category) => ({ value: category, label: scheduleCategoryLabels[category] }))}
              style={{ width: '100%' }}
            />
          </label>
          <label>
            <span>备注（可选）</span>
            <TextArea value={draft.note} maxLength={500} autoSize={{ minRows: 3, maxRows: 5 }} placeholder="地点、需要携带的物品或提醒……" onChange={(event) => setDraft((value) => ({ ...value, note: event.target.value }))} />
          </label>
          <div className="schedule-form-actions">
            <Button onClick={() => setAddOpen(false)}>取消</Button>
            <Button type="primary" icon={<CalendarCheck2 size={16} />} loading={adding} disabled={!draft.title.trim() || !draft.date} onClick={() => void submitDraft()}>加入日历</Button>
          </div>
        </div>
      </Modal>
    </section>
  )
}

const explanationStatusLabels: Record<KnowledgeExplanation['status'], string> = {
  planning: '规划大纲',
  generating: '逐页生成',
  completed: '生成完成',
  cancelled: '已取消',
  error: '生成失败',
}

type KnowledgeImageModel = 'qwen-image-2.0' | 'qwen-image-2.0-pro'

const knowledgeImageModelLabels: Record<KnowledgeImageModel, string> = {
  'qwen-image-2.0': 'QWEN IMAGE 2.0',
  'qwen-image-2.0-pro': 'QWEN IMAGE 2.0 PRO',
}

function knowledgeImageModelLabel(value: string) {
  return value === 'qwen-image-2.0-pro'
    ? knowledgeImageModelLabels['qwen-image-2.0-pro']
    : knowledgeImageModelLabels['qwen-image-2.0']
}

function explanationTime(value: string) {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return ''
  return date.toLocaleString('zh-CN', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  })
}

function explanationPageDisplayStatus(
  page: KnowledgeExplanation['pages'][number],
  explanationStatus: KnowledgeExplanation['status'],
) {
  if (page.status === 'ready') return 'ready'
  if (explanationStatus === 'cancelled') return 'cancelled'
  if (explanationStatus === 'error') return 'error'
  return page.status
}

function KnowledgeExplanationHistoryItem({
  item,
  deleting,
  onSelect,
  onDelete,
}: {
  item: KnowledgeExplanation
  deleting: boolean
  onSelect: (value: KnowledgeExplanation) => void
  onDelete: (value: KnowledgeExplanation) => Promise<void>
}) {
  const active = item.status === 'planning' || item.status === 'generating'
  return (
    <div className="explanation-history-item">
      <button className="explanation-history-open" type="button" onClick={() => onSelect(item)}>
        <span className={`history-lesson-icon ${item.status}`}><Presentation size={18} /></span>
        <span><strong><InlineMath content={item.title || item.question} /></strong><small>{item.page_count || '—'} 页 · {explanationTime(item.created_at)}</small></span>
        <ChevronRight size={15} />
      </button>
      <Popconfirm
        title={active ? '停止并删除这组讲解？' : '删除这组讲解？'}
        description={active ? '正在进行的生成会先停止，已生成页面和记录将永久删除。' : '讲解记录和全部生成页面将永久删除。'}
        okText={active ? '停止并删除' : '删除'}
        cancelText="取消"
        okButtonProps={{ danger: true }}
        onConfirm={() => onDelete(item)}
      >
        <Button
          className="explanation-history-delete"
          type="text"
          danger
          loading={deleting}
          icon={<Trash2 size={13} />}
          aria-label={`删除知识讲解 ${item.title || item.question}`}
        />
      </Popconfirm>
    </div>
  )
}

function KnowledgeExplanationView({
  explanation,
  history,
  pageCount,
  imageModel,
  onPageCountChange,
  onImageModelChange,
  onSelect,
  onBack,
  onDelete,
  deletingId,
}: {
  explanation?: KnowledgeExplanation
  history: KnowledgeExplanation[]
  pageCount: number | null
  imageModel: KnowledgeImageModel
  onPageCountChange: (value: number | null) => void
  onImageModelChange: (value: KnowledgeImageModel) => void
  onSelect: (value: KnowledgeExplanation) => void
  onBack: () => void
  onDelete: (value: KnowledgeExplanation) => Promise<void>
  deletingId: string
}) {
  const [selectedPage, setSelectedPage] = useState(1)
  const [pageCountMode, setPageCountMode] = useState<number | 'custom'>(
    pageCount !== null && [0, 4, 6, 8].includes(pageCount) ? pageCount : 'custom',
  )
  const [customPageCount, setCustomPageCount] = useState<number | null>(
    pageCount !== null && ![0, 4, 6, 8].includes(pageCount) ? pageCount : null,
  )
  useEffect(() => setSelectedPage(1), [explanation?.id])

  const pages = explanation?.pages || []
  const activePage = pages.find((page) => page.index === selectedPage) || pages[0]
  const readyCount = pages.filter((page) => page.status === 'ready').length
  const isActive = explanation?.status === 'planning' || explanation?.status === 'generating'
  const activePageStatus = activePage && explanation
    ? explanationPageDisplayStatus(activePage, explanation.status)
    : 'pending'
  const activePageBusy = isActive && (activePageStatus === 'drawing' || activePageStatus === 'writing')
  const activePageMessage = activePageStatus === 'cancelled'
    ? '生成已取消'
    : activePageStatus === 'error'
      ? '本页未能生成'
      : activePage?.message || '正在准备页面结构…'
  const customPageCountInvalid = customPageCount !== null
    && (customPageCount < 1 || customPageCount > 8)

  const changePageCountMode = (value: number | 'custom') => {
    setPageCountMode(value)
    onPageCountChange(value === 'custom' ? customPageCount : value)
  }

  const changeCustomPageCount = (value: number | null) => {
    const nextValue = value === null ? null : Math.round(value)
    setCustomPageCount(nextValue)
    onPageCountChange(nextValue)
  }

  if (!explanation) {
    return (
      <div className="knowledge-explanation-scroll">
        <section className="knowledge-explanation-empty">
          <div className="explanation-orbit explanation-orbit-a" />
          <div className="explanation-orbit explanation-orbit-b" />
          <span className="explanation-model-kicker"><WandSparkles size={14} /> {knowledgeImageModelLabels[imageModel]} · VISUAL LESSON</span>
          <div className="explanation-empty-icon"><Presentation size={38} /></div>
          <h1>把一个问题，讲成一组好懂的页面</h1>
          <p>AI 会先理解问题的知识结构，再动态规划大纲，并生成紧凑、图文并茂的 16:9 中文讲解页。</p>
          <div className="explanation-process">
            <span><b>01</b><small>分析问题</small></span>
            <ChevronRight size={16} />
            <span><b>02</b><small>规划模块</small></span>
            <ChevronRight size={16} />
            <span><b>03</b><small>逐页绘制</small></span>
          </div>
          <div className="explanation-count-control">
            <div className="explanation-control-copy">
              <strong>讲解页数</strong>
              <small>推荐自动规划，让内容结构随问题变化</small>
            </div>
            <div className="explanation-count-picker">
              <Segmented<number | 'custom'>
                value={pageCountMode}
                onChange={changePageCountMode}
                options={[
                  { label: '自动', value: 0 },
                  { label: '4 页', value: 4 },
                  { label: '6 页', value: 6 },
                  { label: '8 页', value: 8 },
                  { label: '自定义', value: 'custom' },
                ]}
              />
              {pageCountMode === 'custom' && (
                <div className="explanation-custom-field">
                  <InputNumber
                    className="explanation-custom-count"
                    size="small"
                    precision={0}
                    controls={false}
                    value={customPageCount}
                    placeholder="1–8"
                    status={customPageCountInvalid ? 'error' : undefined}
                    addonAfter="页"
                    aria-label="自定义讲解页数，范围一到八页"
                    onChange={changeCustomPageCount}
                  />
                  {customPageCountInvalid && <span>请输入 1–8 页</span>}
                </div>
              )}
            </div>
          </div>
          <div className="explanation-count-control explanation-model-control">
            <div className="explanation-control-copy">
              <strong>生图模型</strong>
              <small>标准版生成更快，Pro 版强化文字渲染与语义遵循</small>
            </div>
            <Segmented<KnowledgeImageModel>
              value={imageModel}
              onChange={onImageModelChange}
              options={[
                { label: 'Qwen Image 2.0', value: 'qwen-image-2.0' },
                { label: 'Qwen Image 2.0 Pro', value: 'qwen-image-2.0-pro' },
              ]}
            />
          </div>
          <div className="explanation-example-row">
            <span>试着提问</span>
            <em>香农定理为什么限制了通信速率？</em>
            <em>傅里叶变换的物理意义是什么？</em>
            <em>PN 结为何单向导电？</em>
          </div>
        </section>
        {history.length > 0 && (
          <section className="explanation-history">
            <header><div><Clock3 size={15} /><strong>最近生成</strong></div><span>{history.length} 组讲解</span></header>
            <div className="explanation-history-grid">
              {history.map((item) => (
                <KnowledgeExplanationHistoryItem
                  key={item.id}
                  item={item}
                  deleting={deletingId === item.id}
                  onSelect={onSelect}
                  onDelete={onDelete}
                />
              ))}
            </div>
          </section>
        )}
      </div>
    )
  }

  return (
    <div className="knowledge-explanation-scroll">
      <section className="explanation-workbench">
        <header className="explanation-workbench-head">
          <div>
            <span className="explanation-model-kicker"><WandSparkles size={13} /> {knowledgeImageModelLabel(explanation.image_model)}</span>
            <h1><InlineMath content={explanation.title || '正在规划知识讲解…'} /></h1>
            <p><InlineMath content={explanation.subtitle || explanation.question} /></p>
          </div>
          <div className="explanation-head-meta">
            <Button
              className="explanation-back-to-create"
              type="text"
              icon={<ChevronLeft size={14} />}
              onClick={onBack}
            >新建讲解</Button>
            <Tag color={explanation.status === 'completed' ? 'green' : explanation.status === 'error' ? 'red' : explanation.status === 'cancelled' ? 'default' : 'blue'}>
              {isActive && <LoaderCircle className="spin" size={12} />}
              {explanationStatusLabels[explanation.status]}
            </Tag>
            <span>{readyCount}/{explanation.page_count || '—'} 页</span>
            <Popconfirm
              title={isActive ? '停止并删除这组讲解？' : '删除这组讲解？'}
              description={isActive ? '正在进行的生成会先停止，已生成页面和记录将永久删除。' : '讲解记录和全部生成页面将永久删除。'}
              okText={isActive ? '停止并删除' : '删除'}
              cancelText="取消"
              okButtonProps={{ danger: true }}
              onConfirm={() => onDelete(explanation)}
            >
              <Button
                className="explanation-delete-current"
                type="text"
                danger
                loading={deletingId === explanation.id}
                icon={<Trash2 size={14} />}
              >删除</Button>
            </Popconfirm>
          </div>
        </header>

        <div className={`explanation-progress-strip ${explanation.status}`}>
          <Progress percent={explanation.progress} showInfo={false} status={explanation.status === 'error' ? 'exception' : explanation.status === 'completed' ? 'success' : 'active'} />
          <span>{explanation.error || explanation.message}</span>
          <b>{explanation.progress}%</b>
        </div>

        {pages.length > 0 && (
          <div className="explanation-stage-grid">
            <div className="explanation-canvas-card">
              <div className="explanation-canvas-toolbar">
                <div>
                  <span>{activePage ? `${activePage.index}/${explanation.page_count}` : '—'}</span>
                  <strong><InlineMath content={activePage?.title || '页面准备中'} /></strong>
                </div>
                {activePage?.image_url && (
                  <a href={activePage.image_url} download={`${explanation.title}-${activePage.index}.png`}>
                    <Download size={14} /> 下载本页
                  </a>
                )}
              </div>
              <div className={`explanation-canvas ${activePageStatus}`}>
                {activePage?.image_url ? (
                  <AntImage
                    src={activePage.image_url}
                    alt={`${explanation.title}第 ${activePage.index} 页：${activePage.title}`}
                    preview={{ mask: <span><Eye size={16} /> 查看大图</span> }}
                  />
                ) : (
                  <div className="explanation-page-placeholder">
                    <span className="placeholder-blueprint"><Presentation size={34} /></span>
                    {activePageBusy
                      ? <LoaderCircle className="spin" size={21} />
                      : activePageStatus === 'cancelled'
                        ? <X size={21} />
                        : activePageStatus === 'error'
                          ? <AlertTriangle size={21} />
                          : <Layers3 size={21} />}
                    <strong>{activePageMessage}</strong>
                    <small>{activePage?.learning_goal ? <InlineMath content={activePage.learning_goal} /> : null}</small>
                  </div>
                )}
              </div>
            </div>

            <aside className="explanation-outline-panel">
              <header><span>LESSON OUTLINE</span><strong>动态讲解大纲</strong></header>
              <div className="explanation-outline-list">
                {pages.map((page) => {
                  const displayStatus = explanationPageDisplayStatus(page, explanation.status)
                  const pageBusy = isActive && (displayStatus === 'drawing' || displayStatus === 'writing')
                  return (
                    <button
                      type="button"
                      key={page.index}
                      className={page.index === activePage?.index ? 'active' : ''}
                      onClick={() => setSelectedPage(page.index)}
                    >
                      <span className={`outline-number ${displayStatus}`}>
                        {displayStatus === 'ready'
                          ? <Check size={13} />
                          : pageBusy
                            ? <LoaderCircle className="spin" size={13} />
                            : displayStatus === 'cancelled'
                              ? <X size={13} />
                              : displayStatus === 'error'
                                ? <AlertTriangle size={13} />
                                : page.index}
                      </span>
                      <span><strong><InlineMath content={page.title} /></strong><small><InlineMath content={page.subtitle} /></small></span>
                    </button>
                  )
                })}
              </div>
              <div className="explanation-goal-card">
                <BrainCircuit size={17} />
                <div><small>本页学习目标</small><p><InlineMath content={activePage?.learning_goal || '等待大纲生成'} /></p></div>
              </div>
            </aside>
          </div>
        )}

        {!pages.length && (
          <div className="explanation-planning-state">
            <div className="planning-page-preview">
              <span /><span /><span /><span />
            </div>
            <LoaderCircle className="spin" size={24} />
            <div><strong>{explanation.message}</strong><small>正在判断概念、原理、推导、示例与应用之间的最佳讲解顺序</small></div>
          </div>
        )}

        {pages.length > 0 && (
          <div className="explanation-thumbnail-rail" aria-label="讲解页缩略图">
            {pages.map((page) => (
              <button key={page.index} type="button" className={page.index === activePage?.index ? 'active' : ''} onClick={() => setSelectedPage(page.index)}>
                <span className={`thumbnail-image ${explanationPageDisplayStatus(page, explanation.status)}`}>
                  {page.image_url ? <img src={page.image_url} alt="" /> : <Presentation size={20} />}
                  <b>{page.index}</b>
                </span>
                <strong><InlineMath content={page.title} /></strong>
              </button>
            ))}
          </div>
        )}
      </section>

      {history.some((item) => item.id !== explanation.id) && (
        <section className="explanation-history compact">
          <header><div><Clock3 size={15} /><strong>其他讲解</strong></div><span>点击切换</span></header>
          <div className="explanation-history-grid">
            {history.filter((item) => item.id !== explanation.id).slice(0, 4).map((item) => (
              <KnowledgeExplanationHistoryItem
                key={item.id}
                item={item}
                deleting={deletingId === item.id}
                onSelect={onSelect}
                onDelete={onDelete}
              />
            ))}
          </div>
        </section>
      )}
    </div>
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
  const activeOCR = useChatStore((state) => state.ocrModelConfig)
  const setModelConfig = useChatStore((state) => state.setModelConfig)
  const setOCRModelConfig = useChatStore((state) => state.setOCRModelConfig)
  const [draft, setDraft] = useState<ModelConfig>(active)
  const [ocrDraft, setOCRDraft] = useState<OCRModelConfig>(activeOCR)
  const { message: toast } = AntApp.useApp()

  useEffect(() => {
    if (open) {
      setDraft(active)
      setOCRDraft(activeOCR)
    }
  }, [open, active, activeOCR])

  const provider = catalog.providers.find((item) => item.id === draft.provider)
    || fallbackModelCatalog.providers[0]
  const selectableModels = (
    draft.provider === 'qwen' ? provider.text_model_options : provider.model_options
  ) || provider.model_options || provider.models.map((model) => ({
    value: model,
    label: model,
    disabled: false,
    description: '',
  }))
  const sharesQwenCredentials = draft.provider === 'qwen'
  const ocrProvider = catalog.ocr.providers.find((item) => item.id === ocrDraft.provider)
    || fallbackModelCatalog.ocr.providers[0]

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
    const selectedOption = selectableModels.find((option) => option.value === draft.model)
    if (selectedOption?.disabled) {
      toast.warning(selectedOption.description || '该模型不能用于当前对话')
      return
    }
    if (ocrDraft.provider === 'api' && !catalog.ocr.api_configured && !ocrDraft.apiToken.trim()) {
      toast.warning('请填写 PaddleOCR-VL API Token，或在后端环境变量中配置')
      return
    }
    setModelConfig({ ...draft, model: draft.model.trim(), baseUrl: draft.baseUrl.trim() })
    setOCRModelConfig({
      ...ocrDraft,
      model: 'PaddleOCR-VL-1.6',
      jobUrl: ocrDraft.jobUrl.trim() || catalog.ocr.api_job_url,
    })
    onClose()
    toast.success(`已切换到 ${draft.model.trim()}`)
  }

  const clearSavedApiKey = () => {
    const cleared = { ...active, apiKey: '' }
    setModelConfig(cleared)
    setDraft((value) => ({ ...value, apiKey: '' }))
    toast.success('已清除当前浏览器保存的 API Key')
  }

  const clearSavedOCRApiToken = () => {
    const cleared = { ...activeOCR, apiToken: '' }
    setOCRModelConfig(cleared)
    setOCRDraft((value) => ({ ...value, apiToken: '' }))
    toast.success('已清除当前浏览器保存的 PaddleOCR-VL API Token')
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
          <p>配置学生交互模型以及知识库 PaddleOCR-VL 的本地/API 运行方式。</p>
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
          <label>{sharesQwenCredentials ? '文本模型' : '模型名称'}</label>
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

        <div className="vision-config-section">
          <div className="vision-config-heading">
            <span className="modal-icon"><ScanLine size={18} /></span>
            <div>
              <strong>知识库 OCR</strong>
              <small>构建教材时可选本地 PaddleOCR-VL 1.6 或官方作业 API。</small>
            </div>
          </div>
          <div className="model-field">
            <label>OCR 运行方式</label>
            <Segmented<OCRModelConfig['provider']>
              block
              value={ocrDraft.provider}
              options={catalog.ocr.providers.map((item) => ({
                value: item.id,
                label: item.id === 'local' ? '本地 GPU/CPU' : 'PaddleOCR API',
              }))}
              onChange={(provider) => setOCRDraft((value) => ({
                ...value,
                provider,
                model: 'PaddleOCR-VL-1.6',
                jobUrl: catalog.ocr.api_job_url,
              }))}
              aria-label="选择知识库 OCR 运行方式"
            />
            <small className="model-status-hint">{ocrProvider.description}</small>
          </div>
          <div className="model-field">
            <label>OCR 模型</label>
            <Input value="PaddleOCR-VL-1.6" disabled prefix={<ScanLine size={15} />} />
          </div>
          {ocrDraft.provider === 'api' && (
            <>
              <div className="model-field">
                <label>PaddleOCR API Token</label>
                <Input.Password
                  value={ocrDraft.apiToken}
                  onChange={(event) => setOCRDraft((value) => ({
                    ...value,
                    apiToken: event.target.value,
                  }))}
                  placeholder={catalog.ocr.api_configured
                    ? '后端已配置；留空即可使用'
                    : '保存后仅在当前浏览器中保留'}
                  prefix={<KeyRound size={15} />}
                  autoComplete="off"
                />
                {activeOCR.apiToken && (
                  <button type="button" className="clear-api-key" onClick={clearSavedOCRApiToken}>
                    清除已保存的 PaddleOCR API Token
                  </button>
                )}
              </div>
              <div className="model-field">
                <label>API Job URL</label>
                <Input
                  value={catalog.ocr.api_job_url}
                  disabled
                  prefix={<Cloud size={15} />}
                />
              </div>
              <div className="model-security-note cloud">
                <ShieldCheck size={16} />
                <span>选择 API 后，教材页面会发送到 Paddle AI Studio；Token 仅保存在后端环境变量或当前浏览器，不写入构建任务文件。</span>
              </div>
            </>
          )}
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
              : sharesQwenCredentials
                ? '文本答疑会使用这套 API；图片理解由服务端固定的 Qwen3.7-Flash 完成，知识库 OCR 使用上方独立的 PaddleOCR 配置。API Key 仅保存在当前浏览器。'
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
  const [explanationHistory, setExplanationHistory] = useState<KnowledgeExplanation[]>([])
  const [activeExplanation, setActiveExplanation] = useState<KnowledgeExplanation>()
  const [deletingExplanationId, setDeletingExplanationId] = useState('')
  const [explanationPageCount, setExplanationPageCount] = useState<number | null>(0)
  const [explanationImageModel, setExplanationImageModel] = useState<KnowledgeImageModel>('qwen-image-2.0')
  const [knowledgeGraph, setKnowledgeGraph] = useState<KnowledgeGraph>()
  const [graphLoading, setGraphLoading] = useState(false)
  const [mistakes, setMistakes] = useState<MistakeItem[]>([])
  const [mistakeCategories, setMistakeCategories] = useState<MistakeCategory[]>([])
  const [mistakeAnalysis, setMistakeAnalysis] = useState<MistakeAnalysis>()
  const [pendingMistakeDrafts, setPendingMistakeDrafts] = useState<MistakeCandidateDraft[]>([])
  const [pendingReferenceCandidate, setPendingReferenceCandidate] = useState<{
    id: string
    summary: string
    question: string
    knowledge_points: string[]
  }>()
  const [savingMistakeDecision, setSavingMistakeDecision] = useState(false)
  const [scheduleItems, setScheduleItems] = useState<ScheduleItem[]>([])
  const [photoPdfBankId, setPhotoPdfBankId] = useState('')
  const studentId = useChatStore((state) => state.studentId)
  const messages = useChatStore((state) => state.messages)
  const sessionId = useChatStore((state) => state.sessionId)
  const mode = useChatStore((state) => state.mode)
  const setMode = useChatStore((state) => state.setMode)
  const setScene = useChatStore((state) => state.setScene)
  const setActivePractice = useChatStore((state) => state.setActivePractice)
  const setActiveQuestionRef = useChatStore((state) => state.setActiveQuestionRef)
  const setActiveFocus = useChatStore((state) => state.setActiveFocus)
  const send = useChatStore((state) => state.send)
  const knowledgeBase = useChatStore((state) => state.knowledgeBase)
  const defaultKnowledgeBase = useChatStore((state) => state.defaultKnowledgeBase)
  const setKnowledgeBase = useChatStore((state) => state.setKnowledgeBase)
  const setDefaultKnowledgeBase = useChatStore((state) => state.setDefaultKnowledgeBase)
  const syncKnowledgeBases = useChatStore((state) => state.syncKnowledgeBases)
  const modelConfig = useChatStore((state) => state.modelConfig)
  const ocrModelConfig = useChatStore((state) => state.ocrModelConfig)
  const setModelConfig = useChatStore((state) => state.setModelConfig)
  const loadSession = useChatStore((state) => state.loadSession)
  const clear = useChatStore((state) => state.clear)
  const { message: toast } = AntApp.useApp()
  const previousBuildStates = useRef<Record<string, KBStatus['state']>>({})
  const handledMistakeProposalId = useRef<string>('')
  const activeBuilds = statuses.filter((item) => item.state === 'building' || item.state === 'cancelling')
  const hasActiveBuilds = activeBuilds.length > 0
  const runningExplanation = explanationHistory.find(
    (item) => item.status === 'planning' || item.status === 'generating',
  )
  const explanationBusy = Boolean(runningExplanation)
  const currentKbStatus = statuses.find((item) => item.id === knowledgeBase)
  const currentKbDisplayName = knowledgeBaseDisplayName(currentKbStatus, knowledgeBase)
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
      const notebook = await fetchMistakeNotebook(studentId)
      setMistakes(notebook.mistakes || [])
      setMistakeCategories(notebook.categories || [])
      setMistakeAnalysis(notebook.analysis)
    } catch {
      setMistakes([])
      setMistakeCategories([])
      setMistakeAnalysis(undefined)
    }
  }

  const refreshSchedule = async () => {
    try {
      setScheduleItems(await fetchSchedule(studentId))
    } catch {
      setScheduleItems([])
    }
  }

  const refreshExplanations = async () => {
    try {
      setExplanationHistory(await fetchKnowledgeExplanations(studentId))
    } catch {
      // Keep the latest successfully loaded explanation history during transient errors.
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
      const preferred = catalog.providers.find((item) => item.id === catalog.default.provider)
      if (allowAutoSwitch && modelConfig.provider === 'ollama' && preferred?.configured && preferred.id !== 'ollama') {
        setModelConfig({ provider: preferred.id, model: catalog.default.model, apiKey: '', baseUrl: preferred.base_url })
        toast.info(`已使用已配置的 ${preferred.label}`)
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
    void refreshSchedule()
    void refreshExplanations()
    void refreshModels(true)
    const timer = window.setInterval(() => {
      void refreshStatuses()
      void refreshSessions()
      void refreshModels()
    }, 5000)
    return () => window.clearInterval(timer)
  }, [])

  useEffect(() => {
    if (!runningExplanation) return
    const taskId = runningExplanation.id
    const previousStatus = runningExplanation.status
    const poll = () => {
      void fetchKnowledgeExplanation(taskId, studentId)
        .then((next) => {
          setActiveExplanation((current) => current?.id === next.id ? next : current)
          setExplanationHistory((current) => [next, ...current.filter((item) => item.id !== next.id)])
          if (!['planning', 'generating'].includes(next.status)) {
            if (previousStatus === 'planning' || previousStatus === 'generating') {
              if (next.status === 'completed') toast.success('知识讲解页面已全部生成')
              if (next.status === 'error') toast.error(`知识讲解生成失败：${next.error || next.message}`)
              if (next.status === 'cancelled') toast.info('知识讲解生成已取消，已完成页面仍可查看')
            }
            void refreshExplanations()
          }
        })
        .catch(() => undefined)
    }
    const timer = window.setInterval(poll, 1500)
    return () => window.clearInterval(timer)
  }, [runningExplanation?.id, runningExplanation?.status, studentId])

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
        const displayName = knowledgeBaseDisplayName(item, item.id)
        if (item.state === 'ready') toast.success(`知识库“${displayName}”构建完成`)
        if (item.state === 'cancelled') toast.info(`知识库“${displayName}”构建已取消，缓存已清理`)
        if (item.state === 'error') toast.error(`知识库“${displayName}”构建失败：${item.message}`)
      }
    })
    previousBuildStates.current = Object.fromEntries(
      statuses.map((item) => [item.id, item.state]),
    )
  }, [statuses])

  useEffect(() => {
    if (activeView !== 'graph') return
    let active = true
    setKnowledgeGraph(undefined)
    setGraphLoading(true)
    void fetchKnowledgeGraph(knowledgeBase)
      .then((nextGraph) => {
        if (active && nextGraph.knowledge_base === knowledgeBase) setKnowledgeGraph(nextGraph)
      })
      .catch(() => { if (active) setKnowledgeGraph(undefined) })
      .finally(() => { if (active) setGraphLoading(false) })
    return () => { active = false }
  }, [activeView, knowledgeBase])

  useEffect(() => {
    const proposalMessage = [...messages].reverse().find((item) => (
      item.role === 'assistant' && item.mistakeProposal && !item.id.startsWith('history-')
    ))
    if (!proposalMessage?.mistakeProposal || handledMistakeProposalId.current === proposalMessage.id) return
    handledMistakeProposalId.current = proposalMessage.id
    if (!proposalMessage.mistakeProposal.question.trim() || !proposalMessage.mistakeProposal.answer.trim()) {
      toast.warning('当前内容还不能整理为错题')
      return
    }
    setPendingMistakeDrafts([proposalMessage.mistakeProposal])
  }, [messages, toast])

  const kbOptions = useMemo(() => {
    const base = statuses.filter((item) => item.runtime_supported === true).map((item) => ({
      value: item.id,
      label: item.id === defaultKnowledgeBase
        ? `${knowledgeBaseDisplayName(item, item.id)}（默认课程）`
        : knowledgeBaseDisplayName(item, item.id),
    }))
    if (knowledgeBase && !base.some((item) => item.value === knowledgeBase)) {
      base.push({
        value: knowledgeBase,
        label: knowledgeBase === defaultKnowledgeBase
          ? `${knowledgeBaseDisplayName(undefined, knowledgeBase)}（默认课程）`
          : knowledgeBaseDisplayName(undefined, knowledgeBase),
      })
    }
    return base
  }, [statuses, knowledgeBase, defaultKnowledgeBase])

  const defaultKbOptions = useMemo(() => statuses
    .filter((item) => item.runtime_supported === true)
    .map((item) => ({
    value: item.id,
    label: item.id === defaultKnowledgeBase
      ? `${knowledgeBaseDisplayName(item, item.id)}（当前默认）`
      : knowledgeBaseDisplayName(item, item.id),
    disabled: item.state !== 'ready' && !item.available,
    })), [statuses, defaultKnowledgeBase])

  const chooseDefaultKnowledgeBase = (id: string) => {
    const target = statuses.find((item) => item.id === id)
    if (!target?.runtime_supported || (target.state !== 'ready' && !target.available)) {
      toast.warning('只有存在可用索引的知识库可以设为默认课程知识库')
      return
    }
    setDefaultKnowledgeBase(id)
    toast.success(`已将“${knowledgeBaseDisplayName(target, id)}”设为默认课程知识库`)
  }

  const ask = (prompt: string, preferredMode?: ChatMode, recognitionConfirmed = false) => {
    if (preferredMode) {
      setMode(preferredMode)
      if (!recognitionConfirmed) setScene('chat')
    }
    void send(prompt, { recognitionConfirmed }).then(() => refreshSessions())
  }

  const generateKnowledgeExplanation = async (prompt: string) => {
    const question = prompt.trim()
    if (!question) return
    if (
      explanationPageCount === null
      || (explanationPageCount !== 0 && (explanationPageCount < 1 || explanationPageCount > 8))
    ) {
      toast.warning('自定义讲解页数请输入 1–8 之间的整数')
      return
    }
    if (explanationBusy) {
      toast.warning('请先等待当前讲解生成完成，或取消当前任务')
      return
    }
    try {
      const explanation = await createKnowledgeExplanation({
        studentId,
        question,
        pageCount: explanationPageCount,
        imageModel: explanationImageModel,
        modelConfig,
      })
      setActiveExplanation(explanation)
      setExplanationHistory((current) => [explanation, ...current.filter((item) => item.id !== explanation.id)])
      toast.info('已开始分析问题并规划讲解大纲')
    } catch (error) {
      toast.error(error instanceof Error ? error.message : '知识讲解任务创建失败')
    }
  }

  const stopKnowledgeExplanation = async () => {
    if (!runningExplanation) return
    try {
      const explanation = await cancelKnowledgeExplanation(runningExplanation.id, studentId)
      setActiveExplanation((current) => current?.id === explanation.id ? explanation : current)
      setExplanationHistory((current) => [explanation, ...current.filter((item) => item.id !== explanation.id)])
    } catch (error) {
      toast.error(error instanceof Error ? error.message : '知识讲解取消失败')
    }
  }

  const removeKnowledgeExplanation = async (explanation: KnowledgeExplanation) => {
    if (deletingExplanationId) return
    setDeletingExplanationId(explanation.id)
    try {
      await deleteKnowledgeExplanation(explanation.id, studentId)
      setExplanationHistory((current) => current.filter((item) => item.id !== explanation.id))
      setActiveExplanation((current) => current?.id === explanation.id ? undefined : current)
      toast.success('知识讲解已删除')
    } catch (error) {
      toast.error(error instanceof Error ? error.message : '知识讲解删除失败')
    } finally {
      setDeletingExplanationId('')
    }
  }

  const uploadPhotoPdf = async (file: File) => {
    if (!/\.pdf$/i.test(file.name)) {
      toast.warning('请选择 PDF 文件')
      return
    }
    try {
      toast.info('正在保存 PDF，保存后将在当前聊天显示拆题进度')
      const bank = await createQuestionBank(
        file,
        file.name.replace(/\.pdf$/i, ''),
        knowledgeBase,
        { studentId, sourceOrigin: 'photo_answer' },
      )
      setPhotoPdfBankId(bank.id)
      toast.success('PDF 已保存，正在聊天页内后台拆题')
    } catch (error) {
      toast.error(error instanceof Error ? error.message : 'PDF 上传失败')
    }
  }

  const askQuestionBankQuestion = (bank: QuestionBank, question: HomeworkQuestion) => {
    const questionRef: QuestionReference = {
      kind: 'question_bank',
      question_bank_id: bank.id,
      question_id: question.id,
    }
    setPhotoPdfBankId('')
    setActiveView('chat')
    setMode('answer')
    setScene('image_answer')
    void send(
      `请结合题库已有参考答案，解释《${bank.title}》${formatQuestionNumber(question.number || question.sequence)}的解题思路、公式来源和中间步骤。`,
      { questionRef },
    ).then(() => refreshSessions())
  }

  const startPracticeAnswer = (practice: PracticeExercise, focus?: ConversationFocus) => {
    if (focus) setActiveFocus(focus)
    setMode('quiz')
    setActivePractice(practice)
    setScene('quiz_grade')
    window.setTimeout(() => {
      document.querySelector('.practice-submit-banner')?.scrollIntoView({
        behavior: 'smooth',
        block: 'center',
      })
    }, 0)
  }

  const generateAnotherPractice = (focus?: ConversationFocus) => {
    if (focus) setActiveFocus(focus)
    ask('请基于刚才这道题再生成一道同构变式题，保持知识点、拓扑和待求量结构，只调整情境或参数。', 'quiz')
  }

  const generateSimilarFromRecommendation = (reference: QuestionReference) => {
    setActiveView('chat')
    setMode('quiz')
    setScene('chat')
    void send('请基于这道原书题生成一道同知识点、同结构的变式题。', {
      questionRef: reference,
    }).then(() => refreshSessions())
  }

  const recommendAnother = (reference: QuestionReference, continuationTaskId?: string, focus?: ConversationFocus) => {
    if (focus) setActiveFocus(focus)
    setMode('recommend')
    setScene('chat')
    void send('再来一道', { questionRef: reference, continuationTaskId })
      .then(() => refreshSessions())
  }

  const startRecommendationAnswer = (recommendation: QuestionRecommendation) => {
    const question = recommendation.question
    setMode('recommend')
    setActivePractice({
      question_type: question.question_type,
      question: question.prompt,
      question_stem: question.prompt,
      question_parts: (question.subquestions || []).map((item) => `(${item.label}) ${item.text}`),
      knowledge_point: recommendation.profile.knowledge_points.join('、'),
      difficulty: recommendation.profile.difficulty,
      solution: '',
      solution_steps: [],
      answer: '',
      answer_items: [],
      common_mistakes: [],
    })
    setScene('quiz_grade')
    setActiveQuestionRef(recommendation.question_ref)
    window.setTimeout(() => {
      document.querySelector('.practice-submit-banner')?.scrollIntoView({
        behavior: 'smooth',
        block: 'center',
      })
    }, 0)
  }

  const requestRecommendationHint = (
    reference: QuestionReference,
    level: 'direction' | 'formula',
  ) => {
    setMode('answer')
    setScene('chat')
    const prompt = level === 'formula'
      ? '只给我解决这道题需要使用的关键公式和每个符号的含义，不要代入数值，不要给最终答案。'
      : '只给我一级思路提示：指出第一步应该判断什么或从哪里开始，不要展开完整步骤，不要给公式计算结果和最终答案。'
    void send(prompt, { questionRef: reference }).then(() => refreshSessions())
  }

  const proposeReferenceMistake = async (reference: QuestionReference) => {
    try {
      setPendingReferenceCandidate(await createQuestionReferenceMistakeCandidate(
        reference,
        studentId,
        sessionId,
        knowledgeBase,
      ))
    } catch (error) {
      toast.error(error instanceof Error ? error.message : '错题候选创建失败')
    }
  }

  const confirmReferenceMistake = async (decision: {
    reason: MistakeReason
    categoryId: string
    title: string
  }) => {
    if (!pendingReferenceCandidate || savingMistakeDecision) return
    setSavingMistakeDecision(true)
    try {
      await confirmMistakeCandidate(studentId, pendingReferenceCandidate.id, {
        ...decision,
        photoRetention: 'original_and_processed',
      })
      setPendingReferenceCandidate(undefined)
      await refreshMistakes()
      toast.success('已加入错题本')
    } catch (error) {
      toast.error(error instanceof Error ? error.message : '加入错题本失败')
    } finally {
      setSavingMistakeDecision(false)
    }
  }

  const proposeMistakes = (drafts: MistakeCandidateDraft | MistakeCandidateDraft[]) => {
    const next = (Array.isArray(drafts) ? drafts : [drafts]).filter((draft) => (
      draft.question.trim() && draft.answer.trim()
    ))
    if (!next.length) {
      toast.warning('当前内容还不能整理为错题')
      return
    }
    setPendingMistakeDrafts(next)
  }

  const materializeMistakeAttachments = async (
    draft: MistakeCandidateDraft,
  ): Promise<MistakeCandidateDraft> => {
    const attachments = [...draft.attachments]
    for (const [index, url] of (draft.attachmentUrls || []).slice(0, 5 - attachments.length).entries()) {
      const response = await fetch(url)
      if (!response.ok) throw new Error('无法读取作业题目或作答图片')
      const blob = await response.blob()
      const suffix = blob.type.includes('png') ? 'png' : blob.type.includes('webp') ? 'webp' : 'jpg'
      const file = new File([blob], `错题图片-${index + 1}.${suffix}`, { type: blob.type || `image/${suffix}` })
      attachments.push(await uploadChatAttachment(file, sessionId))
    }
    return { ...draft, attachments }
  }

  const confirmMistakeDecision = async (decision: {
    reason: MistakeReason
    categoryId: string
    title: string
    photoRetention: MistakePhotoRetention
  }) => {
    if (!pendingMistakeDrafts.length || savingMistakeDecision) return
    setSavingMistakeDecision(true)
    try {
      const saved: MistakeItem[] = []
      for (const [index, originalDraft] of pendingMistakeDrafts.entries()) {
        const draft = await materializeMistakeAttachments(originalDraft)
        const candidate = await createMistakeCandidate(
          studentId,
          sessionId,
          draft,
          modelConfig,
          knowledgeBase,
        )
        saved.push(await confirmMistakeCandidate(studentId, candidate.id, {
          ...decision,
          title: pendingMistakeDrafts.length === 1
            ? decision.title
            : originalDraft.title || `错题 ${index + 1}`,
        }))
      }
      setMistakes((current) => [
        ...saved,
        ...current.filter((existing) => !saved.some((item) => item.id === existing.id)),
      ])
      await refreshMistakes()
      setPendingMistakeDrafts([])
      const points = [...new Set(saved.flatMap((item) => item.knowledge_points))].slice(0, 6)
      toast.success(
        `已由你确认加入 ${saved.length} 道错题${points.length ? `，关联：${points.join('、')}` : ''}`,
      )
    } catch (error) {
      toast.error(error instanceof Error ? error.message : '加入错题本失败')
    } finally {
      setSavingMistakeDecision(false)
    }
  }

  const removeMistake = async (id: string) => {
    try {
      await deleteMistake(studentId, id)
      setMistakes((current) => current.filter((item) => item.id !== id))
      await refreshMistakes()
      toast.success('已从错题本删除')
    } catch (error) {
      toast.error(error instanceof Error ? error.message : '删除错题失败')
    }
  }

  const createScheduleItem = async (draft: ScheduleItemDraft) => {
    try {
      const item = await addScheduleItem(studentId, draft)
      setScheduleItems((current) => [...current, item].sort((a, b) => `${a.date} ${a.time || '99:99'}`.localeCompare(`${b.date} ${b.time || '99:99'}`)))
      toast.success('安排已加入日历')
      return true
    } catch (error) {
      toast.error(error instanceof Error ? error.message : '日程添加失败')
      return false
    }
  }

  const toggleScheduleItem = async (item: ScheduleItem) => {
    try {
      const updated = await setScheduleItemCompleted(studentId, item.id, !item.completed)
      setScheduleItems((current) => current.map((existing) => existing.id === item.id ? updated : existing))
    } catch (error) {
      toast.error(error instanceof Error ? error.message : '日程状态更新失败')
    }
  }

  const removeScheduleItem = async (id: string) => {
    try {
      await deleteScheduleItem(studentId, id)
      setScheduleItems((current) => current.filter((item) => item.id !== id))
      toast.success('安排已删除')
    } catch (error) {
      toast.error(error instanceof Error ? error.message : '日程删除失败')
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
      if (mode === 'explain') {
        setMode('auto')
        setScene('chat')
      }
      setActiveView('chat')
      setSidebarOpen(false)
    } catch (error) {
      toast.error(error instanceof Error ? error.message : '历史会话恢复失败')
    }
  }

  const startNewSession = () => {
    clear()
    if (mode === 'explain') {
      setMode('auto')
      setScene('chat')
    }
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
      const result = await uploadKnowledgeFile(
        options.file as File,
        knowledgeBase,
        undefined,
        ocrModelConfig,
      )
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
      const result = await rebuildKnowledgeBase(knowledgeBase, ocrModelConfig)
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
      const deletedDisplayName = currentKbDisplayName
      const replacement = statuses.find((item) => (
        item.id !== deleted
        && item.runtime_supported === true
        && (item.state === 'ready' || item.available)
      ))
      const nextKnowledgeBase = replacement?.id || ''
      await deleteKnowledgeBase(deleted)
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
      toast.success(`知识库“${deletedDisplayName}”已删除`)
    } catch (error) {
      toast.error(error instanceof Error ? error.message : '删除知识库失败')
    }
  }

  const createKnowledgeBase = async () => {
    if (!newKbFile) {
      toast.warning('请先选择需要上传的教材文件')
      return
    }
    const displayName = newKbName.trim().replace(/\s+/g, ' ')
    if (!displayName || displayName.length > 48) {
      toast.warning('知识库名称不能为空且不能超过 48 个字符')
      return
    }
    if (statuses.some((item) => knowledgeBaseDisplayName(item, item.id) === displayName)) {
      toast.warning(`知识库“${displayName}”已存在，请在下方选择后追加资料`)
      return
    }
    const internalId = createKnowledgeBaseInternalId()
    setCreatingKnowledgeBase(true)
    try {
      const result = await uploadKnowledgeFile(
        newKbFile,
        internalId,
        displayName,
        ocrModelConfig,
      )
      setKnowledgeBase(internalId)
      upsertBuildStatus(result.build)
      setNewKbFile(null)
      setNewKbName('')
      toast.success(`知识库“${displayName}”已创建，正在构建索引`)
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

  const todaySchedule = useMemo(
    () => scheduleItems.filter((item) => item.date === localDateKey()),
    [scheduleItems],
  )

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
              <strong>{activeView === 'graph' ? '知识图谱' : activeView === 'question-bank' ? '题库' : activeView === 'homework' ? '我的作业' : activeView === 'mistakes' ? '错题本' : activeView === 'reports' ? '答案分析报告' : activeView === 'schedule' ? '学习日历' : mode === 'recommend' ? '题库推荐' : mode === 'quiz' ? '同类题生成' : mode === 'answer' ? '课程答疑' : mode === 'explain' ? '知识讲解' : mode === 'plan' ? '学习规划' : '智能学习'}</strong>
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
                  <strong>{item.state === 'cancelling' ? `正在取消“${knowledgeBaseDisplayName(item, item.id)}”` : `正在构建知识库“${knowledgeBaseDisplayName(item, item.id)}”`}</strong>
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
          <section className={`learning-grid ${mode === 'explain' ? 'explanation-learning-grid' : ''}`}>
            <div className="chat-column">
              <div className="chat-scroll">
                {mode === 'explain' ? (
                  <KnowledgeExplanationView
                    explanation={activeExplanation}
                    history={explanationHistory}
                    pageCount={explanationPageCount}
                    imageModel={explanationImageModel}
                    onPageCountChange={setExplanationPageCount}
                    onImageModelChange={setExplanationImageModel}
                    onSelect={setActiveExplanation}
                    onBack={() => setActiveExplanation(undefined)}
                    onDelete={removeKnowledgeExplanation}
                    deletingId={deletingExplanationId}
                  />
                ) : messages.length === 0 ? (
                  <Welcome
                    onAsk={ask}
                    todaySchedule={todaySchedule}
                    onOpenSchedule={() => setActiveView('schedule')}
                    onToggleSchedule={(item) => void toggleScheduleItem(item)}
                  />
                ) : (
                  <Conversation
                    onAddMistake={(draft) => proposeMistakes(draft)}
                    onConfirmPhoto={(content) => ask(content, 'answer', true)}
                    onStartPractice={startPracticeAnswer}
                    onGenerateSimilar={generateAnotherPractice}
                    onRecommendationSimilar={generateSimilarFromRecommendation}
                    onRecommendationBookmark={(reference) => void proposeReferenceMistake(reference)}
                    onRecommendationAnother={recommendAnother}
                    onRecommendationStart={startRecommendationAnswer}
                    onRecommendationHint={requestRecommendationHint}
                  />
                )}
              </div>
              <ChatComposer
                onSend={(value) => mode === 'explain' ? void generateKnowledgeExplanation(value) : ask(value)}
                onPdfUpload={(file) => void uploadPhotoPdf(file)}
                externalBusy={explanationBusy}
                onExternalStop={() => void stopKnowledgeExplanation()}
                explanationImageModel={explanationImageModel}
              />
            </div>
            {mode !== 'explain' && <KnowledgePanel statuses={statuses} onCreate={() => setKbModalOpen(true)} />}
          </section>
        ) : activeView === 'graph' ? (
          <KnowledgeGraphView
            graph={knowledgeGraph}
            loading={graphLoading}
            knowledgeBase={knowledgeBase}
            displayName={currentKbDisplayName}
          />
        ) : activeView === 'question-bank' ? (
          <QuestionBankView
            knowledgeBase={knowledgeBase}
            studentId={studentId}
            initialBankId={photoPdfBankId}
            onAskQuestion={askQuestionBankQuestion}
          />
        ) : activeView === 'mistakes' ? (
          <MistakeBookView
            mistakes={mistakes}
            categories={mistakeCategories}
            analysis={mistakeAnalysis}
            studentId={studentId}
            onDelete={(id) => void removeMistake(id)}
            onPlan={planFromMistakes}
            onChanged={refreshMistakes}
          />
        ) : activeView === 'schedule' ? (
          <ScheduleView
            items={scheduleItems}
            onAdd={createScheduleItem}
            onToggle={(item) => void toggleScheduleItem(item)}
            onDelete={(id) => void removeScheduleItem(id)}
          />
        ) : activeView === 'reports' ? (
          <AnswerReportsView studentId={studentId} />
        ) : (
          <HomeworkView
            studentId={studentId}
            onProposeMistakes={(drafts) => proposeMistakes(drafts)}
          />
        )}
      </main>

      <ModelSettingsModal
        open={modelModalOpen}
        onClose={() => setModelModalOpen(false)}
        catalog={modelCatalog}
      />

      <MistakeConfirmModal
        drafts={pendingMistakeDrafts}
        categories={mistakeCategories}
        saving={savingMistakeDecision}
        onCancel={() => !savingMistakeDecision && setPendingMistakeDrafts([])}
        onConfirm={(decision) => void confirmMistakeDecision(decision)}
      />
      <ReferenceMistakeConfirmModal
        candidate={pendingReferenceCandidate}
        categories={mistakeCategories}
        saving={savingMistakeDecision}
        onCancel={() => !savingMistakeDecision && setPendingReferenceCandidate(undefined)}
        onConfirm={(decision) => void confirmReferenceMistake(decision)}
      />
      <PhotoPdfQuestionPickerModal
        bankId={photoPdfBankId}
        studentId={studentId}
        onClose={() => setPhotoPdfBankId('')}
        onAsk={askQuestionBankQuestion}
      />

      <Modal
        open={kbModalOpen}
        onCancel={closeKnowledgeBaseModal}
        mask={{ closable: !creatingKnowledgeBase }}
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
              setNewKbName((current) => current.trim() || file.name.replace(/\.[^.]+$/, '').slice(0, 48))
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
            <p className="ant-upload-hint">支持 PDF、Word、Markdown 和文本，单个文件最大 200 MB</p>
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
            placeholder="如：模拟电子技术基础"
            prefix={<Plus size={15} />}
            disabled={!newKbFile || creatingKnowledgeBase}
            onPressEnter={() => void createKnowledgeBase()}
          />
          <p className="modal-field-help">支持中文、字母、数字和空格；系统会自动生成内部标识。</p>
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
            <Tooltip title={deleteKbDisabledReason || `删除知识库“${currentKbDisplayName}”`}>
              <span>
                <Popconfirm
                  title={`确认删除知识库“${currentKbDisplayName}”？`}
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
                    aria-label={`删除知识库 ${currentKbDisplayName}`}
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
            <p className="ant-upload-hint">文件将追加到“{currentKbDisplayName}”并触发重建，单个文件最大 200 MB</p>
          </Upload.Dragger>
        </div>
        {(currentKbStatus?.state === 'building' || currentKbStatus?.state === 'cancelling') && (
          <div className="modal-build-progress" aria-label={`${currentKbDisplayName} 构建进度`}>
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
          使用 PaddleOCR-VL 重新构建已有资料
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
