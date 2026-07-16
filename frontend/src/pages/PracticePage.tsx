import { useEffect, useMemo, useRef, useState, type DragEvent } from 'react'
import { Link, Navigate, Route, Routes, useLocation, useNavigate, useParams } from 'react-router-dom'
import { App as AntApp, Button, Image, Input, Modal, Progress, Select, Spin } from 'antd'
import {
  ArrowLeft,
  ArrowRight,
  Bot,
  BookOpenCheck,
  Check,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  CircleAlert,
  CircleX,
  Clock3,
  FileCheck2,
  GraduationCap,
  Home,
  ImagePlus,
  Layers3,
  LoaderCircle,
  LockKeyhole,
  MessageCircle,
  ChartNoAxesCombined,
  LogOut,
  Play,
  Repeat2,
  RotateCcw,
  Send,
  Settings2,
  ShieldCheck,
  Sparkles,
  Square,
  Trash2,
  TriangleAlert,
  UploadCloud,
} from 'lucide-react'
import MathMarkdown from '../components/MathMarkdown'
import { fetchModels, type ModelCatalog, type ModelConfig } from '../lib/api'
import {
  fetchPracticeCatalog,
  fetchActivePracticeSession,
  fetchPracticeQuestion,
  discardEmptyPracticeSession,
  finishPracticeSession,
  gradePracticeSubmission,
  PracticeCatalog,
  PracticeGrade,
  PracticeModelConfig,
  PracticeQuestion,
  PracticeQuestionCatalogItem,
  PracticeSession,
  resolvePracticeSubmission,
  streamPracticeFollowup,
  startPracticeSession,
  submitPracticeAnswer,
  visitPracticeQuestion,
} from '../lib/practiceApi'
import { useChatStore } from '../store/chatStore'
import { PracticeFeedbackDetail, PracticeFeedbackList } from './PracticeFeedbackPage'
import './PracticePage.css'

type SelectedImage = {
  id: string
  file: File
  preview: string
}

const COURSE_PATH = '/practice/electronic-circuits'
const ACCEPTED_IMAGE_TYPES = new Set(['image/png', 'image/jpeg', 'image/webp', 'image/bmp'])
const MAX_IMAGES = 5
const MAX_TOTAL_BYTES = 20 * 1024 * 1024
const PRACTICE_MODEL_STORAGE_KEY = 'circuitmind-practice-vision-model-v1'
const QWEN_VISION_MODELS = [
  { value: 'qwen3-vl-flash', label: 'Qwen3-VL-Flash', description: '速度快，适合日常批改' },
  { value: 'qwen3-vl-plus', label: 'Qwen3-VL-Plus', description: '复杂手写与波形理解更强' },
  { value: 'qwen-vl-max', label: 'Qwen-VL-Max', description: '高质量视觉分析' },
]
const DEFAULT_QWEN_BASE_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1'

function initialPracticeModel(chatModel: ModelConfig): PracticeModelConfig {
  try {
    const saved = JSON.parse(localStorage.getItem(PRACTICE_MODEL_STORAGE_KEY) || '{}')
    if (
      (saved.provider === 'qwen' || saved.provider === 'custom')
      && typeof saved.model === 'string'
      && typeof saved.apiKey === 'string'
      && typeof saved.baseUrl === 'string'
    ) return saved
  } catch {
    // Ignore invalid device-local preferences and fall back to a safe vision model.
  }
  const reusableQwen = chatModel.provider === 'qwen'
  const selectedModel = reusableQwen && QWEN_VISION_MODELS.some((item) => item.value === chatModel.model)
    ? chatModel.model
    : 'qwen3-vl-flash'
  return {
    provider: 'qwen',
    model: selectedModel,
    apiKey: reusableQwen ? chatModel.apiKey : '',
    baseUrl: reusableQwen ? (chatModel.baseUrl || DEFAULT_QWEN_BASE_URL) : DEFAULT_QWEN_BASE_URL,
  }
}

function modelIsReady(config: PracticeModelConfig, qwenConfigured: boolean) {
  if (config.provider === 'custom') return Boolean(config.apiKey && config.baseUrl && config.model)
  return Boolean(config.model && (config.apiKey || qwenConfigured))
}

function practiceQuestionPath(questionId: string) {
  const unit = questionId.split('.')[0] || '1'
  return `${COURSE_PATH}/chapter-${unit}/questions/${encodeURIComponent(questionId)}`
}

function formatSubmissionTime(value: string | null) {
  if (!value) return ''
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return ''
  return date.toLocaleString('zh-CN', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  })
}

const VERDICT_SHORT_LABELS: Record<string, string> = {
  correct: '正确',
  partially_correct: '部分正确',
  incorrect: '错误',
  unreadable: '无法辨认',
}

function questionState(item: PracticeQuestionCatalogItem) {
  if (item.grading_status === 'pending') return { key: 'grading', label: '批改中' }
  if (item.has_submission && !item.resolved) return { key: 'pending', label: '待确认' }
  if (item.completed) return { key: 'mastered', label: '已掌握' }
  return { key: 'new', label: '未作答' }
}

function PracticeSessionBar({
  session,
  stopping,
  onFinish,
}: {
  session: PracticeSession
  stopping: boolean
  onFinish: () => void
}) {
  return (
    <div className="practice-session-bar">
      <span className="practice-session-pulse" />
      <div>
        <strong>本轮练习进行中</strong>
        <small><Clock3 size={12} /> 开始于 {formatSubmissionTime(session.started_at)}</small>
      </div>
      <div className="practice-session-scope">
        <span>已提交 {session.submitted_question_count} 道 · {session.submission_count} 次</span>
        <strong>{session.submitted_question_ids.length ? session.submitted_question_ids.join('、') : '提交答案后才会纳入反馈'}</strong>
      </div>
      <Button danger type="text" icon={<LogOut size={15} />} loading={stopping} onClick={onFinish}>结束本轮</Button>
    </div>
  )
}

function PracticeShell({ children }: { children: React.ReactNode }) {
  const location = useLocation()
  const feedbackActive = location.pathname.startsWith('/practice/feedback')
  return (
    <div className="practice-app">
      <aside className="practice-sidebar">
        <Link to="/student" className="practice-brand" aria-label="返回 CircuitMind 学生工作台">
          <span className="practice-brand-mark"><Sparkles size={18} /></span>
          <span>
            <strong>CircuitMind</strong>
            <small>电路学习空间</small>
          </span>
        </Link>

        <div className="practice-nav-label">学习空间</div>
        <nav className="practice-nav" aria-label="刷题页面导航">
          <Link to="/student"><Home size={17} /><span>智能学习台</span></Link>
          <Link to="/practice" className={feedbackActive ? '' : 'active'}><BookOpenCheck size={17} /><span>题库练习</span></Link>
          <Link to="/practice/feedback" className={feedbackActive ? 'active' : ''}><ChartNoAxesCombined size={17} /><span>学情反馈</span></Link>
        </nav>

        <Link to="/teacher" className="practice-teacher-link">
          <GraduationCap size={17} />
          <span>切换到教师端</span>
          <ChevronRight size={15} />
        </Link>
      </aside>
      <main className="practice-main">{children}</main>
    </div>
  )
}

function PracticeHeader({
  eyebrow,
  title,
  description,
  backTo,
}: {
  eyebrow: string
  title: string
  description: string
  backTo?: string
}) {
  return (
    <header className="practice-header">
      <div className="practice-header-copy">
        {backTo && (
          <Link to={backTo} className="practice-back-link">
            <ArrowLeft size={15} /> 返回上一级
          </Link>
        )}
        <span className="practice-eyebrow">{eyebrow}</span>
        <h1>{title}</h1>
        <p>{description}</p>
      </div>
    </header>
  )
}

function PageLoading() {
  return <div className="practice-page-state"><Spin size="large" /><span>正在加载题库…</span></div>
}

function PageError({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <div className="practice-page-state error">
      <CircleAlert size={28} />
      <strong>页面暂时无法加载</strong>
      <span>{message}</span>
      <Button icon={<RotateCcw size={15} />} onClick={onRetry}>重新加载</Button>
    </div>
  )
}

function PracticeHome({ catalog }: { catalog: PracticeCatalog }) {
  const course = catalog.courses[0]
  const percent = course.question_count
    ? Math.round(course.completed_count / course.question_count * 100)
    : 0
  return (
    <div className="practice-page">
      <PracticeHeader
        eyebrow="PRACTICE LIBRARY"
        title="刷题训练"
        description="按课程和章节稳步练习，每次专注解决一道题。"
      />
      <section className="practice-content-section">
        <div className="practice-section-heading">
          <div><span>课程目录</span><h2>选择一门课程开始</h2></div>
          <small>当前共 1 门课程</small>
        </div>
        <Link to={COURSE_PATH} className="practice-course-card">
          <div className="practice-course-art" aria-hidden="true">
            <span className="practice-chip"><Sparkles size={24} /></span>
            <i className="practice-trace trace-a" /><i className="practice-trace trace-b" />
            <i className="practice-dot dot-a" /><i className="practice-dot dot-b" /><i className="practice-dot dot-c" />
          </div>
          <div className="practice-course-copy">
            <span className="practice-card-kicker">电子信息类 · 核心基础</span>
            <h3>{course.title}</h3>
            <p>{course.description}</p>
            <div className="practice-course-meta">
              <span><Layers3 size={15} /> {course.chapters.length} 个章节</span>
              <span><FileCheck2 size={15} /> {course.question_count} 道习题</span>
            </div>
          </div>
          <div className="practice-course-progress">
            <Progress type="circle" percent={percent} size={68} strokeColor="#4cc3ad" trailColor="rgba(255,255,255,.14)" />
            <span>{course.completed_count}/{course.question_count} 已完成</span>
            <strong>进入课程 <ArrowRight size={16} /></strong>
          </div>
        </Link>
      </section>
    </div>
  )
}

function CoursePage({
  catalog,
  activeSession,
  stopping,
  onBegin,
  onFinish,
}: {
  catalog: PracticeCatalog
  activeSession: PracticeSession | null
  stopping: boolean
  onBegin: (questionId: string) => void
  onFinish: () => void
}) {
  const course = catalog.courses[0]
  const [expandedChapters, setExpandedChapters] = useState<Set<string>>(() => new Set())

  const toggleChapter = (chapterId: string) => {
    setExpandedChapters((current) => {
      const next = new Set(current)
      if (next.has(chapterId)) next.delete(chapterId)
      else next.add(chapterId)
      return next
    })
  }

  return (
    <div className="practice-page">
      <PracticeHeader
        eyebrow="COURSE CHAPTERS"
        title={course.title}
        description="按章节完成练习，经过批改并确认理解后记录进度。"
        backTo="/practice"
      />
      <section className="practice-content-section">
        {activeSession && <PracticeSessionBar session={activeSession} stopping={stopping} onFinish={onFinish} />}
        <div className="practice-section-heading">
          <div><span>章节目录</span><h2>选择章节与题目开始训练</h2></div>
          <small>{course.completed_count}/{course.question_count} 道已确认</small>
        </div>
        {course.chapters.map((chapter, chapterIndex) => {
          const percent = chapter.question_count
            ? Math.round(chapter.completed_count / chapter.question_count * 100)
            : 0
          const startId = chapter.resume_question_id || chapter.questions[0]?.id
          const isUnitTwo = chapter.id === 'chapter-2'
          const tags = isUnitTwo
            ? ['共射放大', '稳定偏置', '电流源', '差分放大']
            : ['半导体载流子', 'PN 结', '二极管模型', '限幅与整流']
          const expanded = expandedChapters.has(chapter.id)
          const questionPanelId = `practice-${chapter.id}-questions`
          return (
            <section className="practice-chapter-block" key={chapter.id}>
              <article className="practice-chapter-card">
                <div className="practice-chapter-number">{String(chapterIndex + 1).padStart(2, '0')}</div>
                <div className="practice-chapter-copy">
                  <span>第{chapterIndex + 1 === 1 ? '一' : '二'}章</span>
                  <h3>{chapter.title.replace(/^第[一二]章\s*/, '')}</h3>
                  <p>{chapter.description}</p>
                  <div className="practice-topic-tags">{tags.map((tag) => <span key={tag}>{tag}</span>)}</div>
                </div>
                <div className="practice-chapter-action">
                  <div className="chapter-progress-copy"><span>章节进度</span><strong>{percent}%</strong></div>
                  <Progress percent={percent} showInfo={false} strokeColor="#159080" trailColor="#e3eeeb" />
                  <button type="button" disabled={!startId} onClick={() => startId && onBegin(startId)}>
                    {activeSession ? '继续本轮练习' : chapter.completed_count ? '继续刷题' : '开始刷题'} <ArrowRight size={16} />
                  </button>
                  <button
                    type="button"
                    className={`chapter-question-toggle ${expanded ? 'is-expanded' : ''}`}
                    aria-expanded={expanded}
                    aria-controls={questionPanelId}
                    onClick={() => toggleChapter(chapter.id)}
                  >
                    {expanded ? '收起题目' : `展开 ${chapter.question_count} 道题`} <ChevronDown size={16} />
                  </button>
                </div>
              </article>
              {expanded && (
                <div id={questionPanelId} className="practice-question-panel">
                  <div className="practice-question-library-heading">
                    <div><span>第{chapterIndex + 1 === 1 ? '一' : '二'}章 · 全部题目</span><h2>选择题目开始或再次练习</h2></div>
                    <small>{chapter.question_count} 道 · 已掌握题目可无限次重练</small>
                  </div>
                  <div className="practice-question-library">
                    {chapter.questions.map((item, questionIndex) => {
                      const state = questionState(item)
                      const included = Boolean(activeSession?.submitted_question_ids.includes(item.id))
                      return (
                        <article key={item.id} className={`practice-question-library-item ${state.key} ${included ? 'in-session' : ''}`}>
                          <div className="practice-question-library-number" title={`教材题号 ${item.id}`}>{questionIndex + 1}</div>
                          <div className="practice-question-library-copy">
                            <div><span>{item.section}</span><em className={state.key}>{state.label}</em>{included && <em className="session">本轮已提交</em>}</div>
                            <h3>{item.title}</h3>
                            <p>
                              {item.attempt_count ? `累计作答 ${item.attempt_count} 次` : '尚未提交作答'}
                              {item.latest_verdict ? ` · 最近${VERDICT_SHORT_LABELS[item.latest_verdict] || item.latest_verdict}` : ''}
                              {item.last_submitted_at ? ` · ${formatSubmissionTime(item.last_submitted_at)}` : ''}
                            </p>
                          </div>
                          <button type="button" onClick={() => onBegin(item.id)}>
                            {item.completed ? <><Repeat2 size={15} /> 再练一次</> : item.has_submission ? <>继续作答 <ArrowRight size={15} /></> : <><Play size={15} /> 开始作答</>}
                          </button>
                        </article>
                      )
                    })}
                  </div>
                </div>
              )}
            </section>
          )
        })}
      </section>
    </div>
  )
}

function PracticeModelSettings({
  open,
  value,
  qwenConfigured,
  qwenBaseUrl,
  onClose,
  onSave,
}: {
  open: boolean
  value: PracticeModelConfig
  qwenConfigured: boolean
  qwenBaseUrl: string
  onClose: () => void
  onSave: (config: PracticeModelConfig) => void
}) {
  const [draft, setDraft] = useState(value)
  const { message } = AntApp.useApp()
  useEffect(() => { if (open) setDraft(value) }, [open, value])

  const save = () => {
    const normalized = {
      ...draft,
      model: draft.model.trim(),
      apiKey: draft.apiKey.trim(),
      baseUrl: draft.baseUrl.trim(),
    }
    if (!normalized.model) {
      message.warning('请填写多模态模型名称')
      return
    }
    if (normalized.provider === 'custom' && (!normalized.apiKey || !normalized.baseUrl)) {
      message.warning('自定义多模态 API 必须填写 API Key 和 Base URL')
      return
    }
    if (normalized.provider === 'qwen' && !normalized.apiKey && !qwenConfigured) {
      message.warning('当前后端未配置千问 API Key，请先填写')
      return
    }
    onSave(normalized)
    onClose()
    message.success('刷题批改模型已保存')
  }

  return (
    <Modal open={open} footer={null} onCancel={onClose} width={590} title={null} className="practice-model-modal">
      <div className="practice-model-modal-heading">
        <span><Bot size={22} /></span>
        <div><h2>配置多模态批改模型</h2><p>此配置仅用于刷题批改和追问，不会改变聊天区模型。</p></div>
      </div>
      <div className="practice-model-form">
        <label>
          <span>服务类型</span>
          <Select
            value={draft.provider}
            options={[
              { value: 'qwen', label: '通义千问视觉模型' },
              { value: 'custom', label: '自定义 OpenAI-compatible 多模态 API' },
            ]}
            onChange={(provider: 'qwen' | 'custom') => setDraft(provider === 'qwen'
              ? { provider, model: 'qwen3-vl-flash', apiKey: '', baseUrl: qwenBaseUrl || DEFAULT_QWEN_BASE_URL }
              : { provider, model: '', apiKey: '', baseUrl: '' })}
          />
        </label>
        <label>
          <span>视觉模型</span>
          {draft.provider === 'qwen' ? (
            <Select
              value={draft.model}
              options={QWEN_VISION_MODELS.map((item) => ({
                value: item.value,
                label: item.label,
                title: item.description,
              }))}
              onChange={(model) => setDraft((current) => ({ ...current, model }))}
            />
          ) : (
            <Input value={draft.model} placeholder="例如：gpt-4.1-mini" onChange={(event) => setDraft((current) => ({ ...current, model: event.target.value }))} />
          )}
        </label>
        <label>
          <span>API Key {draft.provider === 'qwen' && qwenConfigured ? <small>后端已配置，可留空</small> : null}</span>
          <Input.Password value={draft.apiKey} placeholder={qwenConfigured && draft.provider === 'qwen' ? '使用后端配置，或输入临时 Key' : '仅保存在当前浏览器'} onChange={(event) => setDraft((current) => ({ ...current, apiKey: event.target.value }))} />
        </label>
        <label>
          <span>API Base URL</span>
          <Input value={draft.baseUrl} placeholder={DEFAULT_QWEN_BASE_URL} onChange={(event) => setDraft((current) => ({ ...current, baseUrl: event.target.value }))} />
        </label>
      </div>
      <div className="practice-model-security"><LockKeyhole size={15} /><span>API Key 不写入后端作答记录；标准答案始终只在服务器内用于批改。</span></div>
      <div className="practice-model-actions"><Button onClick={onClose}>取消</Button><Button type="primary" onClick={save}>保存配置</Button></div>
    </Modal>
  )
}

const verdictView: Record<PracticeGrade['verdict'], { label: string; icon: React.ReactNode; className: string }> = {
  correct: { label: '作答正确', icon: <CheckCircle2 size={22} />, className: 'correct' },
  partially_correct: { label: '部分正确', icon: <TriangleAlert size={22} />, className: 'partial' },
  incorrect: { label: '需要订正', icon: <CircleX size={22} />, className: 'incorrect' },
  unreadable: { label: '图片无法辨认', icon: <CircleAlert size={22} />, className: 'unreadable' },
}

function GradingAndTutorPanel({
  question,
  grading,
  retrying,
  resolving,
  streaming,
  pendingUser,
  streamingAnswer,
  onRetry,
  onAsk,
  onStop,
  onResolve,
  onOpenSettings,
}: {
  question: PracticeQuestion
  grading: boolean
  retrying: boolean
  resolving: boolean
  streaming: boolean
  pendingUser: string
  streamingAnswer: string
  onRetry: () => void
  onAsk: (message: string) => Promise<boolean>
  onStop: () => void
  onResolve: () => void
  onOpenSettings: () => void
}) {
  const [draft, setDraft] = useState('')
  const submission = question.submission
  const grade = submission.grade

  if (!submission.has_submission) return null
  if (grading) {
    return (
      <section className="practice-grading-panel is-loading">
        <span className="practice-grading-loader"><LoaderCircle className="spin" size={25} /></span>
        <div><strong>AI 正在依据标准答案批改</strong><p>正在识别手写步骤、公式、单位和波形，请稍候。</p></div>
      </section>
    )
  }
  if (submission.grading_status === 'pending') {
    return (
      <section className="practice-grading-panel is-waiting">
        <LoaderCircle size={23} />
        <div><strong>批改请求正在处理，或上次处理被中断</strong><p>若刷新后长时间没有结果，可以安全地恢复批改。</p></div>
        <Button type="primary" loading={retrying} onClick={onRetry}>恢复批改</Button>
      </section>
    )
  }
  if (submission.grading_status === 'ungraded') {
    return (
      <section className="practice-grading-panel is-waiting">
        <Bot size={23} />
        <div><strong>这次作答尚未进行 AI 批改</strong><p>可直接使用已保存的图片开始批改，无需重新上传。</p></div>
        <div className="practice-grading-retry-actions"><Button icon={<Settings2 size={15} />} onClick={onOpenSettings}>模型设置</Button><Button type="primary" loading={retrying} onClick={onRetry}>开始批改</Button></div>
      </section>
    )
  }
  if (submission.grading_status === 'failed') {
    return (
      <section className="practice-grading-panel is-failed">
        <CircleAlert size={23} />
        <div><strong>本次作答已保存，但 AI 批改失败</strong><p>{submission.grading_error || '请检查视觉模型配置后重试。'}</p></div>
        <div className="practice-grading-retry-actions"><Button icon={<Settings2 size={15} />} onClick={onOpenSettings}>模型设置</Button><Button type="primary" loading={retrying} onClick={onRetry}>重新批改</Button></div>
      </section>
    )
  }
  if (!grade) return null

  const verdict = verdictView[grade.verdict]
  const canDiscuss = !submission.resolved && grade.verdict !== 'unreadable'
  return (
    <section className={`practice-feedback-card ${verdict.className}`}>
      <div className="practice-feedback-heading">
        <span className="practice-verdict-icon">{verdict.icon}</span>
        <div><span>AI 批改结果</span><h2>{verdict.label}</h2><div className="practice-feedback-summary"><MathMarkdown content={grade.summary} /></div></div>
        <small>{grade.model} · {formatSubmissionTime(grade.graded_at)}</small>
      </div>

      {grade.strengths.length > 0 && (
        <div className="practice-feedback-section strengths">
          <h3><Check size={16} /> 做得好的地方</h3>
          <ul>{grade.strengths.map((item, index) => <li key={`${index}-${item}`}><MathMarkdown content={item} /></li>)}</ul>
        </div>
      )}
      {grade.issues.length > 0 && (
        <div className="practice-feedback-section issues">
          <h3><TriangleAlert size={16} /> 需要检查的地方</h3>
          <div className="practice-issue-list">
            {grade.issues.map((issue, index) => (
              <article key={`${index}-${issue.location}`}>
                <span>{index + 1}</span>
                <div>
                  <div className="practice-issue-location"><MathMarkdown content={issue.location || '作答过程'} /></div>
                  <div className="practice-issue-problem"><MathMarkdown content={issue.problem} /></div>
                  <div className="practice-issue-correction"><span>建议：</span><MathMarkdown content={issue.correction} /></div>
                </div>
              </article>
            ))}
          </div>
        </div>
      )}
      {grade.solution_markdown && (
        <div className="practice-feedback-section solution">
          <h3><BookOpenCheck size={16} /> 完整解答</h3>
          <div className="practice-solution-markdown"><MathMarkdown content={grade.solution_markdown} /></div>
        </div>
      )}

      {grade.verdict === 'unreadable' ? (
        <div className="practice-unreadable-note"><ImagePlus size={18} /><span>请在上方重新上传光线均匀、文字清晰且页面完整的作答图片。</span></div>
      ) : (
        <div className="practice-tutor-area">
          <div className="practice-tutor-heading"><MessageCircle size={18} /><div><strong>继续追问 AI 助教</strong><span>可以询问某一步为什么这样算，或请它换一种方式讲解。</span></div></div>
          <div className="practice-chat-thread">
            {submission.conversation.map((item) => (
              <div key={item.id} className={`practice-chat-message ${item.role}`}>
                <span>{item.role === 'assistant' ? <Bot size={16} /> : '你'}</span>
                <div>{item.role === 'assistant' ? <MathMarkdown content={item.content} /> : item.content}</div>
              </div>
            ))}
            {pendingUser && <div className="practice-chat-message user pending"><span>你</span><div>{pendingUser}</div></div>}
            {streaming && (
              <div className="practice-chat-message assistant streaming">
                <span><Bot size={16} /></span>
                <div>{streamingAnswer ? <MathMarkdown content={streamingAnswer} /> : <span className="practice-thinking"><i /><i /><i /> 正在组织解答</span>}</div>
              </div>
            )}
          </div>
          {canDiscuss && (
            <div className="practice-chat-composer">
              <Input.TextArea
                value={draft}
                autoSize={{ minRows: 2, maxRows: 5 }}
                maxLength={4000}
                placeholder="例如：为什么这里要先判断二极管是否导通？"
                disabled={streaming}
                onChange={(event) => setDraft(event.target.value)}
                onPressEnter={(event) => {
                  if (!event.shiftKey) {
                    event.preventDefault()
                    const content = draft.trim()
                    if (content) void onAsk(content).then((sent) => sent && setDraft(''))
                  }
                }}
              />
              {streaming ? (
                <Button danger icon={<Square size={14} />} onClick={onStop}>停止</Button>
              ) : (
                <Button type="primary" icon={<Send size={15} />} disabled={!draft.trim()} onClick={() => {
                  const content = draft.trim()
                  if (content) void onAsk(content).then((sent) => sent && setDraft(''))
                }}>发送</Button>
              )}
            </div>
          )}
        </div>
      )}

      {submission.resolved ? (
        <div className="practice-resolved-banner"><CheckCircle2 size={19} /><div><strong>你已确认掌握本题</strong><span>本题已计入章节完成进度。</span></div></div>
      ) : grade.verdict !== 'unreadable' ? (
        <Button className="practice-resolve-button" type="primary" size="large" loading={resolving} disabled={streaming} onClick={onResolve}>
          <CheckCircle2 size={18} /> {question.next_question_id ? '我已弄懂，进入下一题' : '我已弄懂，完成本章'}
        </Button>
      ) : null}
    </section>
  )
}

function UploadArea({
  images,
  submitting,
  onFiles,
  onRemove,
  onSubmit,
}: {
  images: SelectedImage[]
  submitting: boolean
  onFiles: (files: File[]) => void
  onRemove: (id: string) => void
  onSubmit: () => void
}) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)
  const receiveDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    setDragging(false)
    onFiles(Array.from(event.dataTransfer.files))
  }
  return (
    <section className="practice-upload-section">
      <div className="practice-upload-heading">
        <div>
          <span className="practice-upload-icon"><ImagePlus size={19} /></span>
          <div><h3>提交你的解题过程</h3><p>支持手写计算、波形图和电路分析过程</p></div>
        </div>
        <span className="practice-upload-count">{images.length}/{MAX_IMAGES} 张</span>
      </div>
      <input
        ref={inputRef}
        className="practice-file-input"
        type="file"
        accept="image/png,image/jpeg,image/webp,image/bmp"
        multiple
        onChange={(event) => {
          onFiles(Array.from(event.target.files || []))
          event.target.value = ''
        }}
      />
      <div
        className={`practice-dropzone ${dragging ? 'dragging' : ''}`}
        onDragOver={(event) => { event.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={receiveDrop}
      >
        <UploadCloud size={28} />
        <strong>拖入作答图片，或点击选择</strong>
        <span>PNG / JPEG / WebP / BMP，最多 5 张，合计不超过 20MB</span>
        <Button onClick={() => inputRef.current?.click()} disabled={images.length >= MAX_IMAGES}>选择图片</Button>
      </div>
      {images.length > 0 && (
        <div className="practice-preview-grid">
          {images.map((image, index) => (
            <div className="practice-preview-item" key={image.id}>
              <Image src={image.preview} alt={`作答图片 ${index + 1}`} preview={{ mask: '查看大图' }} />
              <span>第 {index + 1} 张</span>
              <button type="button" onClick={() => onRemove(image.id)} aria-label={`移除第 ${index + 1} 张图片`}>
                <Trash2 size={14} />
              </button>
            </div>
          ))}
        </div>
      )}
      <Button
        type="primary"
        className="practice-submit-button"
        icon={<FileCheck2 size={17} />}
        loading={submitting}
        disabled={!images.length}
        onClick={onSubmit}
      >
        {submitting ? '正在保存作答…' : '提交本题作答'}
      </Button>
      <p className="practice-answer-note"><ShieldCheck size={14} /> 提交后由多模态模型依据后端标准答案批改；原始答案库和答案图不会发送到页面。</p>
    </section>
  )
}

function QuestionPage({
  onProgressChanged,
  onSessionRefresh,
  activeSession,
  catalog,
  stopping,
  onBegin,
  onFinish,
  modelConfig,
  onModelConfig,
  qwenConfigured,
  qwenBaseUrl,
}: {
  onProgressChanged: () => Promise<void>
  onSessionRefresh: () => Promise<void>
  activeSession: PracticeSession | null
  catalog: PracticeCatalog
  stopping: boolean
  onBegin: (questionId: string) => void
  onFinish: () => void
  modelConfig: PracticeModelConfig
  onModelConfig: (config: PracticeModelConfig) => void
  qwenConfigured: boolean
  qwenBaseUrl: string
}) {
  const { questionId = '1.1.1' } = useParams()
  const chapterQuestions = catalog.courses[0].chapters.find(
    (chapter) => chapter.questions.some((item) => item.id === questionId),
  )?.questions || []
  const navigate = useNavigate()
  const studentId = useChatStore((state) => state.studentId)
  const { message } = AntApp.useApp()
  const [question, setQuestion] = useState<PracticeQuestion>()
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [images, setImages] = useState<SelectedImage[]>([])
  const [submitting, setSubmitting] = useState(false)
  const [grading, setGrading] = useState(false)
  const [retrying, setRetrying] = useState(false)
  const [resolving, setResolving] = useState(false)
  const [modelOpen, setModelOpen] = useState(false)
  const [streaming, setStreaming] = useState(false)
  const [pendingUser, setPendingUser] = useState('')
  const [streamingAnswer, setStreamingAnswer] = useState('')
  const imagesRef = useRef<SelectedImage[]>([])
  const streamController = useRef<AbortController | undefined>(undefined)

  const loadQuestion = async (showLoading = true) => {
    if (showLoading) setLoading(true)
    setError('')
    try {
      setQuestion(await fetchPracticeQuestion(questionId, studentId))
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : '题目加载失败')
    } finally {
      if (showLoading) setLoading(false)
    }
  }

  useEffect(() => {
    streamController.current?.abort()
    setPendingUser('')
    setStreamingAnswer('')
    setStreaming(false)
    void loadQuestion()
    if (activeSession) {
      void visitPracticeQuestion(activeSession.session_id, questionId, studentId)
        .catch(() => undefined)
    }
  }, [questionId, studentId, activeSession?.session_id])
  useEffect(() => { imagesRef.current = images }, [images])
  useEffect(() => () => {
    streamController.current?.abort()
    imagesRef.current.forEach((image) => URL.revokeObjectURL(image.preview))
  }, [])

  const totalSelectedBytes = useMemo(
    () => images.reduce((total, image) => total + image.file.size, 0),
    [images],
  )

  const addFiles = (files: File[]) => {
    const accepted = files.filter((file) => ACCEPTED_IMAGE_TYPES.has(file.type))
    if (accepted.length !== files.length) message.warning('仅支持 PNG、JPEG、WebP 和 BMP 图片')
    const available = MAX_IMAGES - images.length
    if (accepted.length > available) message.warning(`每道题最多上传 ${MAX_IMAGES} 张图片`)
    const additions = accepted.slice(0, available)
    const nextBytes = totalSelectedBytes + additions.reduce((sum, file) => sum + file.size, 0)
    if (nextBytes > MAX_TOTAL_BYTES) {
      message.error('一次提交的图片合计不能超过 20MB')
      return
    }
    setImages((current) => [...current, ...additions.map((file) => ({
      id: crypto.randomUUID(),
      file,
      preview: URL.createObjectURL(file),
    }))])
  }

  const removeImage = (id: string) => {
    setImages((current) => {
      const target = current.find((image) => image.id === id)
      if (target) URL.revokeObjectURL(target.preview)
      return current.filter((image) => image.id !== id)
    })
  }

  const submit = async () => {
    if (!question || !images.length) return
    if (!activeSession) {
      onBegin(question.id)
      return
    }
    if (!modelIsReady(modelConfig, qwenConfigured)) {
      setModelOpen(true)
      message.warning('请先配置可用的多模态批改模型')
      return
    }
    setSubmitting(true)
    try {
      const result = await submitPracticeAnswer(
        question.id,
        studentId,
        images.map((image) => image.file),
        activeSession.session_id,
      )
      images.forEach((image) => URL.revokeObjectURL(image.preview))
      setImages([])
      await loadQuestion(false)
      setGrading(true)
      try {
        await gradePracticeSubmission(question.id, result.submission_id, studentId, modelConfig)
        message.success(`第 ${result.attempt_number} 次作答已完成 AI 批改`)
      } catch (gradingError) {
        message.error(gradingError instanceof Error ? gradingError.message : '作答已保存，但 AI 批改失败')
      } finally {
        setGrading(false)
        await Promise.all([loadQuestion(false), onProgressChanged(), onSessionRefresh()])
      }
    } catch (nextError) {
      message.error(nextError instanceof Error ? nextError.message : '提交失败，请稍后重试')
    } finally {
      setSubmitting(false)
    }
  }

  const retryGrade = async () => {
    if (!question?.submission.latest_submission_id) return
    if (!modelIsReady(modelConfig, qwenConfigured)) {
      setModelOpen(true)
      message.warning('请先完成多模态模型配置')
      return
    }
    setRetrying(true)
    try {
      await gradePracticeSubmission(question.id, question.submission.latest_submission_id, studentId, modelConfig)
      await loadQuestion(false)
      message.success('AI 已重新完成批改')
    } catch (nextError) {
      await loadQuestion(false)
      message.error(nextError instanceof Error ? nextError.message : '重新批改失败')
    } finally {
      setRetrying(false)
    }
  }

  const askTutor = async (content: string) => {
    const submissionId = question?.submission.latest_submission_id
    if (!question || !submissionId || streaming) return false
    if (!modelIsReady(modelConfig, qwenConfigured)) {
      setModelOpen(true)
      message.warning('请先完成多模态模型配置')
      return false
    }
    const controller = new AbortController()
    streamController.current = controller
    setPendingUser(content)
    setStreamingAnswer('')
    setStreaming(true)
    try {
      await streamPracticeFollowup(
        question.id,
        submissionId,
        studentId,
        content,
        modelConfig,
        {
          onDelta: (chunk) => setStreamingAnswer((current) => current + chunk),
          onDone: () => undefined,
        },
        controller.signal,
      )
      await loadQuestion(false)
      return true
    } catch (nextError) {
      if (nextError instanceof DOMException && nextError.name === 'AbortError') {
        message.info('已停止本次回答')
      } else {
        message.error(nextError instanceof Error ? nextError.message : 'AI 答疑失败')
      }
      return false
    } finally {
      setStreaming(false)
      setPendingUser('')
      setStreamingAnswer('')
      streamController.current = undefined
    }
  }

  const resolve = async () => {
    const submissionId = question?.submission.latest_submission_id
    if (!question || !submissionId) return
    setResolving(true)
    try {
      const result = await resolvePracticeSubmission(question.id, submissionId, studentId)
      await onProgressChanged()
      message.success('本题已确认完成')
      navigate(result.next_question_id ? practiceQuestionPath(result.next_question_id) : COURSE_PATH)
    } catch (nextError) {
      message.error(nextError instanceof Error ? nextError.message : '确认完成失败')
    } finally {
      setResolving(false)
    }
  }

  if (loading) return <PageLoading />
  if (error || !question) return <PageError message={error || '题目不存在'} onRetry={() => void loadQuestion(true)} />
  const percent = Math.round(question.position / question.total * 100)
  return (
    <div className="practice-question-page">
      <div className="practice-question-topbar">
        <Link to={COURSE_PATH}><ArrowLeft size={15} /> 返回章节</Link>
        <div className="practice-question-progress">
          <span>第一章</span>
          <Progress percent={percent} showInfo={false} strokeColor="#159080" trailColor="#dce9e6" />
          <strong>{question.position} / {question.total}</strong>
        </div>
        <div className="practice-question-top-actions">
          <button type="button" className={`practice-model-chip ${modelIsReady(modelConfig, qwenConfigured) ? 'ready' : 'missing'}`} onClick={() => setModelOpen(true)}>
            <Bot size={16} /><span>{modelConfig.model || '配置视觉模型'}</span><Settings2 size={14} />
          </button>
        </div>
      </div>

      {activeSession && (
        <div className="practice-question-session-wrap">
          <PracticeSessionBar session={activeSession} stopping={stopping} onFinish={onFinish} />
        </div>
      )}

      <div className="practice-question-layout">
        <article className="practice-paper">
          <div className="practice-question-meta">
            <div><span>QUESTION {question.number}</span><h1>{question.title}</h1><p>{question.section}</p></div>
            {question.submission.has_submission && !question.submission.resolved ? (
              <span className="practice-complete-badge pending"><Bot size={14} /> 待确认</span>
            ) : question.submission.completed ? (
              <span className="practice-complete-badge"><Check size={14} /> 已掌握</span>
            ) : null}
          </div>
          <div className="practice-prompt"><MathMarkdown content={question.prompt_markdown} /></div>
          {question.figures.length > 0 && (
            <div className="practice-figure-list">
              {question.figures.map((figure) => (
                <figure key={figure.id}>
                  <Image src={figure.url} alt={figure.alt} preview={{ mask: '放大查看' }} />
                  <figcaption>{figure.caption}</figcaption>
                </figure>
              ))}
            </div>
          )}
          {question.submission.has_submission && (
            <div className="practice-submitted-panel">
              <span>{question.submission.completed ? <Check size={17} /> : <Bot size={17} />}</span>
              <div>
                <strong>{question.submission.completed && !question.submission.resolved ? '本题曾经掌握，本次新作答待确认' : question.submission.completed ? '本题已确认掌握，可继续重练' : '本题作答已安全保存'}</strong>
                <p>共提交 {question.submission.attempt_count} 次{question.submission.last_submitted_at ? ` · 最近 ${formatSubmissionTime(question.submission.last_submitted_at)}` : ''}</p>
              </div>
            </div>
          )}
          {activeSession ? (
            <UploadArea
              images={images}
              submitting={submitting}
              onFiles={addFiles}
              onRemove={removeImage}
              onSubmit={() => void submit()}
            />
          ) : (
            <div className="practice-session-gate">
              <Play size={22} />
              <div><strong>开始新一轮练习后提交答案</strong><p>本轮反馈只统计你新上传作答的题目，浏览和切题不会计入。</p></div>
              <Button type="primary" onClick={() => onBegin(question.id)}>开始本轮练习</Button>
            </div>
          )}
          <GradingAndTutorPanel
            question={question}
            grading={grading}
            retrying={retrying}
            resolving={resolving}
            streaming={streaming}
            pendingUser={pendingUser}
            streamingAnswer={streamingAnswer}
            onRetry={() => void retryGrade()}
            onAsk={askTutor}
            onStop={() => streamController.current?.abort()}
            onResolve={() => void resolve()}
            onOpenSettings={() => setModelOpen(true)}
          />
        </article>

        <aside className="practice-question-aside">
          <div className="practice-aside-card">
            <span className="practice-aside-icon"><BookOpenCheck size={18} /></span>
            <h2>本章练习</h2>
            <p>按题号顺序练习，经 AI 批改并确认理解后计入完成进度。</p>
            <div className="practice-question-dots" aria-label={`当前第 ${question.position} 题，共 ${question.total} 题`}>
              {chapterQuestions.map((item, index) => {
                const included = Boolean(activeSession?.submitted_question_ids.includes(item.id))
                const className = [
                  item.completed ? 'passed' : '',
                  included ? 'in-session' : '',
                  item.id === question.id ? 'current' : '',
                ].filter(Boolean).join(' ')
                return <button type="button" key={item.id} className={className} title={`${item.id} ${item.title}`} onClick={() => navigate(practiceQuestionPath(item.id))}>{index + 1}</button>
              })}
            </div>
          </div>
          <div className="practice-tip-card">
            <Sparkles size={18} />
            <div><strong>作答建议</strong><p>请保留计算过程、单位和关键波形，便于 AI 准确对照标准答案。</p></div>
          </div>
        </aside>
      </div>

      <footer className="practice-question-footer">
        <Button
          icon={<ArrowLeft size={16} />}
          disabled={!question.previous_question_id}
          onClick={() => question.previous_question_id && navigate(practiceQuestionPath(question.previous_question_id))}
        >上一题</Button>
        <span>题目 {question.number}</span>
        <Button
          type="primary"
          disabled={!question.next_question_id || !question.submission.resolved}
          onClick={() => question.next_question_id && navigate(practiceQuestionPath(question.next_question_id))}
        >{question.next_question_id ? '下一题' : '已是最后一题'} <ArrowRight size={16} /></Button>
      </footer>
      <PracticeModelSettings
        open={modelOpen}
        value={modelConfig}
        qwenConfigured={qwenConfigured}
        qwenBaseUrl={qwenBaseUrl}
        onClose={() => setModelOpen(false)}
        onSave={onModelConfig}
      />
    </div>
  )
}

function PracticePageContent() {
  const navigate = useNavigate()
  const studentId = useChatStore((state) => state.studentId)
  const chatModel = useChatStore((state) => state.modelConfig)
  const { message, modal } = AntApp.useApp()
  const [catalog, setCatalog] = useState<PracticeCatalog>()
  const [modelCatalog, setModelCatalog] = useState<ModelCatalog>()
  const [practiceModel, setPracticeModel] = useState<PracticeModelConfig>(() => initialPracticeModel(chatModel))
  const [activeSession, setActiveSession] = useState<PracticeSession | null>(null)
  const [stoppingSession, setStoppingSession] = useState(false)
  const [globalModelOpen, setGlobalModelOpen] = useState(false)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const loadCatalog = async () => {
    setError('')
    try {
      setCatalog(await fetchPracticeCatalog(studentId))
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : '题库目录加载失败')
    } finally {
      setLoading(false)
    }
  }
  const loadActiveSession = async () => {
    try {
      setActiveSession(await fetchActivePracticeSession(studentId))
    } catch (nextError) {
      message.error(nextError instanceof Error ? nextError.message : '当前练习状态加载失败')
    }
  }
  useEffect(() => {
    void loadCatalog()
    void loadActiveSession()
  }, [studentId])
  useEffect(() => {
    void fetchModels().then(setModelCatalog).catch(() => setModelCatalog(undefined))
  }, [])

  const qwenInfo = modelCatalog?.providers.find((provider) => provider.id === 'qwen')
  const savePracticeModel = (config: PracticeModelConfig) => {
    localStorage.setItem(PRACTICE_MODEL_STORAGE_KEY, JSON.stringify(config))
    setPracticeModel(config)
  }

  const beginSession = (questionId: string) => {
    if (activeSession) {
      navigate(practiceQuestionPath(questionId))
      return
    }
    modal.confirm({
      title: '开始一轮新的练习？',
      content: '学情反馈只统计本轮新上传的作答。仅浏览题目、切换题号或查看历史批改不会计入。',
      okText: '开始本轮练习',
      cancelText: '暂不开始',
      async onOk() {
        const session = await startPracticeSession(studentId, questionId)
        setActiveSession(session)
        navigate(practiceQuestionPath(questionId))
      },
    })
  }

  const finishSession = async () => {
    if (!activeSession || stoppingSession) return
    const session = await fetchActivePracticeSession(studentId).catch(() => activeSession)
    if (!session) return
    setActiveSession(session)
    const empty = session.submission_count === 0
    if (!empty && !modelIsReady(practiceModel, Boolean(qwenInfo?.configured))) {
      setGlobalModelOpen(true)
      message.warning('请先配置可用的模型，以生成本轮学情反馈')
      return
    }
    modal.confirm({
      title: empty ? '结束本轮练习？' : '结束本轮并生成学情反馈？',
      content: empty
        ? '本轮尚未提交答案，结束后不会生成或保存空反馈。'
        : `反馈只包含本轮提交的 ${session.submitted_question_count} 道题、${session.submission_count} 次作答：${session.submitted_question_ids.join('、')}`,
      okText: empty ? '直接结束' : '结束并生成反馈',
      cancelText: '继续练习',
      async onOk() {
        setStoppingSession(true)
        try {
          const result = empty
            ? await discardEmptyPracticeSession(session.session_id, studentId)
            : await finishPracticeSession(
              session.session_id,
              studentId,
              practiceModel,
            )
          setActiveSession(null)
          await loadCatalog()
          if (result.status === 'discarded') {
            message.info('本轮没有实际作答，已结束且未生成反馈')
            navigate(COURSE_PATH)
          } else {
            message.success('本轮作答记录和学情反馈已保存')
            navigate(`/practice/feedback/${result.session_id}`)
          }
        } catch (nextError) {
          const remaining = await fetchActivePracticeSession(studentId).catch(() => activeSession)
          setActiveSession(remaining)
          if (!remaining) {
            message.error('本轮记录已保存，但反馈生成失败，可以在反馈详情中重试')
            navigate(`/practice/feedback/${session.session_id}`)
            return
          }
          message.error(nextError instanceof Error ? nextError.message : '学情反馈生成失败')
          throw nextError
        } finally {
          setStoppingSession(false)
        }
      },
    })
  }

  return (
    <PracticeShell>
      {loading ? <PageLoading /> : error || !catalog ? (
        <PageError message={error || '题库目录为空'} onRetry={() => { setLoading(true); void loadCatalog() }} />
      ) : (
        <Routes>
          <Route index element={<PracticeHome catalog={catalog} />} />
          <Route path="feedback" element={<PracticeFeedbackList />} />
          <Route path="feedback/:sessionId" element={<PracticeFeedbackDetail modelConfig={practiceModel} />} />
          <Route path="electronic-circuits" element={(
            <CoursePage
              catalog={catalog}
              activeSession={activeSession}
              stopping={stoppingSession}
              onBegin={beginSession}
              onFinish={finishSession}
            />
          )} />
          <Route path="electronic-circuits/:chapterId/questions/:questionId" element={(
            <QuestionPage
              onProgressChanged={loadCatalog}
              onSessionRefresh={loadActiveSession}
              activeSession={activeSession}
              catalog={catalog}
              stopping={stoppingSession}
              onBegin={beginSession}
              onFinish={finishSession}
              modelConfig={practiceModel}
              onModelConfig={savePracticeModel}
              qwenConfigured={Boolean(qwenInfo?.configured)}
              qwenBaseUrl={qwenInfo?.base_url || DEFAULT_QWEN_BASE_URL}
            />
          )} />
          <Route path="*" element={<Navigate to="/practice" replace />} />
        </Routes>
      )}
      <PracticeModelSettings
        open={globalModelOpen}
        value={practiceModel}
        qwenConfigured={Boolean(qwenInfo?.configured)}
        qwenBaseUrl={qwenInfo?.base_url || DEFAULT_QWEN_BASE_URL}
        onClose={() => setGlobalModelOpen(false)}
        onSave={savePracticeModel}
      />
    </PracticeShell>
  )
}

export default function PracticePage() {
  return <AntApp><PracticePageContent /></AntApp>
}
