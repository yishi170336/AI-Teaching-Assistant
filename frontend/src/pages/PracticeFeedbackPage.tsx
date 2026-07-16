import { useEffect, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { App as AntApp, Button, Spin } from 'antd'
import {
  ArrowLeft,
  ArrowRight,
  BookOpenCheck,
  BrainCircuit,
  CalendarClock,
  CheckCircle2,
  CircleAlert,
  ClipboardList,
  Lightbulb,
  Target,
  Trash2,
} from 'lucide-react'
import MathMarkdown from '../components/MathMarkdown'
import {
  deletePracticeSession,
  fetchPracticeSession,
  fetchPracticeSessions,
  finishPracticeSession,
  type PracticeModelConfig,
  type PracticeSession,
} from '../lib/practiceApi'
import { useChatStore } from '../store/chatStore'


function formatDate(value: string | null) {
  if (!value) return '生成中'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return ''
  return date.toLocaleString('zh-CN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  })
}

function FeedbackHeader({ detail = false }: { detail?: boolean }) {
  return (
    <header className="practice-feedback-page-header">
      <div>
        {detail && <Link to="/practice/feedback"><ArrowLeft size={15} /> 返回反馈列表</Link>}
        <span>LEARNING INSIGHTS</span>
        <h1>{detail ? '本次练习反馈' : '学情反馈'}</h1>
        <p>每次停止做题后，AI 会只依据本次练习记录总结表现，不混入以前的作答历史。</p>
      </div>
      <BrainCircuit size={42} />
    </header>
  )
}

export function PracticeFeedbackList() {
  const studentId = useChatStore((state) => state.studentId)
  const { message, modal } = AntApp.useApp()
  const [sessions, setSessions] = useState<PracticeSession[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  const load = async () => {
    setLoading(true)
    setError('')
    try {
      setSessions(await fetchPracticeSessions(studentId))
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : '学情反馈加载失败')
    } finally {
      setLoading(false)
    }
  }
  useEffect(() => { void load() }, [studentId])

  const remove = (session: PracticeSession) => {
    modal.confirm({
      title: '删除这份学情反馈？',
      content: '删除后无法恢复，但不会删除题目作答和批改记录。',
      okText: '删除',
      okButtonProps: { danger: true },
      cancelText: '取消',
      async onOk() {
        await deletePracticeSession(session.session_id, studentId)
        setSessions((current) => current.filter((item) => item.session_id !== session.session_id))
        message.success('学情反馈已删除')
      },
    })
  }

  if (loading) return <div className="practice-page-state"><Spin size="large" /><span>正在读取学情反馈…</span></div>
  if (error) return <div className="practice-page-state error"><CircleAlert size={28} /><strong>学情反馈暂时无法加载</strong><span>{error}</span><Button onClick={() => void load()}>重新加载</Button></div>

  return (
    <div className="practice-feedback-page">
      <FeedbackHeader />
      <section className="practice-feedback-library">
        {sessions.length === 0 ? (
          <div className="practice-feedback-empty">
            <ClipboardList size={36} />
            <h2>还没有练习反馈</h2>
            <p>进入题目开始练习，结束时点击“停止做题并生成反馈”，这里就会保存本次总结。</p>
            <Link to="/practice/electronic-circuits">开始刷题 <ArrowRight size={15} /></Link>
          </div>
        ) : (
          <div className="practice-feedback-list">
            {sessions.map((session) => (
              <article className="practice-feedback-record" key={session.session_id}>
                <Link to={`/practice/feedback/${session.session_id}`}>
                  <div className="practice-feedback-record-icon"><BrainCircuit size={22} /></div>
                  <div className="practice-feedback-record-copy">
                    <span><CalendarClock size={14} /> {formatDate(session.ended_at)}</span>
                    <h2>{session.feedback?.headline || (session.feedback_status === 'failed' ? '本次反馈生成失败' : '本次练习反馈')}</h2>
                    {session.scope_version >= 2 ? (
                      <p>本轮实际作答：{session.submitted_question_ids.join('、')} · {session.submitted_question_count} 道题 / {session.submission_count} 次提交{session.feedback_status === 'failed' ? ' · 点击查看并重新生成' : ''}</p>
                    ) : (
                      <p>旧版记录：按访问范围统计 · {session.question_ids.join('、')}</p>
                    )}
                  </div>
                  <ArrowRight size={18} />
                </Link>
                <button type="button" onClick={() => remove(session)} aria-label="删除这份学情反馈"><Trash2 size={16} /></button>
              </article>
            ))}
          </div>
        )}
      </section>
    </div>
  )
}

export function PracticeFeedbackDetail({ modelConfig }: { modelConfig: PracticeModelConfig }) {
  const { sessionId = '' } = useParams()
  const studentId = useChatStore((state) => state.studentId)
  const navigate = useNavigate()
  const { message, modal } = AntApp.useApp()
  const [session, setSession] = useState<PracticeSession>()
  const [loading, setLoading] = useState(true)
  const [retrying, setRetrying] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    setLoading(true)
    fetchPracticeSession(sessionId, studentId)
      .then(setSession)
      .catch((nextError) => setError(nextError instanceof Error ? nextError.message : '学情反馈加载失败'))
      .finally(() => setLoading(false))
  }, [sessionId, studentId])

  const remove = () => {
    if (!session) return
    modal.confirm({
      title: '删除这份学情反馈？',
      content: '删除后无法恢复，但不会影响原有作答记录。',
      okText: '删除',
      okButtonProps: { danger: true },
      cancelText: '取消',
      async onOk() {
        await deletePracticeSession(session.session_id, studentId)
        message.success('学情反馈已删除')
        navigate('/practice/feedback')
      },
    })
  }

  const retry = async () => {
    if (!session || retrying) return
    setRetrying(true)
    try {
      const result = await finishPracticeSession(
        session.session_id,
        studentId,
        modelConfig,
      )
      setSession(result)
      message.success('学情反馈已重新生成并保存')
    } catch (nextError) {
      message.error(nextError instanceof Error ? nextError.message : '重新生成反馈失败')
    } finally {
      setRetrying(false)
    }
  }

  if (loading) return <div className="practice-page-state"><Spin size="large" /><span>正在读取本次反馈…</span></div>
  if (error || !session) return <div className="practice-page-state error"><CircleAlert size={28} /><strong>无法读取本次反馈</strong><span>{error || '记录不存在'}</span><Button onClick={() => navigate('/practice/feedback')}>返回列表</Button></div>

  const feedback = session.feedback
  return (
    <div className="practice-feedback-page">
      <FeedbackHeader detail />
      <section className="practice-feedback-detail">
        <div className="practice-feedback-detail-meta">
          <span><CalendarClock size={15} /> {formatDate(session.started_at)} — {formatDate(session.ended_at)}</span>
          <span><BookOpenCheck size={15} /> {session.scope_version >= 2 ? `实际作答 ${session.submitted_question_count} 道 / ${session.submission_count} 次提交` : `旧版访问范围 ${session.question_count} 道题`}</span>
          <Button danger type="text" icon={<Trash2 size={15} />} onClick={remove}>删除反馈</Button>
        </div>

        <div className={`practice-feedback-scope-note ${session.scope_version >= 2 ? '' : 'legacy'}`}>
          <Target size={17} />
          <div>
            <strong>{session.scope_version >= 2 ? '本轮反馈统计范围' : '旧版反馈统计范围'}</strong>
            <p>{session.scope_version >= 2
              ? `仅包含本轮新提交的题目：${session.submitted_question_ids.join('、')}。浏览、切题和历史作答均未纳入。`
              : `这份旧记录按当时访问过的题目统计：${session.question_ids.join('、')}。报告内容保持原样。`}</p>
          </div>
        </div>

        {session.feedback_status === 'failed' || !feedback ? (
          <div className="practice-feedback-failed">
            <CircleAlert size={30} />
            <h2>本次记录已保存，但反馈生成失败</h2>
            <p>{session.feedback_error || '可以直接重新生成，不需要重新做题。'}</p>
            <Button type="primary" loading={retrying} onClick={() => void retry()}>重新生成反馈</Button>
          </div>
        ) : (
          <>
            <article className="practice-feedback-overview">
              <span>AI 学情总结</span>
              <h2>{feedback.headline}</h2>
              <MathMarkdown content={feedback.summary_markdown} />
            </article>

            <div className="practice-feedback-detail-grid">
              <section>
                <h2><CheckCircle2 size={18} /> 本次做得好的地方</h2>
                <ul>{feedback.strengths.map((item, index) => <li key={index}><MathMarkdown content={item} /></li>)}</ul>
              </section>
              <section>
                <h2><Target size={18} /> 后续重点</h2>
                <ul>{feedback.focus_areas.map((item, index) => <li key={index}><MathMarkdown content={item} /></li>)}</ul>
              </section>
            </div>

            <section className="practice-feedback-question-section">
              <div className="practice-feedback-section-title"><ClipboardList size={19} /><div><span>逐题回顾</span><h2>本次做了什么、错在什么步骤</h2></div></div>
              <div className="practice-feedback-question-list">
                {feedback.question_reviews.map((review) => (
                  <article key={review.question_id}>
                    <span>题目 {review.question_id}</span>
                    <MathMarkdown content={review.what_was_done} />
                    {review.error_steps.length > 0 && <div className="practice-feedback-errors"><strong>需要订正的步骤</strong>{review.error_steps.map((item, index) => <MathMarkdown key={index} content={item} />)}</div>}
                    {review.advice.length > 0 && <div className="practice-feedback-advice"><strong>本题建议</strong>{review.advice.map((item, index) => <MathMarkdown key={index} content={item} />)}</div>}
                  </article>
                ))}
              </div>
            </section>

            <section className="practice-feedback-next-steps">
              <div><Lightbulb size={20} /><span>下一步学习建议</span></div>
              <ol>{feedback.recommendations.map((item, index) => <li key={index}><MathMarkdown content={item} /></li>)}</ol>
            </section>
          </>
        )}
      </section>
    </div>
  )
}
