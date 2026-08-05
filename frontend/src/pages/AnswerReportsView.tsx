import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  App as AntApp,
  Button,
  Checkbox,
  Empty,
  Input,
  Popconfirm,
  Select,
  Spin,
  Tag,
} from 'antd'
import {
  AlertTriangle,
  BarChart3,
  CheckCircle2,
  FilePlus2,
  History,
  Printer,
  RefreshCw,
  Trash2,
} from 'lucide-react'
import MathMarkdown from '../components/MathMarkdown'
import {
  AnswerReport,
  AnswerReportSummary,
  createAnswerReport,
  deleteAnswerReport,
  fetchAnswerReport,
  fetchAnswerReports,
  fetchPracticeAttempts,
  PracticeAttempt,
} from '../lib/api'

const sourceLabels: Record<string, string> = {
  question_bank: '题库题',
  ai_generated: 'AI 生成题',
  user_uploaded: '用户上传题',
}

const stepStatusLabels: Record<string, string> = {
  correct: '正确',
  partial: '部分正确',
  incorrect: '有误',
  unverifiable: '无法核验',
}

const dimensionLabels: Record<string, string> = {
  correctness: '正确性',
  completeness: '完整性',
  logic: '逻辑性',
  notation: '规范性',
}

function displayTime(value: string) {
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString('zh-CN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  })
}

function percent(value: number | null | undefined) {
  return value == null ? '—' : `${Math.round(value * 1000) / 10}%`
}

function AttemptChoice({
  attempt,
  checked,
  onChange,
}: {
  attempt: PracticeAttempt
  checked: boolean
  onChange: (checked: boolean) => void
}) {
  return (
    <label className={`answer-attempt-choice ${checked ? 'selected' : ''}`}>
      <Checkbox checked={checked} onChange={(event) => onChange(event.target.checked)} />
      <div className="answer-attempt-choice-main">
        <div className="answer-attempt-choice-head">
          <span>{displayTime(attempt.completed_at)}</span>
          <Tag bordered={false}>{sourceLabels[attempt.question.source] || attempt.question.source_label}</Tag>
          <strong>{attempt.grading.max_score > 0 ? `${attempt.grading.score} / ${attempt.grading.max_score}` : '批改未完成'}</strong>
        </div>
        <MathMarkdown content={attempt.question.text} />
        <small>{attempt.knowledge_points.join('、') || '未标注知识点'} · {attempt.answer.submission_mode === 'image' ? '手写图片' : attempt.answer.submission_mode === 'mixed' ? '文字 + 图片' : '文字作答'}</small>
      </div>
    </label>
  )
}

function ReportDocument({ report }: { report: AnswerReport }) {
  const aggregate = report.aggregate
  return (
    <article className="answer-report-print">
      <header className="answer-report-document-head">
        <div>
          <span>学生答案（刷题）分析报告</span>
          <h1>{report.title}</h1>
          <p>生成时间：{displayTime(report.created_at)} · 报告编号：{report.id.slice(0, 10)}</p>
        </div>
        <div className="answer-report-score">
          <strong>{percent(aggregate.average_score_rate)}</strong>
          <span>综合得分率</span>
        </div>
      </header>

      <section className="answer-report-metrics">
        <div><strong>{aggregate.attempt_count}</strong><span>纳入题目</span></div>
        <div><strong>{aggregate.scored_count}</strong><span>有效计分</span></div>
        <div><strong>{aggregate.correct_count}</strong><span>判定正确</span></div>
        <div><strong>{aggregate.partial_correct_count}</strong><span>部分正确</span></div>
        <div><strong>{aggregate.incorrect_count}</strong><span>判定错误</span></div>
        <div><strong>{aggregate.ungradable_count}</strong><span>无法评判</span></div>
        <div><strong>{aggregate.repeated_errors.length}</strong><span>重复错误类型</span></div>
      </section>
      <section className="answer-report-distribution">
        <div><strong>总体表现</strong><span>{aggregate.overall_performance}</span></div>
        <div>
          <strong>题目来源</strong>
          <span>{Object.entries(aggregate.source_counts).map(([source, count]) => `${sourceLabels[source] || source} ${count} 题`).join(' · ') || '无'}</span>
        </div>
        <div><strong>参考答案口径</strong><span>有参考 {aggregate.reference_counts.with_reference} 题 · 无参考 {aggregate.reference_counts.without_reference} 题</span></div>
      </section>

      {aggregate.warnings.length > 0 && (
        <section className="answer-report-warnings">
          <h2><AlertTriangle size={17} /> 数据说明</h2>
          <ul>{aggregate.warnings.map((item) => <li key={item}>{item}</li>)}</ul>
        </section>
      )}

      <section className="answer-report-section">
        <h2><BarChart3 size={17} /> 汇总分析</h2>
        <div className="answer-report-analysis-grid">
          <div>
            <h3>知识点表现</h3>
            {aggregate.knowledge_points.length ? aggregate.knowledge_points.map((item) => (
              <div className="answer-report-stat-row" key={item.knowledge_point}>
                <span>{item.knowledge_point}{item.common_errors.length ? <small>常见：{item.common_errors.join('、')}</small> : null}</span>
                <strong>{percent(item.average_score_rate)}</strong>
                <Tag bordered={false}>{item.status}</Tag>
              </div>
            )) : <p>当前记录未包含可汇总的知识点。</p>}
          </div>
          <div>
            <h3>错误模式</h3>
            {aggregate.issue_patterns.length ? aggregate.issue_patterns.map((item) => (
              <div className="answer-report-stat-row" key={item.type}>
                <span title={item.examples.join('；')}>{item.label}<small>证据题号：{item.attempt_ids.map((id) => report.attempts.findIndex((attempt) => attempt.id === id) + 1).filter(Boolean).join('、')}</small></span>
                <strong>{item.count} 次</strong>
                <Tag bordered={false} color={item.attempt_ids.length >= 2 ? 'error' : 'default'}>
                  {item.attempt_ids.length >= 2 ? '重复出现' : '单次证据'}
                </Tag>
              </div>
            )) : <p>已保存的批改中没有结构化错误项。</p>}
          </div>
        </div>
        <div className="answer-report-recommendations">
          <h3>后续建议</h3>
          {aggregate.recommendations.length
            ? <ol>{aggregate.recommendations.map((item) => <li key={item}>{item}</li>)}</ol>
            : <p>当前没有足够证据生成专项建议，可继续完成练习后创建新报告。</p>}
        </div>
        <div className="answer-report-trend">
          <h3>表现趋势</h3>
          <strong>{aggregate.trend.label}</strong>
          {aggregate.trend.score_rate_change != null && <span>前后半段变化：{aggregate.trend.score_rate_change > 0 ? '+' : ''}{percent(aggregate.trend.score_rate_change)}</span>}
          <p>{aggregate.trend.note}</p>
        </div>
      </section>

      <section className="answer-report-section">
        <h2>逐题证据</h2>
        <div className="answer-report-attempts">
          {report.attempts.map((attempt, index) => (
            <article className="answer-report-attempt" key={attempt.id}>
              <header>
                <div>
                  <span>第 {index + 1} 题</span>
                  <Tag bordered={false}>{sourceLabels[attempt.question.source]}</Tag>
                  <Tag bordered={false} color={attempt.reference.source === 'question_bank' ? 'green' : attempt.reference.source === 'unavailable' ? 'warning' : 'blue'}>
                    {attempt.reference.source_label}
                  </Tag>
                </div>
                <strong>{attempt.grading.max_score > 0 ? `${attempt.grading.score} / ${attempt.grading.max_score}` : '未完成评判'}</strong>
              </header>
              <p className="answer-report-attempt-meta">
                作答时间：{displayTime(attempt.completed_at)} · 难度：{attempt.question.difficulty || '未标注'} · 知识点：{attempt.knowledge_points.join('、') || '未标注'} · 提交方式：{attempt.answer.submission_mode === 'image' ? '手写图片' : attempt.answer.submission_mode === 'mixed' ? '文字 + 图片' : '文字'} · 批改置信度：{attempt.grading.confidence ? percent(attempt.grading.confidence) : '未提供'}
              </p>
              <div className="answer-report-question"><MathMarkdown content={attempt.question.text} /></div>
              {attempt.question.assets.length > 0 && (
                <div className="answer-report-images">
                  {attempt.question.assets.map((asset, assetIndex) => asset.url && (
                    <img key={asset.id || asset.url} src={asset.url} alt={`第 ${index + 1} 题题图 ${assetIndex + 1}`} />
                  ))}
                </div>
              )}
              <div className="answer-report-evidence-grid">
                <div>
                  <h3>学生答案</h3>
                  <MathMarkdown content={attempt.answer.text || '未保存可读文字'} />
                  {attempt.answer.submitted_text && attempt.answer.submitted_text !== attempt.answer.text && (
                    <div className="answer-report-submitted-text"><strong>学生补充文字</strong><MathMarkdown content={attempt.answer.submitted_text} /></div>
                  )}
                  <p className="answer-report-reference-note">识别人工确认：{attempt.answer.recognition_confirmed ? '已确认' : '未确认或不适用'}</p>
                  {attempt.grading.recognition_warnings?.length ? (
                    <ul className="answer-report-recognition-warnings">
                      {attempt.grading.recognition_warnings.map((item) => <li key={item}>{item}</li>)}
                    </ul>
                  ) : null}
                  {attempt.answer.assets.length > 0 && (
                    <div className="answer-report-images answer-images">
                      {attempt.answer.assets.map((asset, assetIndex) => asset.url && (
                        <img key={asset.id || asset.url} src={asset.url} alt={`第 ${index + 1} 题作答图片 ${assetIndex + 1}`} />
                      ))}
                    </div>
                  )}
                </div>
                <div>
                  <h3>{attempt.reference.source_label}</h3>
                  <p className="answer-report-reference-note">{attempt.reference.note}</p>
                  <MathMarkdown content={attempt.reference.answer || attempt.reference.answer_items.join('\n') || '本题无可展示的参考答案'} />
                  {(attempt.reference.solution_steps.length > 0 || attempt.reference.solution) && (
                    <div className="answer-report-reference-steps">
                      <strong>参考步骤</strong>
                      {(attempt.reference.solution_steps.length ? attempt.reference.solution_steps : [attempt.reference.solution]).map((step, stepIndex) => (
                        <MathMarkdown key={`${step}-${stepIndex}`} content={`${stepIndex + 1}. ${step}`} />
                      ))}
                    </div>
                  )}
                </div>
              </div>
              <div className="answer-report-grading">
                <h3>批改结论</h3>
                <MathMarkdown content={attempt.grading.summary} />
                {attempt.grading.dimensions && Object.keys(attempt.grading.dimensions).length > 0 && (
                  <div className="answer-report-dimensions">
                    {Object.entries(attempt.grading.dimensions).map(([name, item]) => item && (
                      <div key={name}>
                        <strong>{dimensionLabels[name] || name}</strong>
                        <span>{item.feedback || '未提供具体说明'}</span>
                      </div>
                    ))}
                  </div>
                )}
                {attempt.grading.final_conclusion_correct != null && (
                  <p className="answer-report-final-conclusion">最终结论：{attempt.grading.final_conclusion_correct ? '正确' : '不正确或证据不足'}</p>
                )}
                {attempt.grading.step_analyses?.length ? (
                  <div className="answer-report-steps">
                    {attempt.grading.step_analyses.map((step, stepIndex) => (
                      <div key={`${step.step}-${stepIndex}`}>
                        <Tag bordered={false} color={step.status === 'correct' ? 'success' : step.status === 'incorrect' ? 'error' : 'warning'}>
                          {stepStatusLabels[step.status]}
                        </Tag>
                        <strong>{step.step}</strong>
                        <MathMarkdown content={step.feedback} />
                        {step.evidence && <small>答案证据：{step.evidence}</small>}
                      </div>
                    ))}
                  </div>
                ) : <p className="answer-report-limit">此历史批改未包含结构化步骤分析，保留原始问题与建议。</p>}
                {attempt.grading.issues.length > 0 && (
                  <div className="answer-report-issues">
                    {attempt.grading.issues.map((issue, issueIndex) => (
                      <div key={`${issue.title}-${issueIndex}`}>
                        <strong>{issue.type_label || issue.title}</strong>
                        <MathMarkdown content={`${issue.detail}${issue.suggestion ? `\n\n建议：${issue.suggestion}` : ''}`} />
                      </div>
                    ))}
                  </div>
                )}
                <div className="answer-report-next-steps">
                  <strong>本题改进建议</strong>
                  {attempt.grading.next_steps.length
                    ? <ul>{attempt.grading.next_steps.map((item) => <li key={item}>{item}</li>)}</ul>
                    : <p>当前没有额外建议。</p>}
                  {attempt.grading.max_score > 0 && attempt.grading.score < attempt.grading.max_score && (
                    <p>建议订正后重新练习；如需加入错题本，仍须由学生在现有错题确认流程中确认。</p>
                  )}
                </div>
              </div>
            </article>
          ))}
        </div>
      </section>
    </article>
  )
}

export default function AnswerReportsView({ studentId }: { studentId: string }) {
  const { message } = AntApp.useApp()
  const [attempts, setAttempts] = useState<PracticeAttempt[]>([])
  const [reports, setReports] = useState<AnswerReportSummary[]>([])
  const [selectedIds, setSelectedIds] = useState<string[]>([])
  const [practiceSessionId, setPracticeSessionId] = useState('all')
  const [title, setTitle] = useState('')
  const [activeReport, setActiveReport] = useState<AnswerReport>()
  const [loading, setLoading] = useState(true)
  const [creating, setCreating] = useState(false)
  const [openingId, setOpeningId] = useState('')

  const refresh = useCallback(async () => {
    setLoading(true)
    try {
      const [nextAttempts, nextReports] = await Promise.all([
        fetchPracticeAttempts(studentId),
        fetchAnswerReports(studentId),
      ])
      setAttempts(nextAttempts)
      setReports(nextReports)
      setSelectedIds((current) => current.filter((id) => nextAttempts.some((item) => item.id === id)))
    } catch (error) {
      message.error(error instanceof Error ? error.message : '答案报告数据读取失败')
    } finally {
      setLoading(false)
    }
  }, [message, studentId])

  useEffect(() => { void refresh() }, [refresh])

  const sessionOptions = useMemo(() => {
    const grouped = new Map<string, PracticeAttempt[]>()
    attempts.forEach((attempt) => grouped.set(
      attempt.practice_session_id,
      [...(grouped.get(attempt.practice_session_id) || []), attempt],
    ))
    return [
      { value: 'all', label: `全部练习（${attempts.length} 题）` },
      ...[...grouped.entries()].map(([id, items], index) => ({
        value: id,
        label: `练习组 ${index + 1} · ${displayTime(items[items.length - 1].completed_at)}（${items.length} 题）`,
      })),
    ]
  }, [attempts])

  const visibleAttempts = useMemo(
    () => attempts.filter((item) => practiceSessionId === 'all' || item.practice_session_id === practiceSessionId),
    [attempts, practiceSessionId],
  )

  const toggleAllVisible = () => {
    const visibleIds = visibleAttempts.map((item) => item.id)
    const allSelected = visibleIds.length > 0 && visibleIds.every((id) => selectedIds.includes(id))
    setSelectedIds((current) => allSelected
      ? current.filter((id) => !visibleIds.includes(id))
      : [...new Set([...current, ...visibleIds])])
  }

  const generate = async () => {
    if (!selectedIds.length) {
      message.warning('请至少选择一道已完成批改的题目')
      return
    }
    setCreating(true)
    try {
      const report = await createAnswerReport(studentId, selectedIds, title)
      setActiveReport(report)
      setReports(await fetchAnswerReports(studentId))
      setTitle('')
      message.success('分析报告已保存')
    } catch (error) {
      message.error(error instanceof Error ? error.message : '报告生成失败')
    } finally {
      setCreating(false)
    }
  }

  const openReport = async (reportId: string) => {
    setOpeningId(reportId)
    try {
      setActiveReport(await fetchAnswerReport(studentId, reportId))
    } catch (error) {
      message.error(error instanceof Error ? error.message : '报告读取失败')
    } finally {
      setOpeningId('')
    }
  }

  const removeReport = async (reportId: string) => {
    try {
      await deleteAnswerReport(studentId, reportId)
      if (activeReport?.id === reportId) setActiveReport(undefined)
      setReports((current) => current.filter((item) => item.id !== reportId))
      message.success('历史报告已删除')
    } catch (error) {
      message.error(error instanceof Error ? error.message : '报告删除失败')
    }
  }

  const printReport = () => {
    if (!activeReport) return
    const previous = document.title
    const safeTitle = activeReport.title.replace(/[^A-Za-z0-9\u4e00-\u9fff_-]+/g, '-').slice(0, 80) || '学生答案分析报告'
    document.title = `${safeTitle}-${activeReport.id.slice(0, 8)}`
    window.print()
    document.title = previous
  }

  if (loading) return <div className="answer-report-loading"><Spin /><span>正在读取练习证据与历史报告…</span></div>

  return (
    <section className="answer-reports-view">
      <div className="answer-report-toolbar no-print">
        <div>
          <span>可复核、可追溯</span>
          <h1>学生答案分析报告</h1>
          <p>从已完成的刷题批改中选择单题或多题。报告保存题目与答案快照，历史查看不会重新调用模型。</p>
        </div>
        <Button icon={<RefreshCw size={15} />} onClick={() => void refresh()}>刷新记录</Button>
      </div>

      <div className="answer-report-workbench no-print">
        <section className="answer-report-picker">
          <div className="answer-report-panel-head">
            <div><FilePlus2 size={18} /><span><strong>选择练习</strong><small>已选择 {selectedIds.length} 题</small></span></div>
            <Button size="small" onClick={toggleAllVisible}>{visibleAttempts.every((item) => selectedIds.includes(item.id)) && visibleAttempts.length ? '取消本组全选' : '全选本组'}</Button>
          </div>
          <Select
            value={practiceSessionId}
            options={sessionOptions}
            onChange={setPracticeSessionId}
            className="answer-report-session-select"
          />
          <div className="answer-attempt-choice-list">
            {visibleAttempts.length ? visibleAttempts.map((attempt) => (
              <AttemptChoice
                key={attempt.id}
                attempt={attempt}
                checked={selectedIds.includes(attempt.id)}
                onChange={(checked) => setSelectedIds((current) => checked
                  ? [...new Set([...current, attempt.id])]
                  : current.filter((id) => id !== attempt.id))}
              />
            )) : <Empty description="还没有可用于报告的已完成刷题记录" />}
          </div>
          <Input
            value={title}
            onChange={(event) => setTitle(event.target.value)}
            placeholder="报告标题（可选）"
            maxLength={120}
          />
          <Button type="primary" icon={<FilePlus2 size={16} />} loading={creating} disabled={!selectedIds.length} onClick={() => void generate()} block>
            生成并保存报告
          </Button>
        </section>

        <section className="answer-report-history">
          <div className="answer-report-panel-head">
            <div><History size={18} /><span><strong>历史报告</strong><small>{reports.length} 份持久化报告</small></span></div>
          </div>
          <div className="answer-report-history-list">
            {reports.length ? reports.map((report) => (
              <div className={`answer-report-history-item ${activeReport?.id === report.id ? 'active' : ''}`} key={report.id}>
                <button onClick={() => void openReport(report.id)}>
                  <strong>{report.title}</strong>
                  <span>{displayTime(report.created_at)} · {report.aggregate.attempt_count} 题 · {percent(report.aggregate.average_score_rate)}</span>
                  {openingId === report.id && <Spin size="small" />}
                </button>
                <Popconfirm title="删除这份历史报告？" description="练习原记录不会被删除。" onConfirm={() => void removeReport(report.id)} okText="删除" cancelText="取消">
                  <Button type="text" danger icon={<Trash2 size={14} />} aria-label="删除报告" />
                </Popconfirm>
              </div>
            )) : <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="暂无历史报告" />}
          </div>
        </section>
      </div>

      {activeReport ? (
        <section className="answer-report-preview">
          <div className="answer-report-preview-actions no-print">
            <div><CheckCircle2 size={17} /><span>已保存，可随时重新打开</span></div>
            <Button type="primary" icon={<Printer size={16} />} onClick={printReport}>打印 / 另存为 PDF</Button>
          </div>
          <ReportDocument report={activeReport} />
        </section>
      ) : (
        <div className="answer-report-empty-preview no-print">
          <BarChart3 size={34} />
          <strong>选择历史报告或生成一份新报告</strong>
          <span>综合分析和逐题证据将在这里展示。</span>
        </div>
      )}
    </section>
  )
}
