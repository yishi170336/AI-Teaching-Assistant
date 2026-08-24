import { Button, Progress, Tag, Tooltip } from 'antd'
import { AlertTriangle, BookOpenCheck, CheckCircle2, Printer, ShieldCheck, TrendingDown, TrendingUp } from 'lucide-react'
import type { HomeworkLearningAdvice, HomeworkLearningMetric, HomeworkLearningReport } from '../lib/api'
import MathMarkdown from './MathMarkdown'

const masteryLabels: Record<HomeworkLearningMetric['status'], { label: string; color: string }> = {
  mastered: { label: '掌握良好', color: 'success' },
  developing: { label: '正在巩固', color: 'processing' },
  needs_support: { label: '需要加强', color: 'warning' },
  insufficient_evidence: { label: '暂不判断', color: 'default' },
}

function AdviceList({ title, items }: { title: string; items?: HomeworkLearningAdvice[] }) {
  if (!items?.length) return null
  return (
    <section className="learning-report-advice-section">
      <h4>{title}</h4>
      <div className="learning-report-advice-list">
        {items.map((item, index) => (
          <article key={`${title}-${index}`}>
            <MathMarkdown content={item.text} />
            <small>{item.question_ids.length ? `关联 ${item.question_ids.length} 道题` : `依据 ${item.submission_ids.length} 份作业`}</small>
          </article>
        ))}
      </div>
    </section>
  )
}

export default function HomeworkLearningReportDocument({
  report,
  studentName,
  showPrint = true,
}: {
  report: HomeworkLearningReport
  studentName?: string
  showPrint?: boolean
}) {
  const metrics = report.metrics
  const scorePercent = Math.round((metrics.score_rate || 0) * 100)
  return (
    <article className="homework-learning-report-document">
      <header className="learning-report-document-header">
        <div>
          <span>CIRCUITMIND · LEARNING REPORT</span>
          <h2>{report.title}</h2>
          <p>{studentName || '学生'} · {report.report_type === 'assignment' ? '单次作业报告' : '累计学情报告'} · {metrics.assignment_count} 份作业</p>
        </div>
        <div className="learning-report-document-actions">
          <Tag color={report.status === 'published' ? 'success' : 'blue'}>{report.status === 'published' ? '已发布' : '教师审核稿'}</Tag>
          {showPrint && <Button icon={<Printer size={14} />} onClick={() => window.print()}>打印报告</Button>}
        </div>
      </header>

      <section className="learning-report-score-overview">
        <Progress type="circle" percent={scorePercent} size={116} strokeColor="#0f766e" />
        <div className="learning-report-score-copy">
          <strong>{metrics.total_score} / {metrics.max_score} 分</strong>
          <span>{metrics.correct_count} 题正确 · {metrics.partial_count} 题部分正确 · {metrics.incorrect_count} 题错误 · {metrics.unscored_count} 题不计分</span>
          {report.diagnosis.summary && <MathMarkdown content={report.diagnosis.summary} />}
        </div>
        <div className="learning-report-trend">
          {metrics.trend.status === 'improving' ? <TrendingUp size={22} /> : metrics.trend.status === 'declining' ? <TrendingDown size={22} /> : <BookOpenCheck size={22} />}
          <strong>{metrics.trend.status === 'improving' ? '表现上升' : metrics.trend.status === 'declining' ? '需要关注' : metrics.trend.status === 'stable' ? '表现平稳' : '趋势证据不足'}</strong>
          <span>{metrics.trend.items.length < 3 ? '至少三份可计分作业后判断趋势' : `前后阶段变化 ${Math.round((metrics.trend.change || 0) * 100)}%`}</span>
        </div>
      </section>

      <section className="learning-report-mastery-section">
        <header><h3>知识点掌握情况</h3><span>阈值：85% 掌握良好 · 60% 正在巩固</span></header>
        {metrics.knowledge_points.length ? (
          <div className="learning-report-mastery-grid">
            {metrics.knowledge_points.map((item) => {
              const status = masteryLabels[item.status]
              const insufficient = item.status === 'insufficient_evidence'
              return (
                <article key={item.knowledge_point}>
                  <div>
                    <strong>{item.knowledge_point}</strong>
                    <Tooltip title={insufficient ? '同一知识点至少需要 2 道可计分题，当前样本不足以判断是否真正掌握。' : undefined}>
                      <Tag color={status.color}>{insufficient ? `仅 ${item.scored_count} 题，${status.label}` : status.label}</Tag>
                    </Tooltip>
                  </div>
                  <Progress percent={Math.round((item.score_rate || 0) * 100)} showInfo={item.score_rate !== null} strokeColor="#14b8a6" />
                  <small>
                    {item.scored_count} 道有效证据题 · {item.score}/{item.max_score} 分
                    {insufficient ? ' · 至少 2 题后判断掌握状态' : ''}
                  </small>
                </article>
              )
            })}
          </div>
        ) : <div className="learning-report-empty"><AlertTriangle size={18} /> 当前作业尚未完成知识点对齐</div>}
      </section>

      <div className="learning-report-advice-grid">
        <AdviceList title="表现优势" items={report.diagnosis.strengths} />
        <AdviceList title="需要加强" items={report.diagnosis.gaps} />
        <AdviceList title="教学建议" items={report.diagnosis.teaching_actions} />
        <AdviceList title="学习建议" items={report.diagnosis.student_actions} />
      </div>

      <footer className={`learning-report-quality ${report.review?.passed ? 'passed' : 'blocked'}`}>
        {report.review?.passed ? <ShieldCheck size={18} /> : <AlertTriangle size={18} />}
        <div>
          <strong>{report.review?.passed ? '独立质量复核通过' : '报告尚未通过独立复核'}</strong>
          <span>{report.review?.review_model || 'qwen3.7-flash'}{report.review?.confidence != null ? ` · 置信度 ${Math.round(report.review.confidence * 100)}%` : ''}</span>
          {report.review?.issues?.map((issue) => <MathMarkdown key={issue} content={issue} />)}
        </div>
        {report.source_available === false ? <Tag icon={<AlertTriangle size={12} />} color="warning">原作业已删除，历史快照保留</Tag> : <Tag icon={<CheckCircle2 size={12} />} color="success">来源快照完整</Tag>}
      </footer>
    </article>
  )
}
