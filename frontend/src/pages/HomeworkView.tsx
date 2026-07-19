import { useCallback, useEffect, useMemo, useState } from 'react'
import { Button, Empty, Modal, Progress, Spin, Tag, Upload, message } from 'antd'
import type { UploadFile } from 'antd'
import {
  AlertTriangle,
  BookOpenCheck,
  Camera,
  CheckCircle2,
  Clock3,
  Eye,
  FileCheck2,
  Image as ImageIcon,
  LoaderCircle,
  RefreshCw,
  ShieldCheck,
  Sparkles,
  UploadCloud,
} from 'lucide-react'
import { fetchHomeworks, Homework, submitHomework } from '../lib/api'

const { Dragger } = Upload

function formatDeadline(value: string) {
  if (!value) return '长期有效'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleString('zh-CN', {
    month: 'long',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

function StudentGrading({ homework }: { homework: Homework }) {
  const submission = homework.submission
  if (!submission) return null
  if (submission.status === 'grading') {
    return (
      <div className="student-grading-wait">
        <LoaderCircle className="spin" size={24} />
        <div><strong>AI 正在批改你的答案</strong><span>qwen3-vl-plus 评分后，还会由 qwen3-vl-flash 独立复核。</span></div>
      </div>
    )
  }
  if (submission.status === 'error') {
    return <div className="student-grading-error"><AlertTriangle size={17} />{submission.processing_error || '自动批改失败，请联系老师'}</div>
  }
  const grading = submission.grading
  if (!grading) return null
  const percent = grading.max_score ? Math.round((grading.total_score / grading.max_score) * 100) : 0
  return (
    <section className="student-grade-report">
      <header>
        <Progress type="circle" percent={percent} size={90} strokeColor="#0f766e" />
        <div>
          <span>本次得分</span>
          <strong>{grading.total_score} <small>/ {grading.max_score} 分</small></strong>
          <p>{grading.summary || '自动批改已完成'}</p>
        </div>
      </header>
      <div className="student-grade-items">
        {grading.items.map((item) => (
          <article key={`${item.question_id}-${item.number}`}>
            <span className={item.is_correct ? 'correct' : 'incorrect'}>
              {item.is_correct ? <CheckCircle2 size={15} /> : <AlertTriangle size={15} />}
            </span>
            <div><strong>第 {item.number} 题</strong><p>{item.feedback || '已完成评分'}</p></div>
            <b>{item.score} / {item.max_score}</b>
          </article>
        ))}
      </div>
      {submission.review && (
        <div className={`student-review-note ${submission.review.passed ? 'passed' : 'pending'}`}>
          {submission.review.passed ? <ShieldCheck size={17} /> : <AlertTriangle size={17} />}
          <div>
            <strong>{submission.review.passed ? '复核通过' : '正在等待教师复查'}</strong>
            <span>{submission.review.passed ? '审查模型确认识别与计分一致' : '审查模型发现疑点，最终结果以教师复查为准'}</span>
          </div>
        </div>
      )}
    </section>
  )
}

export default function HomeworkView({ studentId }: { studentId: string }) {
  const [homeworks, setHomeworks] = useState<Homework[]>([])
  const [loading, setLoading] = useState(true)
  const [detailId, setDetailId] = useState<string | null>(null)
  const [answerFiles, setAnswerFiles] = useState<UploadFile[]>([])
  const [submitting, setSubmitting] = useState(false)

  const load = useCallback(async (withSpinner = false) => {
    if (withSpinner) setLoading(true)
    try {
      setHomeworks(await fetchHomeworks('student', studentId))
    } catch (error) {
      if (withSpinner) message.error(error instanceof Error ? error.message : '作业读取失败')
    } finally {
      if (withSpinner) setLoading(false)
    }
  }, [studentId])

  useEffect(() => { void load(true) }, [load])
  const grading = homeworks.some((homework) => homework.submission?.status === 'grading')
  useEffect(() => {
    if (!grading) return
    const timer = window.setInterval(() => void load(), 2800)
    return () => window.clearInterval(timer)
  }, [grading, load])

  const detail = homeworks.find((homework) => homework.id === detailId) || null
  const progress = useMemo(() => ({
    completed: homeworks.filter((item) => item.submission).length,
    total: homeworks.length,
  }), [homeworks])

  const openHomework = (homeworkId: string) => {
    setAnswerFiles([])
    setDetailId(homeworkId)
  }

  const submit = async () => {
    if (!detail) return
    const files = answerFiles.flatMap((file) => file.originFileObj ? [file.originFileObj] : [])
    if (!files.length) return message.warning('请上传至少一张作答照片')
    setSubmitting(true)
    try {
      await submitHomework(detail.id, studentId, files)
      setAnswerFiles([])
      message.success('答案已提交，正在自动批改与复核')
      await load()
    } catch (error) {
      message.error(error instanceof Error ? error.message : '提交失败')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <section className="feature-view student-homework-view">
      <div className="student-homework-hero">
        <div className="student-homework-hero-copy">
          <span><Sparkles size={14} /> ASSIGNMENTS</span>
          <h1>我的作业</h1>
          <p>查看老师发送的原版题目，完成后拍照提交答案。</p>
        </div>
        <div className="student-homework-progress">
          <div><strong>{progress.completed}</strong><small> / {progress.total}</small></div>
          <span>已提交作业</span>
        </div>
      </div>

      <div className="student-homework-toolbar">
        <div><BookOpenCheck size={17} /><strong>老师布置的作业</strong><Tag>{homeworks.length} 份</Tag></div>
        <Button icon={<RefreshCw size={14} />} onClick={() => void load(true)}>刷新</Button>
      </div>

      {loading ? (
        <div className="student-homework-loading"><Spin /><span>正在读取作业…</span></div>
      ) : homeworks.length === 0 ? (
        <div className="student-homework-empty"><Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="老师还没有发布作业" /></div>
      ) : (
        <div className="student-homework-grid">
          {homeworks.map((homework) => {
            const submission = homework.submission
            const score = submission?.grading
            return (
              <article className="student-homework-card" key={homework.id} onClick={() => openHomework(homework.id)}>
                <header>
                  <span className="student-homework-icon"><FileCheck2 size={20} /></span>
                  {submission?.status === 'grading' ? (
                    <Tag color="processing" icon={<LoaderCircle className="spin" size={12} />}>批改中</Tag>
                  ) : submission ? (
                    <Tag color={submission.status === 'review_required' ? 'warning' : 'success'}>
                      {submission.status === 'review_required' ? '待教师复查' : '已提交'}
                    </Tag>
                  ) : <Tag color="gold">待完成</Tag>}
                </header>
                <h2>{homework.title}</h2>
                <p>{homework.instructions || '请按题目要求完成作答'}</p>
                <div className="student-homework-data">
                  <span><strong>{homework.question_count}</strong>题</span>
                  <span><strong>{homework.max_score}</strong>分</span>
                  {score && <span className="score"><strong>{score.total_score}</strong>得分</span>}
                </div>
                <footer><span><Clock3 size={13} /> {formatDeadline(homework.due_at)} 截止</span><Eye size={16} /></footer>
              </article>
            )
          })}
        </div>
      )}

      <Modal
        open={Boolean(detail)}
        onCancel={() => !submitting && setDetailId(null)}
        footer={null}
        width={980}
        className="student-homework-modal"
        title={null}
        destroyOnHidden
      >
        {detail && (
          <div className="student-homework-detail">
            <header className="student-homework-detail-header">
              <div><span>HOMEWORK</span><h2>{detail.title}</h2><p>{detail.instructions || '请按题目要求完成作答'} · {formatDeadline(detail.due_at)} 截止</p></div>
              <Tag color={detail.submission ? 'success' : 'gold'}>{detail.submission ? '已提交' : '待完成'}</Tag>
            </header>

            <section className="student-question-paper">
              <div className="student-paper-notice"><ShieldCheck size={16} /><span>以下为老师发布的无答案原版题面；题图已与对应题目保留在一起。</span></div>
              {detail.questions.map((question) => (
                <article key={question.id} className="student-question-item">
                  <header><span>第 {question.number} 题</span><Tag bordered={false}>{question.points || 0} 分</Tag>{question.figures.length > 0 && <small><ImageIcon size={12} /> {question.figures.length} 幅题图</small>}</header>
                  <div className="student-layout-stack">
                    {question.layout_images.length ? question.layout_images.map((asset) => (
                      <img src={asset.url} alt={`第 ${question.number} 题`} key={asset.file} />
                    )) : <p>{question.prompt}</p>}
                  </div>
                </article>
              ))}
            </section>

            <section className="student-answer-submit">
              <header><div><Camera size={19} /><span><strong>拍照提交答案</strong><small>按作答顺序上传，确保字迹和电路图清晰</small></span></div><Tag>最多 8 张</Tag></header>
              <Dragger
                accept="image/*"
                multiple
                maxCount={8}
                fileList={answerFiles}
                beforeUpload={() => false}
                onChange={({ fileList }) => setAnswerFiles(fileList.slice(0, 8))}
              >
                <p className="ant-upload-drag-icon"><UploadCloud size={30} /></p>
                <p className="ant-upload-text">点击拍照或选择答案图片</p>
                <p className="ant-upload-hint">支持 PNG、JPG、WEBP、BMP</p>
              </Dragger>
              <Button type="primary" size="large" block icon={<Camera size={17} />} loading={submitting} disabled={!answerFiles.length} onClick={() => void submit()}>
                {detail.submission ? '重新提交并批改' : '提交答案并开始批改'}
              </Button>
            </section>

            {detail.submission && (
              <section className="student-submission-preview">
                <header><span><Eye size={16} /> 已提交的答案图片</span><small>{formatDeadline(detail.submission.created_at)}</small></header>
                <div>{detail.submission.answer_images.map((asset, index) => <a href={asset.url} target="_blank" rel="noreferrer" key={asset.file}><img src={asset.url} alt={`答案 ${index + 1}`} /></a>)}</div>
              </section>
            )}
            <StudentGrading homework={detail} />
          </div>
        )}
      </Modal>
    </section>
  )
}
