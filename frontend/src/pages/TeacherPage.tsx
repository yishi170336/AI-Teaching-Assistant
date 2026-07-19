import { useCallback, useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { Button, Empty, Input, Modal, Popconfirm, Progress, Spin, Tag, Upload, message } from 'antd'
import type { UploadFile } from 'antd'
import {
  AlertTriangle,
  ArrowLeft,
  BookOpenCheck,
  BrainCircuit,
  CheckCircle2,
  ChevronRight,
  Clock3,
  Eye,
  FileCheck2,
  FileText,
  GraduationCap,
  Image as ImageIcon,
  LoaderCircle,
  RefreshCw,
  Send,
  ShieldCheck,
  Sparkles,
  Trash2,
  UploadCloud,
} from 'lucide-react'
import {
  createHomework,
  deleteHomework,
  fetchHomeworks,
  Homework,
  HomeworkSubmission,
  publishHomework,
  reprocessHomework,
} from '../lib/api'

const { Dragger } = Upload
const { TextArea } = Input

function formatTime(value: string) {
  if (!value) return '未设置'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleString('zh-CN', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  })
}

const homeworkStatus = {
  processing: { label: '识别中', color: 'processing', icon: <LoaderCircle className="spin" size={13} /> },
  draft: { label: '待发布', color: 'gold', icon: <FileCheck2 size={13} /> },
  published: { label: '已发布', color: 'success', icon: <CheckCircle2 size={13} /> },
  error: { label: '识别失败', color: 'error', icon: <AlertTriangle size={13} /> },
} as const

const submissionStatus = {
  grading: { label: '自动批改中', color: 'processing' },
  graded: { label: '双模型复核通过', color: 'success' },
  review_required: { label: '需要教师复查', color: 'warning' },
  error: { label: '批改失败', color: 'error' },
} as const

function QuestionPreview({ homework }: { homework: Homework }) {
  return (
    <section className="teacher-question-section">
      <header className="teacher-section-heading">
        <div>
          <span>QUESTION LAYOUT</span>
          <h3>学生将看到的题目</h3>
        </div>
        <small>答案区域已遮除 · 共 {homework.question_count} 题</small>
      </header>
      <div className="teacher-question-list">
        {homework.questions.map((question) => (
          <article className="teacher-question-card" key={question.id}>
            <div className="teacher-question-meta">
              <span>第 {question.number} 题</span>
              <Tag bordered={false}>{question.points || 0} 分</Tag>
              {question.figures.length > 0 && (
                <small><ImageIcon size={12} /> 已关联 {question.figures.length} 幅题图</small>
              )}
              <small>原第 {question.page_start}{question.page_end !== question.page_start ? `–${question.page_end}` : ''} 页</small>
            </div>
            <div className="question-layout-stack">
              {question.layout_images.length ? question.layout_images.map((asset) => (
                <img key={asset.file} src={asset.url} alt={`第 ${question.number} 题原版布局`} />
              )) : (
                <div className="question-text-fallback">{question.prompt || '题目文本待补充'}</div>
              )}
            </div>
            <div className="teacher-answer-panel">
              <div><span>标准答案</span><p>{question.answer || '未识别到答案'}</p></div>
              <div><span>评分标准</span><p>{question.rubric || '按标准答案判定'}</p></div>
            </div>
          </article>
        ))}
      </div>
    </section>
  )
}

function SubmissionPanel({ submission }: { submission: HomeworkSubmission }) {
  const status = submissionStatus[submission.status]
  const grading = submission.grading
  const scorePercent = grading?.max_score
    ? Math.round((grading.total_score / grading.max_score) * 100)
    : 0
  return (
    <article className={`teacher-submission-card status-${submission.status}`}>
      <header>
        <div className="submission-student-mark"><GraduationCap size={18} /></div>
        <div>
          <strong>{submission.student_name || '学生 1'}</strong>
          <span>{formatTime(submission.created_at)} 提交 · {submission.answer_images.length} 张图片</span>
        </div>
        <Tag color={status.color}>{status.label}</Tag>
      </header>
      <div className="submission-body">
        <div className="submission-images">
          {submission.answer_images.map((asset, index) => (
            <a href={asset.url} target="_blank" rel="noreferrer" key={asset.file}>
              <img src={asset.url} alt={`学生答案 ${index + 1}`} />
              <span><Eye size={12} /> 查看原图 {index + 1}</span>
            </a>
          ))}
        </div>
        {submission.status === 'grading' && (
          <div className="submission-processing">
            <LoaderCircle className="spin" size={24} />
            <strong>qwen3-vl-plus 正在识别并逐题评分</strong>
            <span>完成后由 qwen3-vl-flash 独立复核</span>
          </div>
        )}
        {submission.processing_error && (
          <div className="submission-error"><AlertTriangle size={16} />{submission.processing_error}</div>
        )}
        {grading && (
          <div className="grading-result">
            <div className="grading-score">
              <Progress type="circle" percent={scorePercent} size={88} strokeColor="#0f766e" />
              <div><strong>{grading.total_score} / {grading.max_score}</strong><span>{grading.summary || '自动批改已完成'}</span></div>
            </div>
            <div className="grading-items">
              {grading.items.map((item) => (
                <div key={`${item.question_id}-${item.number}`}>
                  <span>第 {item.number} 题</span>
                  <strong>{item.score} / {item.max_score} 分</strong>
                  <p>{item.feedback || item.evidence}</p>
                </div>
              ))}
            </div>
          </div>
        )}
        {submission.review && (
          <div className={`review-result ${submission.review.passed ? 'passed' : 'flagged'}`}>
            {submission.review.passed ? <ShieldCheck size={18} /> : <AlertTriangle size={18} />}
            <div>
              <strong>{submission.review.passed ? '审查模型确认批改无误' : '审查模型发现疑点'}</strong>
              <span>qwen3-vl-flash · 置信度 {Math.round(submission.review.confidence * 100)}%</span>
              {submission.review.issues.map((issue) => <p key={issue}>{issue}</p>)}
              {submission.review.recommendation && <p>{submission.review.recommendation}</p>}
            </div>
          </div>
        )}
      </div>
    </article>
  )
}

export default function TeacherPage() {
  const [homeworks, setHomeworks] = useState<Homework[]>([])
  const [loading, setLoading] = useState(true)
  const [uploadOpen, setUploadOpen] = useState(false)
  const [detailId, setDetailId] = useState<string | null>(null)
  const [fileList, setFileList] = useState<UploadFile[]>([])
  const [title, setTitle] = useState('')
  const [instructions, setInstructions] = useState('')
  const [dueAt, setDueAt] = useState('')
  const [uploading, setUploading] = useState(false)
  const [actionId, setActionId] = useState('')

  const loadHomeworks = useCallback(async (withSpinner = false) => {
    if (withSpinner) setLoading(true)
    try {
      setHomeworks(await fetchHomeworks('teacher'))
    } catch (error) {
      if (withSpinner) message.error(error instanceof Error ? error.message : '作业列表读取失败')
    } finally {
      if (withSpinner) setLoading(false)
    }
  }, [])

  useEffect(() => { void loadHomeworks(true) }, [loadHomeworks])

  const hasRunningTask = homeworks.some((homework) =>
    homework.status === 'processing'
    || homework.submissions?.some((submission) => submission.status === 'grading'),
  )
  useEffect(() => {
    if (!hasRunningTask) return
    const timer = window.setInterval(() => void loadHomeworks(), 2800)
    return () => window.clearInterval(timer)
  }, [hasRunningTask, loadHomeworks])

  const detail = homeworks.find((homework) => homework.id === detailId) || null
  const stats = useMemo(() => ({
    total: homeworks.length,
    published: homeworks.filter((item) => item.status === 'published').length,
    submissions: homeworks.reduce((total, item) => total + (item.submission_count || 0), 0),
    review: homeworks.reduce(
      (total, item) => total + (item.submissions || []).filter((submission) => submission.status === 'review_required').length,
      0,
    ),
  }), [homeworks])

  const resetUpload = () => {
    setUploadOpen(false)
    setFileList([])
    setTitle('')
    setInstructions('')
    setDueAt('')
  }

  const uploadHomework = async () => {
    const file = fileList[0]?.originFileObj
    if (!file) return message.warning('请先选择 PDF、图片或扫描版习题册')
    setUploading(true)
    try {
      const created = await createHomework(file, { title, instructions, dueAt })
      message.success('附件上传成功，正在识别题目与答案区域')
      resetUpload()
      setDetailId(created.id)
      await loadHomeworks()
    } catch (error) {
      message.error(error instanceof Error ? error.message : '上传失败')
    } finally {
      setUploading(false)
    }
  }

  const runAction = async (homeworkId: string, action: 'publish' | 'retry' | 'delete') => {
    setActionId(homeworkId)
    try {
      if (action === 'publish') {
        await publishHomework(homeworkId)
        message.success('作业已发送给学生')
      } else if (action === 'retry') {
        await reprocessHomework(homeworkId)
        message.success('已重新开始识别')
      } else {
        await deleteHomework(homeworkId)
        setDetailId(null)
        message.success('作业已删除')
      }
      await loadHomeworks()
    } catch (error) {
      message.error(error instanceof Error ? error.message : '操作失败')
    } finally {
      setActionId('')
    }
  }

  return (
    <div className="teacher-page">
      <header className="teacher-topbar">
        <div className="brand-row teacher-brand">
          <span className="teacher-brand-icon"><BrainCircuit size={22} /></span>
          <div><strong>CircuitMind</strong><span>教师作业中心</span></div>
        </div>
        <div className="teacher-topbar-actions">
          <span><i /> 双模型作业链路已启用</span>
          <Link to="/student" className="back-student"><ArrowLeft size={16} /> 学生端</Link>
        </div>
      </header>

      <main className="teacher-main">
        <section className="teacher-welcome">
          <div>
            <span className="teacher-eyebrow"><Sparkles size={14} /> AI HOMEWORK STUDIO</span>
            <h1>布置作业，保留题册原貌。</h1>
            <p>上传 PDF、照片或扫描版习题册，自动拆分题目与对应插图；学生只看到无答案原版布局，提交后由两级视觉模型批改与复核。</p>
          </div>
          <Button type="primary" size="large" icon={<UploadCloud size={18} />} onClick={() => setUploadOpen(true)}>
            上传并创建作业
          </Button>
        </section>

        <section className="teacher-stats">
          <article><span><BookOpenCheck size={18} /></span><div><strong>{stats.total}</strong><small>全部作业</small></div></article>
          <article><span><Send size={18} /></span><div><strong>{stats.published}</strong><small>已发布</small></div></article>
          <article><span><FileCheck2 size={18} /></span><div><strong>{stats.submissions}</strong><small>学生提交</small></div></article>
          <article className={stats.review ? 'needs-attention' : ''}><span><ShieldCheck size={18} /></span><div><strong>{stats.review}</strong><small>待人工复查</small></div></article>
        </section>

        <section className="homework-library">
          <header className="teacher-section-heading">
            <div><span>ASSIGNMENTS</span><h2>作业列表</h2></div>
            <Button icon={<RefreshCw size={14} />} onClick={() => void loadHomeworks(true)}>刷新</Button>
          </header>
          {loading ? (
            <div className="teacher-loading"><Spin /><span>正在读取作业…</span></div>
          ) : homeworks.length === 0 ? (
            <button className="teacher-empty" type="button" onClick={() => setUploadOpen(true)}>
              <span><UploadCloud size={28} /></span>
              <strong>上传第一份习题附件</strong>
              <small>支持 PDF、PNG、JPG、WEBP、扫描图片</small>
            </button>
          ) : (
            <div className="homework-grid">
              {homeworks.map((homework) => {
                const status = homeworkStatus[homework.status]
                return (
                  <article className="homework-card" key={homework.id} onClick={() => setDetailId(homework.id)}>
                    <div className="homework-card-top">
                      <span className="homework-file-icon"><FileText size={21} /></span>
                      <Tag color={status.color} icon={status.icon}>{status.label}</Tag>
                    </div>
                    <h3>{homework.title}</h3>
                    <p>{homework.instructions || '未填写作业说明'}</p>
                    {homework.status === 'processing' && <Progress percent={homework.processing_progress || 1} showInfo={false} status="active" />}
                    {homework.processing_error && <div className="homework-card-error">{homework.processing_error}</div>}
                    <div className="homework-card-data">
                      <span><strong>{homework.question_count}</strong>题</span>
                      <span><strong>{homework.max_score}</strong>分</span>
                      <span><strong>{homework.submission_count || 0}</strong>份提交</span>
                    </div>
                    <footer>
                      <span><Clock3 size={13} /> 截止 {formatTime(homework.due_at)}</span>
                      <ChevronRight size={16} />
                    </footer>
                  </article>
                )
              })}
            </div>
          )}
        </section>
      </main>

      <Modal
        open={uploadOpen}
        onCancel={() => !uploading && resetUpload()}
        onOk={() => void uploadHomework()}
        okText="上传并开始识别"
        cancelText="取消"
        confirmLoading={uploading}
        width={650}
        className="homework-upload-modal"
        title={null}
      >
        <div className="homework-modal-heading">
          <span><UploadCloud size={22} /></span>
          <div><small>NEW ASSIGNMENT</small><h2>创建一份新作业</h2><p>视觉模型将自动拆分题目、插图、答案与评分点。</p></div>
        </div>
        <div className="homework-upload-form">
          <label><span>作业标题</span><Input value={title} onChange={(event) => setTitle(event.target.value)} placeholder="留空时使用附件名称" maxLength={120} /></label>
          <label><span>作业说明</span><TextArea value={instructions} onChange={(event) => setInstructions(event.target.value)} placeholder="例如：写出完整计算过程，拍照时保证页面清晰" autoSize={{ minRows: 2, maxRows: 4 }} maxLength={2000} /></label>
          <label><span>截止时间</span><Input type="datetime-local" value={dueAt} onChange={(event) => setDueAt(event.target.value)} /></label>
          <Dragger
            accept=".pdf,.png,.jpg,.jpeg,.webp,.bmp"
            maxCount={1}
            fileList={fileList}
            beforeUpload={() => false}
            onChange={({ fileList: next }) => setFileList(next.slice(-1))}
          >
            <p className="ant-upload-drag-icon"><UploadCloud size={34} /></p>
            <p className="ant-upload-text">拖入 PDF、习题册照片或扫描图片</p>
            <p className="ant-upload-hint">单个附件最大 100 MB · 题图会自动归属到对应题目</p>
          </Dragger>
        </div>
      </Modal>

      <Modal
        open={Boolean(detail)}
        onCancel={() => setDetailId(null)}
        footer={null}
        width={1100}
        className="homework-detail-modal"
        title={null}
        destroyOnHidden
      >
        {detail && (
          <div className="homework-detail">
            <header className="homework-detail-header">
              <div>
                <Tag color={homeworkStatus[detail.status].color}>{homeworkStatus[detail.status].label}</Tag>
                <h2>{detail.title}</h2>
                <p>{detail.instructions || '未填写作业说明'} · 截止 {formatTime(detail.due_at)}</p>
                <div className="model-pipeline">
                  <span><Sparkles size={12} /> {detail.extraction_model}</span>
                  <i />
                  <span><FileText size={12} /> PDF-Extract-Kit</span>
                  <i />
                  <span><ShieldCheck size={12} /> {detail.review_model}</span>
                </div>
              </div>
              <div className="homework-detail-actions">
                {detail.source_url && <Button href={detail.source_url} target="_blank" icon={<Eye size={15} />}>原始附件</Button>}
                {detail.status === 'draft' && (
                  <Button type="primary" icon={<Send size={15} />} loading={actionId === detail.id} onClick={() => void runAction(detail.id, 'publish')}>发布给学生</Button>
                )}
                {detail.status === 'error' && (
                  <Button type="primary" icon={<RefreshCw size={15} />} loading={actionId === detail.id} onClick={() => void runAction(detail.id, 'retry')}>重新识别</Button>
                )}
                <Popconfirm title="删除这份作业？" description="题目、学生提交和批改结果将一并删除。" okText="删除" cancelText="取消" okButtonProps={{ danger: true }} onConfirm={() => void runAction(detail.id, 'delete')}>
                  <Button danger icon={<Trash2 size={15} />} />
                </Popconfirm>
              </div>
            </header>

            {detail.status === 'processing' && (
              <div className="homework-processing-panel"><LoaderCircle className="spin" size={30} /><div><strong>{detail.processing_message || '正在逐页识别题目与答案区域'}</strong><span>进度 {detail.processing_progress || 0}% · 页面较多时需要几分钟，完成后可预览无答案学生版。</span></div></div>
            )}
            {detail.processing_error && <div className="homework-detail-error"><AlertTriangle size={18} /><div><strong>识别未完成</strong><span>{detail.processing_error}</span></div></div>}
            {detail.processing_warnings.length > 0 && <div className="homework-warnings">{detail.processing_warnings.map((warning) => <span key={warning}>{warning}</span>)}</div>}
            {detail.questions.length > 0 && <QuestionPreview homework={detail} />}

            <section className="teacher-submission-section">
              <header className="teacher-section-heading"><div><span>SUBMISSIONS</span><h3>学生提交与批改</h3></div><small>当前演示为 1 名学生</small></header>
              {detail.submissions?.length ? detail.submissions.map((submission) => (
                <SubmissionPanel key={submission.id} submission={submission} />
              )) : <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="学生尚未提交答案" />}
            </section>
          </div>
        )}
      </Modal>
    </div>
  )
}
