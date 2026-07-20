import { useCallback, useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { Button, Checkbox, Empty, Input, Modal, Popconfirm, Progress, Segmented, Select, Spin, Tag, Upload, message } from 'antd'
import type { UploadFile } from 'antd'
import {
  AlertTriangle,
  ArrowLeft,
  BookOpenCheck,
  BookMarked,
  BrainCircuit,
  CheckCircle2,
  ChevronRight,
  Clock3,
  Eye,
  FileCheck2,
  FileText,
  GraduationCap,
  ImagePlus,
  LoaderCircle,
  LibraryBig,
  Pencil,
  Plus,
  Printer,
  RefreshCw,
  Send,
  ShieldCheck,
  Sparkles,
  Trash2,
  UploadCloud,
} from 'lucide-react'
import {
  createHomework,
  createHomeworkFromQuestionBank,
  createQuestionBank,
  deleteDocumentQuestionAsset,
  deleteQuestionBank,
  deleteQuestionBankQuestion,
  deleteHomework,
  fetchHomeworks,
  fetchQuestionBanks,
  Homework,
  HomeworkQuestion,
  HomeworkQuestionUpdate,
  HomeworkSubmission,
  publishHomework,
  QuestionBank,
  reprocessQuestionBank,
  reprocessHomework,
  updateDocumentQuestion,
  uploadDocumentQuestionAsset,
} from '../lib/api'
import HomeworkPaper from '../components/HomeworkPaper'
import MathMarkdown from '../components/MathMarkdown'

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

const questionBankStatus = {
  processing: { label: '提取中', color: 'processing', icon: <LoaderCircle className="spin" size={13} /> },
  ready: { label: '可选题', color: 'success', icon: <CheckCircle2 size={13} /> },
  error: { label: '提取失败', color: 'error', icon: <AlertTriangle size={13} /> },
} as const

const bankQuestionKey = (bankId: string, questionId: string) => `${bankId}:${questionId}`

type EditableQuestionContext = {
  kind: 'homework' | 'question-bank'
  documentId: string
  question: HomeworkQuestion
}

const questionTypeOptions = [
  { value: 'choice', label: '选择题' },
  { value: 'fill_blank', label: '填空题' },
  { value: 'true_false', label: '判断题' },
  { value: 'short_answer', label: '简答题' },
  { value: 'calculation', label: '计算题' },
  { value: 'design', label: '设计题' },
  { value: 'other', label: '其他题型' },
]

function questionTypeLabel(value: string) {
  return questionTypeOptions.find((item) => item.value === value)?.label || '题目'
}

function QuestionManagementList({
  questions,
  deletingQuestionId = '',
  onEditQuestion,
  onDeleteQuestion,
}: {
  questions: HomeworkQuestion[]
  deletingQuestionId?: string
  onEditQuestion: (question: HomeworkQuestion) => void
  onDeleteQuestion?: (questionId: string) => void
}) {
  return (
    <section className="question-bank-question-manager">
      <header className="teacher-section-heading">
        <div><span>QUESTION MANAGEMENT</span><h3>题目管理</h3></div>
        <small>可修正文题、答案、图注，或补充和替换图片</small>
      </header>
      {questions.length ? (
        <div className="question-bank-manage-list">
          {questions.map((question) => (
            <article key={question.id}>
              <div className="question-bank-manage-number">{question.number}</div>
              <div className="question-bank-manage-copy">
                <span>{question.section_title || '题目'} · {questionTypeLabel(question.question_type)}</span>
                <MathMarkdown content={question.prompt || '未识别到题干'} />
                <small>{question.figures?.length || 0} 张题图 · {question.answer_figures?.length || 0} 张答案图 · {question.answer || question.answer_subquestions?.length ? '含参考答案' : '未识别到答案'}</small>
              </div>
              <div className="question-manage-actions">
                <Button type="text" icon={<Pencil size={14} />} onClick={() => onEditQuestion(question)}>编辑</Button>
                {onDeleteQuestion && (
                  <Popconfirm
                    title="从题库中删除这道题？"
                    description="只影响题库，已经布置的作业不会受影响。"
                    okText="删除"
                    cancelText="取消"
                    okButtonProps={{ danger: true }}
                    onConfirm={() => onDeleteQuestion(question.id)}
                  >
                    <Button danger type="text" icon={<Trash2 size={14} />} loading={deletingQuestionId === question.id} />
                  </Popconfirm>
                )}
              </div>
            </article>
          ))}
        </div>
      ) : <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="暂无题目" />}
    </section>
  )
}

function QuestionPreview({
  homework,
  onEditQuestion,
}: {
  homework: Homework
  onEditQuestion: (question: HomeworkQuestion) => void
}) {
  const [mode, setMode] = useState<'questions' | 'answers'>('questions')
  const printPaper = (nextMode: 'questions' | 'answers') => {
    setMode(nextMode)
    window.setTimeout(() => window.print(), 80)
  }
  return (
    <>
      <QuestionManagementList questions={homework.questions} onEditQuestion={onEditQuestion} />
      <section className="teacher-question-section">
      <header className="teacher-section-heading">
        <div>
          <span>STRUCTURED HOMEWORK</span>
          <h3>结构化作业预览</h3>
        </div>
        <div className="homework-paper-actions">
          <Segmented
            value={mode}
            options={[{ label: '作业内容', value: 'questions' }, { label: '参考答案', value: 'answers' }]}
            onChange={(value) => setMode(value as 'questions' | 'answers')}
          />
          <Button icon={<Printer size={14} />} onClick={() => printPaper('questions')}>打印作业内容</Button>
          <Button icon={<Printer size={14} />} onClick={() => printPaper('answers')}>打印参考答案</Button>
        </div>
      </header>
      <p className="homework-reflow-note">仅保留题号、题干、小问、选项、题图和参考答案；教材讲解、目录与无关内容不会进入作业。</p>
      <HomeworkPaper homework={homework} mode={mode} printable />
      </section>
    </>
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
          <span>{formatTime(submission.created_at)} 提交 · {submission.answers?.length || 0} 道结构化答案 · {submission.answer_images.length} 张图片</span>
        </div>
        <Tag color={status.color}>{status.label}</Tag>
      </header>
      <div className="submission-body">
        {submission.answers?.length > 0 && (
          <div className="submission-direct-answers">
            {submission.answers.map((answer) => (
              <article key={answer.question_id}>
                <strong>第 {answer.number || '?'} 题</strong>
                <span>{answer.selected_options?.length ? answer.selected_options.join('、') : answer.answer || answer.subquestion_answers?.map((part) => `（${part.label}）${part.text}`).join('；') || '图片作答'}</span>
              </article>
            ))}
          </div>
        )}
        <div className="submission-images">
          {submission.answer_images.map((asset, index) => (
            <a href={asset.url} target="_blank" rel="noreferrer" key={asset.file}>
              <img src={asset.url} alt={`学生答案 ${index + 1}`} />
              <span><Eye size={12} /> 第 {asset.question_number || '?'} 题 · 查看原图</span>
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

function QuestionBankPreview({
  bank,
  deletingQuestionId,
  onEditQuestion,
  onDeleteQuestion,
}: {
  bank: QuestionBank
  deletingQuestionId: string
  onEditQuestion: (question: HomeworkQuestion) => void
  onDeleteQuestion: (questionId: string) => void
}) {
  const [mode, setMode] = useState<'questions' | 'answers'>('questions')
  const preview: Homework = {
    id: bank.id,
    title: bank.title,
    instructions: '题库题目预览',
    due_at: '',
    status: 'draft',
    source_name: bank.source_name,
    source_url: bank.source_url,
    created_at: bank.created_at,
    updated_at: bank.updated_at,
    published_at: '',
    extraction_model: bank.extraction_model,
    grading_model: '',
    review_model: '',
    processing_error: bank.processing_error,
    processing_warnings: bank.processing_warnings,
    processing_progress: bank.processing_progress,
    processing_message: bank.processing_message,
    page_count: bank.page_count,
    max_score: bank.max_score,
    question_count: bank.question_count,
    questions: bank.questions,
  }
  return (
    <div className="question-bank-detail-body">
      <QuestionManagementList
        questions={bank.questions}
        deletingQuestionId={deletingQuestionId}
        onEditQuestion={onEditQuestion}
        onDeleteQuestion={onDeleteQuestion}
      />
      {bank.questions.length > 0 && (
        <section className="question-bank-paper-preview">
          <header className="teacher-section-heading">
            <div><span>REFLOWED PREVIEW</span><h3>重排效果预览</h3></div>
            <Segmented
              value={mode}
              options={[{ label: '题目', value: 'questions' }, { label: '参考答案', value: 'answers' }]}
              onChange={(value) => setMode(value as 'questions' | 'answers')}
            />
          </header>
          <HomeworkPaper homework={preview} mode={mode} />
        </section>
      )}
    </div>
  )
}

function QuestionEditorModal({
  context,
  onClose,
  onDocumentChange,
}: {
  context: EditableQuestionContext | null
  onClose: () => void
  onDocumentChange: (document: Homework | QuestionBank) => void
}) {
  const [draft, setDraft] = useState<HomeworkQuestion | null>(null)
  const [saving, setSaving] = useState(false)
  const [assetAction, setAssetAction] = useState('')

  useEffect(() => {
    setDraft(context ? JSON.parse(JSON.stringify(context.question)) as HomeworkQuestion : null)
  }, [context])

  if (!context || !draft) return null

  const syncDocument = (document: Homework | QuestionBank) => {
    onDocumentChange(document)
    const nextQuestion = document.questions.find((question) => question.id === context.question.id)
    if (nextQuestion) setDraft(JSON.parse(JSON.stringify(nextQuestion)) as HomeworkQuestion)
  }
  const updatePart = (
    field: 'subquestions' | 'answer_subquestions',
    index: number,
    key: 'label' | 'text',
    value: string,
  ) => {
    const parts = [...(draft[field] || [])]
    parts[index] = { ...parts[index], [key]: value }
    setDraft({ ...draft, [field]: parts })
  }
  const removePart = (field: 'subquestions' | 'answer_subquestions', index: number) => {
    setDraft({ ...draft, [field]: (draft[field] || []).filter((_, itemIndex) => itemIndex !== index) })
  }
  const updateOption = (index: number, key: 'label' | 'text', value: string) => {
    const options = [...(draft.options || [])]
    options[index] = { ...options[index], [key]: value }
    setDraft({ ...draft, options })
  }
  const save = async () => {
    setSaving(true)
    const payload: HomeworkQuestionUpdate = {
      section_key: draft.section_key,
      section_title: draft.section_title,
      number: draft.number,
      question_type: draft.question_type,
      prompt: draft.prompt,
      subquestions: draft.subquestions || [],
      options: draft.options || [],
      option_columns: draft.option_columns,
      figure_position: draft.figure_position,
      points: Number(draft.points) || 0,
      answer: draft.answer || '',
      answer_subquestions: draft.answer_subquestions || [],
      rubric: draft.rubric || '',
      figures: (draft.figures || []).map((asset) => ({
        file: asset.file,
        caption: asset.caption || '',
        position: asset.position || '',
      })),
      answer_figures: (draft.answer_figures || []).map((asset) => ({
        file: asset.file,
        caption: asset.caption || '',
        position: asset.position || '',
      })),
    }
    try {
      const document = await updateDocumentQuestion(
        context.kind,
        context.documentId,
        context.question.id,
        payload,
      )
      syncDocument(document)
      message.success('题目、答案和图注已保存')
    } catch (error) {
      message.error(error instanceof Error ? error.message : '题目保存失败')
    } finally {
      setSaving(false)
    }
  }
  const uploadAsset = async (
    target: 'figures' | 'answer_figures',
    file: File,
    replaceFile = '',
  ) => {
    const action = `${target}:${replaceFile || 'new'}`
    setAssetAction(action)
    try {
      const document = await uploadDocumentQuestionAsset(
        context.kind,
        context.documentId,
        context.question.id,
        target,
        file,
        replaceFile
          ? (draft[target] || []).find((asset) => asset.file === replaceFile)?.caption || ''
          : '',
        replaceFile,
      )
      syncDocument(document)
      message.success(replaceFile ? '图片已替换' : '图片已添加')
    } catch (error) {
      message.error(error instanceof Error ? error.message : '图片上传失败')
    } finally {
      setAssetAction('')
    }
  }
  const deleteAsset = async (target: 'figures' | 'answer_figures', file: string) => {
    setAssetAction(`${target}:${file}`)
    try {
      const document = await deleteDocumentQuestionAsset(
        context.kind,
        context.documentId,
        context.question.id,
        target,
        file,
      )
      syncDocument(document)
      message.success('图片已删除')
    } catch (error) {
      message.error(error instanceof Error ? error.message : '图片删除失败')
    } finally {
      setAssetAction('')
    }
  }
  const renderAssets = (target: 'figures' | 'answer_figures', title: string) => (
    <section className="question-editor-assets">
      <header><div><ImagePlus size={16} /><strong>{title}</strong></div><small>可修改图注、替换错误图片或补图</small></header>
      <div className="question-editor-asset-grid">
        {(draft[target] || []).map((asset, index) => (
          <article key={asset.file}>
            <img src={asset.url} alt={asset.caption || title} />
            <Input
              value={asset.caption || ''}
              placeholder="填写图号或图注"
              onChange={(event) => {
                const assets = [...(draft[target] || [])]
                assets[index] = { ...assets[index], caption: event.target.value }
                setDraft({ ...draft, [target]: assets })
              }}
            />
            <div>
              <Upload
                accept="image/*"
                showUploadList={false}
                beforeUpload={(file) => { void uploadAsset(target, file, asset.file); return false }}
              >
                <Button size="small" loading={assetAction === `${target}:${asset.file}`}>替换图片</Button>
              </Upload>
              <Popconfirm title="删除这张图片？" okText="删除" cancelText="取消" onConfirm={() => void deleteAsset(target, asset.file)}>
                <Button danger size="small" type="text">删除</Button>
              </Popconfirm>
            </div>
          </article>
        ))}
        <Upload
          accept="image/*"
          showUploadList={false}
          beforeUpload={(file) => { void uploadAsset(target, file); return false }}
        >
          <button className="question-editor-add-image" type="button">
            <ImagePlus size={22} /><strong>上传{title}</strong><span>PNG、JPG、WEBP、BMP</span>
          </button>
        </Upload>
      </div>
    </section>
  )

  return (
    <Modal
      open
      onCancel={onClose}
      onOk={() => void save()}
      okText="保存修改"
      cancelText="关闭"
      confirmLoading={saving}
      width={960}
      className="question-editor-modal"
      title={`编辑第 ${draft.number || '?'} 题`}
      destroyOnHidden
    >
      <div className="question-editor-form">
        <div className="question-editor-row columns-4">
          <label><span>题号</span><Input value={draft.number} onChange={(event) => setDraft({ ...draft, number: event.target.value })} /></label>
          <label><span>题型</span><Select value={draft.question_type} options={questionTypeOptions} onChange={(value) => setDraft({ ...draft, question_type: value })} /></label>
          <label><span>分值</span><Input type="number" min={0} value={draft.points} onChange={(event) => setDraft({ ...draft, points: Number(event.target.value) })} /></label>
          <label><span>选项列数</span><Select value={draft.option_columns || 1} options={[1, 2, 4].map((value) => ({ value, label: `${value} 列` }))} onChange={(value) => setDraft({ ...draft, option_columns: value })} /></label>
        </div>
        <div className="question-editor-row columns-2">
          <label><span>章节标识</span><Input value={draft.section_key} onChange={(event) => setDraft({ ...draft, section_key: event.target.value })} /></label>
          <label><span>章节标题</span><Input value={draft.section_title} onChange={(event) => setDraft({ ...draft, section_title: event.target.value })} /></label>
        </div>
        <label className="question-editor-field"><span>题干（支持 Markdown / LaTeX）</span><TextArea autoSize={{ minRows: 4, maxRows: 12 }} value={draft.prompt} onChange={(event) => setDraft({ ...draft, prompt: event.target.value })} /></label>

        <section className="question-editor-list-field">
          <header><strong>题目小问</strong><Button size="small" icon={<Plus size={13} />} onClick={() => setDraft({ ...draft, subquestions: [...(draft.subquestions || []), { label: String((draft.subquestions?.length || 0) + 1), text: '' }] })}>增加小问</Button></header>
          {(draft.subquestions || []).map((part, index) => <div key={`${part.label}-${index}`}><Input className="part-label" value={part.label} onChange={(event) => updatePart('subquestions', index, 'label', event.target.value)} /><TextArea autoSize value={part.text} onChange={(event) => updatePart('subquestions', index, 'text', event.target.value)} /><Button danger type="text" onClick={() => removePart('subquestions', index)}>删除</Button></div>)}
        </section>

        {draft.question_type === 'choice' && (
          <section className="question-editor-list-field">
            <header><strong>选择题选项</strong><Button size="small" icon={<Plus size={13} />} onClick={() => setDraft({ ...draft, options: [...(draft.options || []), { label: String.fromCharCode(65 + (draft.options?.length || 0)), text: '' }] })}>增加选项</Button></header>
            {(draft.options || []).map((option, index) => <div key={`${option.label}-${index}`}><Input className="part-label" value={option.label} onChange={(event) => updateOption(index, 'label', event.target.value)} /><Input value={option.text} onChange={(event) => updateOption(index, 'text', event.target.value)} /><Button danger type="text" onClick={() => setDraft({ ...draft, options: draft.options.filter((_, itemIndex) => itemIndex !== index) })}>删除</Button></div>)}
          </section>
        )}

        <label className="question-editor-field"><span>参考答案（支持 Markdown / LaTeX）</span><TextArea autoSize={{ minRows: 4, maxRows: 12 }} value={draft.answer || ''} onChange={(event) => setDraft({ ...draft, answer: event.target.value })} /></label>
        <section className="question-editor-list-field">
          <header><strong>答案小问</strong><Button size="small" icon={<Plus size={13} />} onClick={() => setDraft({ ...draft, answer_subquestions: [...(draft.answer_subquestions || []), { label: String((draft.answer_subquestions?.length || 0) + 1), text: '' }] })}>增加答案小问</Button></header>
          {(draft.answer_subquestions || []).map((part, index) => <div key={`${part.label}-${index}`}><Input className="part-label" value={part.label} onChange={(event) => updatePart('answer_subquestions', index, 'label', event.target.value)} /><TextArea autoSize value={part.text} onChange={(event) => updatePart('answer_subquestions', index, 'text', event.target.value)} /><Button danger type="text" onClick={() => removePart('answer_subquestions', index)}>删除</Button></div>)}
        </section>
        <label className="question-editor-field"><span>评分标准</span><TextArea autoSize={{ minRows: 2, maxRows: 8 }} value={draft.rubric || ''} onChange={(event) => setDraft({ ...draft, rubric: event.target.value })} /></label>
        <label className="question-editor-field compact"><span>题图位置</span><Select value={draft.figure_position || 'after_question'} options={[{ value: 'before_question', label: '题干前' }, { value: 'after_question', label: '题干后' }, { value: 'after_options', label: '选项后' }]} onChange={(value) => setDraft({ ...draft, figure_position: value })} /></label>
        {renderAssets('figures', '题图')}
        {renderAssets('answer_figures', '答案图')}
      </div>
    </Modal>
  )
}

export default function TeacherPage() {
  const [homeworks, setHomeworks] = useState<Homework[]>([])
  const [questionBanks, setQuestionBanks] = useState<QuestionBank[]>([])
  const [loading, setLoading] = useState(true)
  const [uploadOpen, setUploadOpen] = useState(false)
  const [createMode, setCreateMode] = useState<'upload' | 'bank'>('upload')
  const [detailId, setDetailId] = useState<string | null>(null)
  const [bankDetailId, setBankDetailId] = useState<string | null>(null)
  const [bankUploadOpen, setBankUploadOpen] = useState(false)
  const [fileList, setFileList] = useState<UploadFile[]>([])
  const [bankFileList, setBankFileList] = useState<UploadFile[]>([])
  const [bankTitle, setBankTitle] = useState('')
  const [title, setTitle] = useState('')
  const [instructions, setInstructions] = useState('')
  const [dueAt, setDueAt] = useState('')
  const [selectedBankQuestions, setSelectedBankQuestions] = useState<string[]>([])
  const [uploading, setUploading] = useState(false)
  const [bankUploading, setBankUploading] = useState(false)
  const [actionId, setActionId] = useState('')
  const [bankActionId, setBankActionId] = useState('')
  const [deletingQuestionId, setDeletingQuestionId] = useState('')
  const [editingQuestion, setEditingQuestion] = useState<EditableQuestionContext | null>(null)

  const applyEditedDocument = (document: Homework | QuestionBank) => {
    if (editingQuestion?.kind === 'homework') {
      setHomeworks((current) => current.map((item) => item.id === document.id ? document as Homework : item))
    } else {
      setQuestionBanks((current) => current.map((item) => item.id === document.id ? document as QuestionBank : item))
    }
    if (editingQuestion) {
      const question = document.questions.find((item) => item.id === editingQuestion.question.id)
      if (question) setEditingQuestion({ ...editingQuestion, question })
    }
  }

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

  const loadQuestionBanks = useCallback(async () => {
    try {
      setQuestionBanks(await fetchQuestionBanks())
    } catch (error) {
      message.error(error instanceof Error ? error.message : '题库读取失败')
    }
  }, [])

  useEffect(() => {
    void Promise.all([loadHomeworks(true), loadQuestionBanks()])
  }, [loadHomeworks, loadQuestionBanks])

  const hasRunningTask = questionBanks.some((bank) => bank.status === 'processing')
    || homeworks.some((homework) =>
      homework.status === 'processing'
      || homework.submissions?.some((submission) => submission.status === 'grading'),
    )
  useEffect(() => {
    if (!hasRunningTask) return
    const timer = window.setInterval(() => {
      void loadHomeworks()
      void loadQuestionBanks()
    }, 2800)
    return () => window.clearInterval(timer)
  }, [hasRunningTask, loadHomeworks, loadQuestionBanks])

  const detail = homeworks.find((homework) => homework.id === detailId) || null
  const bankDetail = questionBanks.find((bank) => bank.id === bankDetailId) || null
  const incompleteChoiceNumbers = detail?.questions
    .filter((question) => question.question_type === 'choice' && (question.options?.length || 0) < 2)
    .map((question) => question.number) || []
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
    setSelectedBankQuestions([])
    setTitle('')
    setInstructions('')
    setDueAt('')
  }

  const submitHomework = async () => {
    setUploading(true)
    try {
      let created: Homework
      if (createMode === 'upload') {
        const file = fileList[0]?.originFileObj
        if (!file) {
          message.warning('请先选择 PDF、图片或扫描版习题册')
          return
        }
        created = await createHomework(file, { title, instructions, dueAt })
        message.success('附件上传成功，正在识别题目与答案区域')
      } else {
        if (!selectedBankQuestions.length) {
          message.warning('请至少勾选一道题库题目')
          return
        }
        const grouped = new Map<string, string[]>()
        selectedBankQuestions.forEach((key) => {
          const [bankId, questionId] = key.split(':')
          if (bankId && questionId) grouped.set(bankId, [...(grouped.get(bankId) || []), questionId])
        })
        created = await createHomeworkFromQuestionBank({
          title,
          instructions,
          dueAt,
          selections: Array.from(grouped, ([bank_id, question_ids]) => ({ bank_id, question_ids })),
        })
        message.success(`已从题库选择 ${created.question_count} 道题生成作业`)
      }
      resetUpload()
      setDetailId(created.id)
      await loadHomeworks()
    } catch (error) {
      message.error(error instanceof Error ? error.message : '上传失败')
    } finally {
      setUploading(false)
    }
  }

  const uploadQuestionBank = async () => {
    const file = bankFileList[0]?.originFileObj
    if (!file) return message.warning('请先选择指导书或习题册附件')
    setBankUploading(true)
    try {
      const created = await createQuestionBank(file, bankTitle)
      message.success('题库附件已保存，正在提取可布置题目')
      setBankUploadOpen(false)
      setBankFileList([])
      setBankTitle('')
      setBankDetailId(created.id)
      await loadQuestionBanks()
    } catch (error) {
      message.error(error instanceof Error ? error.message : '题库上传失败')
    } finally {
      setBankUploading(false)
    }
  }

  const runBankAction = async (bankId: string, action: 'retry' | 'delete') => {
    setBankActionId(bankId)
    try {
      if (action === 'retry') {
        await reprocessQuestionBank(bankId)
        message.success('已重新开始识别题库')
      } else {
        await deleteQuestionBank(bankId)
        setBankDetailId(null)
        setSelectedBankQuestions((keys) => keys.filter((key) => !key.startsWith(`${bankId}:`)))
        message.success('题库已删除，已布置作业不受影响')
      }
      await loadQuestionBanks()
    } catch (error) {
      message.error(error instanceof Error ? error.message : '题库操作失败')
    } finally {
      setBankActionId('')
    }
  }

  const removeBankQuestion = async (bankId: string, questionId: string) => {
    setDeletingQuestionId(questionId)
    try {
      await deleteQuestionBankQuestion(bankId, questionId)
      setSelectedBankQuestions((keys) => keys.filter((key) => key !== bankQuestionKey(bankId, questionId)))
      message.success('题目已从题库删除')
      await loadQuestionBanks()
    } catch (error) {
      message.error(error instanceof Error ? error.message : '题目删除失败')
    } finally {
      setDeletingQuestionId('')
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
            <h1>布置作业，重排题目与答案。</h1>
            <p>上传试卷、课后习题、学习指导书、照片或扫描版习题册，自动过滤讲解与无关内容，提取题号、题干、小问、选项、题图和参考答案。</p>
          </div>
          <div className="teacher-welcome-actions">
            <Button size="large" icon={<BookMarked size={18} />} onClick={() => { setCreateMode('bank'); setUploadOpen(true) }}>
              从题库选题
            </Button>
            <Button type="primary" size="large" icon={<UploadCloud size={18} />} onClick={() => { setCreateMode('upload'); setUploadOpen(true) }}>
              上传并创建作业
            </Button>
          </div>
        </section>

        <section className="teacher-stats">
          <article><span><BookOpenCheck size={18} /></span><div><strong>{stats.total}</strong><small>全部作业</small></div></article>
          <article><span><Send size={18} /></span><div><strong>{stats.published}</strong><small>已发布</small></div></article>
          <article><span><FileCheck2 size={18} /></span><div><strong>{stats.submissions}</strong><small>学生提交</small></div></article>
          <article className={stats.review ? 'needs-attention' : ''}><span><ShieldCheck size={18} /></span><div><strong>{stats.review}</strong><small>待人工复查</small></div></article>
        </section>

        <section className="question-bank-library">
          <header className="teacher-section-heading">
            <div><span>QUESTION BANK</span><h2>长期题库</h2></div>
            <div className="question-bank-heading-actions">
              <small>{questionBanks.reduce((total, bank) => total + bank.question_count, 0)} 道可管理题目</small>
              <Button type="primary" icon={<UploadCloud size={14} />} onClick={() => setBankUploadOpen(true)}>上传题库</Button>
            </div>
          </header>
          <p className="question-bank-intro">上传学习指导书或其他习题册后长期保存。系统只保留题号、题目、题图和答案，可逐题清理，并在每次布置作业时重复选用。</p>
          {questionBanks.length === 0 ? (
            <button className="question-bank-empty" type="button" onClick={() => setBankUploadOpen(true)}>
              <span><LibraryBig size={27} /></span>
              <strong>建立第一本长期题库</strong>
              <small>支持 PDF、图片和扫描版习题册</small>
            </button>
          ) : (
            <div className="question-bank-grid">
              {questionBanks.map((bank) => {
                const status = questionBankStatus[bank.status]
                return (
                  <article className="question-bank-card" key={bank.id} onClick={() => setBankDetailId(bank.id)}>
                    <div className="question-bank-card-top">
                      <span><BookMarked size={21} /></span>
                      <Tag color={status.color} icon={status.icon}>{status.label}</Tag>
                    </div>
                    <h3>{bank.title}</h3>
                    <p>{bank.source_name}</p>
                    {bank.status === 'processing' && <Progress percent={bank.processing_progress || 1} showInfo={false} status="active" />}
                    {bank.processing_error && <div className="homework-card-error">{bank.processing_error}</div>}
                    <div className="question-bank-card-data">
                      <span><strong>{bank.question_count}</strong> 道题</span>
                      <span><strong>{bank.page_count || '—'}</strong> 页</span>
                    </div>
                    <footer><span>{formatTime(bank.updated_at)} 更新</span><ChevronRight size={16} /></footer>
                  </article>
                )
              })}
            </div>
          )}
        </section>

        <section className="homework-library">
          <header className="teacher-section-heading">
            <div><span>ASSIGNMENTS</span><h2>作业列表</h2></div>
            <Button icon={<RefreshCw size={14} />} onClick={() => void Promise.all([loadHomeworks(true), loadQuestionBanks()])}>刷新</Button>
          </header>
          {loading ? (
            <div className="teacher-loading"><Spin /><span>正在读取作业…</span></div>
          ) : homeworks.length === 0 ? (
            <button className="teacher-empty" type="button" onClick={() => { setCreateMode('upload'); setUploadOpen(true) }}>
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
                      {homework.max_score > 0
                        ? <span><strong>{homework.max_score}</strong>分</span>
                        : <span><strong>—</strong>未设分值</span>}
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
        onOk={() => void submitHomework()}
        okText={createMode === 'upload' ? '上传并开始识别' : `用已选 ${selectedBankQuestions.length} 题创建`}
        cancelText="取消"
        confirmLoading={uploading}
        width={createMode === 'upload' ? 650 : 860}
        className="homework-upload-modal"
        title={null}
      >
        <div className="homework-modal-heading">
          <span>{createMode === 'upload' ? <UploadCloud size={22} /> : <BookMarked size={22} />}</span>
          <div><small>NEW ASSIGNMENT</small><h2>创建一份新作业</h2><p>{createMode === 'upload' ? '视觉模型将自动拆分题目、插图、答案与评分点。' : '从长期题库勾选题目，立即生成独立的结构化作业。'}</p></div>
        </div>
        <div className="homework-upload-form">
          <Segmented
            block
            value={createMode}
            options={[{ label: '上传新附件', value: 'upload' }, { label: '从题库选题', value: 'bank' }]}
            onChange={(value) => setCreateMode(value as 'upload' | 'bank')}
          />
          <label><span>作业标题</span><Input value={title} onChange={(event) => setTitle(event.target.value)} placeholder="留空时使用附件名称" maxLength={120} /></label>
          <label><span>作业说明</span><TextArea value={instructions} onChange={(event) => setInstructions(event.target.value)} placeholder="例如：写出完整计算过程，拍照时保证页面清晰" autoSize={{ minRows: 2, maxRows: 4 }} maxLength={2000} /></label>
          <label><span>截止时间</span><Input type="datetime-local" value={dueAt} onChange={(event) => setDueAt(event.target.value)} /></label>
          {createMode === 'upload' ? (
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
          ) : (
            <div className="question-bank-picker">
              {questionBanks.filter((bank) => bank.status === 'ready' && bank.questions.length > 0).length === 0 ? (
                <div className="question-bank-picker-empty">
                  <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="暂无可选题库，请先上传并等待识别完成" />
                  <Button icon={<UploadCloud size={14} />} onClick={() => { setUploadOpen(false); setBankUploadOpen(true) }}>上传题库</Button>
                </div>
              ) : questionBanks.filter((bank) => bank.status === 'ready' && bank.questions.length > 0).map((bank) => {
                const bankKeys = bank.questions.map((question) => bankQuestionKey(bank.id, question.id))
                const selectedCount = bankKeys.filter((key) => selectedBankQuestions.includes(key)).length
                return (
                  <section key={bank.id}>
                    <header>
                      <Checkbox
                        checked={selectedCount === bankKeys.length}
                        indeterminate={selectedCount > 0 && selectedCount < bankKeys.length}
                        onChange={(event) => setSelectedBankQuestions((keys) => event.target.checked
                          ? Array.from(new Set([...keys, ...bankKeys]))
                          : keys.filter((key) => !bankKeys.includes(key)))}
                      >
                        <strong>{bank.title}</strong>
                      </Checkbox>
                      <span>已选 {selectedCount}/{bank.questions.length}</span>
                    </header>
                    <div className="question-bank-picker-list">
                      {bank.questions.map((question) => {
                        const key = bankQuestionKey(bank.id, question.id)
                        const figure = question.figures?.[0]
                        return (
                          <article className={selectedBankQuestions.includes(key) ? 'selected' : ''} key={question.id}>
                            <Checkbox
                              checked={selectedBankQuestions.includes(key)}
                              onChange={(event) => setSelectedBankQuestions((keys) => event.target.checked
                                ? [...keys, key]
                                : keys.filter((value) => value !== key))}
                            />
                            <div>
                              <span>{question.section_title || '题目'} · 第 {question.number} 题</span>
                              <MathMarkdown content={question.prompt || '未识别到题干'} />
                              <small>{question.options?.length ? `${question.options.length} 个选项 · ` : ''}{question.subquestions?.length ? `${question.subquestions.length} 个小问 · ` : ''}{question.answer || question.answer_figures?.length ? '含参考答案' : '暂无参考答案'}</small>
                            </div>
                            {figure && <img src={figure.url} alt={figure.caption || `第 ${question.number} 题题图`} />}
                          </article>
                        )
                      })}
                    </div>
                  </section>
                )
              })}
            </div>
          )}
        </div>
      </Modal>

      <Modal
        open={bankUploadOpen}
        onCancel={() => {
          if (bankUploading) return
          setBankUploadOpen(false)
          setBankFileList([])
          setBankTitle('')
        }}
        onOk={() => void uploadQuestionBank()}
        okText="保存并开始提取"
        cancelText="取消"
        confirmLoading={bankUploading}
        width={650}
        className="homework-upload-modal question-bank-upload-modal"
        title={null}
      >
        <div className="homework-modal-heading">
          <span><LibraryBig size={22} /></span>
          <div><small>LONG-TERM QUESTION BANK</small><h2>上传一本长期题库</h2><p>指导书和习题册将长期保存，讲解、目录和无关内容不会入题。</p></div>
        </div>
        <div className="homework-upload-form">
          <label><span>题库名称</span><Input value={bankTitle} onChange={(event) => setBankTitle(event.target.value)} placeholder="留空时使用附件名称" maxLength={120} /></label>
          <Dragger
            accept=".pdf,.png,.jpg,.jpeg,.webp,.bmp"
            maxCount={1}
            fileList={bankFileList}
            beforeUpload={() => false}
            onChange={({ fileList: next }) => setBankFileList(next.slice(-1))}
          >
            <p className="ant-upload-drag-icon"><LibraryBig size={34} /></p>
            <p className="ant-upload-text">拖入学习指导书、习题册或扫描图片</p>
            <p className="ant-upload-hint">自动提取题号、题目、题图与答案 · 题库可长期复用和人为删除</p>
          </Dragger>
        </div>
      </Modal>

      <Modal
        open={Boolean(bankDetail)}
        onCancel={() => setBankDetailId(null)}
        footer={null}
        width={1100}
        className="homework-detail-modal question-bank-detail-modal"
        title={null}
        destroyOnHidden
      >
        {bankDetail && (
          <div className="homework-detail question-bank-detail">
            <header className="homework-detail-header">
              <div>
                <Tag color={questionBankStatus[bankDetail.status].color}>{questionBankStatus[bankDetail.status].label}</Tag>
                <h2>{bankDetail.title}</h2>
                <p>{bankDetail.source_name} · {bankDetail.question_count} 道题 · {bankDetail.page_count || 0} 页</p>
                <div className="model-pipeline">
                  <span><Sparkles size={12} /> {bankDetail.extraction_model}</span>
                  <i />
                  <span><FileText size={12} /> PDF-Extract-Kit</span>
                  <i />
                  <span><BookMarked size={12} /> 长期保存</span>
                </div>
              </div>
              <div className="homework-detail-actions">
                {bankDetail.source_url && <Button href={bankDetail.source_url} target="_blank" icon={<Eye size={15} />}>原始附件</Button>}
                {bankDetail.status === 'ready' && bankDetail.questions.length > 0 && (
                  <Button type="primary" icon={<BookOpenCheck size={15} />} onClick={() => { setBankDetailId(null); setCreateMode('bank'); setUploadOpen(true) }}>选择题目布置</Button>
                )}
                {bankDetail.status === 'error' && (
                  <Button type="primary" icon={<RefreshCw size={15} />} loading={bankActionId === bankDetail.id} onClick={() => void runBankAction(bankDetail.id, 'retry')}>重新识别</Button>
                )}
                <Popconfirm title="删除整本题库？" description="题库文件和题目将删除；已布置的作业不会受影响。" okText="删除" cancelText="取消" okButtonProps={{ danger: true }} onConfirm={() => void runBankAction(bankDetail.id, 'delete')}>
                  <Button danger icon={<Trash2 size={15} />} />
                </Popconfirm>
              </div>
            </header>
            {bankDetail.status === 'processing' && (
              <div className="homework-processing-panel"><LoaderCircle className="spin" size={30} /><div><strong>{bankDetail.processing_message || '正在逐页提取题库内容'}</strong><span>进度 {bankDetail.processing_progress || 0}% · 识别完成后即可长期选用。</span></div></div>
            )}
            {bankDetail.processing_error && <div className="homework-detail-error"><AlertTriangle size={18} /><div><strong>题库识别未完成</strong><span>{bankDetail.processing_error}</span></div></div>}
            {bankDetail.status === 'ready' && (
              <QuestionBankPreview
                bank={bankDetail}
                deletingQuestionId={deletingQuestionId}
                onEditQuestion={(question) => setEditingQuestion({ kind: 'question-bank', documentId: bankDetail.id, question })}
                onDeleteQuestion={(questionId) => void removeBankQuestion(bankDetail.id, questionId)}
              />
            )}
          </div>
        )}
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
                  <Button type="primary" icon={<Send size={15} />} loading={actionId === detail.id} disabled={incompleteChoiceNumbers.length > 0} onClick={() => void runAction(detail.id, 'publish')}>发布给学生</Button>
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
              <div className="homework-processing-panel"><LoaderCircle className="spin" size={30} /><div><strong>{detail.processing_message || '正在逐页筛选题目、题图与参考答案'}</strong><span>进度 {detail.processing_progress || 0}% · 页面较多时需要几分钟，完成后可预览结构化作业内容。</span></div></div>
            )}
            {detail.processing_error && <div className="homework-detail-error"><AlertTriangle size={18} /><div><strong>识别未完成</strong><span>{detail.processing_error}</span></div></div>}
            {detail.status === 'draft' && incompleteChoiceNumbers.length > 0 && (
              <div className="homework-integrity-warning">
                <AlertTriangle size={19} />
                <div>
                  <strong>检测到选择题选项不完整</strong>
                  <span>第 {incompleteChoiceNumbers.slice(0, 12).join('、')} 题仍是旧版识别数据，已禁止发布；重新识别后会保留原选择题形式和 A/B/C/D 选项。</span>
                </div>
                <Button icon={<RefreshCw size={14} />} loading={actionId === detail.id} onClick={() => void runAction(detail.id, 'retry')}>修复并重新识别</Button>
              </div>
            )}
            {detail.questions.length > 0 && (
              <QuestionPreview
                homework={detail}
                onEditQuestion={(question) => setEditingQuestion({ kind: 'homework', documentId: detail.id, question })}
              />
            )}

            <section className="teacher-submission-section">
              <header className="teacher-section-heading"><div><span>SUBMISSIONS</span><h3>学生提交与批改</h3></div><small>当前演示为 1 名学生</small></header>
              {detail.submissions?.length ? detail.submissions.map((submission) => (
                <SubmissionPanel key={submission.id} submission={submission} />
              )) : <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="学生尚未提交答案" />}
            </section>
          </div>
        )}
      </Modal>
      <QuestionEditorModal
        context={editingQuestion}
        onClose={() => setEditingQuestion(null)}
        onDocumentChange={applyEditedDocument}
      />
    </div>
  )
}
