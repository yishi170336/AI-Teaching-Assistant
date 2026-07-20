import { useCallback, useEffect, useMemo, useState } from 'react'
import { Button, Empty, Input, Modal, Progress, Radio, Spin, Tag, Upload, message } from 'antd'
import type { UploadFile } from 'antd'
import {
  AlertTriangle,
  BookOpenCheck,
  Camera,
  CheckCircle2,
  Clock3,
  Eye,
  FileCheck2,
  LoaderCircle,
  Printer,
  RefreshCw,
  ShieldCheck,
  Sparkles,
  UploadCloud,
} from 'lucide-react'
import {
  fetchHomeworks,
  Homework,
  HomeworkQuestion,
  HomeworkStudentAnswer,
  submitHomework,
} from '../lib/api'
import HomeworkPaper from '../components/HomeworkPaper'

const { Dragger } = Upload
const { TextArea } = Input

type StudentAnswerDraft = {
  answer: string
  selected_options: string[]
  subquestion_answers: Array<{ label: string; text: string }>
}

const photoQuestionTypes = new Set(['calculation', 'design', 'other'])

function emptyDraft(question: HomeworkQuestion): StudentAnswerDraft {
  return {
    answer: '',
    selected_options: [],
    subquestion_answers: (question.subquestions || []).map((part) => ({ label: part.label, text: '' })),
  }
}

function responseIsComplete(
  question: HomeworkQuestion,
  response: StudentAnswerDraft,
  files: UploadFile[],
) {
  if (photoQuestionTypes.has(question.question_type)) return files.some((file) => file.originFileObj)
  if (question.question_type === 'choice' || question.question_type === 'true_false') {
    return response.selected_options.length > 0
  }
  if (question.subquestions?.length) {
    return question.subquestions.every((part) => response.subquestion_answers.some(
      (answer) => answer.label === part.label && answer.text.trim(),
    ))
  }
  return Boolean(response.answer.trim() || files.some((file) => file.originFileObj))
}

function QuestionResponseEditor({
  question,
  response,
  files,
  onChange,
  onFilesChange,
}: {
  question: HomeworkQuestion
  response: StudentAnswerDraft
  files: UploadFile[]
  onChange: (next: StudentAnswerDraft) => void
  onFilesChange: (files: UploadFile[]) => void
}) {
  const updatePart = (label: string, text: string) => onChange({
    ...response,
    subquestion_answers: response.subquestion_answers.map((part) => (
      part.label === label ? { ...part, text } : part
    )),
  })
  const requiresPhoto = photoQuestionTypes.has(question.question_type)
  return (
    <section className={`student-question-response ${requiresPhoto ? 'photo-required' : 'direct-answer'}`}>
      <header>
        <div>
          {requiresPhoto ? <Camera size={16} /> : <FileCheck2 size={16} />}
          <strong>{requiresPhoto ? '本题上传作答' : '在此填写答案'}</strong>
        </div>
        <Tag color={responseIsComplete(question, response, files) ? 'success' : 'gold'}>
          {responseIsComplete(question, response, files) ? '已完成' : '待作答'}
        </Tag>
      </header>

      {question.question_type === 'choice' && (
        <Radio.Group
          className="student-choice-response"
          value={response.selected_options[0]}
          onChange={(event) => onChange({ ...response, selected_options: [event.target.value] })}
        >
          {(question.options || []).map((option) => (
            <Radio.Button value={option.label} key={option.label}>{option.label}</Radio.Button>
          ))}
        </Radio.Group>
      )}
      {question.question_type === 'true_false' && (
        <Radio.Group
          value={response.selected_options[0]}
          onChange={(event) => onChange({ ...response, selected_options: [event.target.value] })}
        >
          <Radio value="正确">正确</Radio><Radio value="错误">错误</Radio>
        </Radio.Group>
      )}
      {!requiresPhoto && !['choice', 'true_false'].includes(question.question_type) && (
        question.subquestions?.length ? (
          <div className="student-subquestion-responses">
            {question.subquestions.map((part) => (
              <label key={part.label}>
                <span>（{part.label}）</span>
                {question.question_type === 'fill_blank'
                  ? <Input value={response.subquestion_answers.find((item) => item.label === part.label)?.text || ''} onChange={(event) => updatePart(part.label, event.target.value)} placeholder="填写答案" />
                  : <TextArea autoSize={{ minRows: 2, maxRows: 6 }} value={response.subquestion_answers.find((item) => item.label === part.label)?.text || ''} onChange={(event) => updatePart(part.label, event.target.value)} placeholder="填写本小问答案" />}
              </label>
            ))}
          </div>
        ) : question.question_type === 'fill_blank'
          ? <Input value={response.answer} onChange={(event) => onChange({ ...response, answer: event.target.value })} placeholder="在此填写答案" />
          : <TextArea autoSize={{ minRows: 3, maxRows: 8 }} value={response.answer} onChange={(event) => onChange({ ...response, answer: event.target.value })} placeholder="在此填写答案" />
      )}
      {requiresPhoto && (
        <>
          <TextArea
            autoSize={{ minRows: 2, maxRows: 5 }}
            value={response.answer}
            onChange={(event) => onChange({ ...response, answer: event.target.value })}
            placeholder="可补充文字说明（选填）"
          />
          <Dragger
            accept="image/*"
            multiple
            maxCount={4}
            fileList={files}
            beforeUpload={() => false}
            onChange={({ fileList }) => onFilesChange(fileList.slice(0, 4))}
          >
            <p className="ant-upload-drag-icon"><UploadCloud size={25} /></p>
            <p className="ant-upload-text">拍照或选择第 {question.number} 题答案</p>
            <p className="ant-upload-hint">本题单独上传，最多 4 张</p>
          </Dragger>
        </>
      )}
    </section>
  )
}

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
  const [answers, setAnswers] = useState<Record<string, StudentAnswerDraft>>({})
  const [questionFiles, setQuestionFiles] = useState<Record<string, UploadFile[]>>({})
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
    const homework = homeworks.find((item) => item.id === homeworkId)
    const existing = new Map(
      (homework?.submission?.answers || []).map((answer) => [answer.question_id, answer]),
    )
    setAnswers(Object.fromEntries((homework?.questions || []).map((question) => {
      const saved = existing.get(question.id)
      return [question.id, saved ? {
        answer: saved.answer || '',
        selected_options: saved.selected_options || [],
        subquestion_answers: saved.subquestion_answers?.length
          ? saved.subquestion_answers
          : emptyDraft(question).subquestion_answers,
      } : emptyDraft(question)]
    })))
    setQuestionFiles({})
    setDetailId(homeworkId)
  }

  const submit = async () => {
    if (!detail) return
    const missing = detail.questions.filter((question) => !responseIsComplete(
      question,
      answers[question.id] || emptyDraft(question),
      questionFiles[question.id] || [],
    ))
    if (missing.length) {
      return message.warning(`请先完成第 ${missing.slice(0, 10).map((question) => question.number).join('、')} 题`)
    }
    const structuredAnswers: HomeworkStudentAnswer[] = detail.questions.map((question) => ({
      question_id: question.id,
      ...(answers[question.id] || emptyDraft(question)),
    }))
    const mappedFiles = detail.questions.flatMap((question) => (
      (questionFiles[question.id] || []).flatMap((file) => file.originFileObj
        ? [{ questionId: question.id, file: file.originFileObj }]
        : [])
    ))
    if (mappedFiles.length > 40) return message.warning('整份作业最多上传 40 张答案图片')
    setSubmitting(true)
    try {
      await submitHomework(detail.id, studentId, structuredAnswers, mappedFiles)
      setQuestionFiles({})
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
          <p>查看老师发送的结构化作业内容，完成后拍照提交答案。</p>
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
                  {homework.max_score > 0
                    ? <span><strong>{homework.max_score}</strong>分</span>
                    : <span><strong>—</strong>未设分值</span>}
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
              <div className="student-homework-detail-actions">
                <Button icon={<Printer size={15} />} onClick={() => window.print()}>打印作业</Button>
                <Tag color={detail.submission ? 'success' : 'gold'}>{detail.submission ? '已提交' : '待完成'}</Tag>
              </div>
            </header>

            <section className="student-question-paper">
              <div className="student-paper-notice"><ShieldCheck size={16} /><span>选择题和填空题可在题目下直接作答；计算题、设计题等大题请在各题下方分别拍照上传，学生端不显示参考答案。</span></div>
              <HomeworkPaper
                homework={detail}
                mode="questions"
                printable
                renderQuestionResponse={(question) => (
                  <QuestionResponseEditor
                    question={question}
                    response={answers[question.id] || emptyDraft(question)}
                    files={questionFiles[question.id] || []}
                    onChange={(next) => setAnswers((current) => ({ ...current, [question.id]: next }))}
                    onFilesChange={(files) => setQuestionFiles((current) => ({ ...current, [question.id]: files }))}
                  />
                )}
              />
            </section>

            <section className="student-answer-submit">
              <header><div><CheckCircle2 size={19} /><span><strong>检查并提交整份作业</strong><small>系统会保留每个选择、填空内容以及每张图片对应的题号</small></span></div><Tag>逐题归档</Tag></header>
              <div className="student-answer-completion">
                {detail.questions.map((question) => {
                  const complete = responseIsComplete(
                    question,
                    answers[question.id] || emptyDraft(question),
                    questionFiles[question.id] || [],
                  )
                  return <span className={complete ? 'complete' : ''} key={question.id}>{question.number}</span>
                })}
              </div>
              <Button type="primary" size="large" block icon={<Camera size={17} />} loading={submitting} onClick={() => void submit()}>
                {detail.submission ? '重新提交并批改' : '提交答案并开始批改'}
              </Button>
            </section>

            {detail.submission && (
              <section className="student-submission-preview">
                <header><span><Eye size={16} /> 已提交的逐题答案图片</span><small>{formatDeadline(detail.submission.created_at)}</small></header>
                <div>{detail.submission.answer_images.map((asset, index) => <a href={asset.url} target="_blank" rel="noreferrer" key={asset.file}><img src={asset.url} alt={`第 ${asset.question_number || '?'} 题答案 ${index + 1}`} /><span>第 {asset.question_number || '?'} 题</span></a>)}</div>
              </section>
            )}
            <StudentGrading homework={detail} />
          </div>
        )}
      </Modal>
    </section>
  )
}
