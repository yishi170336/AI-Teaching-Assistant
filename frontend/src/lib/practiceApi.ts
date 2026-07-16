export type PracticeChapter = {
  id: string
  title: string
  description: string
  question_count: number
  completed_count: number
  resume_question_id: string | null
  questions: PracticeQuestionCatalogItem[]
}

export type PracticeQuestionCatalogItem = {
  id: string
  title: string
  section: string
  completed: boolean
  resolved: boolean
  has_submission: boolean
  attempt_count: number
  grading_status: PracticeGradingStatus | null
  latest_verdict: PracticeVerdict | null
  last_submitted_at: string | null
}

export type PracticeCourse = {
  id: string
  title: string
  description: string
  question_count: number
  completed_count: number
  resume_question_id: string
  chapters: PracticeChapter[]
}

export type PracticeCatalog = {
  courses: PracticeCourse[]
}

export type PracticeFigure = {
  id: string
  alt: string
  caption: string
  url: string
}

export type PracticeVerdict = 'correct' | 'partially_correct' | 'incorrect' | 'unreadable'
export type PracticeGradingStatus = 'ungraded' | 'pending' | 'completed' | 'failed'

export type PracticeGradeIssue = {
  location: string
  problem: string
  correction: string
}

export type PracticeGrade = {
  verdict: PracticeVerdict
  summary: string
  strengths: string[]
  issues: PracticeGradeIssue[]
  solution_markdown: string
  model_provider: string
  model: string
  graded_at: string
}

export type PracticeConversationMessage = {
  id: string
  role: 'user' | 'assistant'
  content: string
  created_at: string
}

export type PracticeSubmissionSummary = {
  completed: boolean
  resolved: boolean
  resolved_at: string | null
  mastered_at: string | null
  has_submission: boolean
  attempt_count: number
  last_submitted_at: string | null
  latest_submission_id: string | null
  grading_status: PracticeGradingStatus | null
  grading_error: string | null
  grade: PracticeGrade | null
  conversation: PracticeConversationMessage[]
}

export type PracticeQuestion = {
  id: string
  number: string
  title: string
  section: string
  chapter_id: string
  prompt_markdown: string
  figures: PracticeFigure[]
  position: number
  total: number
  previous_question_id: string | null
  next_question_id: string | null
  submission: PracticeSubmissionSummary
}

export type PracticeSubmission = {
  submission_id: string
  question_id: string
  submitted_at: string
  image_count: number
  attempt_number: number
  grading_status: 'ungraded'
  completed: false
}

export type PracticeModelConfig = {
  provider: 'qwen' | 'custom'
  model: string
  apiKey: string
  baseUrl: string
}

export type PracticeGradeResponse = {
  submission_id: string
  question_id: string
  grading_status: PracticeGradingStatus
  resolved: boolean
  grade: PracticeGrade | null
}

export type PracticeResolveResponse = {
  completed: true
  resolved: true
  resolved_at: string
  next_question_id: string | null
}

export type PracticeQuestionReview = {
  question_id: string
  what_was_done: string
  error_steps: string[]
  advice: string[]
}

export type PracticeSessionFeedback = {
  headline: string
  summary_markdown: string
  question_reviews: PracticeQuestionReview[]
  strengths: string[]
  focus_areas: string[]
  recommendations: string[]
}

export type PracticeSession = {
  session_id: string
  status: 'active' | 'completed' | 'failed' | 'discarded'
  started_at: string
  ended_at: string | null
  feedback_status: 'not_started' | 'pending' | 'completed' | 'failed' | 'skipped'
  feedback_error: string | null
  question_count: number
  question_ids: string[]
  scope_version: number
  scope_label: string
  submitted_question_count: number
  submitted_question_ids: string[]
  submission_count: number
  feedback: PracticeSessionFeedback | null
}

async function responseError(response: Response, fallback: string) {
  const result = await response.json().catch(() => ({}))
  const detail = result.detail || result.error
  if (typeof detail === 'string') return detail
  return fallback
}

function modelPayload(studentId: string, config: PracticeModelConfig) {
  return {
    student_id: studentId,
    model_provider: config.provider,
    model: config.model,
    api_key: config.apiKey,
    base_url: config.baseUrl,
  }
}

export async function fetchPracticeCatalog(studentId: string): Promise<PracticeCatalog> {
  const response = await fetch(`/api/practice/catalog?student_id=${encodeURIComponent(studentId)}`)
  if (!response.ok) throw new Error(await responseError(response, '无法读取刷题目录'))
  return response.json()
}

export async function fetchPracticeQuestion(
  questionId: string,
  studentId: string,
): Promise<PracticeQuestion> {
  const response = await fetch(
    `/api/practice/questions/${encodeURIComponent(questionId)}?student_id=${encodeURIComponent(studentId)}`,
  )
  if (!response.ok) throw new Error(await responseError(response, '无法读取题目'))
  return response.json()
}

export async function submitPracticeAnswer(
  questionId: string,
  studentId: string,
  files: File[],
  sessionId?: string,
): Promise<PracticeSubmission> {
  const data = new FormData()
  data.append('student_id', studentId)
  if (sessionId) data.append('session_id', sessionId)
  files.forEach((file) => data.append('files', file))
  const response = await fetch(
    `/api/practice/questions/${encodeURIComponent(questionId)}/submissions`,
    { method: 'POST', body: data },
  )
  if (!response.ok) throw new Error(await responseError(response, '作答图片提交失败'))
  return response.json()
}

export async function gradePracticeSubmission(
  questionId: string,
  submissionId: string,
  studentId: string,
  config: PracticeModelConfig,
): Promise<PracticeGradeResponse> {
  const response = await fetch(
    `/api/practice/questions/${encodeURIComponent(questionId)}/submissions/${encodeURIComponent(submissionId)}/grade`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(modelPayload(studentId, config)),
    },
  )
  if (!response.ok) throw new Error(await responseError(response, 'AI 批改失败，请稍后重试'))
  return response.json()
}

function parseEvent(block: string) {
  let event = 'message'
  const data: string[] = []
  for (const line of block.split('\n')) {
    if (line.startsWith('event:')) event = line.slice(6).trim()
    if (line.startsWith('data:')) data.push(line.slice(5).trimStart())
  }
  return { event, data: data.join('\n') }
}

export async function streamPracticeFollowup(
  questionId: string,
  submissionId: string,
  studentId: string,
  message: string,
  config: PracticeModelConfig,
  callbacks: {
    onDelta: (content: string) => void
    onDone: (messages: PracticeConversationMessage[]) => void
  },
  signal?: AbortSignal,
) {
  const response = await fetch(
    `/api/practice/questions/${encodeURIComponent(questionId)}/submissions/${encodeURIComponent(submissionId)}/messages`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
      body: JSON.stringify({ ...modelPayload(studentId, config), message }),
      signal,
    },
  )
  if (!response.ok || !response.body) {
    throw new Error(await responseError(response, 'AI 答疑连接失败'))
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let streamError = ''
  const dispatch = (block: string) => {
    if (!block.trim()) return
    const parsed = parseEvent(block)
    if (!parsed.data) return
    const payload = JSON.parse(parsed.data)
    if (parsed.event === 'delta' && typeof payload.content === 'string') {
      callbacks.onDelta(payload.content)
    } else if (parsed.event === 'done') {
      callbacks.onDone(Array.isArray(payload.messages) ? payload.messages : [])
    } else if (parsed.event === 'error') {
      streamError = payload.message || 'AI 答疑失败，请稍后重试'
    }
  }

  while (true) {
    const { value, done } = await reader.read()
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done }).replace(/\r\n/g, '\n')
    let boundary = buffer.indexOf('\n\n')
    while (boundary >= 0) {
      dispatch(buffer.slice(0, boundary))
      buffer = buffer.slice(boundary + 2)
      boundary = buffer.indexOf('\n\n')
    }
    if (done) break
  }
  if (buffer.trim()) dispatch(buffer)
  if (streamError) throw new Error(streamError)
}

export async function resolvePracticeSubmission(
  questionId: string,
  submissionId: string,
  studentId: string,
): Promise<PracticeResolveResponse> {
  const response = await fetch(
    `/api/practice/questions/${encodeURIComponent(questionId)}/submissions/${encodeURIComponent(submissionId)}/resolve`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ student_id: studentId }),
    },
  )
  if (!response.ok) throw new Error(await responseError(response, '无法确认本题完成'))
  return response.json()
}

export async function startPracticeSession(
  studentId: string,
  questionId: string,
): Promise<PracticeSession> {
  const response = await fetch('/api/practice/sessions/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ student_id: studentId, question_id: questionId }),
  })
  if (!response.ok) throw new Error(await responseError(response, '无法开始本次练习记录'))
  return response.json()
}

export async function visitPracticeQuestion(
  sessionId: string,
  questionId: string,
  studentId: string,
): Promise<PracticeSession> {
  const response = await fetch(
    `/api/practice/sessions/${encodeURIComponent(sessionId)}/questions/${encodeURIComponent(questionId)}/visit`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ student_id: studentId }),
    },
  )
  if (!response.ok) throw new Error(await responseError(response, '无法记录本题学习状态'))
  return response.json()
}

export async function finishPracticeSession(
  sessionId: string,
  studentId: string,
  config: PracticeModelConfig,
): Promise<PracticeSession> {
  const response = await fetch(`/api/practice/sessions/${encodeURIComponent(sessionId)}/finish`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(modelPayload(studentId, config)),
  })
  if (!response.ok) throw new Error(await responseError(response, '学情反馈生成失败，请稍后重试'))
  return response.json()
}

export async function discardEmptyPracticeSession(
  sessionId: string,
  studentId: string,
): Promise<PracticeSession> {
  const response = await fetch(`/api/practice/sessions/${encodeURIComponent(sessionId)}/discard`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ student_id: studentId }),
  })
  if (!response.ok) throw new Error(await responseError(response, '无法结束本轮练习'))
  return response.json()
}

export async function fetchPracticeSessions(studentId: string): Promise<PracticeSession[]> {
  const response = await fetch(`/api/practice/sessions?student_id=${encodeURIComponent(studentId)}`)
  if (!response.ok) throw new Error(await responseError(response, '无法读取学情反馈'))
  const result = await response.json()
  return Array.isArray(result.sessions) ? result.sessions : []
}

export async function fetchActivePracticeSession(studentId: string): Promise<PracticeSession | null> {
  const response = await fetch(`/api/practice/sessions/active?student_id=${encodeURIComponent(studentId)}`)
  if (!response.ok) throw new Error(await responseError(response, '无法读取当前练习状态'))
  const result = await response.json()
  return result.session || null
}

export async function fetchPracticeSession(
  sessionId: string,
  studentId: string,
): Promise<PracticeSession> {
  const response = await fetch(
    `/api/practice/sessions/${encodeURIComponent(sessionId)}?student_id=${encodeURIComponent(studentId)}`,
  )
  if (!response.ok) throw new Error(await responseError(response, '无法读取本次学情反馈'))
  return response.json()
}

export async function deletePracticeSession(sessionId: string, studentId: string): Promise<void> {
  const response = await fetch(
    `/api/practice/sessions/${encodeURIComponent(sessionId)}?student_id=${encodeURIComponent(studentId)}`,
    { method: 'DELETE' },
  )
  if (!response.ok) throw new Error(await responseError(response, '无法删除本次学情反馈'))
}
