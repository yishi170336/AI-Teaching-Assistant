export type SourceInfo = {
  id: string
  source: string
  chapter: string
  section: string
  page_start: number | null
  page_end: number | null
  score: number
  doc_type: 'textbook' | 'question' | string
  excerpt?: string
  knowledge_tags?: string[]
  element_type?: string
  vector_score?: number
  bm25_score?: number
  graph_score?: number
  image_score?: number
  rerank_score?: number
  knowledge_base?: string
  historical?: boolean
  citation_index?: number
}

export type KBStatus = {
  id: string
  display_name?: string
  state: 'ready' | 'building' | 'cancelling' | 'cancelled' | 'error' | 'missing'
  documents: number
  chunks: number
  message: string
  available?: boolean
  progress?: number
  stage?: string
  cancellable?: boolean
  started_at?: string
  updated_at?: string
  completed_at?: string
  validation?: { status?: string; question_chunks?: number }
  pipeline_layers?: Record<string, { status?: string }>
}

export type AttachmentInfo = {
  id: string
  name: string
  content_type: string
  size: number
  kind: 'image' | 'document'
  url: string
}

export type ModelProviderId = 'ollama' | 'deepseek' | 'qwen' | 'custom'

export type ModelConfig = {
  provider: ModelProviderId
  model: string
  apiKey: string
  baseUrl: string
}

export type ModelProviderInfo = {
  id: ModelProviderId
  label: string
  description: string
  models: string[]
  default_model: string
  base_url: string
  requires_api_key: boolean
  configured: boolean
  status_message?: string
  model_options?: Array<{
    value: string
    label: string
    disabled?: boolean
    description?: string
  }>
}

export type ModelCatalog = {
  default: { provider: ModelProviderId; model: string }
  providers: ModelProviderInfo[]
  ollama_available?: boolean
}

export type KnowledgeGraphNode = {
  id: string
  type: 'concept' | 'component' | 'circuit' | 'document' | 'page' | string
  name: string
  chunk_id?: string
  page?: number
  pages?: number[]
  evidence_count?: number
  symbol?: string
  component_role?: string
  component_type?: string
}

export type KnowledgeGraphEdge = {
  source: string
  target: string
  type: string
  evidence_count?: number
}

export type ChapterKnowledgeConcept = {
  id: string
  name: string
  evidence_count: number
  pages: number[]
}

export type ChapterKnowledgeSummary = {
  id: string
  name: string
  order: number
  page_start: number | null
  page_end: number | null
  pages: number[]
  sources: string[]
  section_count: number
  concept_count: number
  concepts: ChapterKnowledgeConcept[]
}

export type KnowledgeGraph = {
  knowledge_base: string
  nodes: KnowledgeGraphNode[]
  edges: KnowledgeGraphEdge[]
  chapters?: ChapterKnowledgeSummary[]
  stats: { nodes: number; edges: number; concepts: number; documents?: number; pages?: number; circuits?: number; components?: number; chapters?: number }
}

export type MistakeSource = 'question_bank' | 'ai_generated' | 'user_uploaded'

export type MistakeKnowledgeTag = {
  tag_id: string
  tag_name: string
  tag_source: 'knowledge_graph' | 'custom'
  knowledge_node_id: string | null
  match_type: 'exact' | 'approximate' | 'unmatched'
  confidence: number
  is_exact: boolean
  needs_confirmation: boolean
}

export type MistakePrerequisite = {
  knowledge_node_id: string | null
  name: string
  source: 'knowledge_graph' | 'chapter_order'
  relation: string
  confidence: number
}

export type MistakeAnnotation = {
  id: string
  student_id: string
  mistake_id: string
  content: string
  client_request_id?: string
  created_at: string
  updated_at: string
}

export type MistakeCategory = {
  id: string
  student_id: string
  name: string
  created_at: string
  updated_at: string
}

export type MistakeMessage = {
  role: 'user' | 'assistant'
  content: string
  agent?: string
  model?: string
  created_at?: string
}

export type MistakeItem = {
  id: string
  schema_version?: string
  student_id: string
  session_id: string
  question: string
  answer: string
  content: string
  summary: string
  title: string
  agent: string
  knowledge_base: string
  knowledge_points: string[]
  knowledge_tags: MistakeKnowledgeTag[]
  location: {
    chapter: string
    section: string
    source: 'knowledge_graph' | 'unmatched' | 'unavailable'
    confidence: number
  }
  prerequisites: MistakePrerequisite[]
  source: MistakeSource
  question_bank_id: string
  category_id: string
  messages: MistakeMessage[]
  annotations: MistakeAnnotation[]
  attachments?: AttachmentInfo[]
  created_at: string
  updated_at: string
}

export type MistakeWeakArea = {
  knowledge_point: string
  mistake_count: number
  mistake_ids: string[]
  source_count: number
  score: number
  severity: '轻度薄弱' | '中度薄弱' | '重度薄弱'
  chapter: string
  section: string
  prerequisites: MistakePrerequisite[]
  priority?: number
}

export type MistakeAnalysis = {
  total_mistakes: number
  data_sufficient: boolean
  notice: string
  source_counts: Record<string, number>
  chapter_counts: Record<string, number>
  annotation_count: number
  weak_areas: MistakeWeakArea[]
  recommended_order: MistakeWeakArea[]
  scoring_rule: Record<string, number>
}

export type MistakeNotebook = {
  mistakes: MistakeItem[]
  categories: MistakeCategory[]
  analysis: MistakeAnalysis
}

export type ScheduleCategory = 'exam' | 'study' | 'activity' | 'other'

export type ScheduleItem = {
  id: string
  student_id: string
  title: string
  date: string
  time: string
  category: ScheduleCategory
  note: string
  completed: boolean
  created_at: string
  updated_at: string
}

export type ScheduleItemDraft = Pick<ScheduleItem, 'title' | 'date' | 'time' | 'category' | 'note'>

export type HomeworkStatus = 'processing' | 'draft' | 'published' | 'error'
export type QuestionBankStatus = 'processing' | 'ready' | 'error'
export type HomeworkSubmissionStatus = 'submitted' | 'grading' | 'graded' | 'review_required' | 'error'

export type HomeworkAsset = {
  file: string
  name?: string
  caption?: string
  url: string
  page?: number
  width?: number
  height?: number
  size?: number
  content_type?: string
  redactions_applied?: number
  source_top?: number
  source_left?: number
  position?: 'before_question' | 'after_question' | 'after_options' | string
  question_id?: string
  question_number?: string
}

export type HomeworkOption = { label: string; text: string }
export type HomeworkQuestionPart = { label: string; text: string }

export type HomeworkQuestion = {
  id: string
  section_key: string
  section_title: string
  number: string
  question_type: string
  prompt: string
  subquestions?: HomeworkQuestionPart[]
  options: HomeworkOption[]
  option_columns: number
  figure_position: 'before_question' | 'after_question' | 'after_options' | string
  points: number
  page_start: number | null
  page_end: number | null
  sequence: number
  layout_images: HomeworkAsset[]
  figures: HomeworkAsset[]
  answer?: string
  answer_subquestions?: HomeworkQuestionPart[]
  answer_figures?: HomeworkAsset[]
  rubric?: string
}

export type HomeworkStudentAnswer = {
  question_id: string
  number?: string
  question_type?: string
  answer: string
  selected_options: string[]
  subquestion_answers: HomeworkQuestionPart[]
}

export type HomeworkQuestionUpdate = Partial<Pick<HomeworkQuestion,
  | 'section_key'
  | 'section_title'
  | 'number'
  | 'question_type'
  | 'prompt'
  | 'subquestions'
  | 'options'
  | 'option_columns'
  | 'figure_position'
  | 'points'
  | 'answer'
  | 'answer_subquestions'
  | 'rubric'
>> & {
  figures?: Array<Pick<HomeworkAsset, 'file' | 'caption' | 'position'>>
  answer_figures?: Array<Pick<HomeworkAsset, 'file' | 'caption' | 'position'>>
}

export type HomeworkGradingItem = {
  question_id: string
  number: string
  student_answer: string
  score: number
  max_score: number
  is_scored?: boolean
  is_correct: boolean
  feedback: string
  evidence: string
  subquestion_results?: Array<{
    label: string
    answered: boolean
    student_answer: string
    score: number
    max_score: number
    feedback: string
    completeness_evidence: string
  }>
}

export type HomeworkGrading = {
  items: HomeworkGradingItem[]
  total_score: number
  max_score: number
  summary: string
}

export type HomeworkReview = {
  passed: boolean
  confidence: number
  issues: string[]
  recommendation: string
  review_model: string
}

export type HomeworkSubmission = {
  id: string
  homework_id: string
  student_id: string
  student_name: string
  status: HomeworkSubmissionStatus
  answers: HomeworkStudentAnswer[]
  answer_images: HomeworkAsset[]
  extracted_answer: string
  grading: HomeworkGrading | null
  review: HomeworkReview | null
  processing_error: string
  created_at: string
  updated_at: string
}

export type Homework = {
  id: string
  title: string
  instructions: string
  due_at: string
  status: HomeworkStatus
  source_name: string
  source_url?: string
  created_at: string
  updated_at: string
  published_at: string
  extraction_model: string
  grading_model: string
  review_model: string
  processing_error: string
  processing_warnings: string[]
  processing_progress: number
  processing_message: string
  page_count: number
  max_score: number
  question_count: number
  questions: HomeworkQuestion[]
  submissions?: HomeworkSubmission[]
  submission_count?: number
  submission?: HomeworkSubmission | null
}

export type QuestionBank = {
  id: string
  title: string
  status: QuestionBankStatus
  source_name: string
  source_url?: string
  created_at: string
  updated_at: string
  extraction_model: string
  processing_error: string
  processing_warnings: string[]
  processing_progress: number
  processing_message: string
  page_count: number
  max_score: number
  question_count: number
  questions: HomeworkQuestion[]
}

export type GeneratedPresentation = {
  blob: Blob
  filename: string
  slideCount: number
}

export type SessionSummary = {
  session_id: string
  title: string
  created_at: string
  updated_at: string
  message_count: number
}

export type StoredMessage = {
  role: 'user' | 'assistant'
  content: string
  created_at: string
  agent?: string
  provider?: ModelProviderId
  model?: string
  knowledge_base?: string
  attachments?: AttachmentInfo[]
  sources?: SourceInfo[]
  cited_sources?: SourceInfo[]
}

type SSECallbacks = {
  onStatus: (data: { stage: string; message: string; agent: string }) => void
  onMeta: (data: { intent: string; agent: string; provider: ModelProviderId; model: string; sources: SourceInfo[]; cited_sources: SourceInfo[]; verification?: Record<string, unknown> }) => void
  onDelta: (content: string) => void
  onDone: () => void
  onError: (message: string) => void
}

function parseEvent(block: string) {
  let event = 'message'
  const dataLines: string[] = []
  for (const line of block.split(/\r?\n/)) {
    if (line.startsWith('event:')) event = line.slice(6).trim()
    if (line.startsWith('data:')) dataLines.push(line.slice(5).trimStart())
  }
  const raw = dataLines.join('\n')
  return { event, data: raw ? JSON.parse(raw) : {} }
}

export async function streamChat(
  payload: {
    session_id: string
    message: string
    mode: string
    knowledge_base: string
    attachment_ids: string[]
    model_provider: ModelProviderId
    model: string
    api_key: string
    base_url: string
  },
  callbacks: SSECallbacks,
  signal?: AbortSignal,
) {
  const response = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify(payload),
    signal,
  })
  if (!response.ok || !response.body) {
    const detail = await response.text()
    throw new Error(detail || `请求失败 (${response.status})`)
  }
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let receivedTerminalEvent = false
  const dispatchBlock = (block: string) => {
    if (!block.trim()) return
    const { event, data } = parseEvent(block)
    if (event === 'status') callbacks.onStatus(data)
    if (event === 'meta') callbacks.onMeta(data)
    if (event === 'delta') callbacks.onDelta(data.content || '')
    if (event === 'done') {
      receivedTerminalEvent = true
      callbacks.onDone()
    }
    if (event === 'error') {
      receivedTerminalEvent = true
      callbacks.onError(data.message || '生成失败')
    }
  }
  while (true) {
    const { done, value } = await reader.read()
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done })
    const blocks = buffer.split(/\r?\n\r?\n/)
    buffer = blocks.pop() || ''
    for (const block of blocks) dispatchBlock(block)
    if (done) {
      dispatchBlock(buffer)
      break
    }
  }
  if (!receivedTerminalEvent && !signal?.aborted) {
    throw new Error('回答连接提前结束，已保留收到的内容，请重新生成')
  }
}

export async function fetchModels(): Promise<ModelCatalog> {
  const response = await fetch('/api/models')
  if (!response.ok) throw new Error('无法读取模型列表')
  return response.json()
}

export async function fetchSessions(): Promise<SessionSummary[]> {
  const response = await fetch('/api/sessions')
  if (!response.ok) throw new Error('无法读取历史会话')
  return (await response.json()).sessions || []
}

export async function fetchSession(sessionId: string): Promise<StoredMessage[]> {
  const response = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}`)
  if (!response.ok) throw new Error('无法恢复历史会话')
  return (await response.json()).messages || []
}

export async function deleteSession(sessionId: string): Promise<void> {
  const response = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}`, {
    method: 'DELETE',
  })
  if (!response.ok) {
    const result = await response.json().catch(() => ({}))
    throw new Error(result.detail || '历史会话删除失败')
  }
}

export async function uploadChatAttachment(file: File, sessionId: string): Promise<AttachmentInfo> {
  const data = new FormData()
  data.append('file', file)
  data.append('session_id', sessionId)
  const response = await fetch('/api/attachments', { method: 'POST', body: data })
  const result = await response.json()
  if (!response.ok) throw new Error(result.detail || result.error || '附件上传失败')
  return result.attachment
}

export async function fetchKnowledgeBases(): Promise<KBStatus[]> {
  const response = await fetch('/api/kb/status')
  if (!response.ok) throw new Error('无法读取知识库状态')
  return (await response.json()).knowledge_bases || []
}

export async function fetchKnowledgeGraph(knowledgeBase: string): Promise<KnowledgeGraph> {
  const response = await fetch(`/api/kb/${encodeURIComponent(knowledgeBase)}/graph`)
  const result = await response.json()
  if (!response.ok) throw new Error(result.detail || '知识图谱读取失败')
  return result
}

export function knowledgeBaseSourceUrl(
  knowledgeBase: string,
  source: string,
  page?: number | null,
): string {
  const query = new URLSearchParams({ source })
  const url = `/api/kb/${encodeURIComponent(knowledgeBase)}/source?${query.toString()}`
  return page ? `${url}#page=${page}` : url
}

export async function fetchMistakeNotebook(studentId: string): Promise<MistakeNotebook> {
  const response = await fetch(`/api/mistakes?student_id=${encodeURIComponent(studentId)}`)
  if (!response.ok) throw new Error('错题本读取失败')
  return response.json()
}

export async function fetchMistakes(studentId: string): Promise<MistakeItem[]> {
  return (await fetchMistakeNotebook(studentId)).mistakes || []
}

export async function addMistake(
  studentId: string,
  sessionId: string,
  question: string,
  answer: string,
  agent: string,
  attachments: AttachmentInfo[],
  modelConfig: ModelConfig,
  knowledgeBase = 'default',
  source?: MistakeSource,
  questionBankId = '',
): Promise<MistakeItem> {
  const response = await fetch('/api/mistakes', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      student_id: studentId,
      session_id: sessionId,
      question,
      answer,
      agent,
      knowledge_base: knowledgeBase,
      source,
      question_bank_id: questionBankId,
      messages: [
        { role: 'user', content: question },
        { role: 'assistant', content: answer, agent },
      ],
      attachment_ids: attachments.map((attachment) => attachment.id),
      model_provider: modelConfig.provider,
      model: modelConfig.model,
      api_key: modelConfig.apiKey,
      base_url: modelConfig.baseUrl,
    }),
  })
  const result = await response.json()
  if (!response.ok) throw new Error(result.detail || result.error || '加入错题本失败')
  return result.mistake
}

export async function deleteMistake(studentId: string, mistakeId: string): Promise<void> {
  const response = await fetch(
    `/api/mistakes/${encodeURIComponent(mistakeId)}?student_id=${encodeURIComponent(studentId)}`,
    { method: 'DELETE' },
  )
  if (!response.ok) throw new Error('删除错题失败')
}

export async function updateMistake(
  studentId: string,
  mistakeId: string,
  updates: { title?: string; category_id?: string },
): Promise<MistakeItem> {
  const response = await fetch(`/api/mistakes/${encodeURIComponent(mistakeId)}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ student_id: studentId, ...updates }),
  })
  const result = await response.json()
  if (!response.ok) throw new Error(result.detail || '错题更新失败')
  return result.mistake
}

export async function createMistakeCategory(studentId: string, name: string): Promise<MistakeCategory> {
  const response = await fetch('/api/mistakes/categories', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ student_id: studentId, name }),
  })
  const result = await response.json()
  if (!response.ok) throw new Error(result.detail || '分类创建失败')
  return result.category
}

export async function renameMistakeCategory(
  studentId: string,
  categoryId: string,
  name: string,
): Promise<MistakeCategory> {
  const response = await fetch(`/api/mistakes/categories/${encodeURIComponent(categoryId)}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ student_id: studentId, name }),
  })
  const result = await response.json()
  if (!response.ok) throw new Error(result.detail || '分类重命名失败')
  return result.category
}

export async function deleteMistakeCategory(studentId: string, categoryId: string): Promise<void> {
  const response = await fetch(
    `/api/mistakes/categories/${encodeURIComponent(categoryId)}?student_id=${encodeURIComponent(studentId)}`,
    { method: 'DELETE' },
  )
  const result = await response.json().catch(() => ({}))
  if (!response.ok) throw new Error(result.detail || '分类删除失败')
}

export async function addMistakeAnnotation(
  studentId: string,
  mistakeId: string,
  content: string,
  clientRequestId: string,
): Promise<MistakeAnnotation> {
  const response = await fetch(`/api/mistakes/${encodeURIComponent(mistakeId)}/annotations`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ student_id: studentId, content, client_request_id: clientRequestId }),
  })
  const result = await response.json()
  if (!response.ok) throw new Error(result.detail || '批注保存失败')
  return result.annotation
}

export async function updateMistakeAnnotation(
  studentId: string,
  mistakeId: string,
  annotationId: string,
  content: string,
): Promise<MistakeAnnotation> {
  const response = await fetch(
    `/api/mistakes/${encodeURIComponent(mistakeId)}/annotations/${encodeURIComponent(annotationId)}`,
    {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ student_id: studentId, content }),
    },
  )
  const result = await response.json()
  if (!response.ok) throw new Error(result.detail || '批注更新失败')
  return result.annotation
}

export async function deleteMistakeAnnotation(
  studentId: string,
  mistakeId: string,
  annotationId: string,
): Promise<void> {
  const response = await fetch(
    `/api/mistakes/${encodeURIComponent(mistakeId)}/annotations/${encodeURIComponent(annotationId)}?student_id=${encodeURIComponent(studentId)}`,
    { method: 'DELETE' },
  )
  const result = await response.json().catch(() => ({}))
  if (!response.ok) throw new Error(result.detail || '批注删除失败')
}

export async function fetchSchedule(studentId: string): Promise<ScheduleItem[]> {
  const response = await fetch(`/api/schedule?student_id=${encodeURIComponent(studentId)}`)
  if (!response.ok) throw new Error('日程读取失败')
  return (await response.json()).items || []
}

export async function addScheduleItem(
  studentId: string,
  draft: ScheduleItemDraft,
): Promise<ScheduleItem> {
  const response = await fetch('/api/schedule', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ student_id: studentId, ...draft }),
  })
  const result = await response.json()
  if (!response.ok) throw new Error(result.detail || result.error || '日程添加失败')
  return result.item
}

export async function setScheduleItemCompleted(
  studentId: string,
  itemId: string,
  completed: boolean,
): Promise<ScheduleItem> {
  const response = await fetch(`/api/schedule/${encodeURIComponent(itemId)}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ student_id: studentId, completed }),
  })
  const result = await response.json()
  if (!response.ok) throw new Error(result.detail || result.error || '日程状态更新失败')
  return result.item
}

export async function deleteScheduleItem(studentId: string, itemId: string): Promise<void> {
  const response = await fetch(
    `/api/schedule/${encodeURIComponent(itemId)}?student_id=${encodeURIComponent(studentId)}`,
    { method: 'DELETE' },
  )
  if (!response.ok) {
    const result = await response.json().catch(() => ({}))
    throw new Error(result.detail || '日程删除失败')
  }
}

async function homeworkResponse<T>(response: Response, fallback: string): Promise<T> {
  const result = await response.json().catch(() => ({}))
  if (!response.ok) throw new Error(result.detail || result.error || fallback)
  return result as T
}

export async function fetchHomeworks(
  role: 'teacher' | 'student',
  studentId = 'learner-demo',
): Promise<Homework[]> {
  const query = new URLSearchParams({ role, student_id: studentId })
  const response = await fetch(`/api/homeworks?${query.toString()}`)
  const result = await homeworkResponse<{ homeworks: Homework[] }>(response, '作业列表读取失败')
  return result.homeworks || []
}

export async function createHomework(
  file: File,
  fields: { title: string; instructions: string; dueAt: string },
): Promise<Homework> {
  const data = new FormData()
  data.append('file', file)
  data.append('title', fields.title)
  data.append('instructions', fields.instructions)
  data.append('due_at', fields.dueAt)
  const response = await fetch('/api/homeworks', { method: 'POST', body: data })
  const result = await homeworkResponse<{ homework: Homework }>(response, '作业上传失败')
  return result.homework
}

export async function createHomeworkFromQuestionBank(fields: {
  title: string
  instructions: string
  dueAt: string
  selections: Array<{ bank_id: string; question_ids: string[] }>
}): Promise<Homework> {
  const response = await fetch('/api/homeworks/from-question-bank', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      title: fields.title,
      instructions: fields.instructions,
      due_at: fields.dueAt,
      selections: fields.selections,
    }),
  })
  const result = await homeworkResponse<{ homework: Homework }>(response, '题库作业创建失败')
  return result.homework
}

export async function fetchQuestionBanks(): Promise<QuestionBank[]> {
  const response = await fetch('/api/question-banks')
  const result = await homeworkResponse<{ question_banks: QuestionBank[] }>(response, '题库读取失败')
  return result.question_banks || []
}

export async function createQuestionBank(file: File, title: string): Promise<QuestionBank> {
  const data = new FormData()
  data.append('file', file)
  data.append('title', title)
  const response = await fetch('/api/question-banks', { method: 'POST', body: data })
  const result = await homeworkResponse<{ question_bank: QuestionBank }>(response, '题库上传失败')
  return result.question_bank
}

export async function reprocessQuestionBank(bankId: string): Promise<void> {
  const response = await fetch(`/api/question-banks/${encodeURIComponent(bankId)}/reprocess`, {
    method: 'POST',
  })
  await homeworkResponse(response, '重新识别题库失败')
}

export async function deleteQuestionBank(bankId: string): Promise<void> {
  const response = await fetch(`/api/question-banks/${encodeURIComponent(bankId)}`, {
    method: 'DELETE',
  })
  await homeworkResponse(response, '题库删除失败')
}

export async function deleteQuestionBankQuestion(bankId: string, questionId: string): Promise<void> {
  const response = await fetch(
    `/api/question-banks/${encodeURIComponent(bankId)}/questions/${encodeURIComponent(questionId)}`,
    { method: 'DELETE' },
  )
  await homeworkResponse(response, '题目删除失败')
}

export type EditableQuestionDocument = 'homework' | 'question-bank'

function questionDocumentPath(kind: EditableQuestionDocument, documentId: string) {
  const collection = kind === 'homework' ? 'homeworks' : 'question-banks'
  return `/api/${collection}/${encodeURIComponent(documentId)}`
}

export async function updateDocumentQuestion(
  kind: EditableQuestionDocument,
  documentId: string,
  questionId: string,
  updates: HomeworkQuestionUpdate,
): Promise<Homework | QuestionBank> {
  const response = await fetch(
    `${questionDocumentPath(kind, documentId)}/questions/${encodeURIComponent(questionId)}`,
    {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(updates),
    },
  )
  const result = await homeworkResponse<{ homework?: Homework; question_bank?: QuestionBank }>(response, '题目保存失败')
  const document = kind === 'homework' ? result.homework : result.question_bank
  if (!document) throw new Error('题目保存后未返回文档数据')
  return document
}

export async function uploadDocumentQuestionAsset(
  kind: EditableQuestionDocument,
  documentId: string,
  questionId: string,
  target: 'figures' | 'answer_figures',
  file: File,
  caption: string,
  replaceFile = '',
): Promise<Homework | QuestionBank> {
  const data = new FormData()
  data.append('file', file)
  data.append('target', target)
  data.append('caption', caption)
  data.append('replace_file', replaceFile)
  const response = await fetch(
    `${questionDocumentPath(kind, documentId)}/questions/${encodeURIComponent(questionId)}/assets`,
    { method: 'POST', body: data },
  )
  const result = await homeworkResponse<{ homework?: Homework; question_bank?: QuestionBank }>(response, '题目图片上传失败')
  const document = kind === 'homework' ? result.homework : result.question_bank
  if (!document) throw new Error('图片上传后未返回文档数据')
  return document
}

export async function deleteDocumentQuestionAsset(
  kind: EditableQuestionDocument,
  documentId: string,
  questionId: string,
  target: 'figures' | 'answer_figures',
  assetName: string,
): Promise<Homework | QuestionBank> {
  const query = new URLSearchParams({ target })
  const response = await fetch(
    `${questionDocumentPath(kind, documentId)}/questions/${encodeURIComponent(questionId)}/assets/${encodeURIComponent(assetName)}?${query}`,
    { method: 'DELETE' },
  )
  const result = await homeworkResponse<{ homework?: Homework; question_bank?: QuestionBank }>(response, '题目图片删除失败')
  const document = kind === 'homework' ? result.homework : result.question_bank
  if (!document) throw new Error('图片删除后未返回文档数据')
  return document
}

export async function publishHomework(homeworkId: string): Promise<Homework> {
  const response = await fetch(`/api/homeworks/${encodeURIComponent(homeworkId)}/publish`, {
    method: 'POST',
  })
  const result = await homeworkResponse<{ homework: Homework }>(response, '作业发布失败')
  return result.homework
}

export async function reprocessHomework(homeworkId: string): Promise<void> {
  const response = await fetch(`/api/homeworks/${encodeURIComponent(homeworkId)}/reprocess`, {
    method: 'POST',
  })
  await homeworkResponse(response, '重新识别失败')
}

export async function deleteHomework(homeworkId: string): Promise<void> {
  const response = await fetch(`/api/homeworks/${encodeURIComponent(homeworkId)}`, {
    method: 'DELETE',
  })
  await homeworkResponse(response, '作业删除失败')
}

export async function submitHomework(
  homeworkId: string,
  studentId: string,
  answers: HomeworkStudentAnswer[],
  questionFiles: Array<{ questionId: string; file: File }>,
): Promise<HomeworkSubmission> {
  const data = new FormData()
  data.append('student_id', studentId)
  data.append('answers', JSON.stringify(answers))
  data.append('file_question_ids', JSON.stringify(questionFiles.map((item) => item.questionId)))
  questionFiles.forEach((item) => data.append('files', item.file))
  const response = await fetch(
    `/api/homeworks/${encodeURIComponent(homeworkId)}/submissions`,
    { method: 'POST', body: data },
  )
  const result = await homeworkResponse<{ submission: HomeworkSubmission }>(response, '答案提交失败')
  return result.submission
}

export async function startHomeworkSubmissionGrading(
  submissionId: string,
): Promise<HomeworkSubmission> {
  const response = await fetch(
    `/api/homework-submissions/${encodeURIComponent(submissionId)}/grade`,
    { method: 'POST' },
  )
  const result = await homeworkResponse<{ submission: HomeworkSubmission }>(response, '开始批改失败')
  return result.submission
}

function presentationFilename(disposition: string | null): string {
  if (!disposition) return '学习规划.pptx'
  const encoded = disposition.match(/filename\*=utf-8''([^;]+)/i)?.[1]
  if (encoded) {
    try {
      return decodeURIComponent(encoded.replace(/^"|"$/g, ''))
    } catch {
      // Fall through to the regular filename form.
    }
  }
  return disposition.match(/filename="?([^";]+)"?/i)?.[1]?.trim() || '学习规划.pptx'
}

export async function generateLearningPlanPpt(
  sessionId: string,
  content: string,
  topic: string,
): Promise<GeneratedPresentation> {
  const response = await fetch('/api/learning-plan/ppt', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, content, topic }),
  })
  if (!response.ok) {
    const raw = await response.text()
    let detail = raw
    try {
      const parsed = JSON.parse(raw)
      detail = parsed.detail || parsed.error || raw
    } catch {
      // Keep the plain-text response.
    }
    throw new Error(detail || '学习规划 PPT 生成失败')
  }
  return {
    blob: await response.blob(),
    filename: presentationFilename(response.headers.get('Content-Disposition')),
    slideCount: Number(response.headers.get('X-Slide-Count') || 0),
  }
}

export async function uploadKnowledgeFile(
  file: File,
  knowledgeBase: string,
  modelConfig: ModelConfig,
  displayName?: string,
) {
  const data = new FormData()
  data.append('file', file)
  data.append('knowledge_base', knowledgeBase)
  if (displayName?.trim()) data.append('display_name', displayName.trim())
  data.append('rebuild', 'true')
  data.append('model_provider', modelConfig.provider)
  data.append('model', modelConfig.model)
  data.append('api_key', modelConfig.apiKey)
  data.append('base_url', modelConfig.baseUrl)
  const response = await fetch('/api/upload', { method: 'POST', body: data })
  const result = await response.json()
  if (!response.ok) throw new Error(result.detail || result.error || '上传失败')
  return result
}

export async function rebuildKnowledgeBase(
  knowledgeBase: string,
  modelConfig: ModelConfig,
) {
  const response = await fetch('/api/kb/rebuild', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      knowledge_base: knowledgeBase,
      model_provider: modelConfig.provider,
      model: modelConfig.model,
      api_key: modelConfig.apiKey,
      base_url: modelConfig.baseUrl,
      chapter_limit: null,
    }),
  })
  const result = await response.json()
  if (!response.ok) throw new Error(result.detail || result.error || '重建失败')
  return result
}

export async function cancelKnowledgeBaseBuild(knowledgeBase: string) {
  const response = await fetch(`/api/kb/${encodeURIComponent(knowledgeBase)}/build`, {
    method: 'DELETE',
  })
  const result = await response.json()
  if (!response.ok) throw new Error(result.detail || result.error || '取消构建失败')
  return result
}

export async function deleteKnowledgeBase(knowledgeBase: string) {
  const response = await fetch(`/api/kb/${encodeURIComponent(knowledgeBase)}`, {
    method: 'DELETE',
  })
  const result = await response.json()
  if (!response.ok) throw new Error(result.detail || result.error || '删除知识库失败')
  return result
}
