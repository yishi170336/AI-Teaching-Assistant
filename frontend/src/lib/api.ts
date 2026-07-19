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

export type MistakeItem = {
  id: string
  student_id: string
  session_id: string
  question: string
  answer: string
  content: string
  summary: string
  agent: string
  knowledge_points: string[]
  attachments?: AttachmentInfo[]
  created_at: string
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

export async function fetchMistakes(studentId: string): Promise<MistakeItem[]> {
  const response = await fetch(`/api/mistakes?student_id=${encodeURIComponent(studentId)}`)
  if (!response.ok) throw new Error('错题本读取失败')
  return (await response.json()).mistakes || []
}

export async function addMistake(
  studentId: string,
  sessionId: string,
  question: string,
  answer: string,
  agent: string,
  attachments: AttachmentInfo[],
  modelConfig: ModelConfig,
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
) {
  const data = new FormData()
  data.append('file', file)
  data.append('knowledge_base', knowledgeBase)
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
