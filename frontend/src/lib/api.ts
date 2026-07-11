export type SourceInfo = {
  id: string
  source: string
  chapter: string
  section: string
  page_start: number | null
  page_end: number | null
  score: number
  doc_type: 'textbook' | 'question' | string
}

export type KBStatus = {
  id: string
  state: 'ready' | 'building' | 'error' | 'missing'
  documents: number
  chunks: number
  message: string
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
}

export type ModelCatalog = {
  default: { provider: ModelProviderId; model: string }
  providers: ModelProviderInfo[]
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
}

type SSECallbacks = {
  onStatus: (data: { stage: string; message: string; agent: string }) => void
  onMeta: (data: { intent: string; agent: string; provider: ModelProviderId; model: string; sources: SourceInfo[]; verification?: Record<string, unknown> }) => void
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
