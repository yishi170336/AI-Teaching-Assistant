import { create } from 'zustand'
import { AttachmentInfo, KBStatus, ModelConfig, ModelProviderId, SourceInfo, StoredMessage, streamChat, uploadChatAttachment } from '../lib/api'

export type ChatMode = 'auto' | 'answer' | 'quiz' | 'plan'

export type ChatMessage = {
  id: string
  role: 'user' | 'assistant'
  content: string
  agent?: string
  sources?: SourceInfo[]
  failed?: boolean
  attachments?: AttachmentInfo[]
  model?: string
  provider?: ModelProviderId
}

export type PendingAttachment = {
  localId: string
  name: string
  size: number
  contentType: string
  kind: 'image' | 'document'
  status: 'uploading' | 'ready' | 'error'
  attachment?: AttachmentInfo
  error?: string
}

const sessionKey = 'circuitmind-session-id'
const studentKey = 'circuitmind-student-id'
const modelConfigKey = 'circuitmind-model-config'
const defaultKnowledgeBaseKey = 'circuitmind-default-knowledge-base'

const defaultModelConfig: ModelConfig = {
  provider: 'ollama',
  model: 'qwen3.5:2b',
  apiKey: '',
  baseUrl: 'http://127.0.0.1:11434',
}

function getSessionId() {
  let value = localStorage.getItem(sessionKey)
  if (!value) {
    value = `student-${crypto.randomUUID()}`
    localStorage.setItem(sessionKey, value)
  }
  return value
}

function canonicalModel(provider: ModelProviderId, model: string) {
  const normalized = model.trim()
  if (provider !== 'qwen') return normalized
  const canonical = normalized.toLowerCase()
  if (canonical === 'qwen3-vl-embedding' || canonical === 'qwen3-vl-8b-instruct') {
    return 'qwen3-vl-plus'
  }
  return canonical
}

function getStudentId() {
  let value = localStorage.getItem(studentKey)
  if (!value) {
    value = `learner-${crypto.randomUUID()}`
    localStorage.setItem(studentKey, value)
  }
  return value
}

function getModelConfig(): ModelConfig {
  try {
    const stored = JSON.parse(localStorage.getItem(modelConfigKey) || '{}')
    const providers: ModelProviderId[] = ['ollama', 'deepseek', 'qwen', 'custom']
    if (!providers.includes(stored.provider) || typeof stored.model !== 'string') {
      return defaultModelConfig
    }
    const config = {
      provider: stored.provider,
      model: canonicalModel(stored.provider, stored.model || defaultModelConfig.model),
      apiKey: typeof stored.apiKey === 'string' ? stored.apiKey : '',
      baseUrl: typeof stored.baseUrl === 'string' ? stored.baseUrl : '',
    }
    localStorage.setItem(modelConfigKey, JSON.stringify(config))
    return config
  } catch {
    return defaultModelConfig
  }
}

function getDefaultKnowledgeBase(): string {
  const stored = localStorage.getItem(defaultKnowledgeBaseKey)?.trim() || ''
  return /^[A-Za-z0-9_-]{1,48}$/.test(stored) ? stored : ''
}

const initialKnowledgeBase = getDefaultKnowledgeBase()

type ChatState = {
  studentId: string
  sessionId: string
  mode: ChatMode
  knowledgeBase: string
  defaultKnowledgeBase: string
  modelConfig: ModelConfig
  messages: ChatMessage[]
  streaming: boolean
  stage: string
  stageAgent: string
  activeSources: SourceInfo[]
  pendingAttachments: PendingAttachment[]
  controller?: AbortController
  setMode: (mode: ChatMode) => void
  setKnowledgeBase: (id: string) => void
  setDefaultKnowledgeBase: (id: string) => void
  syncKnowledgeBases: (knowledgeBases: KBStatus[]) => void
  setModelConfig: (config: ModelConfig) => void
  addAttachments: (files: File[]) => Promise<void>
  removeAttachment: (localId: string) => void
  loadSession: (sessionId: string, messages: StoredMessage[]) => void
  send: (message: string) => Promise<void>
  stop: () => void
  clear: () => void
}

export const useChatStore = create<ChatState>((set, get) => ({
  studentId: getStudentId(),
  sessionId: getSessionId(),
  mode: 'auto',
  knowledgeBase: initialKnowledgeBase,
  defaultKnowledgeBase: initialKnowledgeBase,
  modelConfig: getModelConfig(),
  messages: [],
  streaming: false,
  stage: '',
  stageAgent: '',
  activeSources: [],
  pendingAttachments: [],
  setMode: (mode) => set({ mode }),
  setKnowledgeBase: (knowledgeBase) => set({ knowledgeBase }),
  setDefaultKnowledgeBase: (defaultKnowledgeBase) => {
    if (!defaultKnowledgeBase) {
      localStorage.removeItem(defaultKnowledgeBaseKey)
      set({ defaultKnowledgeBase: '', knowledgeBase: '' })
      return
    }
    if (!/^[A-Za-z0-9_-]{1,48}$/.test(defaultKnowledgeBase)) return
    localStorage.setItem(defaultKnowledgeBaseKey, defaultKnowledgeBase)
    set({ defaultKnowledgeBase, knowledgeBase: defaultKnowledgeBase })
  },
  syncKnowledgeBases: (knowledgeBases) => {
    const currentDefault = get().defaultKnowledgeBase
    const available = knowledgeBases.filter((item) => item.state === 'ready' || item.available)
    if (available.some((item) => item.id === currentDefault)) return

    const replacement = available[0]?.id || ''
    if (replacement) {
      localStorage.setItem(defaultKnowledgeBaseKey, replacement)
    } else {
      localStorage.removeItem(defaultKnowledgeBaseKey)
    }
    set((state) => ({
      defaultKnowledgeBase: replacement,
      knowledgeBase:
        !state.knowledgeBase || state.knowledgeBase === currentDefault
          ? replacement
          : state.knowledgeBase,
    }))
  },
  setModelConfig: (modelConfig) => {
    const normalized = { ...modelConfig, model: canonicalModel(modelConfig.provider, modelConfig.model) }
    localStorage.setItem(modelConfigKey, JSON.stringify(normalized))
    set({ modelConfig: normalized })
  },
  addAttachments: async (files) => {
    const available = Math.max(0, 5 - get().pendingAttachments.length)
    const selected = files.slice(0, available)
    const pending = selected.map<PendingAttachment>((file) => ({
      localId: crypto.randomUUID(),
      name: file.name,
      size: file.size,
      contentType: file.type,
      kind: file.type.startsWith('image/') ? 'image' : 'document',
      status: 'uploading',
    }))
    set((state) => ({ pendingAttachments: [...state.pendingAttachments, ...pending] }))
    await Promise.all(
      selected.map(async (file, index) => {
        const localId = pending[index].localId
        try {
          const attachment = await uploadChatAttachment(file, get().sessionId)
          set((state) => ({
            pendingAttachments: state.pendingAttachments.map((item) =>
              item.localId === localId ? { ...item, status: 'ready', attachment } : item,
            ),
          }))
        } catch (error) {
          const detail = error instanceof Error ? error.message : '上传失败'
          set((state) => ({
            pendingAttachments: state.pendingAttachments.map((item) =>
              item.localId === localId ? { ...item, status: 'error', error: detail } : item,
            ),
          }))
        }
      }),
    )
  },
  removeAttachment: (localId) => set((state) => ({
    pendingAttachments: state.pendingAttachments.filter((item) => item.localId !== localId),
  })),
  loadSession: (sessionId, storedMessages) => {
    get().controller?.abort()
    localStorage.setItem(sessionKey, sessionId)
    const messages = storedMessages.map<ChatMessage>((item, index) => ({
      id: `history-${item.created_at}-${index}`,
      role: item.role,
      content: item.content,
      agent: item.agent,
      provider: item.provider,
      model: item.model,
    }))
    set({
      sessionId,
      messages,
      streaming: false,
      stage: '',
      stageAgent: '',
      activeSources: [],
      pendingAttachments: [],
      controller: undefined,
    })
  },
  send: async (rawMessage) => {
    const readyAttachments = get().pendingAttachments
      .filter((item) => item.status === 'ready' && item.attachment)
      .map((item) => item.attachment!)
    const hasUnfinished = get().pendingAttachments.some((item) => item.status !== 'ready')
    const message = rawMessage.trim() || (
      readyAttachments.length
        ? get().mode === 'quiz'
          ? '请根据附件中的原题生成一道同类型新题。'
          : '请识别并解答附件中的电路题。'
        : ''
    )
    if ((!message && !readyAttachments.length) || get().streaming || hasUnfinished) return
    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: message,
      attachments: readyAttachments,
    }
    const assistantId = crypto.randomUUID()
    const selectedModel = get().modelConfig
    const assistantMessage: ChatMessage = {
      id: assistantId,
      role: 'assistant',
      content: '',
      model: selectedModel.model,
      provider: selectedModel.provider,
    }
    const controller = new AbortController()
    set((state) => ({
      messages: [...state.messages, userMessage, assistantMessage],
      streaming: true,
      stage: `正在连接 ${selectedModel.model}…`,
      stageAgent: '系统',
      activeSources: [],
      pendingAttachments: [],
      controller,
    }))
    try {
      await streamChat(
        {
          session_id: get().sessionId,
          message,
          mode: get().mode,
          knowledge_base: get().knowledgeBase,
          attachment_ids: readyAttachments.map((item) => item.id),
          model_provider: selectedModel.provider,
          model: selectedModel.model,
          api_key: selectedModel.apiKey,
          base_url: selectedModel.baseUrl,
        },
        {
          onStatus: (data) => set({ stage: data.message, stageAgent: data.agent }),
          onMeta: (data) => {
            set((state) => ({
              activeSources: data.sources || [],
              messages: state.messages.map((item) =>
                item.id === assistantId
                  ? { ...item, agent: data.agent, provider: data.provider, model: data.model, sources: data.sources }
                  : item,
              ),
            }))
          },
          onDelta: (content) => {
            set((state) => ({
              messages: state.messages.map((item) =>
                item.id === assistantId ? { ...item, content: item.content + content } : item,
              ),
            }))
          },
          onDone: () => set({ streaming: false, stage: '', stageAgent: '', controller: undefined }),
          onError: (error) => {
            set((state) => ({
              streaming: false,
              stage: '',
              messages: state.messages.map((item) =>
                item.id === assistantId
                  ? {
                      ...item,
                      content: item.content
                        ? `${item.content}\n\n> ⚠️ 生成未完整结束：${error}`
                        : `生成失败：${error}`,
                      failed: true,
                    }
                  : item,
              ),
            }))
          },
        },
        controller.signal,
      )
      set({ streaming: false, stage: '', stageAgent: '', controller: undefined })
    } catch (error) {
      if ((error as Error).name === 'AbortError') {
        set({ streaming: false, stage: '已停止生成', stageAgent: '', controller: undefined })
        return
      }
      const detail = error instanceof Error ? error.message : '未知错误'
      set((state) => ({
        streaming: false,
        stage: '',
        controller: undefined,
        messages: state.messages.map((item) =>
          item.id === assistantId
            ? {
                ...item,
                content: item.content
                  ? `${item.content}\n\n> ⚠️ 回答连接提前结束：${detail}`
                  : `连接失败：${detail}`,
                failed: true,
              }
            : item,
        ),
      }))
    }
  },
  stop: () => {
    get().controller?.abort()
  },
  clear: () => {
    const sessionId = `student-${crypto.randomUUID()}`
    localStorage.setItem(sessionKey, sessionId)
    get().controller?.abort()
    set({
      sessionId,
      knowledgeBase: get().defaultKnowledgeBase,
      messages: [],
      streaming: false,
      stage: '',
      activeSources: [],
      pendingAttachments: [],
      controller: undefined,
    })
  },
}))
