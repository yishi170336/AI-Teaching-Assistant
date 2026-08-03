import { create } from 'zustand'
import { AttachmentInfo, ConversationFocus, KBStatus, ModelConfig, ModelProviderId, PhotoRecognition, PracticeExercise, PracticeGrading, QuestionRecommendation, QuestionReference, QuestionSummary, SourceInfo, StoredMessage, streamChat, uploadChatAttachment, VisionModelConfig } from '../lib/api'

export type ChatMode = 'auto' | 'answer' | 'quiz' | 'plan' | 'recommend' | 'explain'
export type ChatScene = 'chat' | 'image_answer' | 'quiz_grade'

export type ChatMessage = {
  id: string
  role: 'user' | 'assistant'
  content: string
  agent?: string
  sources?: SourceInfo[]
  citedSources?: SourceInfo[]
  failed?: boolean
  attachments?: AttachmentInfo[]
  model?: string
  provider?: ModelProviderId
  knowledgeBase?: string
  recognition?: PhotoRecognition
  needsConfirmation?: boolean
  evidenceMode?: 'grounded' | 'mixed' | 'general_only'
  practice?: PracticeExercise
  grading?: PracticeGrading
  questionRef?: QuestionReference
  questionSummary?: QuestionSummary
  recommendation?: QuestionRecommendation
  focus?: ConversationFocus
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
const visionModelConfigKey = 'circuitmind-vision-model-config'
const defaultKnowledgeBaseKey = 'circuitmind-default-knowledge-base'
export const CHAT_MODEL_PROVIDER: ModelProviderId = 'ollama'
export const CHAT_MODEL = 'qwen3.5:2b'
const QWEN_VL_FALLBACK_MODEL = 'qwen3-vl-flash'

const defaultModelConfig: ModelConfig = {
  provider: CHAT_MODEL_PROVIDER,
  model: CHAT_MODEL,
  apiKey: '',
  baseUrl: 'http://127.0.0.1:11434',
}

const defaultVisionModelConfig: VisionModelConfig = {
  model: QWEN_VL_FALLBACK_MODEL,
  apiKey: '',
  baseUrl: 'https://dashscope.aliyuncs.com/compatible-mode/v1',
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
    return QWEN_VL_FALLBACK_MODEL
  }
  return canonical
}

function normalizedModelConfig(value: Partial<ModelConfig>): ModelConfig {
  const providers: ModelProviderId[] = ['ollama', 'deepseek', 'qwen', 'custom']
  if (!value.provider || !providers.includes(value.provider) || typeof value.model !== 'string') {
    return defaultModelConfig
  }
  return {
    provider: value.provider,
    model: canonicalModel(value.provider, value.model || defaultModelConfig.model),
    apiKey: typeof value.apiKey === 'string' ? value.apiKey : '',
    baseUrl: typeof value.baseUrl === 'string' ? value.baseUrl.trim() : '',
  }
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
    const config = normalizedModelConfig(stored)
    localStorage.setItem(modelConfigKey, JSON.stringify(config))
    return config
  } catch {
    return defaultModelConfig
  }
}

function getVisionModelConfig(): VisionModelConfig {
  try {
    const stored = JSON.parse(localStorage.getItem(visionModelConfigKey) || '{}')
    const config = {
      model: typeof stored.model === 'string' && stored.model.trim()
        ? canonicalModel('qwen', stored.model)
        : defaultVisionModelConfig.model,
      apiKey: typeof stored.apiKey === 'string' ? stored.apiKey : '',
      baseUrl: typeof stored.baseUrl === 'string' && stored.baseUrl.trim()
        ? stored.baseUrl.trim()
        : defaultVisionModelConfig.baseUrl,
    }
    localStorage.setItem(visionModelConfigKey, JSON.stringify(config))
    return config
  } catch {
    return defaultVisionModelConfig
  }
}

function getDefaultKnowledgeBase(): string {
  const stored = localStorage.getItem(defaultKnowledgeBaseKey)?.trim() || ''
  return /^[A-Za-z0-9_-]{1,48}$/.test(stored) ? stored : ''
}

const initialKnowledgeBase = getDefaultKnowledgeBase()

function legacySourcesFromContent(content: string): SourceInfo[] {
  const sources: SourceInfo[] = []
  const pattern = /^-\s+\[资料(\d+)\]\s+(.+?)\s+·\s+(.+?)\s+·\s+第\s*(\d+)(?:[–-](\d+))?\s*页\s*$/gm
  for (const match of content.matchAll(pattern)) {
    const pageStart = Number(match[4])
    const pageEnd = Number(match[5] || match[4])
    sources.push({
      id: `history-source-${match[1]}-${match[2]}-${pageStart}`,
      source: match[2].trim(),
      chapter: match[3].trim(),
      section: match[3].trim(),
      page_start: pageStart,
      page_end: pageEnd,
      score: 0,
      doc_type: 'textbook',
      historical: true,
      citation_index: Number(match[1]),
    })
  }
  return sources
}

function citedSourcesFromContent(content: string, sources: SourceInfo[]): SourceInfo[] {
  const citedIndices = new Set<number>()
  for (const match of content.matchAll(/\[资料\s*(\d+)\]/g)) {
    citedIndices.add(Number(match[1]))
  }
  return sources.filter((source, index) =>
    citedIndices.has(source.citation_index || index + 1),
  )
}

type ChatState = {
  studentId: string
  sessionId: string
  mode: ChatMode
  scene: ChatScene
  knowledgeBase: string
  defaultKnowledgeBase: string
  modelConfig: ModelConfig
  visionModelConfig: VisionModelConfig
  messages: ChatMessage[]
  streaming: boolean
  stage: string
  stageAgent: string
  activeSources: SourceInfo[]
  activeCitedSources: SourceInfo[]
  activeMessageId?: string
  pendingAttachments: PendingAttachment[]
  activePractice?: PracticeExercise
  activeQuestionRef?: QuestionReference
  activeFocus?: ConversationFocus
  controller?: AbortController
  setMode: (mode: ChatMode) => void
  setScene: (scene: ChatScene) => void
  setActivePractice: (practice?: PracticeExercise) => void
  setActiveQuestionRef: (reference?: QuestionReference) => void
  setActiveFocus: (focus?: ConversationFocus) => void
  setKnowledgeBase: (id: string) => void
  setDefaultKnowledgeBase: (id: string) => void
  syncKnowledgeBases: (knowledgeBases: KBStatus[]) => void
  setModelConfig: (config: ModelConfig) => void
  setVisionModelConfig: (config: VisionModelConfig) => void
  addAttachments: (files: File[]) => Promise<void>
  removeAttachment: (localId: string) => void
  activateMessage: (messageId: string, bindFocus?: boolean) => void
  loadSession: (sessionId: string, messages: StoredMessage[]) => void
  send: (message: string, options?: { recognitionConfirmed?: boolean; questionRef?: QuestionReference; focusId?: string }) => Promise<void>
  stop: () => void
  clear: () => void
}

export const useChatStore = create<ChatState>((set, get) => ({
  studentId: getStudentId(),
  sessionId: getSessionId(),
  mode: 'auto',
  scene: 'chat',
  knowledgeBase: initialKnowledgeBase,
  defaultKnowledgeBase: initialKnowledgeBase,
  modelConfig: getModelConfig(),
  visionModelConfig: getVisionModelConfig(),
  messages: [],
  streaming: false,
  stage: '',
  stageAgent: '',
  activeSources: [],
  activeCitedSources: [],
  activeMessageId: undefined,
  pendingAttachments: [],
  activePractice: undefined,
  activeQuestionRef: undefined,
  activeFocus: undefined,
  setMode: (mode) => set({ mode }),
  setScene: (scene) => set({
    scene,
    ...(scene === 'quiz_grade' ? {} : { activePractice: undefined }),
  }),
  setActivePractice: (activePractice) => set({ activePractice }),
  setActiveQuestionRef: (activeQuestionRef) => set({ activeQuestionRef }),
  setActiveFocus: (activeFocus) => set({
    activeFocus,
    activeQuestionRef: activeFocus?.question_ref,
  }),
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
    const normalized = normalizedModelConfig(modelConfig)
    localStorage.setItem(modelConfigKey, JSON.stringify(normalized))
    set({ modelConfig: normalized })
  },
  setVisionModelConfig: (visionModelConfig) => {
    const normalized = {
      model: canonicalModel('qwen', visionModelConfig.model || defaultVisionModelConfig.model),
      apiKey: visionModelConfig.apiKey,
      baseUrl: visionModelConfig.baseUrl.trim() || defaultVisionModelConfig.baseUrl,
    }
    localStorage.setItem(visionModelConfigKey, JSON.stringify(normalized))
    set({ visionModelConfig: normalized })
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
    const startsNewPhotoQuestion = get().scene === 'image_answer' && pending.some((item) => item.kind === 'image')
    set((state) => ({
      pendingAttachments: [...state.pendingAttachments, ...pending],
      ...(startsNewPhotoQuestion ? { activeFocus: undefined, activeQuestionRef: undefined } : {}),
    }))
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
  activateMessage: (messageId, bindFocus = false) => set((state) => {
    const message = state.messages.find((item) => item.id === messageId)
    if (!message || message.role !== 'assistant') return state
    return {
      activeMessageId: messageId,
      activeSources: message.sources || [],
      activeCitedSources: message.citedSources || [],
      ...(bindFocus && message.focus
        ? {
            activeFocus: message.focus,
            activeQuestionRef: message.focus.question_ref,
          }
        : {}),
    }
  }),
  loadSession: (sessionId, storedMessages) => {
    get().controller?.abort()
    localStorage.setItem(sessionKey, sessionId)
    const messages = storedMessages.map<ChatMessage>((item, index) => {
      const sources = item.sources?.length
        ? item.sources
        : legacySourcesFromContent(item.content)
      const citedSources = item.cited_sources?.length
        ? item.cited_sources
        : citedSourcesFromContent(item.content, sources)
      const safeCreatedAt = item.created_at.replace(/[^A-Za-z0-9_.:-]/g, '-')
      return {
        id: `history-${safeCreatedAt}-${index}`,
        role: item.role,
        content: item.content,
        agent: item.agent,
        provider: item.provider,
        model: item.model,
        knowledgeBase: item.knowledge_base,
        attachments: item.attachments || [],
        sources,
        citedSources,
        recognition: item.recognition,
        needsConfirmation: item.needs_confirmation,
        evidenceMode: item.evidence_mode,
        practice: item.practice,
        grading: item.grading,
        questionRef: item.question_ref,
        questionSummary: item.question_summary,
        recommendation: item.recommendation,
        focus: item.conversation_focus,
      }
    })
    const latestAssistant = [...messages].reverse().find((item) => item.role === 'assistant')
    const latestQuestion = [...messages].reverse().find((item) => item.questionRef)?.questionRef
    const latestFocus = [...messages].reverse().find((item) => item.focus)?.focus
    set({
      sessionId,
      messages,
      streaming: false,
      stage: '',
      stageAgent: '',
      activeSources: latestAssistant?.sources || [],
      activeCitedSources: latestAssistant?.citedSources || [],
      activeMessageId: latestAssistant?.id,
      pendingAttachments: [],
      activePractice: undefined,
      activeQuestionRef: latestFocus ? latestFocus.question_ref : latestQuestion,
      activeFocus: latestFocus,
      controller: undefined,
    })
  },
  send: async (rawMessage, options) => {
    const startsNewPhotoQuestion = get().scene === 'image_answer'
      && get().pendingAttachments.some((item) => item.status === 'ready' && item.kind === 'image')
    const questionRef = startsNewPhotoQuestion ? undefined : options?.questionRef || get().activeQuestionRef
    const focusId = startsNewPhotoQuestion ? undefined : options?.focusId || get().activeFocus?.id
    const readyAttachments = get().pendingAttachments
      .filter((item) => (
        item.status === 'ready'
        && item.attachment
        && (!['image_answer', 'quiz_grade'].includes(get().scene) || item.kind === 'image')
      ))
      .map((item) => item.attachment!)
    const hasUnfinished = get().pendingAttachments.some((item) => item.status !== 'ready')
    const message = rawMessage.trim() || (
      readyAttachments.length
        ? get().scene === 'quiz_grade'
          ? '请批改我上传的作答，并指出具体错误和改进方法。'
          : get().mode === 'quiz'
          ? '请根据附件中的原题生成一道同类型新题。'
          : '请识别并解答附件中的电路题。'
        : questionRef ? '请解答选中的题库题目。' : ''
    )
    if ((!message && !readyAttachments.length && !questionRef) || get().streaming || hasUnfinished) return
    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: message,
      attachments: readyAttachments,
      knowledgeBase: get().knowledgeBase,
      questionRef,
      focus: get().activeFocus,
    }
    const assistantId = crypto.randomUUID()
    const requestScene = get().scene
    const selectedModel = get().modelConfig
    const selectedVisionModel = get().visionModelConfig
    const assistantMessage: ChatMessage = {
      id: assistantId,
      role: 'assistant',
      content: '',
      model: selectedModel.model,
      provider: selectedModel.provider,
      knowledgeBase: get().knowledgeBase,
    }
    const controller = new AbortController()
    set((state) => ({
      messages: [...state.messages, userMessage, assistantMessage],
      streaming: true,
      stage: `正在连接 ${selectedModel.model}…`,
      stageAgent: '系统',
      activeSources: [],
      activeCitedSources: [],
      activeMessageId: assistantId,
      pendingAttachments: [],
      activeQuestionRef: questionRef,
      controller,
    }))
    try {
      await streamChat(
        {
          session_id: get().sessionId,
          message,
          mode: get().mode,
          scene: get().scene,
          recognition_confirmed: Boolean(options?.recognitionConfirmed),
          student_id: get().studentId,
          question_ref: questionRef,
          focus_id: focusId,
          knowledge_base: get().knowledgeBase,
          attachment_ids: readyAttachments.map((item) => item.id),
          model_provider: selectedModel.provider,
          model: selectedModel.model,
          api_key: selectedModel.apiKey,
          base_url: selectedModel.baseUrl,
          vision_model: selectedVisionModel.model,
          vision_api_key: selectedVisionModel.apiKey,
          vision_base_url: selectedVisionModel.baseUrl,
        },
        {
          onStatus: (data) => set({ stage: data.message, stageAgent: data.agent }),
          onMeta: (data) => {
            set((state) => {
              const sources = data.sources || []
              const assistant = state.messages.find((item) => item.id === assistantId)
              const citedSources = data.cited_sources?.length
                ? data.cited_sources
                : citedSourcesFromContent(assistant?.content || '', sources)
              return {
                activeSources:
                  state.activeMessageId === assistantId
                    ? sources
                    : state.activeSources,
                activeCitedSources:
                  state.activeMessageId === assistantId
                    ? citedSources
                    : state.activeCitedSources,
                messages: state.messages.map((item) =>
                  item.id === assistantId
                    ? {
                        ...item,
                        agent: data.agent,
                        provider: data.provider,
                        model: data.model,
                        sources,
                        citedSources,
                        recognition: data.recognition,
                        needsConfirmation: data.needs_confirmation,
                        evidenceMode: data.evidence_mode,
                        practice: data.practice,
                        grading: data.grading,
                        questionRef: data.question_ref,
                        questionSummary: data.question_summary,
                        recommendation: data.recommendation,
                        focus: data.conversation_focus,
                      }
                    : item,
                ),
                activeFocus: data.conversation_focus || state.activeFocus,
                activeQuestionRef: data.conversation_focus
                  ? data.conversation_focus.question_ref
                  : data.question_ref || state.activeQuestionRef,
              }
            })
          },
          onDelta: (content) => {
            set((state) => ({
              messages: state.messages.map((item) =>
                item.id === assistantId ? { ...item, content: item.content + content } : item,
              ),
            }))
          },
          onDone: () => set({
            streaming: false,
            stage: '',
            stageAgent: '',
            controller: undefined,
            ...(
              requestScene === 'quiz_grade'
                ? { scene: 'chat' as ChatScene, activePractice: undefined }
                : requestScene === 'image_answer'
                  ? { scene: 'chat' as ChatScene }
                  : {}
            ),
          }),
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
      activeCitedSources: [],
      activeMessageId: undefined,
      pendingAttachments: [],
      activePractice: undefined,
      activeQuestionRef: undefined,
      activeFocus: undefined,
      controller: undefined,
    })
  },
}))
