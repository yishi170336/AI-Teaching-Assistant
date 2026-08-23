import { create } from 'zustand'
import { AttachmentInfo, ContextStateSummary, ConversationFocus, KBStatus, MistakeCandidateDraft, ModelConfig, ModelProviderId, OCRModelConfig, PhotoRecognition, PracticeExercise, PracticeGrading, QuestionRecommendation, QuestionReference, QuestionSummary, ResolvedContext, SourceInfo, StoredMessage, streamChat, uploadChatAttachment } from '../lib/api'

export type ChatMode = 'auto' | 'answer' | 'quiz' | 'plan' | 'recommend' | 'explain'
export type ChatScene = 'chat' | 'image_answer' | 'quiz_grade'

export type ChatMessage = {
  id: string
  role: 'user' | 'assistant'
  content: string
  status?: 'running' | 'completed' | 'cancelled' | 'failed' | 'error'
  agent?: string
  sources?: SourceInfo[]
  citedSources?: SourceInfo[]
  failed?: boolean
  attachments?: AttachmentInfo[]
  model?: string
  provider?: ModelProviderId
  knowledgeBase?: string
  practiceSessionId?: string
  recognition?: PhotoRecognition
  needsConfirmation?: boolean
  evidenceMode?: 'grounded' | 'mixed' | 'general_only'
  practice?: PracticeExercise
  grading?: PracticeGrading
  questionRef?: QuestionReference
  questionSummary?: QuestionSummary
  recommendation?: QuestionRecommendation
  focus?: ConversationFocus
  resolvedContext?: ResolvedContext
  contextState?: ContextStateSummary
  mistakeProposal?: MistakeCandidateDraft
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
const ocrModelConfigKey = 'circuitmind-ocr-model-config'
const defaultKnowledgeBaseKey = 'circuitmind-default-knowledge-base'
export const CHAT_MODEL_PROVIDER: ModelProviderId = 'qwen'
export const CHAT_MODEL = 'qwen3.7-flash'
const QWEN_TEXT_MODELS = ['qwen3.7-flash', 'qwen3.7-plus', 'qwen3.7-max'] as const

const defaultModelConfig: ModelConfig = {
  provider: CHAT_MODEL_PROVIDER,
  model: CHAT_MODEL,
  apiKey: '',
  baseUrl: 'https://dashscope.aliyuncs.com/compatible-mode/v1',
}

const defaultOCRModelConfig: OCRModelConfig = {
  provider: 'local',
  model: 'PaddleOCR-VL-1.6',
  apiToken: '',
  jobUrl: 'https://paddleocr.aistudio-app.com/api/v2/ocr/jobs',
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
  return normalized.toLowerCase()
}

function normalizedQwenTextModel(model: string) {
  const canonical = canonicalModel('qwen', model)
  return QWEN_TEXT_MODELS.some((item) => item === canonical)
    ? canonical
    : CHAT_MODEL
}

function normalizedModelConfig(value: Partial<ModelConfig>): ModelConfig {
  const providers: ModelProviderId[] = ['ollama', 'deepseek', 'qwen', 'custom']
  if (!value.provider || !providers.includes(value.provider) || typeof value.model !== 'string') {
    return defaultModelConfig
  }
  return {
    provider: value.provider,
    model: value.provider === 'qwen'
      ? normalizedQwenTextModel(value.model || defaultModelConfig.model)
      : canonicalModel(value.provider, value.model || defaultModelConfig.model),
    // The built-in Qwen provider always uses the server-owned credential.
    // Clear historical browser keys so a stale/restricted key can never
    // override the validated backend configuration.
    apiKey: value.provider === 'qwen'
      ? ''
      : typeof value.apiKey === 'string' ? value.apiKey : '',
    baseUrl: value.provider === 'qwen'
      ? defaultModelConfig.baseUrl
      : typeof value.baseUrl === 'string' ? value.baseUrl.trim() : '',
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
    const previous = normalizedModelConfig(stored)
    const config = {
      ...defaultModelConfig,
      apiKey: '',
      baseUrl: defaultModelConfig.baseUrl,
    }
    localStorage.setItem(modelConfigKey, JSON.stringify(config))
    return config
  } catch {
    return defaultModelConfig
  }
}

function normalizedOCRModelConfig(value: Partial<OCRModelConfig>): OCRModelConfig {
  return {
    provider: value.provider === 'api' ? 'api' : 'local',
    model: 'PaddleOCR-VL-1.6',
    apiToken: typeof value.apiToken === 'string' ? value.apiToken : '',
    jobUrl: typeof value.jobUrl === 'string' && value.jobUrl.trim()
      ? value.jobUrl.trim()
      : defaultOCRModelConfig.jobUrl,
  }
}

function getOCRModelConfig(): OCRModelConfig {
  try {
    const stored = JSON.parse(localStorage.getItem(ocrModelConfigKey) || '{}')
    const config = normalizedOCRModelConfig(stored)
    localStorage.setItem(ocrModelConfigKey, JSON.stringify(config))
    return config
  } catch {
    return defaultOCRModelConfig
  }
}

function getDefaultKnowledgeBase(): string {
  const stored = localStorage.getItem(defaultKnowledgeBaseKey)?.trim() || ''
  return /^[A-Za-z0-9_-]{1,48}$/.test(stored) ? stored : ''
}

const initialKnowledgeBase = getDefaultKnowledgeBase()

const emptyContextState = (): ContextStateSummary => ({
  schema_version: 1,
  revision: 0,
  active_subject_focus_id: '',
  active_task_id: '',
  continuation_task_id: '',
  focus_stack: [],
  tasks: {},
})

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
  ocrModelConfig: OCRModelConfig
  messages: ChatMessage[]
  streaming: boolean
  stage: string
  stageAgent: string
  activeSources: SourceInfo[]
  activeCitedSources: SourceInfo[]
  activeMessageId?: string
  pendingAttachments: PendingAttachment[]
  activePractice?: PracticeExercise
  activePracticeSessionId?: string
  activeQuestionRef?: QuestionReference
  activeFocus?: ConversationFocus
  contextState: ContextStateSummary
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
  setOCRModelConfig: (config: OCRModelConfig) => void
  addAttachments: (files: File[]) => Promise<void>
  removeAttachment: (localId: string) => void
  activateMessage: (messageId: string, bindFocus?: boolean) => void
  loadSession: (sessionId: string, messages: StoredMessage[]) => void
  send: (message: string, options?: { recognitionConfirmed?: boolean; questionRef?: QuestionReference; focusId?: string; continuationTaskId?: string; attachmentRole?: 'auto' | 'question' | 'answer' | 'reference' }) => Promise<void>
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
  ocrModelConfig: getOCRModelConfig(),
  messages: [],
  streaming: false,
  stage: '',
  stageAgent: '',
  activeSources: [],
  activeCitedSources: [],
  activeMessageId: undefined,
  pendingAttachments: [],
  activePractice: undefined,
  activePracticeSessionId: undefined,
  activeQuestionRef: undefined,
  activeFocus: undefined,
  contextState: emptyContextState(),
  setMode: (mode) => set({ mode }),
  setScene: (scene) => set({
    scene,
    ...(scene === 'quiz_grade' ? {} : { activePractice: undefined }),
  }),
  setActivePractice: (activePractice) => set((state) => ({
    activePractice,
    ...(activePractice && !state.activePracticeSessionId
      ? { activePracticeSessionId: `practice-${crypto.randomUUID()}` }
      : {}),
  })),
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
    const available = knowledgeBases.filter((item) => (
      item.runtime_supported === true && (item.state === 'ready' || item.available)
    ))
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
  setOCRModelConfig: (ocrModelConfig) => {
    const normalized = normalizedOCRModelConfig(ocrModelConfig)
    localStorage.setItem(ocrModelConfigKey, JSON.stringify(normalized))
    set({ ocrModelConfig: normalized })
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
        status: item.status,
        agent: item.agent,
        failed: item.status === 'failed' || item.status === 'error',
        provider: item.provider,
        model: item.model,
        knowledgeBase: item.knowledge_base,
        practiceSessionId: item.practice_session_id,
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
        resolvedContext: item.resolved_context,
        contextState: item.context_state,
        mistakeProposal: item.mistake_proposal,
      }
    })
    const latestAssistant = [...messages].reverse().find((item) => item.role === 'assistant')
    const latestQuestion = [...messages].reverse().find((item) => item.questionRef)?.questionRef
    const latestFocus = [...messages].reverse().find((item) => item.focus)?.focus
    const latestPracticeSessionId = [...messages].reverse().find((item) => item.practiceSessionId)?.practiceSessionId
    const latestContextState = [...messages].reverse().find((item) => item.contextState)?.contextState
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
      activePracticeSessionId: latestPracticeSessionId,
      activeQuestionRef: latestFocus ? latestFocus.question_ref : latestQuestion,
      activeFocus: latestFocus,
      contextState: latestContextState || emptyContextState(),
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
    const requestContextState = get().contextState
    const selectedModel = get().modelConfig
    const assistantMessage: ChatMessage = {
      id: assistantId,
      role: 'assistant',
      content: '',
      status: 'running',
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
          target_focus_id: focusId,
          submission_target_focus_id: requestScene === 'quiz_grade' ? focusId : undefined,
          continuation_task_id: options?.continuationTaskId,
          attachment_role: options?.attachmentRole || (
            requestScene === 'quiz_grade' ? 'answer'
              : requestScene === 'image_answer' ? 'question'
                : 'auto'
          ),
          expected_context_revision: requestContextState.revision,
          practice_session_id: requestScene === 'quiz_grade'
            ? get().activePracticeSessionId
            : undefined,
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
                        focus: data.conversation_focus || undefined,
                        resolvedContext: data.resolved_context,
                        contextState: data.context_state,
                        mistakeProposal: data.mistake_proposal,
                      }
                    : item,
                ),
                activeFocus: data.conversation_focus || state.activeFocus,
                activeQuestionRef: data.conversation_focus
                  ? data.conversation_focus.question_ref
                  : data.question_ref || state.activeQuestionRef,
                contextState: data.context_state || state.contextState,
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
          onDone: () => set((state) => ({
            streaming: false,
            stage: '',
            stageAgent: '',
            controller: undefined,
            messages: state.messages.map((item) =>
              item.id === assistantId ? { ...item, status: 'completed' } : item,
            ),
            ...(
              requestScene === 'quiz_grade'
                ? { scene: 'chat' as ChatScene, activePractice: undefined }
                : requestScene === 'image_answer'
                  ? { scene: 'chat' as ChatScene }
                  : {}
            ),
          })),
          onError: (error) => {
            set((state) => ({
              streaming: false,
              stage: '',
              messages: state.messages.map((item) =>
                item.id === assistantId
                  ? {
                      ...item,
                      status: 'failed',
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
        set((state) => ({
          streaming: false,
          stage: '',
          stageAgent: '',
          controller: undefined,
          messages: state.messages.map((item) =>
            item.id === assistantId ? { ...item, status: 'cancelled' } : item,
          ),
        }))
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
                status: 'failed',
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
      activePracticeSessionId: undefined,
      activeQuestionRef: undefined,
      activeFocus: undefined,
      contextState: emptyContextState(),
      controller: undefined,
    })
  },
}))
