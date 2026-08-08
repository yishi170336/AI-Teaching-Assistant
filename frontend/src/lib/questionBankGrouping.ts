import type { HomeworkQuestion } from './api'

export type QuestionBankQuestionGroup = {
  key: string
  title: string
  questions: HomeworkQuestion[]
}

export type QuestionBankQuestionLayout = {
  grouped: boolean
  paper: boolean
  groups: QuestionBankQuestionGroup[]
}

type ChapterIdentity = Pick<QuestionBankQuestionGroup, 'key' | 'title'>

const CHINESE_DIGITS: Record<string, number> = {
  零: 0,
  〇: 0,
  一: 1,
  二: 2,
  两: 2,
  三: 3,
  四: 4,
  五: 5,
  六: 6,
  七: 7,
  八: 8,
  九: 9,
}

function chapterNumber(value: string): number | null {
  const normalized = value.trim()
  if (/^\d+$/.test(normalized)) return Number(normalized)
  if (!normalized || !/^[零〇一二两三四五六七八九十百]+$/.test(normalized)) return null

  let total = 0
  let digit = 0
  for (const character of normalized) {
    if (character === '十' || character === '百') {
      const unit = character === '十' ? 10 : 100
      total += (digit || 1) * unit
      digit = 0
    } else {
      digit = CHINESE_DIGITS[character]
    }
  }
  return total + digit
}

function chineseChapterNumber(value: number): string {
  const digits = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九']
  if (value <= 0 || value >= 100) return String(value)
  if (value < 10) return digits[value]
  const tens = Math.floor(value / 10)
  const ones = value % 10
  return `${tens === 1 ? '' : digits[tens]}十${ones ? digits[ones] : ''}`
}

function normalizedChapter(value: string): ChapterIdentity | null {
  const number = chapterNumber(value)
  if (number === null || !Number.isFinite(number)) return null
  return {
    key: `chapter-${number}`,
    title: `第${chineseChapterNumber(number)}章`,
  }
}

function explicitChapter(value: string): ChapterIdentity | null {
  const match = value.match(/第\s*([0-9零〇一二两三四五六七八九十百]+)\s*章/)
  return match ? normalizedChapter(match[1]) : null
}

function numberedChapter(value: string): ChapterIdentity | null {
  const normalized = value.trim().replace(/^(?:例题?|习题|题)\s*/u, '')
  const chapterKey = normalized.match(/^chapter[-_\s]*(\d+)/i)
  if (chapterKey) return normalizedChapter(chapterKey[1])
  const dottedNumber = normalized.match(/^(\d+)\s*[.．]\s*\d+/)
  return dottedNumber ? normalizedChapter(dottedNumber[1]) : null
}

function questionChapter(question: HomeworkQuestion): ChapterIdentity | null {
  const structuralValues = [question.section_title || '', question.section_key || '']
  for (const value of structuralValues) {
    const chapter = explicitChapter(value)
    if (chapter) return chapter
  }
  for (const value of [...structuralValues, String(question.number || '')]) {
    const chapter = numberedChapter(value)
    if (chapter) return chapter
  }
  return null
}

/**
 * Chapter-based books use explicit chapter headings or hierarchical numbers such
 * as 2.3 / 2.3.1. Exam papers normally use flat numbers and question-type
 * sections, so they intentionally retain the original flat list.
 */
export function groupQuestionBankQuestions(
  questions: HomeworkQuestion[],
  sourceOrigin = '',
  bankLabel = '',
  documentKind = '',
): QuestionBankQuestionLayout {
  const flatLayout: QuestionBankQuestionLayout = {
    grouped: false,
    paper: false,
    groups: [{ key: 'all-questions', title: '', questions }],
  }
  const paperSectionPattern = /(?:选择|填空|判断|简答|计算|论述|作图|综合|分析|设计).*题/u
  const hasFlatPaperSection = questions.some((question) => (
    paperSectionPattern.test(question.section_title || '')
    && !questionChapter({ ...question, number: '' })
  ))
  const hasPaperLabel = /(?:测试题|考试|测验|模拟卷|期中|期末|[a-zＡ-Ｚ]卷)/iu.test(bankLabel)
  const isPaper = documentKind === 'paper'
    || (sourceOrigin || '').trim().toLowerCase() === 'photo_answer'
    || hasFlatPaperSection
    || hasPaperLabel
  if (
    !questions.length
    || isPaper
  ) return { ...flatLayout, paper: isPaper }

  const chapters = questions.map(questionChapter)
  if (!chapters.some(Boolean)) return flatLayout

  const groups: QuestionBankQuestionGroup[] = []
  let activeChapter: ChapterIdentity | null = null
  questions.forEach((question, index) => {
    activeChapter = chapters[index] || activeChapter
    const chapter = activeChapter || { key: 'unassigned', title: '未定位章节' }
    let group = groups.find((item) => item.key === chapter.key)
    if (!group) {
      group = { ...chapter, questions: [] }
      groups.push(group)
    }
    group.questions.push(question)
  })

  return { grouped: true, paper: false, groups }
}
