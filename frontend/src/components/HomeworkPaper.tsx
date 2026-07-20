import MathMarkdown from './MathMarkdown'
import type { Homework, HomeworkAsset, HomeworkQuestion, HomeworkQuestionPart } from '../lib/api'
import type { ReactNode } from 'react'

type PaperMode = 'questions' | 'answers'

const numberedPartPattern = /(?:\(|（)\s*(\d{1,2})\s*(?:\)|）)/g

function splitNumberedParts(content: string) {
  const matches: Array<{ label: string; start: number; end: number }> = []
  let expected = 1
  for (const match of content.matchAll(numberedPartPattern)) {
    const number = Number(match[1])
    if (number !== expected || match.index === undefined) continue
    matches.push({ label: match[1], start: match.index, end: match.index + match[0].length })
    expected += 1
  }
  if (!matches.length) return { stem: content, parts: [] as HomeworkQuestionPart[] }
  return {
    stem: content.slice(0, matches[0].start).trim(),
    parts: matches.map((match, index) => ({
      label: match.label,
      text: content.slice(match.end, matches[index + 1]?.start ?? content.length).trim(),
    })).filter((part) => part.text),
  }
}

function structuredContent(content: string, parts?: HomeworkQuestionPart[]) {
  if (!parts?.length) return splitNumberedParts(content)
  const parsed = splitNumberedParts(content)
  return { stem: parsed.parts.length ? parsed.stem : content, parts }
}

function NumberedParts({ parts, answer = false }: { parts: HomeworkQuestionPart[]; answer?: boolean }) {
  if (!parts.length) return null
  return (
    <div className={answer ? 'homework-paper-answer-subquestions' : 'homework-paper-subquestions'}>
      {parts.map((part, index) => (
        <div className="homework-paper-subquestion" key={`${part.label}-${index}`}>
          <strong>（{part.label}）</strong>
          <MathMarkdown content={part.text} />
        </div>
      ))}
    </div>
  )
}

function groupedSections(questions: HomeworkQuestion[]) {
  const sections: Array<{ key: string; title: string; questions: HomeworkQuestion[] }> = []
  questions.forEach((question) => {
    const key = question.section_key || 'questions'
    let section = sections.at(-1)
    if (!section || section.key !== key) {
      section = {
        key,
        title: question.section_title || `${key}、题目`,
        questions: [],
      }
      sections.push(section)
    } else if (section.title === `${key}、题目` && question.section_title) {
      section.title = question.section_title
    }
    section.questions.push(question)
  })
  return sections
}

function QuestionFigures({ figures, label }: { figures: HomeworkAsset[]; label: string }) {
  if (!figures.length) return null
  return (
    <div className={`homework-paper-figures count-${Math.min(figures.length, 3)}`}>
      {figures.map((figure, index) => (
        <figure key={figure.file}>
          <img src={figure.url} alt={figure.caption || `${label}题图 ${index + 1}`} />
          {figure.caption && <figcaption>{figure.caption}</figcaption>}
        </figure>
      ))}
    </div>
  )
}

function ReflowedQuestion({
  question,
  renderQuestionResponse,
}: {
  question: HomeworkQuestion
  renderQuestionResponse?: (question: HomeworkQuestion) => ReactNode
}) {
  const position = question.figure_position || 'after_question'
  const figures = question.figures || []
  const options = question.options || []
  const content = structuredContent(question.prompt, question.subquestions)
  return (
    <article className={`homework-paper-question ${question.number.length > 2 ? 'has-long-number' : ''}`}>
      <div className="homework-paper-number">{question.number}.</div>
      <div className="homework-paper-question-body">
        {position === 'before_question' && <QuestionFigures figures={figures} label={`第 ${question.number} 题`} />}
        {content.stem && <div className="homework-paper-stem"><MathMarkdown content={content.stem} /></div>}
        <NumberedParts parts={content.parts} />
        {position === 'after_question' && <QuestionFigures figures={figures} label={`第 ${question.number} 题`} />}
        {options.length > 0 && (
          <div className={`homework-paper-options columns-${Math.max(1, Math.min(4, question.option_columns || 1))}`}>
            {options.map((option) => (
              <div key={option.label}>
                <strong>{option.label}.</strong>
                <MathMarkdown content={option.text} />
              </div>
            ))}
          </div>
        )}
        {position === 'after_options' && <QuestionFigures figures={figures} label={`第 ${question.number} 题`} />}
        {renderQuestionResponse?.(question)}
        {question.points > 0 && <span className="homework-paper-points">（{question.points} 分）</span>}
      </div>
    </article>
  )
}

function ReflowedAnswer({ question }: { question: HomeworkQuestion }) {
  const content = structuredContent(question.answer || '', question.answer_subquestions)
  const answerFigures = question.answer_figures || []
  return (
    <article className={`homework-paper-answer ${question.number.length > 2 ? 'has-long-number' : ''} ${question.points > 0 ? '' : 'is-unscored'}`}>
      <div className="homework-paper-number">{question.number}.</div>
      <div>
        <div className="homework-paper-answer-content">
          <QuestionFigures figures={answerFigures} label={`第 ${question.number} 题参考答案`} />
          {content.stem
            ? <MathMarkdown content={content.stem} />
            : !content.parts.length && <MathMarkdown content="未识别到参考答案" />}
          <NumberedParts parts={content.parts} answer />
        </div>
        {question.rubric && (
          <div className="homework-paper-rubric">
            <span>评分标准</span>
            <MathMarkdown content={question.rubric} />
          </div>
        )}
      </div>
      {question.points > 0 && <strong className="homework-paper-answer-score">{question.points} 分</strong>}
    </article>
  )
}

export default function HomeworkPaper({
  homework,
  mode,
  printable = false,
  renderQuestionResponse,
}: {
  homework: Homework
  mode: PaperMode
  printable?: boolean
  renderQuestionResponse?: (question: HomeworkQuestion) => ReactNode
}) {
  const sections = groupedSections(homework.questions)
  const dueDate = homework.due_at ? new Date(homework.due_at) : null
  const deadline = dueDate && !Number.isNaN(dueDate.getTime()) ? dueDate.toLocaleString('zh-CN') : ''
  return (
    <div className={`homework-reflow-paper mode-${mode} ${printable ? 'homework-print-target' : ''}`}>
      <header className="homework-paper-header">
        <span>CIRCUITMIND · {mode === 'questions' ? 'ASSIGNMENT' : 'REFERENCE ANSWERS'}</span>
        <h1>{homework.title}</h1>
        <p>{mode === 'questions' ? homework.instructions || '请按题目要求作答' : '参考答案与评分标准'}</p>
        <div>
          <span>共 {homework.question_count} 题</span>
          {homework.max_score > 0 && <span>满分 {homework.max_score} 分</span>}
          {deadline && <span>截止 {deadline}</span>}
        </div>
      </header>
      <main className="homework-paper-content">
        {sections.map((section) => (
          <section className="homework-paper-section" key={`${section.key}-${section.questions[0]?.sequence ?? 0}`}>
            <h2>{section.title}</h2>
            {section.questions.map((question) => mode === 'questions'
              ? <ReflowedQuestion key={question.id} question={question} renderQuestionResponse={renderQuestionResponse} />
              : <ReflowedAnswer key={question.id} question={question} />)}
          </section>
        ))}
      </main>
    </div>
  )
}
