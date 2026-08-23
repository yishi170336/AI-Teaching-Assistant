import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import { normalizeLatex } from '../lib/latex'

const markdownPlugins: Parameters<typeof ReactMarkdown>[0]['remarkPlugins'] = [
  [remarkGfm, { singleTilde: false }],
  remarkMath,
]
const katexPlugins: Parameters<typeof ReactMarkdown>[0]['rehypePlugins'] = [
  [rehypeKatex, { strict: false, throwOnError: false, trust: false }],
]

export type KnowledgeEntityLink = {
  id: string
  name: string
  aliases?: string[]
}

const entityLinkPrefix = '#knowledge-entity='
const protectedMarkdownPattern = /(```[\s\S]*?```|`[^`\n]*`|\$\$[\s\S]*?\$\$|\$[^$\n]*\$|!?\[[^\]]*\]\([^)]+\))/g

function escapeRegularExpression(value: string) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

/** Link only names supplied by the retriever, while leaving Markdown and LaTeX intact. */
export function linkKnowledgeEntities(content: string, entities: KnowledgeEntityLink[] = []) {
  const terms = new Map<string, { id: string; text: string }>()
  entities.forEach((entity) => {
    const values = [entity.name, ...(entity.aliases || [])]
    values.forEach((rawValue) => {
      const text = String(rawValue || '').trim()
      if (text.length < 2 || /^[$\\]/.test(text)) return
      const key = text.toLocaleLowerCase('zh-CN')
      if (!terms.has(key)) terms.set(key, { id: entity.id, text })
    })
  })
  const candidates = [...terms.values()].sort((left, right) => right.text.length - left.text.length)
  if (!candidates.length) return content
  const pattern = new RegExp(candidates.map((item) => escapeRegularExpression(item.text)).join('|'), 'giu')
  const linkedEntityIds = new Set<string>()
  return content.split(protectedMarkdownPattern).map((segment, index) => {
    if (index % 2 === 1) return segment
    return segment.replace(pattern, (match) => {
      const entity = terms.get(match.toLocaleLowerCase('zh-CN'))
      if (!entity || linkedEntityIds.has(entity.id)) return match
      linkedEntityIds.add(entity.id)
      return `[${match}](${entityLinkPrefix}${encodeURIComponent(entity.id)})`
    })
  }).join('')
}

export default function MathMarkdown({
  content,
  entityLinks = [],
  onEntityClick,
}: {
  content: string
  entityLinks?: KnowledgeEntityLink[]
  onEntityClick?: (entityId: string) => void
}) {
  const linkedContent = linkKnowledgeEntities(content, entityLinks)
  return (
    <div className="math-markdown">
      <ReactMarkdown
        remarkPlugins={markdownPlugins}
        rehypePlugins={katexPlugins}
        components={{
          a: ({ href, children }) => {
            if (href?.startsWith(entityLinkPrefix) && onEntityClick) {
              const entityId = decodeURIComponent(href.slice(entityLinkPrefix.length))
              return (
                <a
                  href={href}
                  className="knowledge-entity-link"
                  title="在知识图谱中查看该实体"
                  onClick={(event) => {
                    event.preventDefault()
                    event.stopPropagation()
                    onEntityClick(entityId)
                  }}
                >
                  {children}
                </a>
              )
            }
            return <a href={href}>{children}</a>
          },
        }}
      >
        {normalizeLatex(linkedContent)}
      </ReactMarkdown>
    </div>
  )
}

/** Render formula-bearing labels and short UI text without introducing a block wrapper. */
export function InlineMath({ content, className = '' }: { content: string; className?: string }) {
  return (
    <span className={`math-markdown math-markdown-inline ${className}`.trim()}>
      <ReactMarkdown
        remarkPlugins={markdownPlugins}
        rehypePlugins={katexPlugins}
        components={{ p: ({ children }) => <>{children}</> }}
      >
        {normalizeLatex(content)}
      </ReactMarkdown>
    </span>
  )
}
