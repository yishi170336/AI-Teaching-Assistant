import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import { normalizeLatex } from '../lib/latex'

const markdownPlugins = [remarkGfm, remarkMath]
const katexPlugins: Parameters<typeof ReactMarkdown>[0]['rehypePlugins'] = [
  [rehypeKatex, { strict: false, throwOnError: false, trust: false }],
]

export default function MathMarkdown({ content }: { content: string }) {
  return (
    <div className="math-markdown">
      <ReactMarkdown
        remarkPlugins={markdownPlugins}
        rehypePlugins={katexPlugins}
      >
        {normalizeLatex(content)}
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

