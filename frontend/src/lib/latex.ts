export function normalizeLatex(input: string): string {
  let text = input
    .replace(/\r\n?/g, '\n')
    .replace(/＄/g, '$')
    .replace(/\\\[([\s\S]*?)\\\]/g, (_, body) => `\n$$${body.trim()}$$\n`)
    .replace(/\\\(([\s\S]*?)\\\)/g, (_, body) => `$${body.trim()}$`)
    .replace(/\\begin\{(?:equation\*?|displaymath)\}([\s\S]*?)\\end\{(?:equation\*?|displaymath)\}/g, (_, body) => `\n$$${body.trim()}$$\n`)
    // Repair one malformed `$$$` delimiter, but never merge two valid
    // adjacent display blocks (`$$...$$\n$$...$$`).
    .replace(/(?<!\$)\${3}(?!\$)/g, '$$')

  // Compatible APIs occasionally emit HTML-style subscripts even when the
  // prompt requests LaTeX.  ReactMarkdown escapes raw HTML, so normalize the
  // common variable form without enabling unsafe HTML rendering.
  text = text
    .replace(/([A-Za-zΑ-Ωα-ω])<sub>([A-Za-z0-9Α-Ωα-ω,+\-]+)<\/sub>/gi, (_, base, subscript) => `$${base}_{\\mathrm{${subscript}}}$`)
    .replace(/([A-Za-zΑ-Ωα-ω])<sup>([A-Za-z0-9Α-Ωα-ω,+\-]+)<\/sup>/gi, (_, base, superscript) => `$${base}^{\\mathrm{${superscript}}}$`)

  const protectedBlocks: string[] = []
  text = text.replace(/\$\$[\s\S]*?\$\$/g, (block) => {
    protectedBlocks.push(block)
    return `@@MATH_BLOCK_${protectedBlocks.length - 1}@@`
  })

  // Some imported question-bank answers place the opening `$` at the end of
  // a prose line and the formula on the following line. Markdown does not
  // consistently recognize that as inline math. More importantly, the
  // variable/unit repair below would otherwise add another pair of `$` inside
  // it and produce invalid nested math. Collapse only paired inline-math line
  // breaks while display blocks are protected by placeholders.
  text = text.replace(
    /(?<!\\)\$(?!\$)([\s\S]*?)(?<!\\)\$(?!\$)/g,
    (_, body) => `$${String(body).replace(/[ \t]*\n[ \t]*/g, ' ').trim()}$`,
  )
  const singleDollarCount = (text.match(/(?<!\\)\$/g) || []).length
  if (singleDollarCount % 2 === 1) {
    const trailingBackslashes = text.match(/\\+$/)?.[0].length || 0
    text += trailingBackslashes % 2 === 1 ? ' $' : '$'
  }
  text = text.replace(/@@MATH_BLOCK_(\d+)@@/g, (_, index) => protectedBlocks[Number(index)])

  const completeMath: string[] = []
  text = text.replace(/\$\$[\s\S]*?\$\$|\$(?:\\.|[^$\n])*?\$/g, (block) => {
    completeMath.push(block)
    return `@@PROTECTEDMATH${completeMath.length - 1}@@`
  })

  text = text.replace(
    /(^|[^A-Za-z0-9_$\\])([A-Za-z])_([A-Za-z][A-Za-z0-9]*(?:\([A-Za-z]+\))?)(['′])?(?=$|[^A-Za-z0-9_])/g,
    (_, prefix, base, subscript, prime) => `${prefix}$${base}_{${subscript}}${prime ? "'" : ''}$`,
  )
  text = text.replace(
    /(^|[^A-Za-z0-9_$\\])(β|ω)(?=$|[^A-Za-z0-9_])/g,
    (_, prefix, symbol) => `${prefix}$\\${symbol === 'β' ? 'beta' : 'omega'}$`,
  )
  text = text.replace(
    /(^|[^A-Za-z0-9_$\\])([±+-]?\d+(?:\.\d+)?)\s*([fpnumkM]?)(Ω|V|A|F|Hz|W)(?=$|[^A-Za-z0-9_])/g,
    (_, prefix, value, unitPrefix, unit) => {
      const unitLatex = unit === 'Ω'
        ? `${unitPrefix ? `\\mathrm{${unitPrefix}}` : ''}\\Omega`
        : `\\mathrm{${unitPrefix}${unit}}`
      return `${prefix}$${value}\\,${unitLatex}$`
    },
  )
  text = text.replace(
    /@@PROTECTEDMATH(\d+)@@/g,
    (_, index) => completeMath[Number(index)],
  )
  return text
}
