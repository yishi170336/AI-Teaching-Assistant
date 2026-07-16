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

  // Vision models often indent only the opening `$$` beneath a Markdown list
  // item, while leaving the body and closing delimiter at column zero.
  // Canonicalize display blocks at the top level so remark-math cannot swallow
  // the following explanation as part of a malformed formula.
  text = text.replace(
    /^[ \t]*\$\$[ \t]*\n([\s\S]*?)\n[ \t]*\$\$[ \t]*$/gm,
    (_, body) => `\n\n$$\n${body.trim()}\n$$\n\n`,
  )
  text = text.replace(
    /^[ \t]*\$\$([^\n$]+?)\$\$[ \t]*$/gm,
    (_, body) => `\n\n$$\n${body.trim()}\n$$\n\n`,
  )

  const protectedBlocks: string[] = []
  text = text.replace(/\$\$[\s\S]*?\$\$/g, (block) => {
    protectedBlocks.push(block)
    return `@@MATH_BLOCK_${protectedBlocks.length - 1}@@`
  })
  const singleDollarCount = (text.match(/(?<!\\)\$/g) || []).length
  if (singleDollarCount % 2 === 1) text += '$'
  text = text.replace(/@@MATH_BLOCK_(\d+)@@/g, (_, index) => protectedBlocks[Number(index)])
  return text
}
