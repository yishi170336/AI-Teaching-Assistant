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
