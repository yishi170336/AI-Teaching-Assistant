import assert from 'node:assert/strict'
import katex from 'katex'
import { unified } from 'unified'
import remarkParse from 'remark-parse'
import remarkMath from 'remark-math'

import { normalizeLatex } from '../src/lib/latex.ts'

const adjacentDisplayMath = String.raw`因此完整路径为：
$$
V_{BB} \xrightarrow{+} \text{（串 } u_i\text{）} \to R_b \to B\text{（基极）} \to Q_T \to E\text{（发射极）} \to \text{GND}
$$
$$
V_{CC} \xrightarrow{+} \to R_c \to C\text{（集电极）} \to Q_T \to E \to \text{GND}
$$`

const normalized = normalizeLatex(adjacentDisplayMath)
const blocks = [...normalized.matchAll(/\$\$([\s\S]*?)\$\$/g)].map((match) => match[1].trim())

assert.equal(blocks.length, 2, 'adjacent display formulas must remain two complete blocks')
for (const block of blocks) {
  katex.renderToString(block, { strict: false, throwOnError: true })
}
assert.equal(normalized.includes('\\xrightarrow'), true)

const indentedModelFeedback = String.raw`- 空穴浓度（多数载流子）：
  
  $$
p_0 \approx N_A = 2\times10^{20}\,m^{-3}
$$

- 电子浓度（少数载流子）：
  
  $$
n_0 = \frac{n_i^2}{p_0} = 1.125\times10^{12}\,m^{-3}
$$

**573 K 时，$n_i \approx 3\times10^{21}\,m^{-3}$，近似本征导电。**`

const feedbackTree = unified()
  .use(remarkParse)
  .use(remarkMath)
  .parse(normalizeLatex(indentedModelFeedback))
const formulaNodes = []
const rawLatexTextNodes = []
const visit = (node) => {
  if (node.type === 'math' || node.type === 'inlineMath') formulaNodes.push(node)
  if (node.type === 'text' && /\\(?:frac|times|approx|text)\b/.test(node.value || '')) {
    rawLatexTextNodes.push(node.value)
  }
  for (const child of node.children || []) visit(child)
}
visit(feedbackTree)
assert.equal(formulaNodes.length, 3, 'indented model formulas must become math nodes')
assert.deepEqual(rawLatexTextNodes, [], 'LaTeX commands must not remain raw text')
for (const node of formulaNodes) {
  katex.renderToString(node.value, { strict: false, throwOnError: true })
}

console.log('LaTeX normalization regression check passed')
