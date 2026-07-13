import assert from 'node:assert/strict'
import katex from 'katex'

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

console.log('LaTeX normalization regression check passed')
