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

const circuitNotation = normalizeLatex(
  "β = 150，V_T = 26mV，V_BE(on) = 0.7V，r_bb' = 100 Ω，R_B1 = 60kΩ，A_v1 = v_o / v_i。",
)
assert.equal(circuitNotation.includes('$\\beta$'), true)
assert.equal(circuitNotation.includes('$V_{T}$'), true)
assert.equal(circuitNotation.includes('$26\\,\\mathrm{mV}$'), true)
assert.equal(circuitNotation.includes('$V_{BE(on)}$'), true)
assert.equal(circuitNotation.includes("$r_{bb}'$"), true)
assert.equal(circuitNotation.includes('$100\\,\\Omega$'), true)
assert.equal(circuitNotation.includes('$R_{B1}$'), true)
assert.equal(circuitNotation.includes('$60\\,\\mathrm{k}\\Omega$'), true)
assert.equal(circuitNotation.includes('$A_{v1}$ = $v_{o}$ / $v_{i}$'), true)

const escapedMarkdownNotation = normalizeLatex(
  String.raw`**u\_min**、u\_o(t)、U_o,max 与 v_1 均应显示为数学符号。`,
)
assert.equal(escapedMarkdownNotation.includes(String.raw`**$u_{min}$**`), true)
assert.equal(escapedMarkdownNotation.includes(String.raw`$u_{o}(t)$`), true)
assert.equal(escapedMarkdownNotation.includes(String.raw`$U_{o,max}$`), true)
assert.equal(escapedMarkdownNotation.includes(String.raw`$v_{1}$`), true)
assert.equal(
  normalizeLatex('输出在 0~10ms 下降，再在 10~20ms 上升。'),
  '输出在 0～10ms 下降，再在 10～20ms 上升。',
)
for (const inline of [...escapedMarkdownNotation.matchAll(/\$([^$]+)\$/g)].map((match) => match[1])) {
  katex.renderToString(inline, { strict: false, throwOnError: true })
}

const boundQuestionSummary = normalizeLatex(
  String.raw`主要参数为：$\beta_0=80$，$r_{b'e}=50\,\Omega$，$C_{b'e}=2\,\mathrm{pF}$，$f_T=400\,\mathrm{MHz}$。`,
)
for (const inline of [...boundQuestionSummary.matchAll(/\$([^$]+)\$/g)].map((match) => match[1])) {
  katex.renderToString(inline, { strict: false, throwOnError: true })
}
assert.equal(boundQuestionSummary.includes(String.raw`$\beta_0=80$`), true)

const importedMultilineInlineMath = normalizeLatex(
  String.raw`当 $V_1 = 12.5\,\mathrm{V}$ 时，$
\Delta V_1 = 0.5\,\mathrm{V}$，则 $
\Delta V_O = \frac{r_z}{R + r_z} \Delta V_1 \approx \frac{50}{R + 50} \times 0.5\,\mathrm{V}$。`,
)
assert.equal(importedMultilineInlineMath.includes('$\n'), false)
assert.equal(importedMultilineInlineMath.includes('$$V_{1}'), false)
for (const inline of [...importedMultilineInlineMath.matchAll(/\$([^$]+)\$/g)].map((match) => match[1])) {
  katex.renderToString(inline, { strict: false, throwOnError: true })
}
assert.equal(
  importedMultilineInlineMath.includes(String.raw`$\Delta V_O = \frac{r_z}{R + r_z}`),
  true,
)

const tableFormula = normalizeLatex(String.raw`| 知识点 | 依据 | 为何重要 |
|---|---|---|
| 直流工作点 | 需用公式：$$ I_D=\frac{1}{2}k_p\left(\frac{W}{L}\right)(V_{GS}-V_{th})^2\left(1+\gamma\sqrt{|V_{SB}|}\right) $$ | 决定后续参数 |`)
const formulaRow = tableFormula.split('\n').at(-1)
assert.equal(formulaRow.match(/\|/g)?.length, 4, 'math absolute-value bars must not split GFM table cells')
assert.equal(formulaRow.includes('$$'), false, 'display math embedded in a table cell must become inline math')
assert.equal(formulaRow.includes(String.raw`\sqrt{\vert{}V_{SB}\vert{}}`), true)
for (const inline of [...tableFormula.matchAll(/\$([^$]+)\$/g)].map((match) => match[1])) {
  katex.renderToString(inline, { strict: false, throwOnError: true })
}

console.log('LaTeX normalization regression check passed')
