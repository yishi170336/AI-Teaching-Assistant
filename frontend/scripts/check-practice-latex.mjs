import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'
import katex from 'katex'

import { normalizeLatex } from '../src/lib/latex.ts'

const here = dirname(fileURLToPath(import.meta.url))
const catalogPath = resolve(here, '../../backend/app/practice/catalog.json')
const catalog = JSON.parse(readFileSync(catalogPath, 'utf8'))
const unit2CatalogPath = resolve(here, '../../backend/app/practice/catalog_unit2.json')
const unit2Catalog = JSON.parse(readFileSync(unit2CatalogPath, 'utf8'))
const questions = [...catalog.questions, ...unit2Catalog.questions]

assert.equal(questions.length, 45, 'practice bank must contain 45 questions')
let formulaCount = 0
for (const question of questions) {
  const normalized = normalizeLatex(question.prompt_markdown)
  assert.equal(normalized.includes('\\('), false, `${question.id} contains raw inline delimiters`)
  assert.equal(normalized.includes('\\['), false, `${question.id} contains raw display delimiters`)

  const displayBlocks = [...normalized.matchAll(/\$\$([\s\S]*?)\$\$/g)]
  const withoutDisplay = normalized.replace(/\$\$[\s\S]*?\$\$/g, '')
  const inlineBlocks = [...withoutDisplay.matchAll(/(?<!\\)\$([^$\n]+?)\$/g)]
  for (const match of [...displayBlocks, ...inlineBlocks]) {
    katex.renderToString(match[1].trim(), { strict: false, throwOnError: true })
    formulaCount += 1
  }
}

assert.ok(formulaCount >= 100, 'expected formulas across the practice bank')
console.log(`Practice LaTeX check passed: ${formulaCount} formulas in 45 questions`)
