# Competition case-design slides

`build_case_design.mjs` imports the user's three-slide local template, preserves
its navigation and competition artwork, and adds editable Microsoft YaHei text,
framework and extraction-process diagrams. It does not change the platform.

Run from the repository root with the bundled Node runtime. `CASE_TEMPLATE` and
`CASE_RUNTIME_MODULES` optionally override the template and bundled package paths.
The script writes previews and a draft to `.cache/case-design-build`, and graph
exports to `output/case-design`. After reviewing previews, add `--finalize` to
run the package, geometry, font and import checks and write the final PPTX.
The final output must not already exist. Python checks use the `llm` environment.

Required private inputs in `.cache/case-design-build/assets`:

- `qa-full.png`, `recommend-full.png`, `variant-full.png`, `grading-full.png`,
  `plan-full.png`: actual 1080×945 screenshots of the August 30 five-turn session.
- `current-source-visual.png`: text-free generated Norton-model visual.
- `graph-snapshot.json`: `viewBox`, `nodes`, `edges` read from the actual
  application's complete graph at 100%, with no selected node or visible label.
  Each node includes `transform`, `r`, `fill`, `stroke`, `sw`; each edge includes
  `x1`, `y1`, `x2`, `y2`, `stroke`, `sw`, `opacity`.

The graph snapshot must contain all 1339 nodes and 3319 edges. Browser export
must split the edges into batches smaller than 2000 to avoid array truncation.
Private screenshots, input templates, chat records and build outputs are not
included in this source-code commit. Crops are literal image crops and do not
rewrite the historic answers. Sources and the generated illustration's status
are recorded in each slide's speaker notes.
