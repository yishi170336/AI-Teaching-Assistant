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

## Revision from the user's edited deck

`revise_case_design.mjs` imports the user's subsequently edited desktop PPTX and
retains its changed wording, navigation, footer and second slide. It adds native
editable agent/teacher/student icons, shortens agent descriptions, and updates
the two lower application panels. `--framework-preview` renders only the first
slide while the new teaching illustration is being generated.

`export_mirror_explanation_prompt.py` runs the actual platform explanation
pipeline using the `llm` Python environment and the configured Qwen text client.
It saves its task in `.cache/case-design-revision/private-platform-run`, with an
image-boundary adapter that exports the final prompt before any image-provider
call. It uses the same planner, reviewers, page writer, layout designer and
prompt compiler. The private run permits four audited attempts at a stage; it
does not bypass the checks or change the running app. `--repair-plan` uses a
previous rejected plan at `mirror-plan-for-repair.json` through the platform's
existing plan-repair interface, with guidance to satisfy its length constraints.

The exported `mirror-image-prompt.txt` is passed to the built-in image-generation
tool. Copy the user's selected complete generated lecture page into
`.cache/case-design-revision/assets/mirror-selected-page.png` before building.
It is embedded without cropping or distortion, as explicitly requested.
The other private inputs are `mistakes-full.png` and `planning-new-full.png`,
both real 1280×720 screenshots captured from the platform. The latter is a new
planning response requested without fixed day/week allocations. Run the revision
builder, inspect its previews, and then run it with `--finalize`.
`CASE_REVISION_FILENAME` can choose a fresh output filename; final output and
matching validation receipt files must not already exist. Only source
code and this documentation are committed; screenshots, model traces and deck
assets remain private.

## Thevenin lecture page

`export_thevenin_explanation_prompt.py` runs the same platform pipeline for a
single Thevenin lesson. Its short planning question is followed by explicit
circuit and numerical-example requirements at the audited page-writing stage.
Run it with the `llm` environment. The private traces, platform prompt and task
record are written to `.cache/thevenin-explanation`. The image-boundary adapter
captures the prompt without invoking the platform's image provider; render that
prompt with the built-in image-generation tool. The optional `--repair-plan`
flag uses `plan-for-repair.json` through the platform's existing repair interface.
For an author-corrected platform draft, `--review-repaired-draft` reads
`approved-plan.json` and `repaired-page-draft.json` from that private directory,
normalizes them and reruns the platform's independent outline and page auditors.
Both reviews must pass before the standard layout designer and image-prompt
compiler run. No review result is overridden. `--thinking` optionally enables
the configured text model's thinking mode. The exported record indicates when
an author-corrected draft was used.
