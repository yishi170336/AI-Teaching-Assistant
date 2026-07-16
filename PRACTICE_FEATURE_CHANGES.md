# 刷题模块与多模态 AI 批改变更说明

## 1. 功能概览

学生端现已提供完整的“作答 → AI 批改 → 追问 → 自主确认 → 下一题”流程：

1. 进入“刷题训练”；
2. 选择“电子电路基础及通信电子电路”；
3. 选择第一章“半导体基础知识及二极管电路”或第二章“双极型晶体管及其放大电路”；
4. 每次查看一道题并上传 1～5 张作答图片；
5. 系统保存原始图片，并调用独立的多模态模型对照后端标准答案批改；
6. 查看“正确 / 部分正确 / 错误 / 无法辨认”结论、错误说明及必要的完整解答；
7. 继续向 AI 文字追问；
8. 学生点击“我已弄懂，进入下一题”后，本题才计入章节进度；
9. 学生可随时点击“停止做题”，系统只汇总本轮明确绑定的新提交、批改和对应追问，生成并保存 AI 学情反馈；
10. 在刷题训练内部的“学情反馈”栏目查看逐题错误步骤和后续建议，也可删除不再需要的反馈。

当前会话已升级为范围明确的 v2 规则：学生从章节题目列表明确开始一轮练习，只有携带本轮 `session_id` 的新图片提交会进入反馈。浏览、切题、查看历史批改均不会被统计。已掌握题可以“再练一次”，且不会撤销原有章节完成进度。

### 第二章题库（2026-07-15）

- 新增第二章“双极型晶体管及其放大电路”，包含用户复核通过的 30 道题：2.3.1 至 2.7.1 的指定题号。
- 30 道题均使用 `output/unit2_selected_30/` 中已核对电路拓扑的 SVG 题图；对应 600 DPI PNG 仅供多模态模型批改参考。
- 题干、标准答案、答案要点与教材页码依据《电子电路基础及通信电子电路学习指导书》第二章原书第 48–70 页整理。
- 第一章和第二章共 45 道题。课程页按章节分别展示进度和题目列表；单题页的上一题、下一题及 1–30 题号导航严格限制在当前章节内。
- 课程页的章节题目清单默认收起；第一章和第二章可分别点击“展开题目/收起题目”，避免 45 道题同时铺满页面。
- 展开后的题目卡按章节内顺序显示连续编号（第一章 1–15、第二章 1–30）；教材原题号仍保留在后端主键、题目小节和悬停提示中，确保历史记录与标准答案映射不变。
- 第二章题库拆分为独立 `catalog_unit2.json` 和 `answer_key_unit2.json`，运行时由后端合并，避免改写第一章数据文件并降低团队分支冲突。

刷题模型与聊天区模型相互独立，默认使用 `qwen3-vl-flash`。页面支持千问视觉模型和自定义 OpenAI-compatible 多模态 API，不允许已知纯文本千问模型执行图片批改。

## 2. 主要代码变更

### 后端

- `backend/app/services/model_client_factory.py`
  - 抽取聊天和刷题共用的模型客户端创建逻辑。
  - 保持原 Ollama、DeepSeek、Qwen 和自定义 API 的 Key/Base URL 解析行为。
- `backend/app/main.py`
  - 原 `select_model_client` 保留为兼容包装，内部调用共享工厂。
  - 原聊天、RAG、附件、错题本和模型选择流程未改变。
- `backend/app/practice/grader.py`
  - 读取私有标准答案、答案要点、题图和答案参考图。
  - 将学生图片规范化后发送给多模态模型。
  - 构造防提示词注入的批改与答疑提示词。
  - 校验模型 JSON，维护批改幂等锁，并流式生成追问回答。
- `backend/app/practice/service.py`
  - 将提交状态扩展为 `ungraded / pending / completed / failed`。
  - 保存结构化批改结果、追问记录和 `resolved_at`。
  - 章节完成数和断点续刷改为依据学生确认状态计算。
  - “曾经掌握”和“最新尝试已确认”分开计算；重练不会让已获得的章节进度倒退。
  - 目录接口增加 45 道题的公开状态摘要，提交元数据可绑定 v2 练习会话。
  - 支持独立单元题库文件的运行时合并，并按章节计算题目位置、上一题和下一题。
- `backend/app/practice/catalog_unit2.json`
  - 维护第二章 30 道公开题目、固定顺序、章节归属和公开题图登记。
- `backend/app/practice/answer_key_unit2.json`
  - 维护第二章 30 道私有标准答案、逐题批改要点和教材页码；不提供公开查询接口。
- `backend/app/practice/schemas.py`
  - 校验学生 ID、视觉模型、API 地址和最长 4000 字追问。
- `backend/app/practice/router.py`
  - 在原目录、题目、题图和上传接口上增加批改、SSE 追问、完成确认和本次练习会话接口。
- `backend/app/practice/session_feedback.py`
  - 独立维护练习会话的开始、题目访问、结束、反馈生成、查询和删除。
  - 结束时按会话起止时间提取当次记录，排除更早的历史作答，再复用刷题模型生成结构化反馈。
  - 仅公开模型生成的学情总结，不公开答案库、内部提交元数据或 API Key。
  - 反馈请求同时包含 `system` 约束与 `user` 任务消息，兼容千问对话接口对消息角色的要求。
  - v2 反馈只扫描明确绑定该会话的新提交；同题多次提交按时间顺序交给模型分析修改过程。
  - 零提交会话直接标记为跳过，不调用模型、不进入反馈列表；旧版活动会话不再自动续接。
- `backend/app/practice/assets/grading/`
  - 保存第一章 18 张和第二章 30 张仅供模型识别的 PNG；第二章 PNG 全部为题目主图。
  - 该目录没有公开静态挂载，也没有读取 API。
- `backend/app/practice/assets/prompts/`
  - 新增第二章 30 幅公开 SVG 题图，接口仍只允许读取当前题目登记的图，禁止跨题访问。

### 前端

- `frontend/src/pages/PracticePage.tsx`
  - 增加刷题专用视觉模型配置弹窗，配置独立保存在当前浏览器。
  - 上传完成后自动批改；失败时保留上传记录并支持直接重试。
  - 增加结构化批改卡、KaTeX 完整解答、流式追问、停止生成和历史恢复。
  - 确认理解前禁用“下一题”，最后一题确认后返回章节页。
  - 在刷题训练内部增加“学情反馈”子栏目，并在单题页增加“停止做题”操作。
  - 移除侧栏“答案安全隔离”说明卡和页面头部“第一章题库”徽章。
  - 第一章展示 15 道题、第二章展示 30 道题；每章独立显示状态和进度，已掌握题提供“再练一次”。
  - 课程页和单题路由改为多章节结构，题号导航只显示当前章节。
  - 点击题目时显式确认开始新一轮；章节页和单题页持续显示开始时间、本轮题号和提交次数。
- `frontend/src/pages/PracticeFeedbackPage.tsx`
  - 增加反馈列表、反馈详情、逐题回顾、错误步骤、学习建议和删除确认界面。
  - 生成失败的会话保留原记录，并可在详情页使用当前刷题模型配置直接重新生成。
  - 新版报告明确显示实际提交题号与提交次数；旧报告标记“旧版记录：按访问范围统计”。
- `frontend/src/pages/PracticePage.css`
  - 增加模型设置、批改状态、错误列表、完整解答、答疑消息和移动端样式。
- `frontend/src/lib/practiceApi.ts`
  - 增加批改状态、批改结果、追问消息、模型配置和确认结果类型。
  - 增加批改请求、SSE 追问解析和确认完成请求。
  - 增加练习会话开始、结束、反馈查询和删除请求。
- 原有入口仍由 `frontend/src/App.tsx` 和 `frontend/src/pages/StudentPage.tsx` 以最小改动接入。

### 测试

- `tests/test_practice.py` 现有 16 项测试，覆盖：
  - 15 道题、15 份答案和 18 幅图的完整映射；
  - 图片数量、格式、真实内容、总大小、学生 ID 与题号校验；
  - 标准答案、答案字段和私有图片路径不进入公开响应；
  - 标准答案文字、全部要点和私有 PNG 确实进入模型批改上下文；
  - 正常批改、无法辨认、模型 JSON 错误、失败重试和并发幂等；
  - SSE 追问、历史持久化、跨学生隔离；
  - 确认前不计进度、新尝试重新打开状态、确认后断点续刷。
  - 会话开始前的历史记录不会进入本次反馈上下文；反馈默认保存并支持学生隔离删除。
  - 重练不撤销掌握进度、浏览题不进入 v2 反馈、同题多次提交聚合、空会话跳过和旧会话兼容。

## 3. 后端接口

### 原有接口

- `GET /api/practice/catalog?student_id=...`
  - 返回课程、章节、按“曾经掌握”计算的完成数、断点题号和逐题公开状态摘要。
- `GET /api/practice/questions/{question_id}?student_id=...`
  - 返回公开题目、公开题图、最新提交状态、公开批改结果和追问历史。
- `GET /api/practice/questions/{question_id}/figures/{figure_id}`
  - 仅允许读取该题登记的公开 SVG；私有答案图和跨题访问返回 404。
- `POST /api/practice/questions/{question_id}/submissions`
  - `multipart/form-data`：`student_id` 和 1～5 个 `files`。
  - 接受 PNG、JPEG、WebP、BMP，真实图片合计最大 20 MB。
  - 上传成功只表示图片已保存，返回 `grading_status: "ungraded"` 和 `completed: false`。

### 新增接口

#### `POST /api/practice/questions/{question_id}/submissions/{submission_id}/grade`

JSON 请求字段：

- `student_id`
- `model_provider`：`qwen` 或 `custom`
- `model`
- `api_key`
- `base_url`

返回提交 ID、批改状态、是否已确认及公开 `grade`：

- `verdict`
- `summary`
- `strengths`
- `issues`，每项包含 `location / problem / correction`
- `solution_markdown`
- `model_provider / model / graded_at`

不返回数字分数，也不返回答案库字段、原书页码或私有资源路径。相同提交成功批改后再次请求会直接返回已保存结果。

#### `POST /api/practice/questions/{question_id}/submissions/{submission_id}/messages`

- JSON 请求在模型字段之外增加 `message`，最长 4000 字。
- 响应为 SSE：`connected`、多个 `delta`、`done` 或 `error`。
- 只有完整生成成功的问答才写入 `conversation.json`。

#### `POST /api/practice/questions/{question_id}/submissions/{submission_id}/resolve`

- JSON 请求字段为 `student_id`。
- 仅允许确认本题最新、批改成功且不是 `unreadable` 的提交。
- 返回 `resolved_at` 和 `next_question_id`。

#### 本次练习与学情反馈接口

- `POST /api/practice/sessions/start`
  - JSON：`student_id`、`question_id`。
  - 学生确认后创建 `scope_version: 2` 会话；刷新或切题只续接当前 v2 会话。
- `GET /api/practice/sessions/active?student_id=...`
  - 返回当前活动 v2 会话及本轮实际提交题号、题数和提交次数；没有活动会话时返回 `null`。
- `POST /api/practice/sessions/{session_id}/questions/{question_id}/visit`
  - JSON：`student_id`，记录本次会话访问过的题目。
- `POST /api/practice/sessions/{session_id}/finish`
  - JSON 使用刷题模型配置字段。
  - 锁定本次起止时间，汇总本次题目、提交、批改结论、错误步骤和追问次数，并调用 LLM 生成结构化反馈。
- `POST /api/practice/sessions/{session_id}/discard`
  - 仅用于结束零提交会话，不需要模型配置，也不会生成反馈记录。
- `GET /api/practice/sessions?student_id=...`
  - 返回当前学生已结束且默认保存的反馈列表。
- `GET /api/practice/sessions/{session_id}?student_id=...`
  - 返回一份公开学情反馈，不包含标准答案、答案要点、私有路径或 API Key。
- `DELETE /api/practice/sessions/{session_id}?student_id=...`
  - 删除反馈会话；不删除原有作答图片、批改结果和追问记录。

## 4. 数据目录与状态

运行时数据继续保存在：

`data/practice/submissions/{student_id}/{question_id}/{submission_id}/`

每次尝试包含：

- 原始作答图片 `answer-1.*` 至 `answer-5.*`；
- `metadata.json`：提交、批改、模型和确认状态；
- `conversation.json`：该次作答的成功追问记录，首次追问后创建。

API Key 不写入上述文件。重新提交会创建新的 `submission_id`，不会覆盖旧图片、旧批改或旧追问；最新尝试会重新进入未确认状态。

新版前端上传时同时提交活动 `session_id`，后端在该次 `metadata.json` 中保存 `practice_session_id`。反馈范围只以这个绑定字段为准；没有绑定字段的旧提交继续保留，但不会混入新一轮反馈。

学情反馈运行时数据保存在：

`data/practice/sessions/{student_id}/{session_id}/metadata.json`

该文件保存本次开始/结束时间、访问题号、反馈状态和 LLM 生成的公开反馈。题目提交和对话仍保留在原 submissions 目录，删除反馈不会连带删除作答。

## 5. 答案隔离与模型约束

- `catalog.json` 与 `catalog_unit2.json` 仅维护公开题目；`answer_key.json` 与 `answer_key_unit2.json` 仅供后端批改服务读取。
- 公开题图、私有 SVG 答案图和模型专用 PNG 分目录保存。
- 公开 DTO 使用白名单逐字段构造，批改结果也使用独立白名单过滤。
- 模型收到标准答案、答案要点及必要参考图，但浏览器只收到模型生成的教学反馈。
- 对错误或部分正确作答返回完整解答属于本功能的教学行为；原始答案对象、字段名、页码和答案图仍不会返回前端。
- 学生图片和追问被明确标记为不可信内容，不能改变服务器标准答案或要求模型泄露内部提示词。
- 学生 BMP、WebP、带 EXIF 方向和透明通道的图片会在内存中规范化；磁盘上的原图不变。

## 6. 公式与界面

- 题目和 AI 解答统一复用 `MathMarkdown`、LaTeX 规范化及 KaTeX。
- 批改摘要、正确点、错误位置、错误说明和修改建议也统一经过 `MathMarkdown`，避免模型返回的 `$...$` 公式以源码形式显示。
- LaTeX 规范化会把模型在列表中错误缩进的 `$$...$$` 独立公式还原为顶层公式块；已有批改记录刷新页面后也能正确显示，无需重新批改。
- 批改和追问提示统一要求模型使用 `\\(...\\)` 与 `\\[...\\]`，减少后续响应产生不稳定美元符分隔符的概率。
- 45 道公开题目的 237 个公式均通过 KaTeX 编译检查。
- 批改结果不显示数字分数，只显示结论、正确点、错误点和必要解答。
- 追问回答流式显示，刷新后从后端恢复；停止生成时不保存不完整回答。
- 学情反馈的总结、逐题错误步骤和建议同样通过 `MathMarkdown` 与 KaTeX 呈现。
- 桌面、平板和移动端均保留题目、模型设置、批改结果和答疑操作。

## 7. 验证结果

- `python -m pytest tests/test_practice.py -q`：16 项全部通过。
- `python -m pytest tests -q --tb=no`：120 项通过，2 项既有多模态公式回退测试失败；失败位于 `tests/test_multimodal_rag.py`，与刷题模块无关。
- `python -m pytest -q`：仍在收集 `scripts/pdf_extract_kit_smoke_test.py` 时因当前环境缺少 `cv2` 失败，这是仓库既有环境问题。
- 第二章专项完整性检查：30 个题号顺序正确、30 份答案一一对应、30 幅题图均存在；首题和末题的章节内导航正确，公开目录不含答案字段或答案正文。
- `npm run check:practice`：45 道题、237 个公式通过。
- `npm run check:latex`：LaTeX 规范化回归检查通过，并覆盖列表内错位 `$$` 公式；实际已保存解答解析为 8 个公式节点且没有裸露 LaTeX 文本。
- `npm run build`：TypeScript 检查和 Vite 生产构建通过；仅有原项目的大包体提示。
- 当前后端没有配置可用的云端视觉模型 Key，因此自动化使用模拟多模态客户端；真实 API 调用需要在刷题页面填写 Key，或在后端配置 Qwen Key。

## 8. 后续扩展点

- 可在现有 `grade` 结构中增加教师复核、批改置信度或逐要点评语，但本版不显示数字分数。
- 可将当前本机文件存储替换为数据库和对象存储，接口无需改变。
- 可在确认完成时选择性同步到错题本；当前版本刻意不修改原错题本逻辑。
