# 学生答案分析报告

学生端“答案分析报告”读取已经完成的刷题批改，支持选择单题、一个连续练习组或跨组多题生成独立报告。生成后的报告包含题目、答案、图片引用、参考依据和批改结果快照；题库后续修改、聊天历史继续增长或模型配置变化都不会改写旧报告。

## 数据与边界

- `practice_session_id` 由学生端开始一组练习时创建，和普通聊天会话分离；同一组后续变式题复用该标识。
- `attempt_id` 由聊天会话和批改轮次稳定推导。历史会话在首次读取时可以回填，但已落盘的尝试不会被重新扫描结果覆盖。
- 只在答案提交时使用当前文本/视觉模型完成批改和手写识别。打开、汇总和打印报告不再调用模型。
- 题目来源分为 `question_bank`、`ai_generated`、`user_uploaded`；参考依据分为 `question_bank`、`teacher_provided`、`ai_inferred`、`unavailable`。AI 推断不会显示为题库标准答案。
- 失败或取消的批改可作为“无法评判”记录进入报告，成功题仍可形成部分报告。
- 跨题汇总是确定性计算。重复错误至少需要两道题的记录证据；趋势至少需要四道按时间有序且题型或知识点可比的计分题。

运行数据保存在 `data/practice_attempts.json` 和 `data/answer_reports.json`，两者均被 Git 忽略。报告保存完整必要快照，不保存 API Key。

## API

```text
GET    /api/practice-attempts?student_id=...&practice_session_id=...
POST   /api/answer-reports
GET    /api/answer-reports?student_id=...
GET    /api/answer-reports/{report_id}?student_id=...
DELETE /api/answer-reports/{report_id}?student_id=...
```

创建请求示例：

```json
{
  "student_id": "learner-demo",
  "attempt_ids": ["32位作答记录标识"],
  "title": "戴维南定理阶段练习"
}
```

所有接口按 `student_id` 过滤并校验资源归属。当前项目仍是单教师/单学生 MVP，`student_id` 不是完整身份认证；生产部署应在这些归属检查之前接入可信登录主体，不能依赖客户端自行提交的 ID。

## 导出

详情页的“打印 / 另存为 PDF”复用浏览器打印能力。打印内容来自已经通过报告详情 API 读取的持久化快照，包含报告标识、生成时间、综合分析、逐题内容、公式和图片。打印 CSS 使用 A4 页面并避免在题目卡片内部强制分页；最终分页和字体效果仍取决于浏览器的打印引擎。

## 兼容与回滚

旧批改记录缺少结构化步骤或置信度时，报告保留旧反馈并显示数据限制，不补造分析。部署不需要数据库迁移；首次访问会从已有会话中兼容回填。回滚代码不会影响聊天和错题本，运行时的两个 JSON 文件可以保留供重新部署后继续使用，或在确认不再需要历史报告后单独备份并删除。
