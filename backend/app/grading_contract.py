"""Shared bounded vocabulary for practice grading and answer reports."""

GRADING_ISSUE_TYPE_LABELS = {
    "concept": "概念理解",
    "setup": "建模与列式",
    "calculation": "计算与代数",
    "unit": "单位与量纲",
    "sign_direction": "符号与参考方向",
    "conclusion": "结论表达",
    "recognition": "手写识别",
    "incomplete": "步骤不完整",
    "other": "其他",
}

GRADING_STEP_STATUSES = {"correct", "partial", "incorrect", "unverifiable"}
GRADING_DIMENSION_STATUSES = {"good", "mixed", "needs_improvement", "unverifiable"}

REFERENCE_SOURCE_LABELS = {
    "question_bank": "题库参考答案",
    "teacher_provided": "教师提供答案",
    "ai_inferred": "AI 推断参考",
    "unavailable": "无可用参考答案",
}

QUESTION_SOURCE_LABELS = {
    "question_bank": "题库题",
    "ai_generated": "AI 生成题",
    "user_uploaded": "用户上传题",
}
