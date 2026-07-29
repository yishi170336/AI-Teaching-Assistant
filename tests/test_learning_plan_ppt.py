from __future__ import annotations

from pathlib import Path

from pptx import Presentation

from backend.app.services.learning_plan_ppt import (
    build_learning_plan_ppt,
    generate_learning_plan_ppt,
    parse_learning_plan,
    presentation_filename,
    validate_learning_plan_ppt,
)


SAMPLE_PLAN = r"""
# 晶体管放大电路学习规划

**总体目标：** 建立从静态工作点到动态参数分析的完整知识链，能够独立分析基本共射放大电路。

### 学习现状与执行原则

- 当前基础：已经掌握欧姆定律，但对晶体管工作区与等效模型不熟悉。
- 先补前置，再进入参数计算；每个阶段都通过练习与自测收口。

## 阶段一：诊断与前置补全（1～2 小时）

- **目标：** 确认直流通路、交流通路和晶体管三个工作区的理解缺口。
- **具体行动：** 画出典型共射电路的直流通路与交流通路。
- 回顾欧姆定律、基尔霍夫定律以及电容的隔直通交作用。
- 完成 5 道前置自测题，并记录不会的知识点。
- **完成标准：** 能不看资料说明截止区、放大区和饱和区的判据。
- **资料依据：** [资料1] 基本放大电路工作原理。

## 阶段二：静态工作点分析（2～3 小时）

- **目标：** 掌握 $I_{BQ}$、$I_{CQ}$ 与 $U_{CEQ}$ 的计算路径。
- **具体行动：** 整理固定偏置和分压偏置两类电路的计算模板。
- 独立完成 6 道静态分析题，对每个结果标注单位。
- 用负载线检查计算结果是否处于放大区。
- **完成标准：** 静态参数计算正确率达到 85%，并能解释 Q 点偏移的影响。

## 阶段三：动态参数与等效模型（3～4 小时）

- **目标：** 使用 h 参数等效模型求解电压增益、输入电阻和输出电阻。
- **具体行动：** 先画交流等效电路，再列出输入与输出回路方程。
- 对比旁路电容存在与不存在时增益和输入电阻的变化。
- 推导 $A_u=-\frac{\beta R'_L}{r_{be}}$，并解释各参数对电压增益的影响。
- 完成 4 道综合计算题，并用量纲与数量级复核答案。
- **完成标准：** 能独立画出等效电路，三项动态参数正确率不低于 80%。

## 阶段四：专项训练与复盘（2 小时）

- **目标：** 把静态与动态分析串联起来，形成稳定的解题流程。
- **具体行动：** 完成一组限时综合题，按“识图—静态—动态—校验”书写。
- 将错题按概念、建模、计算和单位四类归因。
- 24 小时后重做错题，并用自己的话讲解完整思路。
- **完成标准：** 综合题正确率达到 80%，同类错误不连续出现两次。

### 7 天学习安排

- Day 1：完成诊断与前置知识补全。
- Day 2～3：集中训练静态工作点计算。
- Day 4～5：学习等效模型并完成动态参数练习。
- Day 6：完成综合题与错题归因。
- Day 7：闭卷自测、复盘并形成一页知识地图。

### 量化验收指标

- 能在 3 分钟内画出直流通路与交流通路。
- 静态工作点计算正确率达到 85%。
- 动态参数综合题正确率达到 80%。
- 能口头解释失真类型、产生原因与调整方向。
""".strip()


REALISTIC_PLAN = r"""
依据学生画像与错题摘要，本规划聚焦 16 个细粒度标签。根据 schedule_guidance 推荐跨 3–6 周推进。

### 整体阶段划分原则

- 窄范围合并：反馈机制、三种组态与旁路电容按关联模块合并。
- 系统范围拆分：按知识依赖关系推进。
- 所有阶段严格遵循诊断→学习→练习→验收。

## 第一阶段：诊断与前置补全（2–4 课次，4–8 小时）

- 目标：定位错题中的前置知识缺口。
- 资料依据：[资料1]第256页、[资料7]第300页
- 具体行动：
  - 对照错题本逐条标注未掌握的前置点。
  - 精读 [资料1] 并绘制思维导图。
  - 完成 2 个基础题。
- 完成标准：
  - 能区分直流反馈与交流反馈。

## 第二阶段：反馈机制与组态特性融合学习（4–6 课次，8–12 小时）

- 目标：建立反馈类型、性能影响和电路结构的映射关系。
- 资料依据：[资料1]第282页、[资料3]第300页
- 具体行动：
  - 梳理反馈类型—性能改善—电路实现三元矩阵。
  - 对比三种组态的关键参数。
  - 完成 3 类典型题。
- 完成标准：
  - 能独立说明闭环增益近似式及其适用条件。

## 第三阶段：专项练习与错题重构（4–6 课次）

- 目标：把理论转化为解题能力。
- 资料依据：[资料6]第267页、[资料8]第277页
- 具体行动：
  - 精做 3 类题型。
  - 建立错题标签库。
  - 自主设计 2 个反例验证。
- 完成标准：
  - 同类错题重做正确率达到 95%。

## 第四阶段：复盘验收与迁移应用（2–4 课次）

- 目标：检验系统理解力与迁移能力。
- 资料依据：[资料1]第282页、[资料8]第277页
- 具体行动：
  - 进行综合电路分析模拟测试。
  - 写一份反馈机制知识图谱。
  - 回顾错题本并撰写反思。
- 完成标准：
  - 综合题得分不低于 85%。

### 可量化验收指标

| 指标项 | 具体内容 | 达标阈值 |
| --- | --- | --- |
| 知识覆盖度 | 所有模块均有学习任务 | 100% |
| 错题转化率 | 同类题型重做正确率 | ≥90% |
| 分析独立性 | 无提示完成反馈判定 | 100% |
| 公式应用准确性 | 公式与场景匹配 | ≥95% |

> 本规划严格遵循 schedule_guidance，不生成日历。
""".strip()


def test_parse_learning_plan_builds_a_logical_story() -> None:
    plan = parse_learning_plan(SAMPLE_PLAN, "请帮我系统学习晶体管放大电路")

    assert plan.title == "晶体管放大电路学习规划"
    assert "完整知识链" in plan.goal
    assert len(plan.stages) == 3
    assert all("复盘" not in stage.title for stage in plan.stages)
    assert "直流通路" in plan.stages[0].actions[0]
    assert "完成 5 道前置自测题" in plan.stages[0].practices[0]
    assert "I_BQ" in plan.stages[1].goal
    assert "旁路电容" in " ".join(plan.stages[2].actions)
    assert "完成 4 道综合计算题" in " ".join(plan.stages[2].practices)
    assert "85%" in " ".join(plan.metrics)
    assert not hasattr(plan, "schedule")
    assert all("小时" not in stage.title for stage in plan.stages)

    generic = parse_learning_plan(
        "# 学习规划\n\n目标：掌握反馈放大电路。\n\n## 第一模块：基础概念\n- 行动：完成概念复习",
        "系统掌握反馈放大电路",
    )
    assert generic.title == "系统掌握反馈放大电路 · 学习规划"
    assert generic.stages[0].title == "基础概念"


def test_build_learning_plan_ppt_is_editable_and_in_bounds(tmp_path: Path) -> None:
    output = tmp_path / "plan.pptx"
    plan, slide_count = build_learning_plan_ppt(SAMPLE_PLAN, output)

    assert output.stat().st_size > 30_000
    assert slide_count == 9
    presentation = Presentation(str(output))
    assert len(presentation.slides) == slide_count
    all_text = "\n".join(
        shape.text
        for slide in presentation.slides
        for shape in slide.shapes
        if getattr(shape, "has_text_frame", False)
    )
    assert "晶体管放大电路学习规划" in all_text
    assert "学习目录" in all_text
    assert "学习目标与方法" in all_text
    assert "静态工作点分析" in all_text
    assert "学什么，怎么做" in all_text
    assert "练什么，达到什么要求" in all_text
    assert "完成 5 道前置自测题" in all_text
    assert "完成 4 道综合计算题" in all_text
    assert "$" not in all_text
    assert "\\frac" not in all_text
    assert "…" not in all_text
    for forbidden in (
        "1～2 小时",
        "7 天学习安排",
        "Day 1",
        "24 小时后",
        "3 分钟内",
        "资料索引",
        "对应资料",
        "复盘验收",
        "迁移应用",
        "什么时候算真正学会",
        "现在就开始",
    ):
        assert forbidden not in all_text
    contents_text = "\n".join(
        shape.text
        for shape in presentation.slides[1].shapes
        if getattr(shape, "has_text_frame", False)
    )
    body_text = "\n".join(
        shape.text
        for slide in list(presentation.slides)[2:]
        for shape in slide.shapes
        if getattr(shape, "has_text_frame", False)
    )
    for item in ["学习目标与方法", *(stage.title for stage in plan.stages)]:
        assert item in contents_text
        assert item in body_text
    final_slide_text = "\n".join(
        shape.text
        for shape in presentation.slides[-1].shapes
        if getattr(shape, "has_text_frame", False)
    )
    assert plan.stages[-1].title in final_slide_text
    assert validate_learning_plan_ppt(output, plan) == []

    for slide in presentation.slides:
        for shape in slide.shapes:
            assert shape.left >= 0
            assert shape.top >= 0
            assert shape.left + shape.width <= presentation.slide_width + 2
            assert shape.top + shape.height <= presentation.slide_height + 2


def test_realistic_plan_does_not_promote_meta_sections_or_table_headers(tmp_path: Path) -> None:
    topic = (
        "依据我的错题本制定知识补全与巩固学习规划。"
        "薄弱知识点：反馈极性的判别、共射电压放大能力、共集输入电阻、旁路电容。"
    )
    plan = parse_learning_plan(REALISTIC_PLAN, topic)

    assert plan.title == "反馈机制、三种基本组态、旁路电容 · 学习规划"
    assert len(plan.stages) == 3
    assert all("整体阶段划分原则" not in stage.title for stage in plan.stages)
    assert all("复盘" not in stage.title and "迁移" not in stage.title for stage in plan.stages)
    assert len(plan.stages[0].actions) == 2
    assert len(plan.stages[0].practices) == 1
    assert len(plan.metrics) == 4
    assert all("指标项" not in metric for metric in plan.metrics)
    assert all("schedule_guidance" not in metric for metric in plan.metrics)
    assert "16 个" not in plan.summary
    assert "3–6 周" not in plan.summary

    output = tmp_path / "realistic-plan.pptx"
    _, slide_count = build_learning_plan_ppt(REALISTIC_PLAN, output, topic)
    presentation = Presentation(str(output))
    all_text = "\n".join(
        shape.text
        for slide in presentation.slides
        for shape in slide.shapes
        if getattr(shape, "has_text_frame", False)
    )

    assert slide_count == 9
    assert "学习目录" in all_text
    assert "学什么，怎么做" in all_text
    assert "练什么，达到什么要求" in all_text
    assert "整体阶段划分原则" not in all_text
    assert "schedule_guidance" not in all_text
    assert "资料索引" not in all_text
    assert "对应资料" not in all_text
    assert "复盘验收与迁移应用" not in all_text
    assert "什么时候算真正学会" not in all_text
    assert "现在就开始" not in all_text
    assert "课次" not in all_text
    assert "小时" not in all_text
    assert "3–6 周" not in all_text
    assert "…" not in all_text
    assert "\n>" not in all_text
    assert validate_learning_plan_ppt(output, plan) == []


def test_complete_student_guide_paginates_without_dropping_content(tmp_path: Path) -> None:
    markdown = """
# 反馈放大电路学习规划

总体目标：能够从电路结构判断反馈类型，并用计算结果解释反馈对性能的影响。

### 学习原则

- 先识别输出取样与输入求和，再判断反馈极性。
- 每完成一个模块，都要留下可检查的图、表或解题过程。
- 错题必须归因并重做，直到能独立讲清判断依据。
- 用反例检查公式适用条件，避免只背结论。

## 阶段一：反馈结构识别

- 目标：建立结构、极性和组态之间的对应关系。
- 核心内容：电压取样与电流取样的结构差异。
- 核心内容：串联求和与并联求和对输入电阻的影响。
- 核心内容：瞬时极性法的判断步骤与常见误区。
- 具体行动：标出典型电路的输出取样点和输入求和节点。
- 具体行动：绘制四种反馈组态的结构对照表。
- 具体行动：为每种组态写出判断依据。
- 具体行动：对照错题本标记发生误判的步骤。
- 具体行动：用自己的话解释负反馈形成条件。
- 练习与复盘：完成四种反馈组态的判断题并逐题写出依据。
- 练习与复盘：设计一个容易误判的反例并完成纠错说明。
- 完成标准：能独立标出反馈网络、取样点和求和节点。
- 完成标准：能说明四种反馈组态对输入输出电阻的影响。
- 完成标准：反馈类型判断正确率达到 90%。
- 完成标准：能解释至少两个常见误判原因。
- 资料依据：[资料1] 反馈的基本概念与分类。
- 资料依据：[资料2] 负反馈放大电路的四种组态。

### 可量化验收指标

- 能独立完成反馈类型判断。
- 能用结构依据解释反馈极性。
- 能把错题归入概念、识图或计算错误。
- 能完成一页反馈组态对照表。
- 同类错题重做正确率达到 95%。
- 能根据新电路迁移判断方法。

### 检索依据

- [资料1] 电子电路基础 · 第五章 · 反馈的基本概念。
- [资料2] 电子电路基础 · 第五章 · 负反馈放大电路的四种组态。
- [资料3] 电子电路基础 · 第五章 · 1.0 mA · 第 300 页。
""".strip()

    output = tmp_path / "complete-guide.pptx"
    plan, slide_count = build_learning_plan_ppt(markdown, output)
    presentation = Presentation(str(output))
    all_text = "\n".join(
        shape.text
        for slide in presentation.slides
        for shape in slide.shapes
        if getattr(shape, "has_text_frame", False)
    )

    assert slide_count >= 7
    expected_fragments = [
        *plan.principles,
        *plan.stages[0].concepts,
        *plan.stages[0].actions,
        *plan.stages[0].practices,
        *plan.stages[0].standards,
    ]
    assert expected_fragments
    assert all(fragment in all_text for fragment in expected_fragments)
    assert "1.0 mA" not in all_text
    assert "资料索引" not in all_text
    assert "检索依据" not in all_text
    assert "[资料3]" not in all_text
    assert "现在就开始" not in all_text
    assert "…" not in all_text
    assert validate_learning_plan_ppt(output, plan) == []


def test_generate_learning_plan_ppt_reuses_content_cache(tmp_path: Path) -> None:
    first_path, first_title, first_count = generate_learning_plan_ppt(
        tmp_path,
        "student-session",
        SAMPLE_PLAN,
        "晶体管放大电路",
    )
    first_mtime = first_path.stat().st_mtime_ns
    second_path, second_title, second_count = generate_learning_plan_ppt(
        tmp_path,
        "student-session",
        SAMPLE_PLAN,
        "晶体管放大电路",
    )

    assert second_path == first_path
    assert second_path.stat().st_mtime_ns == first_mtime
    assert second_title == first_title
    assert second_count == first_count
    assert presentation_filename(first_title).endswith(".pptx")
