# PaddleOCR-VL 本地试验记录

试验日期：2026-08-21。生产知识库 OCR 已切换至 PaddleOCR-VL；Qwen 仅保留课程相关图片/表格总结和文本知识抽取，不参与公式识别。

## 环境

- Windows、NVIDIA GeForce RTX 3060 Laptop GPU（6 GB）
- PaddleOCR GitHub 提交 `2661c7c0ef5c613e8f93c6e93b2e052399f0f854`
- PaddleOCR-VL `v1.6`，模型 `PaddleOCR-VL-1.6-0.9B`
- Transformers `5.15.1`、PyTorch `2.12.0+cu130`
- GPU 推理参数：`engine="transformers"`、`device="gpu:0"`、FP16

模型缓存约 2 GB。PaddleOCR-VL 模型和版面模型由 PaddleOCR 首次运行时自动下载。

## 实测结果

| 样例 | 冷启动 | 单页推理 | 峰值显存 | 观察 |
| --- | ---: | ---: | ---: | --- |
| 官方多栏报纸页 | 18.75 秒 | 75.65 秒 | 2.14 GB | 正确恢复多栏阅读顺序、标题、正文，并抽取图片 |
| 项目教材第 1 页 | 11.49 秒 | 15.30 秒 | 2.14 GB | 中文正文完整，能输出 Markdown 标题层级 |
| 主 `llm` 环境教材封面页 | 11.45 秒 | 4.11 秒 | 2.139 GB | Transformers 5.15.1、RTX 3060 生产依赖验证通过 |

同一教材页与已有 Qwen OCR 缓存去除空白和格式字符后的文本一致度为
`99.06%`。这是两个模型输出的一致度，不是基于人工真值的准确率。

教材页上的差异：

- PaddleOCR-VL 把“纯净的具有晶体结构……”排到了“一、半导体”标题之后，视觉原页和 Qwen 输出均在标题之前。
- PaddleOCR-VL 漏掉了 5 个项目符号中的 2 个，正文文字仍然存在。
- 生产适配层已用几何拓扑排序修复同栏倒序，并根据相邻项目的缩进恢复缺失项目符号的列表类型。
- `chapter`、`section` 从 Paddle 标题块、目录和编号规则中确定，`concepts` 只从已转写正文抽取，不再由 OCR 模型生成。

## 结论

PaddleOCR-VL 在 6 GB 显存上稳定运行。模型只在知识库构建子进程中加载一次，逐页串行解析，OCR 完成后释放显存再启动 GraphRAG 和 Embedding。页面、标题、公式、表格转写不会回退到 Qwen；单页失败会终止候选构建并保留活动索引。

缓存 schema 为 `3.0-paddleocr-vl-layout-evidence`，包含模型 revision、设备、dtype、渲染比例、原始 Paddle 结果和规范化块证据。旧 Qwen OCR 缓存会被强制失效。
