# PaddleOCR-VL 本地试验记录

试验日期：2026-08-21。生产 OCR 未切换，仍使用 `qwen3-vl-flash`。

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

同一教材页与已有 Qwen OCR 缓存去除空白和格式字符后的文本一致度为
`99.06%`。这是两个模型输出的一致度，不是基于人工真值的准确率。

教材页上的差异：

- PaddleOCR-VL 把“纯净的具有晶体结构……”排到了“一、半导体”标题之后，视觉原页和 Qwen 输出均在标题之前。
- PaddleOCR-VL 漏掉了 5 个项目符号中的 2 个，正文文字仍然存在。
- Qwen 现有输出直接包含项目所需的 `chapter`、`section` 和 `concepts`；PaddleOCR-VL 的优势是原生 Markdown、版面块和图片提取，接入现有知识库前还需要适配层。

## 结论

PaddleOCR-VL 在 6 GB 显存上可以稳定运行，教材单页速度可接受，但当前样例没有显示出足以直接替换 Qwen 的质量优势。更适合先作为本地离线兜底或版面解析对照引擎。

不要把当前 PaddleOCR-VL 直接导入 FastAPI 进程：GitHub 版要求
Transformers `>=5.8`，而项目依赖仍固定在 `<5`。Windows 下 Paddle 与 Torch 的 DLL
加载顺序也可能互相影响。试验和后续集成应保持独立环境及独立进程。
