from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.config import settings
from backend.app.services.qwen_multimodal_client import (
    QwenMultimodalEmbeddingClient,
    QwenVisionClient,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="验证 Qwen3-VL 课程图片/表格总结和多模态向量 API 配置"
    )
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "tmp" / "qwen_multimodal_smoke.json"
    )
    args = parser.parse_args()

    if not settings.qwen_api_key:
        raise RuntimeError("QWEN_API_KEY 未配置")
    image = args.image.read_bytes()
    prompt = """总结该教材图片表达的课程知识，只返回 JSON：
{"is_course_relevant":true,"visual_type":"circuit|characteristic_curve|waveform|physical_structure|device_photo|system_block_diagram|illustration|other","caption":"","summary":"","knowledge_points":[],"is_circuit":false,"components":[],"nets":[],"description":"","confidence":0.0}
不得进行页面 OCR 或公式识别；只记录图中可核验的信息，看不清的内容不得猜测。"""

    with QwenVisionClient(
        api_key=settings.qwen_api_key,
        model=settings.qwen_visual_summary_model,
        base_url=settings.qwen_base_url,
    ) as vision_client:
        visual_summary = vision_client.complete_json(prompt, image_bytes=image)
    with QwenMultimodalEmbeddingClient(
        api_key=settings.qwen_api_key
    ) as embedding_client:
        text_vector = embedding_client.embed_text("基本共射放大电路")
        image_vector = embedding_client.embed_image(image)

    report = {
        "vision_model": settings.qwen_visual_summary_model,
        "embedding_model": settings.qwen_multimodal_embedding_model,
        "visual_summary": visual_summary,
        "text_embedding": {
            "dimension": len(text_vector),
            "l2_norm": round(sum(value * value for value in text_vector) ** 0.5, 6),
        },
        "image_embedding": {
            "dimension": len(image_vector),
            "l2_norm": round(sum(value * value for value in image_vector) ** 0.5, 6),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
