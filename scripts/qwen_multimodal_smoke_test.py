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
        description="验证 Qwen3-VL 电路结构化和多模态向量 API 配置"
    )
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "tmp" / "qwen_multimodal_smoke.json"
    )
    args = parser.parse_args()

    if not settings.qwen_api_key:
        raise RuntimeError("QWEN_API_KEY 未配置")
    image = args.image.read_bytes()
    prompt = """识别该电路，只返回 JSON：
{"is_circuit":true,"caption":"","components":[{"id":"R1","type":"resistor","value":null,"terminals":["n1","n2"],"bbox":[]}],"nets":[{"id":"n1","terminals":["R1.1"]}],"netlist":"","description":"","confidence":0.0}
列出所有可见元件及连接；看不清的值写 null，不得猜测。"""

    with QwenVisionClient(api_key=settings.qwen_api_key) as vision_client:
        circuit = vision_client.complete_json(prompt, image_bytes=image)
    with QwenMultimodalEmbeddingClient(
        api_key=settings.qwen_api_key
    ) as embedding_client:
        text_vector = embedding_client.embed_text("基本共射放大电路")
        image_vector = embedding_client.embed_image(image)

    report = {
        "vision_model": settings.qwen_circuit_vision_model,
        "embedding_model": settings.qwen_multimodal_embedding_model,
        "circuit": circuit,
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
