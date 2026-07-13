from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.config import settings
from backend.app.main import app


def _events(response) -> list[tuple[str, dict]]:
    values: list[tuple[str, dict]] = []
    event = "message"
    for line in response.iter_lines():
        if line.startswith("event:"):
            event = line.removeprefix("event:").strip()
        elif line.startswith("data:"):
            values.append((event, json.loads(line.removeprefix("data:").strip())))
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="验证图片附件 + Qdrant/图谱混合检索 + Qwen3-VL 最终回答"
    )
    parser.add_argument("image", type=Path)
    parser.add_argument("--knowledge-base", default="pdfkit_smoke")
    args = parser.parse_args()
    if not settings.qwen_api_key:
        raise RuntimeError("QWEN_API_KEY 未配置")

    session_id = "qwen-rag-e2e-smoke"
    with TestClient(app) as client:
        with args.image.open("rb") as handle:
            upload = client.post(
                "/api/attachments",
                data={"session_id": session_id},
                files={"file": (args.image.name, handle, "image/png")},
            )
        upload.raise_for_status()
        attachment_id = upload.json()["attachment"]["id"]
        with client.stream(
            "POST",
            "/api/chat",
            headers={"Accept": "text/event-stream"},
            json={
                "session_id": session_id,
                "message": "请识别这张电路图，说明 Rb、Rc 和晶体管的连接，并核对 Netlist。",
                "mode": "answer",
                "knowledge_base": args.knowledge_base,
                "attachment_ids": [attachment_id],
                "model_provider": "qwen",
                "model": settings.qwen_vision_model,
            },
        ) as response:
            response.raise_for_status()
            events = _events(response)

    errors = [data for event, data in events if event == "error"]
    if errors:
        raise RuntimeError(str(errors[0].get("message") or errors[0]))
    answer = "".join(
        str(data.get("content", "")) for event, data in events if event == "delta"
    )
    meta = next((data for event, data in events if event == "meta"), {})
    report = {
        "model": settings.qwen_vision_model,
        "knowledge_base": args.knowledge_base,
        "event_counts": {
            name: sum(event == name for event, _ in events)
            for name in sorted({event for event, _ in events})
        },
        "answer_chars": len(answer),
        "source_count": len(meta.get("sources", [])),
        "top_source": (meta.get("sources") or [{}])[0],
        "mentions_netlist": "netlist" in answer.lower(),
        "preview": answer[:600],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
