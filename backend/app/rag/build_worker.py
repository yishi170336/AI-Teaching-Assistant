from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from backend.app.rag.multimodal import BuildModelConfig
from backend.app.rag.pipeline import build_knowledge_base


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def run(job_path: Path) -> int:
    job = json.loads(job_path.read_text(encoding="utf-8"))
    progress_path = Path(job["progress_path"])
    result_path = Path(job["result_path"])
    model_value = job.get("model_config")
    if model_value:
        model_value["api_key"] = os.environ.get("CIRCUITMIND_BUILD_API_KEY", "")
    model_config = BuildModelConfig(**model_value) if model_value else None

    def report(progress: int, stage: str, message: str) -> None:
        _write_json(progress_path, {
            "progress": progress,
            "stage": stage,
            "message": message,
        })

    try:
        build_knowledge_base(
            Path(job["resources_dir"]),
            Path(job["output_dir"]),
            Path(job["embedding_model_path"]),
            chapter_limit=job.get("chapter_limit"),
            model_config=model_config,
            knowledge_base_id=job["knowledge_base"],
            sync_graph_store=False,
            progress_callback=report,
        )
    except Exception as exc:
        _write_json(result_path, {"ok": False, "error": str(exc)})
        return 1
    _write_json(result_path, {"ok": True})
    return 0


def main() -> int:
    if len(sys.argv) != 2:
        return 2
    return run(Path(sys.argv[1]).resolve())


if __name__ == "__main__":
    raise SystemExit(main())
