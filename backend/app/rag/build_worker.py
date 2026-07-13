from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from backend.app.rag.multimodal import BuildModelConfig
from backend.app.rag.pipeline import build_knowledge_base


def _write_json(
    path: Path,
    value: dict[str, Any],
    *,
    best_effort: bool = False,
) -> None:
    """Write worker state without letting transient Windows locks abort a build."""

    payload = json.dumps(value, ensure_ascii=False)
    temporary = path.with_name(
        f"{path.name}.{os.getpid()}.{uuid4().hex}.tmp"
    )
    replace_error: OSError | None = None
    try:
        try:
            temporary.write_text(payload, encoding="utf-8")
        except OSError:
            if best_effort:
                return
            raise

        for delay in (0.0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.4):
            if delay:
                time.sleep(delay)
            try:
                os.replace(temporary, path)
                return
            except OSError as exc:
                replace_error = exc
                if (
                    not isinstance(exc, PermissionError)
                    and getattr(exc, "winerror", None) not in {5, 32}
                ):
                    raise

        # Antivirus/indexers may keep the destination open longer than the
        # retry window. A direct overwrite is not atomic, but the manager
        # already ignores incomplete JSON and reads again on the next poll.
        try:
            path.write_text(payload, encoding="utf-8")
            return
        except OSError as direct_error:
            if best_effort:
                return
            raise direct_error from replace_error
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            # A stale uniquely named temp file is harmless and can be removed
            # after the external scanner releases it.
            pass


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
        }, best_effort=True)

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
