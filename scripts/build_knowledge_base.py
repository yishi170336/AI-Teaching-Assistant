from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.config import settings
from backend.app.rag.pipeline import build_knowledge_base
from backend.app.rag.multimodal import BuildModelConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="清洗课程资料并构建混合检索向量库")
    parser.add_argument("--knowledge-base", default="default", help="知识库标识")
    parser.add_argument("--chapter-limit", type=int, default=settings.initial_chapter_limit)
    parser.add_argument("--full", action="store_true", help="索引教材全部章节")
    parser.add_argument(
        "--without-multimodal-llm",
        action="store_true",
        help="不调用 DeepSeek 做语义清洗和电路图理解，仅运行可审计的本地降级流程",
    )
    args = parser.parse_args()

    if args.knowledge_base == "default":
        resources_dir = settings.resources_dir
    else:
        resources_dir = settings.resources_dir / "knowledge_bases" / args.knowledge_base
    output_dir = settings.vector_stores_dir / args.knowledge_base
    meta = build_knowledge_base(
        resources_dir,
        output_dir,
        settings.embedding_model_path,
        chapter_limit=None if args.full else args.chapter_limit,
        model_config=(
            None
            if args.without_multimodal_llm
            else BuildModelConfig(
                provider="deepseek",
                model=settings.deepseek_model,
                api_key=settings.deepseek_api_key,
                base_url=settings.deepseek_base_url,
            )
        ),
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
