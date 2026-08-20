from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_REPOSITORY = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_OUTPUT = ROOT_DIR / "models" / "Qwen3-Embedding-0.6B"

# The upstream repository also contains duplicate ONNX, OpenVINO, TensorFlow,
# and legacy PyTorch exports. The application loads safetensors through
# SentenceTransformers, so only the files below are required for local use.
REQUIRED_PATTERNS = (
    "1_Pooling/config.json",
    "config.json",
    "config_sentence_transformers.json",
    "modules.json",
    "model.safetensors",
    "sentence_bert_config.json",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "unigram.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the minimal embedding model files required by CircuitMind."
    )
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--revision", default="main")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repository,
        revision=args.revision,
        local_dir=output,
        allow_patterns=list(REQUIRED_PATTERNS),
    )
    print(f"Embedding model is ready: {output}")


if __name__ == "__main__":
    main()
