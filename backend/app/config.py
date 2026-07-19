from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]


def _load_dotenv() -> None:
    env_path = ROOT_DIR / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()


@dataclass(frozen=True)
class Settings:
    root_dir: Path = ROOT_DIR
    resources_dir: Path = ROOT_DIR / "RAG_Resources"
    vector_stores_dir: Path = ROOT_DIR / "data" / "vector_stores"
    embedding_model_path: Path = ROOT_DIR / os.getenv(
        "EMBEDDING_MODEL_PATH", "models/paraphrase-multilingual-MiniLM-L12-v2"
    )
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen3.5:2b")
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    deepseek_base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    qwen_api_key: str = os.getenv("QWEN_API_KEY", os.getenv("DASHSCOPE_API_KEY", ""))
    qwen_base_url: str = os.getenv(
        "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    qwen_vision_model: str = os.getenv("QWEN_VISION_MODEL", "qwen3-vl-flash")
    qwen_circuit_vision_model: str = os.getenv(
        "QWEN_CIRCUIT_VISION_MODEL", "qwen3-vl-flash"
    )
    qwen_cleaning_model: str = os.getenv("QWEN_CLEANING_MODEL", "qwen3.7-plus")
    qwen_homework_extraction_model: str = os.getenv(
        "QWEN_HOMEWORK_EXTRACTION_MODEL", "qwen3-vl-plus"
    )
    qwen_homework_grading_model: str = os.getenv(
        "QWEN_HOMEWORK_GRADING_MODEL", "qwen3-vl-plus"
    )
    qwen_homework_review_model: str = os.getenv(
        "QWEN_HOMEWORK_REVIEW_MODEL", "qwen3-vl-flash"
    )
    qwen_vision_max_tokens: int = int(os.getenv("QWEN_VISION_MAX_TOKENS", "8192"))
    qwen_multimodal_embedding_model: str = os.getenv(
        "QWEN_MULTIMODAL_EMBEDDING_MODEL", "qwen3-vl-embedding"
    )
    qwen_multimodal_embedding_url: str = os.getenv(
        "QWEN_MULTIMODAL_EMBEDDING_URL",
        "https://dashscope.aliyuncs.com/api/v1/services/embeddings/"
        "multimodal-embedding/multimodal-embedding",
    )
    qwen_multimodal_embedding_dimension: int = int(
        os.getenv("QWEN_MULTIMODAL_EMBEDDING_DIMENSION", "1024")
    )
    qwen_multimodal_timeout_seconds: float = float(
        os.getenv("QWEN_MULTIMODAL_TIMEOUT_SECONDS", "180")
    )
    circuit_image_embedding_instruct: str = os.getenv(
        "CIRCUIT_IMAGE_EMBEDDING_INSTRUCT",
        (
            "Represent this analog circuit diagram for retrieval. Focus on topology, "
            "components, terminal connections, signal direction, and biasing; ignore "
            "typography, scan quality, and page layout."
        ),
    )
    circuit_image_retrieval_min_score: float = float(
        os.getenv("CIRCUIT_IMAGE_RETRIEVAL_MIN_SCORE", "0.70")
    )
    circuit_image_retrieval_max_references: int = int(
        os.getenv("CIRCUIT_IMAGE_RETRIEVAL_MAX_REFERENCES", "2")
    )
    circuit_image_retrieval_candidates: int = int(
        os.getenv("CIRCUIT_IMAGE_RETRIEVAL_CANDIDATES", "12")
    )
    redis_url: str = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
    memory_turns: int = int(os.getenv("MEMORY_TURNS", "8"))
    session_history_messages: int = int(os.getenv("SESSION_HISTORY_MESSAGES", "100"))
    max_ollama_concurrency: int = int(os.getenv("MAX_OLLAMA_CONCURRENCY", "2"))
    remote_max_tokens: int = int(os.getenv("REMOTE_MAX_TOKENS", "8192"))
    remote_max_continuations: int = int(os.getenv("REMOTE_MAX_CONTINUATIONS", "2"))
    initial_chapter_limit: int = int(os.getenv("INITIAL_CHAPTER_LIMIT", "1"))
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "80"))
    max_attachment_mb: int = int(os.getenv("MAX_ATTACHMENT_MB", "20"))
    max_homework_upload_mb: int = int(os.getenv("MAX_HOMEWORK_UPLOAD_MB", "100"))
    max_homework_answer_images: int = int(
        os.getenv("MAX_HOMEWORK_ANSWER_IMAGES", "8")
    )
    max_chat_attachments: int = int(os.getenv("MAX_CHAT_ATTACHMENTS", "5"))
    max_chat_document_images: int = int(os.getenv("MAX_CHAT_DOCUMENT_IMAGES", "6"))
    qdrant_url: str = os.getenv("QDRANT_URL", "")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
    neo4j_uri: str = os.getenv("NEO4J_URI", "")
    neo4j_http_url: str = os.getenv("NEO4J_HTTP_URL", "http://127.0.0.1:7474")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "")
    neo4j_database: str = os.getenv("NEO4J_DATABASE", "neo4j")
    pdf_extract_kit_output_dir: str = os.getenv("PDF_EXTRACT_KIT_OUTPUT_DIR", "")
    pdf_extract_kit_dir: str = os.getenv("PDF_EXTRACT_KIT_DIR", "third_party/PDF-Extract-Kit")
    pdf_extract_kit_page_limit: int = int(os.getenv("PDF_EXTRACT_KIT_PAGE_LIMIT", "0"))
    rerank_model_path: str = os.getenv("RERANK_MODEL_PATH", "")
    multimodal_image_limit: int = int(os.getenv("MULTIMODAL_IMAGE_LIMIT", "0"))
    multimodal_min_image_area: int = int(os.getenv("MULTIMODAL_MIN_IMAGE_AREA", "12000"))
    formula_vl_retry_count: int = int(os.getenv("FORMULA_VL_RETRY_COUNT", "1"))
    frontend_origins: tuple[str, ...] = tuple(
        value.strip()
        for value in os.getenv(
            "FRONTEND_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
        ).split(",")
        if value.strip()
    )


settings = Settings()
