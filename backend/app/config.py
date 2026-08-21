from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]


def _load_dotenv() -> None:
    for env_path in (ROOT_DIR / ".env.local", ROOT_DIR / ".env"):
        if not env_path.exists():
            continue
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
        "EMBEDDING_MODEL_PATH", "models/Qwen3-Embedding-0.6B"
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
    qwen_chat_model: str = os.getenv("QWEN_CHAT_MODEL", "qwen3.7-plus")
    qwen_graph_model: str = os.getenv("QWEN_GRAPH_MODEL", "qwen3.7-flash")
    qwen_vision_model: str = os.getenv("QWEN_VISION_MODEL", "qwen3-vl-flash")
    qwen_visual_summary_model: str = os.getenv(
        "QWEN_VISUAL_SUMMARY_MODEL",
        os.getenv("QWEN_CIRCUIT_VISION_MODEL", "qwen3.7-flash"),
    )
    # Backward-compatible alias for existing deployments and index metadata.
    qwen_circuit_vision_model: str = os.getenv(
        "QWEN_CIRCUIT_VISION_MODEL", qwen_visual_summary_model
    )
    qwen_cleaning_model: str = os.getenv("QWEN_CLEANING_MODEL", "qwen3.7-flash")
    qwen_homework_extraction_model: str = os.getenv(
        "QWEN_HOMEWORK_EXTRACTION_MODEL", "qwen3-vl-flash"
    )
    qwen_homework_grading_model: str = os.getenv(
        "QWEN_HOMEWORK_GRADING_MODEL", "qwen3-vl-flash"
    )
    qwen_homework_review_model: str = os.getenv(
        "QWEN_HOMEWORK_REVIEW_MODEL", "qwen3-vl-8b-instruct"
    )
    qwen_vision_max_tokens: int = int(os.getenv("QWEN_VISION_MAX_TOKENS", "8192"))
    qwen_image_model: str = os.getenv("QWEN_IMAGE_MODEL", "qwen-image-2.0")
    qwen_image_endpoint: str = os.getenv(
        "QWEN_IMAGE_ENDPOINT",
        "https://dashscope.aliyuncs.com/api/v1/services/aigc/"
        "multimodal-generation/generation",
    )
    qwen_image_size: str = os.getenv("QWEN_IMAGE_SIZE", "2688*1536")
    qwen_image_timeout_seconds: float = float(
        os.getenv("QWEN_IMAGE_TIMEOUT_SECONDS", "360")
    )
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
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "200"))
    max_attachment_mb: int = int(os.getenv("MAX_ATTACHMENT_MB", "20"))
    max_homework_upload_mb: int = int(os.getenv("MAX_HOMEWORK_UPLOAD_MB", "100"))
    max_homework_answer_images: int = int(
        os.getenv("MAX_HOMEWORK_ANSWER_IMAGES", "40")
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
    paddleocr_device: str = os.getenv("PADDLEOCR_DEVICE", "gpu:0")
    paddleocr_engine: str = os.getenv("PADDLEOCR_ENGINE", "transformers")
    paddleocr_dtype: str = os.getenv("PADDLEOCR_DTYPE", "float16")
    paddleocr_provider: str = os.getenv("PADDLEOCR_PROVIDER", "local").strip().lower()
    paddleocr_api_token: str = os.getenv("PADDLEOCR_API_TOKEN", "")
    paddleocr_api_job_url: str = os.getenv(
        "PADDLEOCR_API_JOB_URL",
        "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs",
    )
    paddleocr_api_model: str = os.getenv("PADDLEOCR_API_MODEL", "PaddleOCR-VL-1.6")
    paddleocr_api_poll_interval_seconds: float = float(
        os.getenv("PADDLEOCR_API_POLL_INTERVAL_SECONDS", "5")
    )
    paddleocr_api_timeout_seconds: float = float(
        os.getenv("PADDLEOCR_API_TIMEOUT_SECONDS", "900")
    )
    paddleocr_pipeline_version: str = os.getenv(
        "PADDLEOCR_PIPELINE_VERSION", "v1.6"
    )
    paddleocr_model_source: str = os.getenv("PADDLEOCR_MODEL_SOURCE", "BOS")
    paddleocr_render_scale: float = float(os.getenv("PADDLEOCR_RENDER_SCALE", "2.0"))
    paddleocr_retry_scale: float = float(os.getenv("PADDLEOCR_RETRY_SCALE", "3.0"))
    paddleocr_key_block_confidence: float = float(
        os.getenv("PADDLEOCR_KEY_BLOCK_CONFIDENCE", "0.70")
    )
    rerank_model_path: str = os.getenv("RERANK_MODEL_PATH", "")
    multimodal_image_limit: int = int(os.getenv("MULTIMODAL_IMAGE_LIMIT", "0"))
    multimodal_min_image_area: int = int(os.getenv("MULTIMODAL_MIN_IMAGE_AREA", "12000"))
    formula_vl_retry_count: int = int(os.getenv("FORMULA_VL_RETRY_COUNT", "1"))
    semantic_quality_min_text_units: int = int(
        os.getenv("SEMANTIC_QUALITY_MIN_TEXT_UNITS", "100")
    )
    semantic_min_fact_evidence_coverage: float = float(
        os.getenv("SEMANTIC_MIN_FACT_EVIDENCE_COVERAGE", "0.9882")
    )
    semantic_min_multimodal_fact_coverage: float = float(
        os.getenv("SEMANTIC_MIN_MULTIMODAL_FACT_COVERAGE", "0.9952")
    )
    frontend_origins: tuple[str, ...] = tuple(
        value.strip()
        for value in os.getenv(
            "FRONTEND_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
        ).split(",")
        if value.strip()
    )


settings = Settings()
