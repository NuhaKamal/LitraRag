"""LitraRag settings. Every tunable value lives here, never inside a stage.

Override any setting with an environment variable, e.g. in PowerShell:
    $env:RAG_LLM_URL = "https://your-name.ngrok-free.app"
"""
import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(key: str, default):
    """Read an environment variable, converted to the default's type."""
    value = os.getenv(key)
    if value is None:
        return default
    return type(default)(value)


@dataclass(frozen=True)
class ResourceLimits:
    """Knobs that control CPU, RAM and disk use. Tuned for an 8 GB laptop."""
    embed_batch_size: int = _env("RAG_EMBED_BATCH", 16)    # chunks per embedding call
    max_workers: int = _env("RAG_MAX_WORKERS", 2)          # parallel OCR pages
    ocr_dpi: int = _env("RAG_OCR_DPI", 200)                # higher = sharper but slower
    ocr_min_chars: int = _env("RAG_OCR_MIN_CHARS", 30)     # fewer chars = scanned page
    llm_num_ctx: int = _env("RAG_NUM_CTX", 2048)           # LLM context size (uses RAM)
    ollama_keep_alive: str = _env("RAG_KEEP_ALIVE", "2m")  # unload idle models


@dataclass(frozen=True)
class Settings:
    # Paths
    data_dir: Path = Path(_env("RAG_DATA_DIR", "data"))
    sqlite_path: Path = Path(_env("RAG_SQLITE", "data/litrarag.db"))
    log_dir: str = _env("RAG_LOG_DIR", "logs")
    log_level: str = _env("RAG_LOG_LEVEL", "INFO")

    # OCR
    tesseract_cmd: str = _env("RAG_TESSERACT", r"C:\Program Files\Tesseract-OCR\tesseract.exe")

    # Vector store: "embedded" (no Docker) or "server" (Docker)
    qdrant_mode: str = _env("RAG_QDRANT_MODE", "embedded")
    qdrant_path: Path = Path(_env("RAG_QDRANT_PATH", "data/qdrant"))
    qdrant_url: str = _env("RAG_QDRANT_URL", "http://localhost:6333")
    collection: str = _env("RAG_COLLECTION", "litrarag")

    # Models: embeddings always local, LLM local or Colab
    embed_url: str = _env("RAG_EMBED_URL", "http://localhost:11434")
    embed_model: str = _env("RAG_EMBED_MODEL", "nomic-embed-text")
    llm_url: str = _env("RAG_LLM_URL", "http://localhost:11434")
    llm_model: str = _env("RAG_LLM_MODEL", "llama3.2:1b")

    # Chunking
    chunk_size: int = _env("RAG_CHUNK_SIZE", 800)
    chunk_overlap: int = _env("RAG_CHUNK_OVERLAP", 100)

    limits: ResourceLimits = field(default_factory=ResourceLimits)


settings = Settings()


def _demo() -> None:
    from litrarag.logging_setup import get_logger, setup_logging

    setup_logging(settings.log_level, settings.log_dir)
    log = get_logger("litrarag.config")

    log.info("LLM: %s at %s", settings.llm_model, settings.llm_url)
    log.info("Embeddings: %s at %s", settings.embed_model, settings.embed_url)
    log.info("Qdrant mode: %s", settings.qdrant_mode)
    log.info("Limits: %s", settings.limits)

    # Validate what we can check right now
    if Path(settings.tesseract_cmd).exists():
        log.info("Tesseract found at %s", settings.tesseract_cmd)
    else:
        log.error("Tesseract NOT found at %s", settings.tesseract_cmd)


if __name__ == "__main__":
    _demo()