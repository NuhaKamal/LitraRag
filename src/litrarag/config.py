"""LitraRag settings. Every tunable value lives here, never inside a stage.

Any field can be overridden by an environment variable named
RAG_ + the field name in uppercase. In PowerShell:
    $env:RAG_LLM_URL = "https://your-name.ngrok-free.app"
    $env:RAG_MAX_WORKERS = "1"
"""

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Self

ENV_PREFIX = "RAG_"


def _convert(raw: str, target: type, env_name: str) -> Any:
    """Convert an environment string to the field's type, with a clear error."""
    try:
        if target is bool:
            # bool("false") is True in Python, so booleans need their own rule
            return raw.strip().lower() in {"1", "true", "yes", "on"}
        return target(raw)
    except ValueError as err:
        raise ValueError(f"{env_name}={raw!r} is not a valid {target.__name__}") from err


def _load[T](cls: type[T], **fixed: Any) -> T:
    """Build a dataclass from its defaults, overridden by RAG_<FIELD> env variables.

    Fields given in `fixed` are used as they are, not read from the environment.
    """
    values: dict[str, Any] = dict(fixed)
    for f in fields(cls):
        if f.name in fixed:
            continue
        env_name = ENV_PREFIX + f.name.upper()
        raw = os.environ.get(env_name)
        if raw is not None:
            values[f.name] = _convert(raw, f.type, env_name)
    return cls(**values)


@dataclass(frozen=True)
class ResourceLimits:
    """Knobs that control CPU, RAM and disk use. Tuned for an 8 GB laptop."""

    embed_batch_size: int = 16  # chunks per embedding call
    max_workers: int = 2  # parallel OCR pages
    ocr_dpi: int = 200  # higher = sharper but slower
    ocr_min_chars: int = 30  # fewer chars = scanned page
    llm_num_ctx: int = 2048  # LLM context size (uses RAM)
    ollama_keep_alive: str = "2m"  # unload idle models

    @classmethod
    def from_env(cls) -> Self:
        return _load(cls)


@dataclass(frozen=True)
class Settings:
    # Paths and logging
    data_dir: Path = Path("data")
    sqlite_path: Path = Path("data/litrarag.db")
    log_dir: str = "logs"
    log_level: str = "INFO"

    # OCR
    tesseract_cmd: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    # Vector store: "embedded" (no Docker) or "server" (Docker)
    qdrant_mode: str = "embedded"
    qdrant_path: Path = Path("data/qdrant")
    qdrant_url: str = "http://localhost:6333"
    collection: str = "litrarag"

    # Models: embeddings always local, LLM local or Colab
    embed_url: str = "http://localhost:11434"
    embed_model: str = "nomic-embed-text"
    llm_url: str = "http://localhost:11434"
    llm_model: str = "llama3.2:1b"

    # Chunking
    chunk_size: int = 800
    chunk_overlap: int = 100

    limits: ResourceLimits = field(default_factory=ResourceLimits)

    @classmethod
    def from_env(cls) -> Self:
        """Read the environment now. Call again to pick up later changes."""
        return _load(cls, limits=ResourceLimits.from_env())


# The app-wide settings, read once at startup
settings = Settings.from_env()


def _demo() -> None:
    from litrarag.logging_setup import get_logger, setup_logging

    setup_logging(settings.log_level, settings.log_dir)
    log = get_logger("litrarag.config")

    log.info("LLM: %s at %s", settings.llm_model, settings.llm_url)
    log.info("Embeddings: %s at %s", settings.embed_model, settings.embed_url)
    log.info("Qdrant mode: %s", settings.qdrant_mode)
    log.info("Limits: %s", settings.limits)

    if Path(settings.tesseract_cmd).exists():
        log.info("Tesseract found at %s", settings.tesseract_cmd)
    else:
        log.error("Tesseract NOT found at %s", settings.tesseract_cmd)


if __name__ == "__main__":
    _demo()
