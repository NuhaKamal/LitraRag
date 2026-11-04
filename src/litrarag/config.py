"""LitraRag settings. Every tunable value lives here, never inside a stage.

Any field can be overridden by an environment variable named
RAG_ + the field name in uppercase. In PowerShell:
    $env:RAG_LLM_URL = "https://your-name.ngrok-free.app"
    $env:RAG_MAX_WORKERS = "1"

Settings are validated on creation: invalid values fail at startup, not deep in a stage.
"""

import os
import shutil
from dataclasses import dataclass, field, fields
from pathlib import Path
from types import UnionType
from typing import TYPE_CHECKING, Any, Literal, Self, get_args, get_origin

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

ENV_PREFIX = "RAG_"

QdrantMode = Literal["embedded", "server"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]


def _convert(raw: str, target: Any, env_name: str) -> Any:
    """Convert an environment string to the field's type, with a clear error."""
    if get_origin(target) is Literal:
        return raw  # allowed values are checked in __post_init__
    if isinstance(target, UnionType):  # e.g. str | None -> convert as str
        target = next(t for t in get_args(target) if t is not type(None))
    try:
        if target is bool:
            # bool("false") is True in Python, so booleans need their own rule
            return raw.strip().lower() in {"1", "true", "yes", "on"}
        return target(raw)
    except ValueError as err:
        raise ValueError(f"{env_name}={raw!r} is not a valid {target.__name__}") from err


def _load[T: DataclassInstance](cls: type[T], **fixed: Any) -> T:
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


def _literal_errors(obj: Any) -> list[str]:
    """Check every Literal-typed field holds one of its allowed values."""
    errors = []
    for f in fields(obj):
        if get_origin(f.type) is Literal:
            value, allowed = getattr(obj, f.name), get_args(f.type)
            if value not in allowed:
                errors.append(f"{f.name}={value!r} must be one of {allowed}")
    return errors


def _raise_if_errors(name: str, errors: list[str]) -> None:
    if errors:
        raise ValueError(f"Invalid {name}: " + "; ".join(errors))


@dataclass(frozen=True)
class ResourceLimits:
    """Knobs that control speed and resource use. Tune them; never hard-code them."""

    embed_batch_size: int = 16  # chunks per embedding call
    max_workers: int = 4  # parallel OCR pages (benchmarked: 2.8x faster than 1)
    ocr_dpi: int = 200  # higher = sharper but slower
    ocr_min_chars: int = 30  # fewer chars = scanned page
    llm_num_ctx: int = 2048  # LLM context size (uses RAM)
    ollama_keep_alive: str = "2m"  # unload idle models

    def __post_init__(self) -> None:
        errors = _literal_errors(self)
        if self.embed_batch_size < 1:
            errors.append(f"embed_batch_size={self.embed_batch_size} must be >= 1")
        if self.max_workers < 1:
            errors.append(f"max_workers={self.max_workers} must be >= 1")
        if not 72 <= self.ocr_dpi <= 600:
            errors.append(f"ocr_dpi={self.ocr_dpi} must be between 72 and 600")
        if self.ocr_min_chars < 0:
            errors.append(f"ocr_min_chars={self.ocr_min_chars} must be >= 0")
        if self.llm_num_ctx < 512:
            errors.append(f"llm_num_ctx={self.llm_num_ctx} must be >= 512")
        _raise_if_errors("ResourceLimits", errors)

    @classmethod
    def from_env(cls) -> Self:
        return _load(cls)


@dataclass(frozen=True)
class Settings:
    # Paths and logging
    data_dir: Path = Path("data")
    sqlite_path: Path = Path("data/litrarag.db")
    log_dir: str = "logs"
    log_level: LogLevel = "INFO"
    embedding_cache_path: Path = Path("data/cache/embeddings.db")

    # OCR: None means "find tesseract on PATH"
    tesseract_cmd: str | None = None

    # Vector store
    qdrant_mode: QdrantMode = "embedded"
    qdrant_path: Path = Path("data/qdrant")
    qdrant_url: str = "http://localhost:6333"
    collection: str = "litrarag"

    # Models: embeddings local by default, LLM local or Colab
    embed_url: str = "http://localhost:11434"
    embed_model: str = "nomic-embed-text"
    llm_url: str = "http://localhost:11434"
    llm_model: str = "llama3.2:1b"

    # Chunking
    chunk_size: int = 800
    chunk_overlap: int = 100

    limits: ResourceLimits = field(default_factory=ResourceLimits)

    def __post_init__(self) -> None:
        errors = _literal_errors(self)
        if self.chunk_size <= 0:
            errors.append(f"chunk_size={self.chunk_size} must be > 0")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            errors.append(
                f"chunk_overlap={self.chunk_overlap} must be >= 0 "
                f"and smaller than chunk_size={self.chunk_size}"
            )
        _raise_if_errors("Settings", errors)

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

    tesseract = settings.tesseract_cmd or shutil.which("tesseract")
    if tesseract:
        log.info("Tesseract: %s", tesseract)
    else:
        log.error("Tesseract not found: install it or set RAG_TESSERACT_CMD")


if __name__ == "__main__":
    _demo()
