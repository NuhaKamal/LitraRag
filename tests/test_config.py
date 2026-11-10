"""Unit tests for config.py."""

import dataclasses
from pathlib import Path

import pytest

from litrarag.config import ResourceLimits, Settings


def test_defaults_when_no_env(monkeypatch):
    monkeypatch.delenv("RAG_LLM_MODEL", raising=False)
    monkeypatch.delenv("RAG_CHUNK_SIZE", raising=False)
    s = Settings.from_env()
    assert s.llm_model == "llama3.2:1b"
    assert s.chunk_size == 800


def test_env_overrides_string(monkeypatch):
    monkeypatch.setenv("RAG_LLM_MODEL", "gemma2:9b")
    assert Settings.from_env().llm_model == "gemma2:9b"


def test_env_value_is_converted_to_int(monkeypatch):
    monkeypatch.setenv("RAG_CHUNK_SIZE", "500")
    s = Settings.from_env()
    assert s.chunk_size == 500
    assert isinstance(s.chunk_size, int)


def test_env_value_is_converted_to_path(monkeypatch):
    monkeypatch.setenv("RAG_SQLITE_PATH", "other/test.db")
    assert Settings.from_env().sqlite_path == Path("other/test.db")


def test_env_overrides_resource_limits(monkeypatch):
    monkeypatch.setenv("RAG_MAX_WORKERS", "4")
    assert Settings.from_env().limits.max_workers == 4


def test_invalid_number_gives_clear_error(monkeypatch):
    monkeypatch.setenv("RAG_CHUNK_SIZE", "big")
    with pytest.raises(ValueError, match="RAG_CHUNK_SIZE"):
        Settings.from_env()


def test_settings_are_frozen():
    s = Settings.from_env()
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.chunk_size = 1  # pyright: ignore[reportAttributeAccessIssue]


# --- Validation ---------------------------------------------------------------


def test_invalid_qdrant_mode_from_env(monkeypatch):
    monkeypatch.setenv("RAG_QDRANT_MODE", "embeded")  # typo on purpose
    with pytest.raises(ValueError, match="qdrant_mode"):
        Settings.from_env()


def test_invalid_qdrant_mode_direct():
    with pytest.raises(ValueError, match="qdrant_mode"):
        Settings(qdrant_mode="embeded")  # pyright: ignore[reportArgumentType]


def test_invalid_log_level():
    with pytest.raises(ValueError, match="log_level"):
        Settings(log_level="LOUD")  # pyright: ignore[reportArgumentType]


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError, match="chunk_overlap"):
        Settings(chunk_size=800, chunk_overlap=900)


def test_all_errors_reported_together():
    with pytest.raises(ValueError) as exc_info:
        Settings(chunk_size=0, qdrant_mode="x")  # pyright: ignore[reportArgumentType]
    message = str(exc_info.value)
    assert "chunk_size" in message
    assert "qdrant_mode" in message


def test_resource_limits_reject_zero_workers():
    with pytest.raises(ValueError, match="max_workers"):
        ResourceLimits(max_workers=0)


# --- Tesseract default -------------------------------------------------------


def test_tesseract_cmd_defaults_to_none(monkeypatch):
    monkeypatch.delenv("RAG_TESSERACT_CMD", raising=False)
    assert Settings.from_env().tesseract_cmd is None


def test_tesseract_cmd_can_be_set_from_env(monkeypatch):
    monkeypatch.setenv("RAG_TESSERACT_CMD", r"D:\tools\tesseract.exe")
    assert Settings.from_env().tesseract_cmd == r"D:\tools\tesseract.exe"


def test_candidates_must_cover_top_k():
    with pytest.raises(ValueError, match="retrieval_candidates"):
        Settings(top_k=10, retrieval_candidates=5)
