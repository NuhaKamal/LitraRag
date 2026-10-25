"""Unit tests for config.py."""

import dataclasses
from pathlib import Path

import pytest

from litrarag.config import Settings


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
        s.chunk_size = 1
