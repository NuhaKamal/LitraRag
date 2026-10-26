"""Unit tests for logging_setup.py."""

import logging

import pytest

from litrarag import logging_setup
from litrarag.logging_setup import get_logger, log_timing, setup_logging


@pytest.fixture
def fresh_logging(monkeypatch):
    """Give a test unconfigured logging, then undo everything it added."""
    root = logging.getLogger()
    old_handlers = root.handlers[:]
    old_level = root.level
    monkeypatch.setattr(logging_setup, "_CONFIGURED", False)

    yield  # the test runs here

    for handler in root.handlers[:]:
        if handler not in old_handlers:
            handler.close()  # Windows: release the log file
            root.removeHandler(handler)
    root.setLevel(old_level)


def test_setup_writes_to_log_file(fresh_logging, tmp_path):
    setup_logging("INFO", str(tmp_path))
    get_logger("test.file").info("hello file")

    for handler in logging.getLogger().handlers:
        handler.flush()
    text = (tmp_path / "litrarag.log").read_text(encoding="utf-8")
    assert "hello file" in text


def test_setup_is_idempotent(fresh_logging, tmp_path):
    root = logging.getLogger()
    before = len(root.handlers)

    setup_logging("INFO", str(tmp_path))
    setup_logging("INFO", str(tmp_path))  # second call must do nothing

    assert len(root.handlers) == before + 2  # console + file, added once


def test_log_timing_reports_duration(caplog):
    log = get_logger("test.timing")
    with caplog.at_level(logging.INFO), log_timing(log, "unit work"):
        pass
    assert "unit work took" in caplog.text
    assert "ms" in caplog.text


def test_log_timing_logs_even_when_block_fails(caplog):
    log = get_logger("test.timing")
    with (
        caplog.at_level(logging.INFO),
        pytest.raises(RuntimeError),
        log_timing(log, "failing work"),
    ):
        raise RuntimeError("boom")
    assert "failing work took" in caplog.text
