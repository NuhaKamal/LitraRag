"""LitraRag logging: one setup for the whole app, console + rotating file."""

import logging
import time
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path

_CONFIGURED = False


def setup_logging(level: str = "INFO", log_dir: str = "logs") -> None:
    """Configure logging once. Safe to call many times."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")

    console = logging.StreamHandler()
    console.setFormatter(fmt)

    file = RotatingFileHandler(
        Path(log_dir) / "litrarag.log",
        maxBytes=5_000_000,  # 5 MB per file
        backupCount=3,  # keep 3 old files, delete older
        encoding="utf-8",  # needed for Arabic text in logs
    )
    file.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(console)
    root.addHandler(file)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Each module calls get_logger(__name__) to get its own named logger."""
    return logging.getLogger(name)


@contextmanager
def log_timing(logger: logging.Logger, label: str):
    """Measure and log how long a block takes.

    Usage:
        with log_timing(log, "embed batch"):
            ...
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        ms = (time.perf_counter() - start) * 1000
        logger.info("%s took %.1f ms", label, ms)


def _demo() -> None:
    setup_logging("DEBUG")
    log = get_logger(__name__)
    log.debug("debug: detailed info for developers")
    log.info("info: normal progress messages")
    log.warning("warning: something unexpected, but we continue")
    log.info("welcome test: Hello LitraRag")
    with log_timing(log, "fake work"):
        time.sleep(0.2)


if __name__ == "__main__":
    _demo()
