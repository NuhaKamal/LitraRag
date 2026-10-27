"""Shared test helpers."""

import shutil

import pytest

needs_tesseract = pytest.mark.skipif(
    shutil.which("tesseract") is None, reason="Tesseract is not installed"
)
