"""Tests for ocr.py: pure logic as unit tests, real Tesseract as integration tests."""

import pytest
from PIL import Image, ImageDraw, ImageFont

from litrarag import ocr
from litrarag.ocr import TesseractOCR, parse_tesseract_data, resolve_tesseract
from tests.helpers import needs_tesseract


def tess_data(words, confs, line_nums):
    """Build data shaped like pytesseract.image_to_data(..., output_type=DICT)."""
    n = len(words)
    return {
        "text": words,
        "conf": confs,
        "block_num": [1] * n,
        "par_num": [1] * n,
        "line_num": line_nums,
    }


# --- parse_tesseract_data: pure, no Tesseract needed -------------------------


def test_parse_groups_words_into_lines():
    result = parse_tesseract_data(tess_data(["Hello", "world", "again"], [90, 80, 70], [1, 1, 2]))
    assert result.text == "Hello world\nagain"
    assert result.word_count == 3
    assert result.confidence == pytest.approx(80.0)


def test_parse_skips_non_words():
    # Tesseract reports layout boxes with conf -1, and sometimes blank "words"
    result = parse_tesseract_data(tess_data(["", "Hi", "  "], [-1, 90, 50], [1, 1, 1]))
    assert result.text == "Hi"
    assert result.word_count == 1


def test_parse_empty_page_has_zero_confidence():
    result = parse_tesseract_data(tess_data(["", ""], [-1, -1], [1, 1]))
    assert result.text == ""
    assert result.confidence == 0.0
    assert result.word_count == 0


# --- resolve_tesseract --------------------------------------------------------


def test_resolve_explicit_path_that_exists(tmp_path):
    fake_exe = tmp_path / "tesseract.exe"
    fake_exe.write_text("")
    assert resolve_tesseract(str(fake_exe)) == str(fake_exe)


def test_resolve_explicit_path_missing_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="RAG_TESSERACT_CMD"):
        resolve_tesseract(str(tmp_path / "nope.exe"))


def test_resolve_not_on_path_gives_clear_error(monkeypatch):
    monkeypatch.setattr(ocr.shutil, "which", lambda name: None)
    with pytest.raises(FileNotFoundError, match="PATH"):
        resolve_tesseract(None)


# --- Real Tesseract: integration ---------------------------------------------


@pytest.mark.integration
@needs_tesseract
def test_tesseract_reads_rendered_text():
    image = Image.new("RGB", (700, 120), "white")
    font = ImageFont.load_default(size=48)
    ImageDraw.Draw(image).text((20, 30), "HELLO LITRARAG", fill="black", font=font)

    result = TesseractOCR().read(image)

    assert "LITRARAG" in result.text.upper()
    assert result.confidence > 70
