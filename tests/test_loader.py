"""Tests for loader.py. Unit tests use fakes; one integration test uses real Tesseract."""

import io
import shutil
from pathlib import Path

import pymupdf
import pytest
from PIL import Image, ImageDraw, ImageFont

from litrarag.loader import DocumentLoader, UnsupportedFileError
from litrarag.ocr import TesseractOCR
from tests.fakes import FakeOCR, SizeOCR
from tests.helpers import needs_tesseract

LONG_TEXT = "This page has a real text layer with plenty of characters."


def make_pdf(path: Path, pages: list[str | None], widths: list[int] | None = None) -> Path:
    """Create a PDF. A None entry is a page with no text layer, like a scan."""
    pdf = pymupdf.open()
    for n, text in enumerate(pages):
        page = pdf.new_page(width=widths[n] if widths else 595, height=842)
        if text:
            page.insert_text((50, 72), text)
    pdf.save(path)
    pdf.close()
    return path


# --- File types --------------------------------------------------------------


def test_text_file_is_read_directly(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("hello litrarag", encoding="utf-8")
    fake = FakeOCR()

    doc = DocumentLoader(fake).load(path)

    assert doc.pages[0].text == "hello litrarag"
    assert doc.pages[0].method == "text"
    assert doc.file_type == "txt"
    assert fake.calls == 0


def test_image_file_goes_to_ocr(tmp_path):
    path = tmp_path / "photo.png"
    Image.new("RGB", (50, 50), "white").save(path)
    fake = FakeOCR()

    doc = DocumentLoader(fake).load(path)

    assert fake.calls == 1
    assert doc.pages[0].method == "ocr"
    assert doc.file_type == "png"


def test_unsupported_type_raises(tmp_path):
    path = tmp_path / "report.docx"
    path.write_bytes(b"not really a docx")
    with pytest.raises(UnsupportedFileError, match="docx"):
        DocumentLoader(FakeOCR()).load(path)


# --- PDFs: the OCR decision ---------------------------------------------------


def test_pdf_with_text_layer_skips_ocr(tmp_path):
    path = make_pdf(tmp_path / "typed.pdf", [LONG_TEXT, LONG_TEXT])
    fake = FakeOCR()

    doc = DocumentLoader(fake).load(path)

    assert fake.calls == 0
    assert [p.method for p in doc.pages] == ["text", "text"]
    assert "real text layer" in doc.pages[0].text


def test_only_scanned_pages_are_ocred(tmp_path):
    path = make_pdf(tmp_path / "mixed.pdf", [LONG_TEXT, None, LONG_TEXT, None])
    fake = FakeOCR()

    doc = DocumentLoader(fake).load(path)

    assert fake.calls == 2
    assert [p.method for p in doc.pages] == ["text", "ocr", "text", "ocr"]
    assert doc.pages[1].text == "fake ocr text"
    assert doc.pages[1].ocr_confidence == 95.0


def test_short_text_layer_counts_as_scanned(tmp_path):
    # A scan with only a page number in its text layer still needs OCR
    path = make_pdf(tmp_path / "short.pdf", ["Page 7"])
    fake = FakeOCR()

    doc = DocumentLoader(fake, ocr_min_chars=30).load(path)

    assert fake.calls == 1
    assert doc.pages[0].method == "ocr"


def test_page_numbers_start_at_one(tmp_path):
    path = make_pdf(tmp_path / "pages.pdf", [LONG_TEXT] * 3)
    doc = DocumentLoader(FakeOCR()).load(path)
    assert [p.page_number for p in doc.pages] == [1, 2, 3]


def test_parallel_ocr_keeps_page_order(tmp_path):
    widths = [200, 300, 400, 500, 600, 700, 800]
    path = make_pdf(tmp_path / "order.pdf", [None] * len(widths), widths=widths)
    ocr = SizeOCR()

    # At 72 DPI, one PDF point = one pixel, so each page's text is its width
    doc = DocumentLoader(ocr, ocr_dpi=72, max_workers=3).load(path)

    assert [p.text for p in doc.pages] == [str(w) for w in widths]
    assert ocr.calls == len(widths)


# --- doc_id: content, not name -----------------------------------------------


def test_renamed_file_keeps_doc_id(tmp_path):
    original = make_pdf(tmp_path / "original.pdf", [LONG_TEXT])
    renamed = tmp_path / "renamed.pdf"
    shutil.copy(original, renamed)
    loader = DocumentLoader(FakeOCR())

    assert loader.load(original).doc_id == loader.load(renamed).doc_id


def test_changed_content_gets_new_doc_id(tmp_path):
    path = tmp_path / "notes.txt"
    loader = DocumentLoader(FakeOCR())

    path.write_text("version 1", encoding="utf-8")
    first = loader.load(path).doc_id
    path.write_text("version 2", encoding="utf-8")

    assert loader.load(path).doc_id != first


# --- Real Tesseract ----------------------------------------------------------


@pytest.mark.integration
@needs_tesseract
def test_real_ocr_reads_scanned_pdf_page(tmp_path):
    image = Image.new("L", (1200, 300), 255)
    font = ImageFont.load_default(size=72)
    ImageDraw.Draw(image).text((40, 100), "HELLO LITRARAG", fill=0, font=font)
    png = io.BytesIO()
    image.save(png, format="PNG")

    pdf = pymupdf.open()
    pdf.new_page().insert_image(pymupdf.Rect(72, 72, 540, 189), stream=png.getvalue())
    path = tmp_path / "scan.pdf"
    pdf.save(path)
    pdf.close()

    doc = DocumentLoader(TesseractOCR()).load(path)

    assert doc.pages[0].method == "ocr"
    assert "LITRARAG" in doc.pages[0].text.upper()
