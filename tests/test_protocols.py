"""Check that implementations satisfy their Protocols."""

from PIL import Image

from litrarag.ocr import TesseractOCR
from litrarag.protocols import OCREngine
from tests.fakes import FakeOCR


def test_fake_ocr_satisfies_protocol():
    engine: OCREngine = FakeOCR()  # pyright verifies this assignment
    result = engine.read(Image.new("RGB", (10, 10)))
    assert result.text == "fake ocr text"
    assert result.word_count == 3


def test_fake_ocr_counts_calls():
    fake = FakeOCR()
    image = Image.new("RGB", (10, 10))
    fake.read(image)
    fake.read(image)
    assert fake.calls == 2


def _tesseract_fits_protocol(engine: TesseractOCR) -> OCREngine:
    return engine  # never called: pyright checks this line statically
