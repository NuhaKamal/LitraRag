"""Test doubles for LitraRag Protocols: fast, deterministic, no external tools."""

from PIL import Image

from litrarag.schemas import OCRResult


class FakeOCR:
    """Returns a fixed result instantly, and counts how often it was called."""

    def __init__(self, text: str = "fake ocr text", confidence: float = 95.0) -> None:
        self.result = OCRResult(text, confidence, len(text.split()))
        self.calls = 0

    def read(self, image: Image.Image) -> OCRResult:
        self.calls += 1
        return self.result
