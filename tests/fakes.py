"""Test doubles for LitraRag Protocols: fast, deterministic, no external tools."""

import threading

from PIL import Image

from litrarag.schemas import OCRResult


class FakeOCR:
    """Returns a fixed result instantly, and counts how often it was called."""

    def __init__(self, text: str = "fake ocr text", confidence: float = 95.0) -> None:
        self.result = OCRResult(text, confidence, len(text.split()))
        self.calls = 0
        self._lock = threading.Lock()

    def read(self, image: Image.Image) -> OCRResult:
        with self._lock:  # several threads may call read() at the same time
            self.calls += 1
        return self.result


class SizeOCR:
    """Returns the image width as its text, so tests can check page ORDER."""

    def __init__(self) -> None:
        self.calls = 0
        self._lock = threading.Lock()

    def read(self, image: Image.Image) -> OCRResult:
        with self._lock:
            self.calls += 1
        return OCRResult(str(image.width), 90.0, 1)
