"""Test doubles for LitraRag Protocols: fast, deterministic, no external tools."""

import hashlib
import re
import threading
from collections.abc import Sequence

import numpy as np
from PIL import Image

from litrarag.schemas import OCRResult, Vector, Vectors

_WORDS = re.compile(r"\w+")


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


class FakeEmbedder:
    """Deterministic bag-of-words embedder: texts sharing words get similar vectors."""

    def __init__(self, dimension: int = 64, model_name: str = "fake-embedder") -> None:
        self._dimension = dimension
        self._model_name = model_name
        self.calls = 0
        self.texts_embedded = 0
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: Sequence[str]) -> Vectors:
        with self._lock:
            self.calls += 1
            self.texts_embedded += len(texts)
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)
        return np.stack([self._vector(text) for text in texts])

    def embed_query(self, text: str) -> Vector:
        with self._lock:
            self.calls += 1
            self.texts_embedded += 1
        return self._vector(text)

    def _vector(self, text: str) -> Vector:
        vector = np.zeros(self._dimension, dtype=np.float32)
        for word in _WORDS.findall(text.lower()):
            # hashlib, not hash(): Python's hash() of a str changes on every run
            digest = hashlib.blake2b(word.encode("utf-8"), digest_size=4).digest()
            vector[int.from_bytes(digest, "little") % self._dimension] += 1.0
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 0 else vector
