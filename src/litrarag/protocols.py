"""Behavior contracts between LitraRag stages.

schemas.py defines the DATA that flows between stages.
This file defines what each swappable stage must be able to DO.
"""

from typing import Protocol

from PIL import Image

from litrarag.schemas import Chunk, Document, OCRResult


class OCREngine(Protocol):
    """Anything that can read text from an image."""

    def read(self, image: Image.Image) -> OCRResult: ...


class Chunker(Protocol):
    """Anything that can split a Document into Chunks."""

    def split(self, doc: Document) -> tuple[Chunk, ...]: ...
