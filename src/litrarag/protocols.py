"""Behavior contracts between LitraRag stages.

schemas.py defines the DATA that flows between stages.
This file defines what each swappable stage must be able to DO.
"""

from typing import Protocol

from PIL import Image

from litrarag.schemas import OCRResult


class OCREngine(Protocol):
    """Anything that can read text from an image."""

    def read(self, image: Image.Image) -> OCRResult: ...
