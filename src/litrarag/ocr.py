"""Tesseract adapter for the OCREngine protocol: image -> OCRResult.

Knows nothing about files, PDFs or pages. The loader decides WHEN to OCR;
this module only knows HOW.
"""

import shutil
import time
from pathlib import Path
from typing import Any

import pytesseract
from PIL import Image

from litrarag.logging_setup import get_logger
from litrarag.schemas import OCRResult

log = get_logger(__name__)


def resolve_tesseract(cmd: str | None) -> str:
    """Return the path of the Tesseract program, or raise a clear error."""
    if cmd is not None:
        if Path(cmd).is_file():
            return cmd
        raise FileNotFoundError(f"Tesseract not found at {cmd!r} (set by RAG_TESSERACT_CMD)")
    found = shutil.which("tesseract")
    if found is None:
        raise FileNotFoundError("Tesseract not found on PATH. Install it or set RAG_TESSERACT_CMD.")
    return found


def parse_tesseract_data(data: dict[str, list[Any]]) -> OCRResult:
    """Turn Tesseract's word-level output into text + confidence. Pure function."""
    lines: dict[tuple[int, int, int], list[str]] = {}
    confidences: list[float] = []
    for i, word in enumerate(data["text"]):
        conf = float(data["conf"][i])
        if conf < 0 or not str(word).strip():  # -1 = layout box, not a word
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(str(word))
        confidences.append(conf)

    text = "\n".join(" ".join(words) for words in lines.values())
    average = sum(confidences) / len(confidences) if confidences else 0.0
    return OCRResult(text=text, confidence=average, word_count=len(confidences))


class TesseractOCR:
    """OCREngine backed by the Tesseract program."""

    def __init__(self, cmd: str | None = None, lang: str = "eng") -> None:
        self.cmd = resolve_tesseract(cmd)  # fail now, not on the first page
        self.lang = lang

    def read(self, image: Image.Image) -> OCRResult:
        # pytesseract stores the program path in a module-level variable.
        # Setting it on each call keeps this instance's choice in effect.
        pytesseract.pytesseract.tesseract_cmd = self.cmd

        start = time.perf_counter()
        data = pytesseract.image_to_data(
            image.convert("L"), lang=self.lang, output_type=pytesseract.Output.DICT
        )
        result = parse_tesseract_data(data)
        ms = (time.perf_counter() - start) * 1000

        log.debug(
            "OCR %dx%d px: %d words, confidence %.1f, %.0f ms",
            image.width,
            image.height,
            result.word_count,
            result.confidence,
            ms,
        )
        return result


def _render_sample(font_size: int) -> Image.Image:
    """Draw two lines of text at a given size, to simulate different scan resolutions."""
    from PIL import ImageDraw, ImageFont

    lines = ["LitraRag reads scanned pages.", "Invoice total: 1,234.50 GBP"]
    font = ImageFont.load_default(size=font_size)
    line_height = int(font_size * 1.6)
    image = Image.new("RGB", (font_size * 18, line_height * len(lines) + font_size), "white")
    draw = ImageDraw.Draw(image)
    for n, line in enumerate(lines):
        draw.text((font_size // 2, font_size // 2 + n * line_height), line, fill="black", font=font)
    return image


def _demo() -> None:
    """Mini benchmark: how text size (~ scan DPI) affects speed and accuracy."""
    from litrarag.config import settings
    from litrarag.logging_setup import setup_logging

    setup_logging("INFO")
    engine = TesseractOCR(cmd=settings.tesseract_cmd)
    log.info("Using Tesseract at %s", engine.cmd)

    log.info("%5s | %11s | %8s | %5s | %s", "font", "image px", "ms (avg)", "conf", "text")
    for font_size in (8, 12, 20, 32, 64):
        image = _render_sample(font_size)
        result = engine.read(image)

        runs = 3
        start = time.perf_counter()
        for _ in range(runs):
            engine.read(image)
        avg_ms = (time.perf_counter() - start) * 1000 / runs
        size = f"{image.width}x{image.height}"
        first_line = result.text.splitlines()[0] if result.text else "(nothing)"
        log.info(
            "%5d | %11s | %8.0f | %5.1f | %s",
            font_size,
            size,
            avg_ms,
            result.confidence,
            first_line,
        )


if __name__ == "__main__":
    _demo()
