"""Loader stage: file -> Document.

Decides WHEN to OCR (pages without a usable text layer) and delegates HOW
to an injected OCREngine. Knows nothing about Tesseract.
"""

import hashlib
import io
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pymupdf
from PIL import Image

from litrarag.logging_setup import get_logger
from litrarag.protocols import OCREngine
from litrarag.schemas import Document, OCRResult, Page

log = get_logger(__name__)

PDF_TYPES = {".pdf"}
IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
TEXT_TYPES = {".txt", ".md"}


class UnsupportedFileError(ValueError):
    """Raised for file types LitraRag cannot load."""


def file_id(path: Path) -> str:
    """Hash of the file's CONTENT, streamed so large files never sit fully in memory."""
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()[:16]


def page_text(page: pymupdf.Page) -> str:
    """Return a page's text layer as a str, checking PyMuPDF's loosely typed result."""
    text = page.get_text("text")
    if not isinstance(text, str):
        raise TypeError(f"PyMuPDF returned {type(text).__name__}, expected str")
    return text


class DocumentLoader:
    """Turns files into Documents, using OCR only for pages that need it."""

    def __init__(
        self,
        ocr: OCREngine,
        *,
        ocr_dpi: int = 200,
        ocr_min_chars: int = 30,
        max_workers: int = 2,
    ) -> None:
        self.ocr = ocr
        self.ocr_dpi = ocr_dpi
        self.ocr_min_chars = ocr_min_chars
        self.max_workers = max_workers

    def load(self, path: Path) -> Document:
        suffix = path.suffix.lower()
        start = time.perf_counter()

        if suffix in PDF_TYPES:
            pages = self._load_pdf(path)
        elif suffix in IMAGE_TYPES:
            pages = (self._load_image(path),)
        elif suffix in TEXT_TYPES:
            text = path.read_text(encoding="utf-8", errors="replace")
            pages = (Page(1, text, "text"),)
        else:
            raise UnsupportedFileError(f"Cannot load {path.name!r}: unsupported type {suffix!r}")

        doc = Document(
            doc_id=file_id(path),
            source_path=str(path),
            file_type=suffix.lstrip("."),
            pages=pages,
            metadata={"file_name": path.name, "size_bytes": path.stat().st_size},
        )
        ms = (time.perf_counter() - start) * 1000
        log.info(
            "Loaded %s: %d pages (%d via OCR) in %.0f ms",
            path.name,
            doc.page_count,
            doc.ocr_page_count,
            ms,
        )
        return doc

    def _load_image(self, path: Path) -> Page:
        with Image.open(path) as image:
            result = self.ocr.read(image)
        return Page(1, result.text, "ocr", result.confidence)

    def _load_pdf(self, path: Path) -> tuple[Page, ...]:
        with pymupdf.open(path) as pdf:
            texts = [page_text(page) for page in pdf]
            scanned = [i for i, text in enumerate(texts) if len(text.strip()) < self.ocr_min_chars]
            ocr_results = self._ocr_pages(pdf, scanned)

        pages = []
        for i, text in enumerate(texts):
            if i in ocr_results:
                result = ocr_results[i]
                pages.append(Page(i + 1, result.text, "ocr", result.confidence))
            else:
                pages.append(Page(i + 1, text, "text"))
        return tuple(pages)

    def _ocr_pages(self, pdf: pymupdf.Document, indexes: Sequence[int]) -> dict[int, OCRResult]:
        """OCR pages in parallel, holding at most `max_workers` page images in memory."""
        results: dict[int, OCRResult] = {}
        if not indexes:
            return results

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            for start in range(0, len(indexes), self.max_workers):
                batch = indexes[start : start + self.max_workers]
                # Render in this thread: PyMuPDF is not thread-safe
                images = [self._render(pdf[i]) for i in batch]
                for i, result in zip(batch, pool.map(self.ocr.read, images), strict=True):
                    results[i] = result
        return results

    def _render(self, page: pymupdf.Page) -> Image.Image:
        pixmap = page.get_pixmap(dpi=self.ocr_dpi, colorspace=pymupdf.csGRAY)
        return Image.frombytes("L", (pixmap.width, pixmap.height), pixmap.samples)


def _make_demo_pdf(path: Path, text_pages: int, scanned_pages: int) -> None:
    """A PDF with typed pages and image-only 'scanned' pages."""
    from PIL import ImageDraw, ImageFont

    pdf = pymupdf.open()
    for n in range(text_pages):
        pdf.new_page().insert_text((72, 72), f"Typed page {n + 1}: read from the text layer.")

    font = ImageFont.load_default(size=36)
    for n in range(scanned_pages):
        image = Image.new("L", (1200, 300), 255)
        ImageDraw.Draw(image).text((40, 100), f"Scanned page {n + 1} needs OCR", fill=0, font=font)
        png = io.BytesIO()
        image.save(png, format="PNG")
        pdf.new_page().insert_image(pymupdf.Rect(72, 72, 540, 189), stream=png.getvalue())

    pdf.save(path)
    pdf.close()


def _demo() -> None:
    """Load a mixed PDF, then benchmark OCR worker counts."""
    from litrarag.config import settings
    from litrarag.logging_setup import setup_logging
    from litrarag.ocr import TesseractOCR

    setup_logging("INFO")
    scanned = 8
    path = settings.data_dir / "demo" / "loader_demo.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    _make_demo_pdf(path, text_pages=4, scanned_pages=scanned)

    engine = TesseractOCR(cmd=settings.tesseract_cmd)
    limits = settings.limits

    doc = DocumentLoader(engine, ocr_dpi=limits.ocr_dpi).load(path)  # warm-up
    for page in doc.pages[2:6]:
        log.info("Page %d [%s]: %s", page.page_number, page.method, page.text.strip()[:50])

    log.info("workers | total ms | scanned pages/s")
    for workers in (1, 2, 4, 6):
        loader = DocumentLoader(
            engine,
            ocr_dpi=limits.ocr_dpi,
            ocr_min_chars=limits.ocr_min_chars,
            max_workers=workers,
        )
        start = time.perf_counter()
        loader.load(path)
        ms = (time.perf_counter() - start) * 1000
        log.info("%7d | %8.0f | %15.1f", workers, ms, scanned / (ms / 1000))


if __name__ == "__main__":
    _demo()
