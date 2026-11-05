"""All data shapes passed between LitraRag stages.

Stages import from here and never from each other.
"""

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

ExtractionMethod = Literal["text", "ocr"]
HitSource = Literal["dense", "sparse", "hybrid"]
Vector = npt.NDArray[np.float32]  # one embedding, shape (dimension,)
Vectors = npt.NDArray[np.float32]  # a batch, shape (n, dimension); row i belongs to text i


def stable_id(*parts: str) -> str:
    """Deterministic ID: the same input always gives the same ID."""
    joined = "|".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def _freeze_metadata(obj: Any) -> None:
    """Replace obj.metadata with a read-only copy. Called from __post_init__."""
    # object.__setattr__ is the one allowed way to set a field on a frozen dataclass,
    # and only during construction.
    object.__setattr__(obj, "metadata", MappingProxyType(dict(obj.metadata)))


@dataclass(frozen=True)
class Page:
    """One page of text, however it was extracted."""

    page_number: int
    text: str
    method: ExtractionMethod
    ocr_confidence: float | None = None  # only set when method == "ocr"


@dataclass(frozen=True)
class OCRResult:
    """Text read from one image. Output of an OCR engine."""

    text: str
    confidence: float  # average word confidence 0-100; 0.0 if no words found
    word_count: int


@dataclass(frozen=True)
class Document:
    """A loaded file. Output of the loader."""

    doc_id: str  # hash of the file's CONTENT
    source_path: str
    file_type: str  # "pdf", "png", "txt", ...
    pages: tuple[Page, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        _freeze_metadata(self)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def ocr_page_count(self) -> int:
        return sum(1 for p in self.pages if p.method == "ocr")


@dataclass(frozen=True)
class Chunk:
    """A piece of a document small enough to embed. Output of the chunker."""

    chunk_id: str  # hash of doc_id + position
    doc_id: str
    text: str
    page_number: int
    index: int  # position within the document
    metadata: Mapping[str, Any] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        _freeze_metadata(self)


@dataclass(frozen=True)
class SearchHit:
    """One search result. Output of the retriever."""

    chunk: Chunk
    score: float
    source: HitSource  # which search found it


@dataclass(frozen=True)
class Answer:
    """The final result. Output of the generator."""

    question: str
    text: str
    hits: tuple[SearchHit, ...]  # the sources used
    model: str
    latency_ms: float


def _demo() -> None:
    from litrarag.logging_setup import get_logger, setup_logging

    setup_logging("INFO")
    log = get_logger("litrarag.schemas")

    pages = (
        Page(1, "LitraRag reads documents.", "text"),
        Page(2, "This page was scanned.", "ocr", ocr_confidence=91.5),
    )
    doc = Document(stable_id("demo file content"), "demo.pdf", "pdf", pages)
    log.info("Document %s: %d pages, %d via OCR", doc.doc_id, doc.page_count, doc.ocr_page_count)

    chunk = Chunk(stable_id(doc.doc_id, "0"), doc.doc_id, pages[0].text, 1, 0)
    log.info("Chunk %s from page %d", chunk.chunk_id, chunk.page_number)

    log.info("Same input, same ID: %s", stable_id("a") == stable_id("a"))

    try:
        chunk.text = "changed!"  # pyright: ignore[reportAttributeAccessIssue]
    except Exception as e:
        log.info("Frozen works, change blocked: %s", type(e).__name__)


if __name__ == "__main__":
    _demo()
