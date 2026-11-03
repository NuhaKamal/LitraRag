"""Chunker stage: Document -> Chunks.

LangChain's RecursiveCharacterTextSplitter does the splitting INSIDE this adapter.
Callers only ever see LitraRag's own Chunk schema.
"""

import re
import time

from langchain_text_splitters import RecursiveCharacterTextSplitter

from litrarag.logging_setup import get_logger
from litrarag.schemas import Chunk, Document, stable_id

log = get_logger(__name__)

_SPACES = re.compile(r"[ \t]+")
_BLANK_LINES = re.compile(r"\n\s*\n+")


def normalize_text(text: str) -> str:
    """Clean extracted text: one space between words, at most one blank line. Pure."""
    text = text.replace("\r\n", "\n")
    lines = (_SPACES.sub(" ", line).strip() for line in text.split("\n"))
    text = "\n".join(lines)
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip()


class LangChainChunker:
    """Chunker that splits each page at paragraph, line, then word boundaries."""

    def __init__(self, *, chunk_size: int = 800, chunk_overlap: int = 100) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Built once and reused for every document
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def split(self, doc: Document) -> tuple[Chunk, ...]:
        start = time.perf_counter()
        source = doc.metadata.get("file_name", doc.source_path)
        chunks: list[Chunk] = []

        for page in doc.pages:
            text = normalize_text(page.text)
            if not text:
                continue
            for piece in self._splitter.split_text(text):
                index = len(chunks)
                chunks.append(
                    Chunk(
                        chunk_id=stable_id(doc.doc_id, str(index)),
                        doc_id=doc.doc_id,
                        text=piece,
                        page_number=page.page_number,
                        index=index,
                        metadata={"source": source, "method": page.method},
                    )
                )

        ms = (time.perf_counter() - start) * 1000
        log.info(
            "Chunked %s: %d pages -> %d chunks in %.1f ms",
            source,
            doc.page_count,
            len(chunks),
            ms,
        )
        return tuple(chunks)


def _demo() -> None:
    """Show a chunk, then compare chunk sizes: count, average length, overlap cost."""
    from litrarag.config import settings
    from litrarag.logging_setup import setup_logging
    from litrarag.schemas import Page

    setup_logging("INFO")

    paragraphs = [
        " ".join(f"Sentence {p}.{s} explains how LitraRag handles part {s}." for s in range(6))
        for p in range(150)
    ]
    pages = tuple(
        Page(n + 1, "\n\n".join(paragraphs[n * 10 : (n + 1) * 10]), "text") for n in range(15)
    )
    doc = Document(stable_id("chunker demo"), "demo.txt", "txt", pages, {"file_name": "demo.txt"})
    total_chars = sum(len(normalize_text(p.text)) for p in pages)

    default = LangChainChunker(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    first = default.split(doc)[0]
    log.info("First chunk (page %d, %d chars):\n%s", first.page_number, len(first.text), first.text)

    log.info("%5s | %7s | %6s | %9s | %s", "size", "overlap", "chunks", "avg chars", "extra stored")
    for size in (300, 500, 800, 1200):
        overlap = size // 8
        chunks = LangChainChunker(chunk_size=size, chunk_overlap=overlap).split(doc)
        stored = sum(len(c.text) for c in chunks)
        log.info(
            "%5d | %7d | %6d | %9.0f | %+.0f%%",
            size,
            overlap,
            len(chunks),
            stored / len(chunks),
            (stored / total_chars - 1) * 100,
        )


if __name__ == "__main__":
    _demo()
