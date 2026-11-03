"""Tests for chunker.py. No external services: plain Python and LangChain."""

from litrarag.chunker import LangChainChunker, normalize_text
from litrarag.schemas import Document, ExtractionMethod, Page


def make_doc(*page_texts: str, method: ExtractionMethod = "text") -> Document:
    pages = tuple(Page(n + 1, text, method) for n, text in enumerate(page_texts))
    return Document("doc1", "notes.pdf", "pdf", pages, {"file_name": "notes.pdf"})


def numbered_words(count: int) -> str:
    return " ".join(f"w{i}" for i in range(count))


# --- normalize_text: pure ----------------------------------------------------


def test_normalize_collapses_spaces_and_blank_lines():
    assert normalize_text("a   b\t c\n\n\n\nd  \n e") == "a b c\n\nd\ne"


def test_normalize_keeps_paragraph_breaks():
    assert normalize_text("para one\n\npara two") == "para one\n\npara two"


def test_normalize_handles_windows_line_endings():
    assert normalize_text("line one\r\nline two") == "line one\nline two"


# --- Splitting ----------------------------------------------------------------


def test_short_page_becomes_one_chunk():
    chunks = LangChainChunker(chunk_size=200, chunk_overlap=20).split(make_doc("Short page."))
    assert len(chunks) == 1
    assert chunks[0].text == "Short page."


def test_long_page_is_split_within_size():
    chunks = LangChainChunker(chunk_size=100, chunk_overlap=10).split(make_doc(numbered_words(500)))
    assert len(chunks) > 1
    assert all(len(c.text) <= 100 for c in chunks)


def test_consecutive_chunks_overlap():
    chunks = LangChainChunker(chunk_size=100, chunk_overlap=20).split(make_doc(numbered_words(300)))
    last_word_of_first = chunks[0].text.split()[-1]
    assert last_word_of_first in chunks[1].text.split()


def test_empty_pages_are_skipped():
    chunks = LangChainChunker().split(make_doc("Page one text.", "   \n\n  ", "Page three text."))
    assert [c.page_number for c in chunks] == [1, 3]


# --- Chunk fields ---------------------------------------------------------------


def test_chunks_record_their_page_number():
    doc = make_doc("First page.", numbered_words(300))
    chunks = LangChainChunker(chunk_size=100, chunk_overlap=10).split(doc)
    assert chunks[0].page_number == 1
    assert all(c.page_number == 2 for c in chunks[1:])


def test_index_is_continuous_across_pages():
    doc = make_doc(numbered_words(200), numbered_words(200))
    chunks = LangChainChunker(chunk_size=100, chunk_overlap=10).split(doc)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_chunk_ids_are_deterministic_and_unique():
    doc = make_doc(numbered_words(400))
    chunker = LangChainChunker(chunk_size=100, chunk_overlap=10)
    first, second = chunker.split(doc), chunker.split(doc)

    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert len({c.chunk_id for c in first}) == len(first)


def test_metadata_records_source_and_method():
    chunks = LangChainChunker().split(make_doc("Scanned text here.", method="ocr"))
    assert chunks[0].metadata["source"] == "notes.pdf"
    assert chunks[0].metadata["method"] == "ocr"
