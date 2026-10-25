"""Unit tests for schemas.py."""

import dataclasses

import pytest

from litrarag.schemas import Chunk, Document, Page, stable_id


def make_chunk(**overrides) -> Chunk:
    """Build a valid Chunk; tests override only the fields they care about."""
    values = {"chunk_id": "c1", "doc_id": "d1", "text": "hello", "page_number": 1, "index": 0}
    values.update(overrides)
    return Chunk(**values)


# --- stable_id ---------------------------------------------------------------


def test_stable_id_is_deterministic():
    assert stable_id("a", "b") == stable_id("a", "b")


def test_stable_id_differs_for_different_input():
    assert stable_id("a") != stable_id("b")


def test_stable_id_keeps_parts_separate():
    # Without the "|" separator, both would hash "abc" and collide
    assert stable_id("ab", "c") != stable_id("a", "bc")


# --- Document ----------------------------------------------------------------


def test_document_counts_ocr_pages():
    pages = (Page(1, "typed", "text"), Page(2, "scanned", "ocr", ocr_confidence=90.0))
    doc = Document("d1", "a.pdf", "pdf", pages)
    assert doc.page_count == 2
    assert doc.ocr_page_count == 1


# --- Immutability (Issue 1) --------------------------------------------------


def test_chunk_fields_cannot_be_reassigned():
    chunk = make_chunk()
    with pytest.raises(dataclasses.FrozenInstanceError):
        chunk.text = "changed"


def test_chunk_metadata_cannot_be_modified():
    chunk = make_chunk(metadata={"source": "a.pdf"})
    with pytest.raises(TypeError):
        chunk.metadata["source"] = "b.pdf"


def test_chunk_is_hashable():
    chunk = make_chunk(metadata={"source": "a.pdf"})
    assert len({chunk, chunk}) == 1  # a set needs hashing


def test_chunk_copies_metadata_from_caller():
    original = {"source": "a.pdf"}
    chunk = make_chunk(metadata=original)
    original["source"] = "b.pdf"  # the caller changes their own dict later
    assert chunk.metadata["source"] == "a.pdf"
