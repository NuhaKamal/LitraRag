"""Tests for keyword_index.py: SQLite FTS5 with BM25 ranking."""

import pytest

from litrarag.keyword_index import SQLiteKeywordIndex, fts_query
from litrarag.schemas import Chunk, stable_id


@pytest.fixture
def index(tmp_path):
    store = SQLiteKeywordIndex(tmp_path / "keywords.db")
    yield store
    store.close()


def make_chunks(doc_id: str, texts: list[str]) -> list[Chunk]:
    return [
        Chunk(
            stable_id(doc_id, str(i)), doc_id, text, i + 1, i, {"source": "a.pdf", "method": "text"}
        )
        for i, text in enumerate(texts)
    ]


# --- fts_query: pure -----------------------------------------------------------


def test_fts_query_quotes_words_and_joins_with_or():
    assert fts_query("battery low") == '"battery" OR "low"'


def test_fts_query_strips_punctuation():
    assert fts_query('How does "OCR" work?') == '"How" OR "does" OR "OCR" OR "work"'


def test_fts_query_of_no_words_is_empty():
    assert fts_query("?!  ...") == ""


# --- Storing -----------------------------------------------------------------


def test_new_index_is_empty(index):
    assert index.count() == 0


def test_replace_source_stores_chunks(index):
    index.replace_source("a.pdf", make_chunks("doc1", ["one", "two", "three"]))
    assert index.count() == 3


def test_new_version_replaces_old_chunks(index):
    index.replace_source("a.pdf", make_chunks("v1", ["old alpha", "old beta", "old gamma"]))
    index.replace_source("a.pdf", make_chunks("v2", ["new alpha"]))

    assert index.count() == 1
    assert all(hit.chunk.doc_id == "v2" for hit in index.search("alpha"))


def test_replacing_one_file_leaves_others(index):
    index.replace_source("a.pdf", make_chunks("docA", ["alpha"]))
    index.replace_source("b.pdf", make_chunks("docB", ["beta", "gamma"]))
    index.replace_source("a.pdf", make_chunks("docA2", ["alpha edited"]))
    assert index.count() == 3


def test_has_document(index):
    index.replace_source("a.pdf", make_chunks("doc1", ["text"]))
    assert index.has_document("doc1")
    assert not index.has_document("unknown")


def test_index_survives_reopening(tmp_path):
    path = tmp_path / "keywords.db"
    with SQLiteKeywordIndex(path) as first:
        first.replace_source("a.pdf", make_chunks("doc1", ["persisted words"]))
    with SQLiteKeywordIndex(path) as second:
        assert second.count() == 1


# --- Searching -----------------------------------------------------------------


def test_exact_code_is_found(index):
    index.replace_source(
        "manual.pdf",
        make_chunks(
            "doc1",
            [
                "General safety instructions for the panel.",
                "Error AIQ001440 means the sensor battery is low.",
                "Battery care: replace batteries every two years.",
            ],
        ),
    )
    hits = index.search("What does AIQ001440 mean?")
    assert "AIQ001440" in hits[0].chunk.text


def test_stemming_matches_word_forms(index):
    index.replace_source(
        "plan.pdf", make_chunks("doc1", ["A weekly workout plan.", "Banana bread."])
    )
    hits = index.search("workouts")
    assert len(hits) == 1
    assert "workout" in hits[0].chunk.text


def test_rare_words_outrank_common_ones(index):
    index.replace_source(
        "a.pdf",
        make_chunks(
            "doc1",
            [
                "the panel the sensor the battery",
                "the panel the sensor the battery",
                "the panel shows error AIQ001440",
            ],
        ),
    )
    hits = index.search("the AIQ001440")
    assert "AIQ001440" in hits[0].chunk.text


def test_hits_are_complete_ranked_sparse_results(index):
    original = make_chunks("doc1", ["the only chunk about tesseract"])[0]
    index.replace_source("a.pdf", [original])

    hit = index.search("tesseract")[0]

    assert hit.chunk == original  # every field survives the round trip
    assert hit.source == "sparse"
    assert hit.score > 0  # higher = better, like the vector store


def test_k_limits_results(index):
    index.replace_source("a.pdf", make_chunks("doc1", ["word one", "word two", "word three"]))
    assert len(index.search("word", k=2)) == 2


def test_question_with_no_words_returns_nothing(index):
    index.replace_source("a.pdf", make_chunks("doc1", ["text"]))
    assert index.search("?!") == ()
