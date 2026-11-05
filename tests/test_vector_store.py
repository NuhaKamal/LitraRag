"""Tests for vector_store.py, using Qdrant's in-memory mode and FakeEmbedder."""

import pytest
from qdrant_client import QdrantClient

from litrarag.schemas import Chunk, stable_id
from litrarag.vector_store import (
    EmbeddingMismatchError,
    QdrantVectorStore,
    VectorStoreLockedError,
    open_qdrant,
)
from tests.fakes import FakeEmbedder


@pytest.fixture
def embedder():
    return FakeEmbedder(dimension=256)


@pytest.fixture
def client():
    qdrant = QdrantClient(":memory:")
    yield qdrant
    qdrant.close()


@pytest.fixture
def store(client, embedder):
    return QdrantVectorStore(
        client, collection="test", model_name=embedder.model_name, dimension=embedder.dimension
    )


def make_chunks(doc_id: str, texts: list[str]) -> list[Chunk]:
    return [
        Chunk(stable_id(doc_id, str(i)), doc_id, text, 1, i, {"source": "a.pdf", "method": "text"})
        for i, text in enumerate(texts)
    ]


def add(store, embedder, path: str, doc_id: str, texts: list[str]) -> list[Chunk]:
    chunks = make_chunks(doc_id, texts)
    store.replace_source(path, chunks, embedder.embed_documents(texts))
    return chunks


# --- Storing -------------------------------------------------------------------


def test_new_store_is_empty(store):
    assert store.count() == 0


def test_replace_source_stores_chunks(store, embedder):
    add(store, embedder, "a.pdf", "doc1", ["one", "two", "three"])
    assert store.count() == 3


def test_storing_same_file_twice_does_not_duplicate(store, embedder):
    add(store, embedder, "a.pdf", "doc1", ["one", "two"])
    add(store, embedder, "a.pdf", "doc1", ["one", "two"])
    assert store.count() == 2


def test_new_version_replaces_old_chunks(store, embedder):
    add(store, embedder, "a.pdf", "v1", ["old one", "old two", "old three"])
    add(store, embedder, "a.pdf", "v2", ["new one", "new two"])  # edited file: new doc_id

    assert store.count() == 2
    hits = store.search(embedder.embed_query("old one"), k=5)
    assert all(hit.chunk.doc_id == "v2" for hit in hits)


def test_replacing_one_file_leaves_others(store, embedder):
    add(store, embedder, "a.pdf", "docA", ["alpha text"])
    add(store, embedder, "b.pdf", "docB", ["beta text", "gamma text"])
    add(store, embedder, "a.pdf", "docA2", ["alpha edited"])
    assert store.count() == 3


def test_has_document(store, embedder):
    add(store, embedder, "a.pdf", "doc1", ["text"])
    assert store.has_document("doc1")
    assert not store.has_document("unknown")


def test_mismatched_lengths_are_rejected(store, embedder):
    chunks = make_chunks("doc1", ["one", "two"])
    with pytest.raises(ValueError, match="2 chunks"):
        store.replace_source("a.pdf", chunks, embedder.embed_documents(["only one"]))


# --- Searching -----------------------------------------------------------------


def test_search_returns_nearest_first(store, embedder):
    add(
        store,
        embedder,
        "a.pdf",
        "doc1",
        [
            "ocr reads scanned pages",
            "banana bread recipe with walnuts",
            "tesseract is an ocr engine for scanned pages",
        ],
    )
    hits = store.search(embedder.embed_query("ocr scanned pages"), k=3)

    assert "ocr" in hits[0].chunk.text
    assert hits[0].score >= hits[1].score >= hits[2].score
    assert "banana" in hits[-1].chunk.text


def test_search_returns_complete_chunks(store, embedder):
    original = add(store, embedder, "a.pdf", "doc1", ["the only chunk"])[0]
    hit = store.search(embedder.embed_query("the only chunk"), k=1)[0]

    assert hit.chunk == original  # every field survives the round trip
    assert hit.source == "dense"


def test_k_limits_results(store, embedder):
    add(store, embedder, "a.pdf", "doc1", ["one", "two", "three", "four"])
    assert len(store.search(embedder.embed_query("one"), k=2)) == 2


# --- Safety checks -------------------------------------------------------------


def test_different_dimension_is_refused(client, store):
    with pytest.raises(EmbeddingMismatchError, match="dimension"):
        QdrantVectorStore(client, collection="test", model_name="fake-embedder", dimension=128)


def test_different_model_is_refused(client, store, embedder):
    add(store, embedder, "a.pdf", "doc1", ["text"])
    with pytest.raises(EmbeddingMismatchError, match="built with"):
        QdrantVectorStore(client, collection="test", model_name="another-model", dimension=256)


def test_second_embedded_client_gets_clear_error(tmp_path):
    first = open_qdrant("embedded", path=tmp_path / "qdrant", url="")
    try:
        with pytest.raises(VectorStoreLockedError, match="RAG_QDRANT_MODE=server"):
            open_qdrant("embedded", path=tmp_path / "qdrant", url="")
    finally:
        first.close()
