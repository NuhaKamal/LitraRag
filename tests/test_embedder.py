"""Tests for embedder.py. A recording client replaces Ollama for unit tests."""

import numpy as np
import pytest

from litrarag.embedder import EmbeddingError, OllamaEmbedder, prefixes_for
from tests.helpers import needs_ollama_embeddings


class RecordingClient:
    """Stands in for LangChain's OllamaEmbeddings: records calls, returns tiny vectors."""

    def __init__(self, fail: bool = False) -> None:
        self.batches: list[list[str]] = []
        self.queries: list[str] = []
        self.fail = fail

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.fail:
            raise ConnectionError("connection refused")
        self.batches.append(texts)
        return [[float(len(t)), 1.0, 0.0] for t in texts]  # first value = text length

    def embed_query(self, text: str) -> list[float]:
        if self.fail:
            raise ConnectionError("connection refused")
        self.queries.append(text)
        return [float(len(text)), 0.0, 1.0]


def make_embedder(client: RecordingClient, **kwargs) -> OllamaEmbedder:
    return OllamaEmbedder("nomic-embed-text", client=client, **kwargs)


# --- Prefixes ---------------------------------------------------------------


def test_prefixes_for_known_model_ignores_tag():
    assert prefixes_for("nomic-embed-text:latest") == ("search_document: ", "search_query: ")


def test_documents_get_document_prefix():
    client = RecordingClient()
    make_embedder(client).embed_documents(["hello"])
    assert client.batches == [["search_document: hello"]]


def test_query_gets_query_prefix():
    client = RecordingClient()
    make_embedder(client).embed_query("what is ocr?")
    assert client.queries == ["search_query: what is ocr?"]


def test_unknown_model_gets_no_prefix():
    client = RecordingClient()
    OllamaEmbedder("some-other-model", client=client).embed_documents(["hi"])
    assert client.batches == [["hi"]]


def test_explicit_prefix_overrides_default():
    client = RecordingClient()
    make_embedder(client, document_prefix="").embed_documents(["hi"])
    assert client.batches == [["hi"]]


# --- Batching -----------------------------------------------------------------


def test_texts_are_split_into_batches():
    client = RecordingClient()
    vectors = make_embedder(client, batch_size=2).embed_documents(["a", "b", "c", "d", "e"])
    assert [len(batch) for batch in client.batches] == [2, 2, 1]
    assert vectors.shape == (5, 3)


def test_batching_keeps_text_order():
    client = RecordingClient()
    texts = ["a", "bb", "ccc", "dddd", "eeeee"]
    vectors = make_embedder(client, batch_size=2, document_prefix="").embed_documents(texts)
    assert vectors[:, 0].tolist() == [1, 2, 3, 4, 5]  # row i belongs to text i


def test_vectors_are_float32():
    vectors = make_embedder(RecordingClient()).embed_documents(["a"])
    assert vectors.dtype == np.float32


# --- Dimension and errors ----------------------------------------------------


def test_dimension_is_probed_once():
    client = RecordingClient()
    embedder = make_embedder(client)
    assert embedder.dimension == 3
    assert embedder.dimension == 3
    assert len(client.queries) == 1


def test_service_failure_gives_clear_error():
    with pytest.raises(EmbeddingError, match="Is Ollama running"):
        make_embedder(RecordingClient(fail=True)).embed_documents(["x"])


# --- Real Ollama ----------------------------------------------------------------


@pytest.mark.integration
@needs_ollama_embeddings
def test_real_ollama_puts_related_text_closer():
    embedder = OllamaEmbedder("nomic-embed-text")
    docs = embedder.embed_documents(
        ["Tesseract reads text from scanned images.", "A recipe for banana bread with walnuts."]
    )
    query = embedder.embed_query("How does OCR work?")

    similarity = docs @ query / (np.linalg.norm(docs, axis=1) * np.linalg.norm(query))

    assert docs.shape == (2, embedder.dimension)
    assert similarity[0] > similarity[1]
