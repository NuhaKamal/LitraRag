"""Check that implementations satisfy their Protocols."""

import numpy as np
from PIL import Image

from litrarag.chunker import LangChainChunker
from litrarag.embedder import OllamaEmbedder
from litrarag.embedding_cache import CachedEmbedder
from litrarag.ocr import TesseractOCR
from litrarag.protocols import Chunker, Embedder, OCREngine
from tests.fakes import FakeEmbedder, FakeOCR


def _ollama_embedder_fits_protocol(embedder: OllamaEmbedder) -> Embedder:
    return embedder  # never called: pyright checks this line statically


def test_fake_ocr_satisfies_protocol():
    engine: OCREngine = FakeOCR()  # pyright verifies this assignment
    result = engine.read(Image.new("RGB", (10, 10)))
    assert result.text == "fake ocr text"
    assert result.word_count == 3


def test_fake_ocr_counts_calls():
    fake = FakeOCR()
    image = Image.new("RGB", (10, 10))
    fake.read(image)
    fake.read(image)
    assert fake.calls == 2


def _tesseract_fits_protocol(engine: TesseractOCR) -> OCREngine:
    return engine  # never called: pyright checks this line statically


def _langchain_chunker_fits_protocol(chunker: LangChainChunker) -> Chunker:
    return chunker  # never called: pyright checks this line statically


def cosine(a, b) -> float:
    return float(np.dot(a, b))  # vectors are normalized, so dot product = cosine


def test_fake_embedder_satisfies_protocol():
    embedder: Embedder = FakeEmbedder(dimension=32)
    vectors = embedder.embed_documents(["one text", "another text"])
    assert vectors.shape == (2, 32)
    assert vectors.dtype == np.float32


def test_fake_embedder_is_deterministic():
    embedder = FakeEmbedder()
    assert np.array_equal(embedder.embed_query("same words"), embedder.embed_query("same words"))


def test_fake_embedder_similar_texts_are_closer():
    embedder = FakeEmbedder(dimension=256)
    question = embedder.embed_query("how does ocr work")
    related = embedder.embed_query("ocr work explained")
    unrelated = embedder.embed_query("banana bread recipe")
    assert cosine(question, related) > cosine(question, unrelated)


def test_fake_embedder_counts_texts():
    embedder = FakeEmbedder()
    embedder.embed_documents(["a", "b", "c"])
    embedder.embed_query("d")
    assert embedder.calls == 2
    assert embedder.texts_embedded == 4


def test_fake_embedder_handles_empty_batch():
    assert FakeEmbedder(dimension=8).embed_documents([]).shape == (0, 8)


def _cached_embedder_fits_protocol(embedder: CachedEmbedder) -> Embedder:
    return embedder  # never called: pyright checks this line statically
