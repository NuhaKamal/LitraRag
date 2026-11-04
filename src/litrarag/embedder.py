"""Embedder stage: text -> vectors, via Ollama.

LangChain's OllamaEmbeddings does the HTTP work INSIDE this adapter.
Callers get NumPy arrays and never see LangChain types.
"""

import time
from collections.abc import Sequence
from typing import Protocol

import numpy as np
from langchain_ollama import OllamaEmbeddings

from litrarag.logging_setup import get_logger
from litrarag.schemas import Vector, Vectors

log = get_logger(__name__)

# Task prefixes some models were trained with: (document prefix, query prefix)
KNOWN_PREFIXES: dict[str, tuple[str, str]] = {
    "nomic-embed-text": ("search_document: ", "search_query: "),
}


def prefixes_for(model: str) -> tuple[str, str]:
    """Return (document_prefix, query_prefix) for a model; no prefixes if unknown."""
    base_name = model.split(":")[0]  # "nomic-embed-text:latest" -> "nomic-embed-text"
    return KNOWN_PREFIXES.get(base_name, ("", ""))


class EmbeddingError(RuntimeError):
    """The embedding service failed. The message says what to check."""


class EmbeddingsClient(Protocol):
    """The part of LangChain's embeddings interface this adapter uses."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class OllamaEmbedder:
    """Embedder backed by an Ollama server, local or remote (e.g. Colab)."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://localhost:11434",
        batch_size: int = 16,
        document_prefix: str | None = None,
        query_prefix: str | None = None,
        client: EmbeddingsClient | None = None,
    ) -> None:
        default_document, default_query = prefixes_for(model)
        self._model = model
        self.base_url = base_url
        self.batch_size = batch_size
        self.document_prefix = default_document if document_prefix is None else document_prefix
        self.query_prefix = default_query if query_prefix is None else query_prefix
        self._client: EmbeddingsClient = client or OllamaEmbeddings(model=model, base_url=base_url)
        self._dimension: int | None = None

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        """Vector length, asked from the model once, then remembered."""
        if self._dimension is None:
            self._dimension = len(self.embed_query("dimension probe"))
        return self._dimension

    def embed_documents(self, texts: Sequence[str]) -> Vectors:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        start = time.perf_counter()
        batches = []
        for first in range(0, len(texts), self.batch_size):
            batch = [self.document_prefix + t for t in texts[first : first + self.batch_size]]
            batches.append(self._embed_batch(batch))
        vectors = np.vstack(batches)

        seconds = max(time.perf_counter() - start, 1e-9)  # avoid dividing by zero
        log.info(
            "Embedded %d texts in %.0f ms (%.1f texts/s, batch size %d)",
            len(texts),
            seconds * 1000,
            len(texts) / seconds,
            self.batch_size,
        )
        return vectors

    def embed_query(self, text: str) -> Vector:
        try:
            raw = self._client.embed_query(self.query_prefix + text)
        except Exception as err:
            raise self._error(err) from err
        return np.asarray(raw, dtype=np.float32)

    def _embed_batch(self, batch: list[str]) -> Vectors:
        try:
            raw = self._client.embed_documents(batch)
        except Exception as err:
            raise self._error(err) from err
        return np.asarray(raw, dtype=np.float32)

    def _error(self, err: Exception) -> EmbeddingError:
        return EmbeddingError(
            f"Embedding with {self._model!r} at {self.base_url} failed: {err}. "
            "Is Ollama running, and is the model pulled?"
        )


def _demo() -> None:
    """Semantic check, then benchmark batch sizes."""
    from litrarag.config import settings
    from litrarag.logging_setup import setup_logging

    setup_logging("INFO")
    embedder = OllamaEmbedder(settings.embed_model, base_url=settings.embed_url)
    log.info(
        "Model %s: %d dimensions, prefixes %r / %r",
        embedder.model_name,
        embedder.dimension,
        embedder.document_prefix,
        embedder.query_prefix,
    )

    docs = embedder.embed_documents(
        ["Tesseract reads text from scanned images.", "A recipe for banana bread with walnuts."]
    )
    query = embedder.embed_query("How does OCR work?")
    similarity = docs @ query / (np.linalg.norm(docs, axis=1) * np.linalg.norm(query))
    log.info("First 5 values: %s", np.round(query[:5], 4).tolist())
    log.info("'How does OCR work?' vs OCR text: %.3f | vs banana bread: %.3f", *similarity)

    texts = [
        f"Chunk {i}: "
        + " ".join(f"Sentence {i}.{s} explains how LitraRag handles part {s}." for s in range(12))
        for i in range(64)
    ]  # ~650 characters each, like real 800-character chunks

    log.info("batch | total ms | texts/s")
    for batch_size in (64, 16, 4, 1):
        timed = OllamaEmbedder(
            settings.embed_model, base_url=settings.embed_url, batch_size=batch_size
        )
        timed.embed_documents(texts[:batch_size])  # warm-up for this batch size, not timed
        start = time.perf_counter()
        timed.embed_documents(texts)
        seconds = time.perf_counter() - start
        log.info("%5d | %8.0f | %7.1f", batch_size, seconds * 1000, len(texts) / seconds)


if __name__ == "__main__":
    _demo()
