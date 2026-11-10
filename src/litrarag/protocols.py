"""Behavior contracts between LitraRag stages.

schemas.py defines the DATA that flows between stages.
This file defines what each swappable stage must be able to DO.
"""

from collections.abc import Iterator, Sequence
from typing import Protocol

from PIL import Image

from litrarag.schemas import Answer, Chunk, Document, OCRResult, SearchHit, Vector, Vectors


class OCREngine(Protocol):
    """Anything that can read text from an image."""

    def read(self, image: Image.Image) -> OCRResult: ...


class Chunker(Protocol):
    """Anything that can split a Document into Chunks."""

    def split(self, doc: Document) -> tuple[Chunk, ...]: ...


class Embedder(Protocol):
    """Anything that turns text into vectors.

    Documents and queries are separate methods because some models
    (like nomic-embed-text) expect a different prefix for each.
    """

    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> Vectors: ...

    def embed_query(self, text: str) -> Vector: ...


class VectorStore(Protocol):
    """Anything that stores chunk vectors and finds the nearest ones."""

    def count(self) -> int: ...

    def has_document(self, doc_id: str) -> bool: ...

    def replace_source(
        self, source_path: str, chunks: Sequence[Chunk], vectors: Vectors
    ) -> None: ...

    def search(self, query: Vector, k: int = 5) -> tuple[SearchHit, ...]: ...


class KeywordIndex(Protocol):
    """Anything that finds chunks by their exact words."""

    def count(self) -> int: ...

    def has_document(self, doc_id: str) -> bool: ...

    def replace_source(self, source_path: str, chunks: Sequence[Chunk]) -> None: ...

    def search(self, query: str, k: int = 5) -> tuple[SearchHit, ...]: ...


class Retriever(Protocol):
    """Anything that finds the chunks most relevant to a question."""

    def retrieve(self, question: str, k: int = 5) -> tuple[SearchHit, ...]: ...


class Generator(Protocol):
    """Anything that answers a question from retrieved chunks."""

    model_name: str

    def answer(self, question: str, hits: Sequence[SearchHit]) -> Answer: ...

    def stream(self, question: str, hits: Sequence[SearchHit]) -> Iterator[str]: ...
