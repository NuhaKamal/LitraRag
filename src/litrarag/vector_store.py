"""Vector store stage: keep chunk vectors in Qdrant and find the nearest ones.

Works in embedded mode (a folder, no server) and server mode (Docker).
Only open_qdrant() knows the difference.
"""

import time
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    VectorParams,
)

from litrarag.config import QdrantMode
from litrarag.logging_setup import get_logger
from litrarag.schemas import Chunk, SearchHit, Vector, Vectors

log = get_logger(__name__)

# Fixed namespace: the same chunk_id always maps to the same Qdrant point id
_ID_NAMESPACE = uuid.UUID("6f1c2a9e-3b7d-4c55-9a21-8e4f0d7b6c13")
_UPSERT_BATCH = 256


class VectorStoreLockedError(RuntimeError):
    """Embedded storage is already open in another process."""


class EmbeddingMismatchError(RuntimeError):
    """The collection was built with a different model or vector size."""


def open_qdrant(mode: QdrantMode, *, path: Path, url: str) -> QdrantClient:
    """Open Qdrant in embedded or server mode, with a clear error if storage is locked."""
    if mode == "server":
        return QdrantClient(url=url)
    path.mkdir(parents=True, exist_ok=True)
    try:
        return QdrantClient(path=str(path))
    except RuntimeError as err:
        if "already accessed" in str(err):
            raise VectorStoreLockedError(
                f"Qdrant storage at {path} is in use by another LitraRag process "
                "(probably the chat app). Close it first, or run Qdrant in Docker "
                "and set RAG_QDRANT_MODE=server."
            ) from err
        raise


def point_id(chunk_id: str) -> str:
    """Qdrant needs integer or UUID ids: derive a stable UUID from the chunk id."""
    return str(uuid.uuid5(_ID_NAMESPACE, chunk_id))


def _match(key: str, value: str) -> Filter:
    return Filter(must=[FieldCondition(key=key, match=MatchValue(value=value))])


def _chunk_from(payload: dict[str, Any]) -> Chunk:
    return Chunk(
        chunk_id=payload["chunk_id"],
        doc_id=payload["doc_id"],
        text=payload["text"],
        page_number=payload["page_number"],
        index=payload["index"],
        metadata=payload.get("metadata", {}),
    )


class QdrantVectorStore:
    """VectorStore backed by a Qdrant collection."""

    def __init__(
        self, client: QdrantClient, *, collection: str, model_name: str, dimension: int
    ) -> None:
        self.client = client
        self.collection = collection
        self.model_name = model_name
        self.dimension = dimension
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                self.collection,
                vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE),
            )
            log.info(
                "Created collection %r for %s (%d dimensions)",
                self.collection,
                self.model_name,
                self.dimension,
            )
            return

        vectors = self.client.get_collection(self.collection).config.params.vectors
        if not isinstance(vectors, VectorParams):
            raise EmbeddingMismatchError(f"Collection {self.collection!r} has an unexpected layout")
        if vectors.size != self.dimension:
            raise EmbeddingMismatchError(
                f"Collection {self.collection!r} holds {vectors.size}-dimension vectors, but "
                f"{self.model_name!r} makes {self.dimension}-dimension vectors. Re-index into "
                "a new collection (RAG_COLLECTION) or switch back to the original model."
            )
        stored = self._stored_model()
        if stored is not None and stored != self.model_name:
            raise EmbeddingMismatchError(
                f"Collection {self.collection!r} was built with {stored!r}, not "
                f"{self.model_name!r}. Re-index into a new collection (RAG_COLLECTION) "
                "or switch back to the original model."
            )

    def _stored_model(self) -> str | None:
        points, _ = self.client.scroll(self.collection, limit=1, with_payload=True)
        if not points or not points[0].payload:
            return None
        return points[0].payload.get("model")

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count

    def has_document(self, doc_id: str) -> bool:
        """Is this exact version (content hash) already indexed?"""
        found = self.client.count(
            self.collection, count_filter=_match("doc_id", doc_id), exact=True
        )
        return found.count > 0

    def replace_source(self, source_path: str, chunks: Sequence[Chunk], vectors: Vectors) -> None:
        """Remove everything stored for this file, then store its new chunks."""
        if len(chunks) != len(vectors):
            raise ValueError(f"Got {len(chunks)} chunks but {len(vectors)} vectors")
        if len(vectors) and vectors.shape[1] != self.dimension:
            raise EmbeddingMismatchError(
                f"Vectors have {vectors.shape[1]} dimensions; collection expects {self.dimension}"
            )

        start = time.perf_counter()
        self.client.delete(
            self.collection,
            points_selector=FilterSelector(filter=_match("source_path", source_path)),
        )
        points = [
            PointStruct(
                id=point_id(chunk.chunk_id),
                vector=vector.tolist(),
                payload={
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "source_path": source_path,
                    "text": chunk.text,
                    "page_number": chunk.page_number,
                    "index": chunk.index,
                    "metadata": dict(chunk.metadata),
                    "model": self.model_name,
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        for first in range(0, len(points), _UPSERT_BATCH):
            self.client.upsert(self.collection, points=points[first : first + _UPSERT_BATCH])
        log.info(
            "Stored %d chunks for %s in %.0f ms",
            len(points),
            source_path,
            (time.perf_counter() - start) * 1000,
        )

    def search(self, query: Vector, k: int = 5) -> tuple[SearchHit, ...]:
        start = time.perf_counter()
        result = self.client.query_points(
            self.collection, query=query.tolist(), limit=k, with_payload=True
        )
        hits = tuple(
            SearchHit(_chunk_from(point.payload or {}), point.score, "dense")
            for point in result.points
        )
        log.debug(
            "Searched %r: %d hits in %.1f ms",
            self.collection,
            len(hits),
            (time.perf_counter() - start) * 1000,
        )
        return hits


def _demo() -> None:
    """Index a few passages with real embeddings, then ask questions."""
    from litrarag.config import settings
    from litrarag.embedder import OllamaEmbedder
    from litrarag.embedding_cache import CachedEmbedder, SQLiteVectorCache
    from litrarag.logging_setup import setup_logging
    from litrarag.schemas import stable_id

    setup_logging("INFO")
    passages = [
        "Tesseract reads text from scanned pages and photos.",
        "LitraRag renders scanned PDF pages at 200 DPI before running OCR.",
        "Embeddings turn text into vectors, so similar meanings end up close together.",
        "The embedding cache keeps vectors in SQLite, so unchanged text is never embedded twice.",
        "Qdrant stores vectors and finds the ones nearest to a question.",
        "Banana bread needs ripe bananas, flour, butter and sugar.",
    ]
    doc_id = stable_id("vector store demo")
    chunks = [
        Chunk(
            stable_id(doc_id, str(i)), doc_id, text, 1, i, {"source": "demo.txt", "method": "text"}
        )
        for i, text in enumerate(passages)
    ]

    with SQLiteVectorCache(settings.embedding_cache_path) as cache:
        embedder = CachedEmbedder(
            OllamaEmbedder(settings.embed_model, base_url=settings.embed_url), cache
        )
        client = open_qdrant("embedded", path=settings.data_dir / "demo" / "qdrant", url="")
        try:
            store = QdrantVectorStore(
                client,
                collection="demo",
                model_name=embedder.model_name,
                dimension=embedder.dimension,
            )
            store.replace_source("demo.txt", chunks, embedder.embed_documents(passages))
            log.info("Store holds %d chunks", store.count())

            for question in (
                "How does OCR work?",
                "Why is re-ingesting fast?",
                "What do I need for baking?",
            ):
                query = embedder.embed_query(question)
                start = time.perf_counter()
                hits = store.search(query, k=2)
                log.info("Q: %s  (search %.1f ms)", question, (time.perf_counter() - start) * 1000)
                for hit in hits:
                    log.info("   %.3f  %s", hit.score, hit.chunk.text)
        finally:
            client.close()


if __name__ == "__main__":
    _demo()
