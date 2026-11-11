"""Embedding cache: never embed the same text with the same model twice.

CachedEmbedder wraps any Embedder and is itself an Embedder, so the rest of
LitraRag cannot tell whether a cache is in use (the decorator pattern).
"""

import hashlib
import sqlite3
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from types import TracebackType
from typing import Literal, Self

import numpy as np

from litrarag.logging_setup import get_logger
from litrarag.protocols import Embedder
from litrarag.schemas import Vector, Vectors

log = get_logger(__name__)

EmbedKind = Literal["document", "query"]
_LOOKUP_BATCH = 500  # stay well below SQLite's limit on parameters per query


def cache_key(model: str, kind: EmbedKind, text: str) -> str:
    """Key for one vector: the same model, kind and text always give the same key."""
    raw = "\x1f".join((model, kind, text)).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:32]  # 128 bits: collisions practically impossible


class SQLiteVectorCache:
    """Vectors stored as raw float32 bytes in a SQLite file."""

    def __init__(self, path: Path | str) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        # Streamlit may call us from different threads; Chat serializes access with a lock
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")  # faster writes, readers not blocked
        self._conn.execute("PRAGMA synchronous=NORMAL")  # safe with WAL, fewer disk syncs
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS vectors ("
            " key TEXT PRIMARY KEY, model TEXT NOT NULL, vector BLOB NOT NULL)"
        )
        self._conn.commit()

    def get_many(self, keys: Sequence[str]) -> dict[str, Vector]:
        """Return the cached vectors for whichever keys exist."""
        found: dict[str, Vector] = {}
        for start in range(0, len(keys), _LOOKUP_BATCH):
            batch = keys[start : start + _LOOKUP_BATCH]
            placeholders = ",".join("?" * len(batch))  # "?,?,?": values are never in the SQL text
            rows = self._conn.execute(
                f"SELECT key, vector FROM vectors WHERE key IN ({placeholders})", batch
            )
            for key, blob in rows:
                found[key] = np.frombuffer(blob, dtype=np.float32)
        return found

    def put_many(self, items: Iterable[tuple[str, str, Vector]]) -> None:
        """Store (key, model, vector) items in a single transaction."""
        with self._conn:  # one transaction: one disk sync instead of one per vector
            self._conn.executemany(
                "INSERT OR REPLACE INTO vectors (key, model, vector) VALUES (?, ?, ?)",
                (
                    (key, model, np.asarray(vector, dtype=np.float32).tobytes())
                    for key, model, vector in items
                ),
            )

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class CachedEmbedder:
    """An Embedder that remembers every vector it has produced."""

    def __init__(self, inner: Embedder, cache: SQLiteVectorCache) -> None:
        self.inner = inner
        self.cache = cache
        self.hits = 0
        self.misses = 0

    @property
    def model_name(self) -> str:
        return self.inner.model_name

    @property
    def dimension(self) -> int:
        return self.inner.dimension

    def embed_documents(self, texts: Sequence[str]) -> Vectors:
        start = time.perf_counter()
        misses_before = self.misses
        vectors = self._embed(texts, "document")
        new = self.misses - misses_before
        log.info(
            "Embedded %d texts: %d from cache, %d new, in %.0f ms",
            len(texts),
            len(texts) - new,
            new,
            (time.perf_counter() - start) * 1000,
        )
        return vectors

    def embed_query(self, text: str) -> Vector:
        return self._embed([text], "query")[0]

    def _embed(self, texts: Sequence[str], kind: EmbedKind) -> Vectors:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        model = self.inner.model_name
        keys = [cache_key(model, kind, text) for text in texts]
        found = self.cache.get_many(list(dict.fromkeys(keys)))  # unique keys, order kept

        # Texts still missing, each unique text only once
        missing: dict[str, str] = {}
        for key, text in zip(keys, texts, strict=True):
            if key not in found and key not in missing:
                missing[key] = text

        if missing:
            new_texts = list(missing.values())
            if kind == "document":
                new_vectors = self.inner.embed_documents(new_texts)
            else:
                new_vectors = np.stack([self.inner.embed_query(t) for t in new_texts])
            new_by_key = dict(zip(missing, new_vectors, strict=True))
            self.cache.put_many((key, model, vector) for key, vector in new_by_key.items())
            found.update(new_by_key)

        self.hits += len(texts) - len(missing)
        self.misses += len(missing)
        return np.stack([found[key] for key in keys])  # back in the caller's order


def _demo() -> None:
    """Cold cache, warm cache, then a few changed texts, against real Ollama."""
    from litrarag.config import settings
    from litrarag.embedder import OllamaEmbedder
    from litrarag.logging_setup import setup_logging

    setup_logging("INFO")
    path = settings.data_dir / "demo" / "embedding_cache_demo.db"
    for suffix in ("", "-wal", "-shm"):  # start with an empty cache
        Path(f"{path}{suffix}").unlink(missing_ok=True)

    texts = [
        f"Chunk {i}: "
        + " ".join(f"Sentence {i}.{s} explains how LitraRag handles part {s}." for s in range(12))
        for i in range(32)
    ]
    edited = texts[:28] + [text + " This chunk was edited." for text in texts[28:]]

    inner = OllamaEmbedder(
        settings.embed_model,
        base_url=settings.embed_url,
        batch_size=settings.limits.embed_batch_size,
    )
    with SQLiteVectorCache(path) as cache:
        embedder = CachedEmbedder(inner, cache)
        for label, batch in (
            ("cold: empty cache", texts),
            ("warm: same texts", texts),
            ("4 of 32 texts edited", edited),
        ):
            start = time.perf_counter()
            embedder.embed_documents(batch)
            log.info("%-22s %8.0f ms", label, (time.perf_counter() - start) * 1000)
        log.info(
            "Cache holds %d vectors (%d hits, %d misses)",
            cache.count(),
            embedder.hits,
            embedder.misses,
        )


if __name__ == "__main__":
    _demo()
