"""Keyword index: exact-word search with SQLite FTS5 and BM25 ranking.

Complements the vector store: embeddings match MEANING, keywords match EXACT
terms such as part numbers, error codes and names.
"""

import json
import re
import sqlite3
import time
from collections.abc import Sequence
from pathlib import Path
from types import TracebackType
from typing import Self

from litrarag.logging_setup import get_logger
from litrarag.schemas import Chunk, SearchHit

log = get_logger(__name__)

_WORDS = re.compile(r"\w+")


def fts_query(text: str) -> str:
    """Turn free text into a safe FTS5 query: each word quoted, joined by OR. Pure."""
    return " OR ".join(f'"{word}"' for word in _WORDS.findall(text))


class SQLiteKeywordIndex:
    """KeywordIndex backed by an SQLite FTS5 table."""

    def __init__(self, path: Path | str) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        # Streamlit may call us from different threads; Chat serializes access with a lock
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        # Only `text` is searched; UNINDEXED columns are stored alongside it
        self._conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5("
            " text, chunk_id UNINDEXED, doc_id UNINDEXED, source_path UNINDEXED,"
            " page_number UNINDEXED, idx UNINDEXED, metadata UNINDEXED,"
            " tokenize = 'porter unicode61 remove_diacritics 2')"
        )
        self._conn.commit()

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def has_document(self, doc_id: str) -> bool:
        row = self._conn.execute("SELECT 1 FROM chunks WHERE doc_id = ? LIMIT 1", (doc_id,))
        return row.fetchone() is not None

    def replace_source(self, source_path: str, chunks: Sequence[Chunk]) -> None:
        """Remove everything stored for this file, then store its new chunks."""
        start = time.perf_counter()
        with self._conn:  # one transaction: delete + insert succeed or fail together
            self._conn.execute("DELETE FROM chunks WHERE source_path = ?", (source_path,))
            self._conn.executemany(
                "INSERT INTO chunks"
                " (text, chunk_id, doc_id, source_path, page_number, idx, metadata)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    (
                        c.text,
                        c.chunk_id,
                        c.doc_id,
                        source_path,
                        c.page_number,
                        c.index,
                        json.dumps(dict(c.metadata)),
                    )
                    for c in chunks
                ),
            )
        log.info(
            "Keyword-indexed %d chunks for %s in %.0f ms",
            len(chunks),
            source_path,
            (time.perf_counter() - start) * 1000,
        )

    def search(self, query: str, k: int = 5) -> tuple[SearchHit, ...]:
        match = fts_query(query)
        if not match:
            return ()
        start = time.perf_counter()
        rows = self._conn.execute(
            "SELECT chunk_id, doc_id, text, page_number, idx, metadata, bm25(chunks) AS score"
            " FROM chunks WHERE chunks MATCH ? ORDER BY score LIMIT ?",
            (match, k),
        ).fetchall()
        hits = tuple(
            SearchHit(
                Chunk(chunk_id, doc_id, text, page_number, idx, json.loads(metadata)),
                -score,  # SQLite's bm25(): lower is better. Flip it: higher is better.
                "sparse",
            )
            for chunk_id, doc_id, text, page_number, idx, metadata, score in rows
        )
        log.debug(
            "Keyword search %r: %d hits in %.1f ms",
            query,
            len(hits),
            (time.perf_counter() - start) * 1000,
        )
        return hits

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


def _demo() -> None:
    """Index a few passages, then show what keyword search finds, and misses."""
    from litrarag.config import settings
    from litrarag.logging_setup import setup_logging
    from litrarag.schemas import stable_id

    setup_logging("INFO")
    path = settings.data_dir / "demo" / "keyword_demo.db"
    for suffix in ("", "-wal", "-shm"):  # start fresh
        Path(f"{path}{suffix}").unlink(missing_ok=True)

    passages = [
        "Tesseract reads text from scanned pages and photos.",
        "LitraRag renders scanned PDF pages at 200 DPI before running OCR.",
        "The embedding cache keeps vectors in SQLite, so unchanged text is never embedded twice.",
        "Error AIQ001440 on the panel means the sensor battery is low.",
        "Workout plan: squats, lunges and glute bridges, three times a week.",
        "Banana bread needs ripe bananas, flour, butter and sugar.",
    ]
    doc_id = stable_id("keyword demo")
    chunks = [
        Chunk(
            stable_id(doc_id, str(i)), doc_id, text, 1, i, {"source": "demo.txt", "method": "text"}
        )
        for i, text in enumerate(passages)
    ]

    with SQLiteKeywordIndex(path) as index:
        index.replace_source("demo.txt", chunks)
        for question in (
            "What does AIQ001440 mean?",
            "workouts for my legs",
            "How does OCR work?",
            "Why is re-ingesting fast?",
        ):
            start = time.perf_counter()
            hits = index.search(question, k=2)
            log.info(
                "Q: %s  (%.1f ms)  query = %s",
                question,
                (time.perf_counter() - start) * 1000,
                fts_query(question),
            )
            for hit in hits or ():
                log.info("   %5.2f  %s", hit.score, hit.chunk.text)
            if not hits:
                log.info("   (no keyword matches)")


if __name__ == "__main__":
    _demo()
