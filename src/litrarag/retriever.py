"""Retrieval: find the chunks most relevant to a question.

DenseRetriever and KeywordRetriever each wrap one search method.
HybridRetriever combines ANY retrievers with Reciprocal Rank Fusion,
and is itself a Retriever (the composite pattern).
"""

import time
from collections.abc import Sequence

from litrarag.logging_setup import get_logger
from litrarag.protocols import Embedder, KeywordIndex, Retriever, VectorStore
from litrarag.schemas import Chunk, SearchHit

log = get_logger(__name__)


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[SearchHit]], *, rrf_k: int = 60
) -> list[SearchHit]:
    """Fuse rankings by rank alone: score = sum of 1 / (rrf_k + rank). Pure."""
    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            chunk_id = hit.chunk.chunk_id
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
            chunks.setdefault(chunk_id, hit.chunk)

    # sorted() is stable: equal scores keep the order in which chunks were first seen
    ordered = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)
    return [SearchHit(chunks[chunk_id], scores[chunk_id], "hybrid") for chunk_id in ordered]


class DenseRetriever:
    """Search by meaning: embed the question, find the nearest vectors."""

    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self.embedder = embedder
        self.store = store

    def retrieve(self, question: str, k: int = 5) -> tuple[SearchHit, ...]:
        return self.store.search(self.embedder.embed_query(question), k)


class KeywordRetriever:
    """Search by exact words, ranked with BM25."""

    def __init__(self, index: KeywordIndex) -> None:
        self.index = index

    def retrieve(self, question: str, k: int = 5) -> tuple[SearchHit, ...]:
        return self.index.search(question, k)


class HybridRetriever:
    """Fuse several retrievers' rankings with Reciprocal Rank Fusion."""

    def __init__(
        self, retrievers: Sequence[Retriever], *, candidates: int = 20, rrf_k: int = 60
    ) -> None:
        self.retrievers = tuple(retrievers)
        self.candidates = candidates
        self.rrf_k = rrf_k

    def retrieve(self, question: str, k: int = 5) -> tuple[SearchHit, ...]:
        start = time.perf_counter()
        depth = max(k, self.candidates)  # look deeper than k, so agreement lower down counts
        rankings = [retriever.retrieve(question, depth) for retriever in self.retrievers]
        fused = tuple(reciprocal_rank_fusion(rankings, rrf_k=self.rrf_k)[:k])
        log.debug(
            "Hybrid retrieval: %s candidates -> %d hits in %.1f ms",
            "+".join(str(len(r)) for r in rankings),
            len(fused),
            (time.perf_counter() - start) * 1000,
        )
        return fused


def _demo() -> None:
    """Same questions, three retrievers, side by side."""
    from qdrant_client import QdrantClient

    from litrarag.config import settings
    from litrarag.embedder import OllamaEmbedder
    from litrarag.embedding_cache import CachedEmbedder, SQLiteVectorCache
    from litrarag.keyword_index import SQLiteKeywordIndex
    from litrarag.logging_setup import setup_logging
    from litrarag.schemas import stable_id
    from litrarag.vector_store import QdrantVectorStore

    setup_logging("INFO")
    passages = [
        "Tesseract reads text from scanned pages and photos.",
        "LitraRag renders scanned PDF pages at 200 DPI before running OCR.",
        "The embedding cache keeps vectors in SQLite, so unchanged text is never embedded twice.",
        "Error AIQ001440 on the panel means the sensor battery is low.",
        "Workout plan: squats, lunges and glute bridges, three times a week.",
        "Banana bread needs ripe bananas, flour, butter and sugar.",
    ]
    doc_id = stable_id("retriever demo")
    chunks = [
        Chunk(
            stable_id(doc_id, str(i)), doc_id, text, 1, i, {"source": "demo.txt", "method": "text"}
        )
        for i, text in enumerate(passages)
    ]

    with (
        SQLiteVectorCache(settings.embedding_cache_path) as cache,
        SQLiteKeywordIndex(":memory:") as keywords,
    ):
        embedder = CachedEmbedder(
            OllamaEmbedder(settings.embed_model, base_url=settings.embed_url), cache
        )
        client = QdrantClient(":memory:")  # in-memory: no folder, no lock
        try:
            store = QdrantVectorStore(
                client,
                collection="demo",
                model_name=embedder.model_name,
                dimension=embedder.dimension,
            )
            store.replace_source("demo.txt", chunks, embedder.embed_documents(passages))
            keywords.replace_source("demo.txt", chunks)

            dense = DenseRetriever(embedder, store)
            keyword = KeywordRetriever(keywords)
            hybrid = HybridRetriever([dense, keyword], candidates=len(passages))

            for question in (
                "What does AIQ001440 mean?",
                "workouts for my legs",
                "How does OCR work?",
                "Why is re-ingesting fast?",
            ):
                log.info("Q: %s", question)
                for name, retriever in (("dense", dense), ("keyword", keyword), ("hybrid", hybrid)):
                    top = retriever.retrieve(question, k=2)
                    summary = " | ".join(hit.chunk.text[:38] for hit in top) or "(none)"
                    log.info("   %-7s %s", name, summary)
        finally:
            client.close()


if __name__ == "__main__":
    _demo()
