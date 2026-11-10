"""Query-side composition root: build the retrievers from settings.

The counterpart of open_pipeline(): ingestion WRITES the stores, this READS them.
"""

from collections.abc import Iterator
from contextlib import contextmanager

from litrarag.config import Settings
from litrarag.embedder import OllamaEmbedder
from litrarag.embedding_cache import CachedEmbedder, SQLiteVectorCache
from litrarag.keyword_index import SQLiteKeywordIndex
from litrarag.protocols import Retriever
from litrarag.retriever import DenseRetriever, HybridRetriever, KeywordRetriever
from litrarag.vector_store import QdrantVectorStore, open_qdrant


@contextmanager
def open_retrievers(settings: Settings) -> Iterator[dict[str, Retriever]]:
    """Yield the dense, keyword and hybrid retrievers over the real stores."""
    with (
        SQLiteVectorCache(settings.embedding_cache_path) as cache,
        SQLiteKeywordIndex(settings.sqlite_path) as keywords,
    ):
        embedder = CachedEmbedder(
            OllamaEmbedder(settings.embed_model, base_url=settings.embed_url), cache
        )
        client = open_qdrant(
            settings.qdrant_mode, path=settings.qdrant_path, url=settings.qdrant_url
        )
        try:
            store = QdrantVectorStore(
                client,
                collection=settings.collection,
                model_name=embedder.model_name,
                dimension=embedder.dimension,
            )
            dense = DenseRetriever(embedder, store)
            keyword = KeywordRetriever(keywords)
            hybrid = HybridRetriever(
                [dense, keyword],
                candidates=settings.retrieval_candidates,
                rrf_k=settings.rrf_k,
            )
            yield {"dense": dense, "keyword": keyword, "hybrid": hybrid}
        finally:
            client.close()
