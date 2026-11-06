"""Ingestion pipeline: folder -> searchable chunks.

IngestPipeline connects the stages it is GIVEN, so tests can pass fakes.
open_pipeline() is the composition root: the ONE place that reads settings
and builds the real stages.
"""

import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from litrarag.chunker import LangChainChunker
from litrarag.config import Settings
from litrarag.embedder import EmbeddingError, OllamaEmbedder
from litrarag.embedding_cache import CachedEmbedder, SQLiteVectorCache
from litrarag.loader import SUPPORTED_TYPES, DocumentLoader, file_id
from litrarag.logging_setup import get_logger
from litrarag.ocr import TesseractOCR
from litrarag.protocols import Chunker, Embedder, VectorStore
from litrarag.schemas import FileResult, IngestReport
from litrarag.vector_store import QdrantVectorStore, open_qdrant

log = get_logger(__name__)


class IngestPipeline:
    """file -> Document -> Chunks -> vectors -> vector store, skipping unchanged files."""

    def __init__(
        self,
        *,
        loader: DocumentLoader,
        chunker: Chunker,
        embedder: Embedder,
        store: VectorStore,
    ) -> None:
        self.loader = loader
        self.chunker = chunker
        self.embedder = embedder
        self.store = store

    def ingest_file(self, path: Path) -> FileResult:
        source = str(path.resolve())  # one identity per file, however the path was typed
        start = time.perf_counter()
        try:
            doc_id = file_id(path)
            if self.store.has_document(doc_id):  # unchanged: skip ALL the work
                return FileResult(source, "unchanged", seconds=time.perf_counter() - start)

            doc = self.loader.load(path)
            chunks = self.chunker.split(doc)
            vectors = self.embedder.embed_documents([chunk.text for chunk in chunks])
            self.store.replace_source(source, chunks, vectors)
        except EmbeddingError:
            raise  # the service is down: every remaining file would fail the same way
        except Exception as err:
            log.warning("Failed to ingest %s: %s", path.name, err)
            return FileResult(
                source,
                "failed",
                seconds=time.perf_counter() - start,
                error=f"{type(err).__name__}: {err}",
            )
        return FileResult(
            source, "indexed", chunks=len(chunks), seconds=time.perf_counter() - start
        )

    def ingest_folder(self, folder: Path) -> IngestReport:
        if not folder.is_dir():
            raise FileNotFoundError(f"Folder not found: {folder}")

        files = sorted(
            p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_TYPES
        )
        start = time.perf_counter()
        results = []
        for n, path in enumerate(files, start=1):
            result = self.ingest_file(path)
            results.append(result)
            log.info(
                "[%d/%d] %s: %s (%d chunks, %.1f s)",
                n,
                len(files),
                path.name,
                result.status,
                result.chunks,
                result.seconds,
            )
        return IngestReport(tuple(results), time.perf_counter() - start)


def format_report(report: IngestReport) -> str:
    """A short, human-readable summary of an ingestion run. Pure."""
    lines = [
        f"Ingested {len(report.files)} files in {report.seconds:.1f} s",
        f"  {report.indexed} indexed, {report.unchanged} unchanged, "
        f"{report.failed} failed, {report.chunks} chunks",
    ]
    indexing_seconds = sum(f.seconds for f in report.files if f.status == "indexed")
    if report.chunks and indexing_seconds > 0:
        lines.append(f"  {report.chunks / indexing_seconds:.1f} chunks/s while indexing")
    for f in report.files:
        if f.status == "failed":
            lines.append(f"  FAILED {Path(f.path).name}: {f.error}")
    return "\n".join(lines)


@contextmanager
def open_pipeline(settings: Settings) -> Iterator[IngestPipeline]:
    """Build the real pipeline from settings; close the cache and Qdrant afterwards."""
    limits = settings.limits
    loader = DocumentLoader(
        TesseractOCR(cmd=settings.tesseract_cmd),
        ocr_dpi=limits.ocr_dpi,
        ocr_min_chars=limits.ocr_min_chars,
        max_workers=limits.max_workers,
    )
    chunker = LangChainChunker(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)

    with SQLiteVectorCache(settings.embedding_cache_path) as cache:
        embedder = CachedEmbedder(
            OllamaEmbedder(
                settings.embed_model,
                base_url=settings.embed_url,
                batch_size=limits.embed_batch_size,
            ),
            cache,
        )
        client = open_qdrant(
            settings.qdrant_mode, path=settings.qdrant_path, url=settings.qdrant_url
        )
        try:
            store = QdrantVectorStore(
                client,
                collection=settings.collection,
                model_name=embedder.model_name,
                dimension=embedder.dimension,  # asks Ollama now: fails fast if it's down
            )
            yield IngestPipeline(loader=loader, chunker=chunker, embedder=embedder, store=store)
        finally:
            client.close()
