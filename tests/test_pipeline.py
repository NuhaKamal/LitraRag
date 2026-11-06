"""End-to-end ingestion tests: real loader, chunker and Qdrant; fake OCR and embeddings."""

from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from litrarag.chunker import LangChainChunker
from litrarag.cli import main
from litrarag.embedder import EmbeddingError
from litrarag.loader import DocumentLoader
from litrarag.pipeline import IngestPipeline, format_report
from litrarag.schemas import FileResult, IngestReport
from litrarag.vector_store import QdrantVectorStore
from tests.fakes import FakeEmbedder, FakeOCR


class FailingEmbedder(FakeEmbedder):
    """Simulates Ollama being down."""

    def embed_documents(self, texts):
        raise EmbeddingError("Ollama is not running")


@pytest.fixture
def setup():
    embedder = FakeEmbedder(dimension=256)
    client = QdrantClient(":memory:")
    store = QdrantVectorStore(client, collection="t", model_name=embedder.model_name, dimension=256)
    pipeline = IngestPipeline(
        loader=DocumentLoader(FakeOCR()),
        chunker=LangChainChunker(chunk_size=200, chunk_overlap=20),
        embedder=embedder,
        store=store,
    )
    yield pipeline, embedder, store
    client.close()


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


# --- One file ------------------------------------------------------------------


def test_new_file_is_indexed_and_searchable(setup, tmp_path):
    pipeline, embedder, store = setup
    path = write(tmp_path / "notes.txt", "Tesseract reads scanned pages for LitraRag.")

    result = pipeline.ingest_file(path)

    assert result.status == "indexed"
    assert result.chunks == 1
    hits = store.search(embedder.embed_query("tesseract scanned pages"), k=1)
    assert "Tesseract" in hits[0].chunk.text


def test_unchanged_file_is_skipped_without_work(setup, tmp_path):
    pipeline, embedder, _ = setup
    path = write(tmp_path / "notes.txt", "Some text worth indexing.")
    pipeline.ingest_file(path)
    embedded_before = embedder.texts_embedded

    result = pipeline.ingest_file(path)

    assert result.status == "unchanged"
    assert embedder.texts_embedded == embedded_before  # no embedding work at all


def test_edited_file_replaces_old_chunks(setup, tmp_path):
    pipeline, embedder, store = setup
    path = write(tmp_path / "notes.txt", "The original version of this note.")
    pipeline.ingest_file(path)

    write(path, "A completely rewritten note.")
    result = pipeline.ingest_file(path)

    assert result.status == "indexed"
    assert store.count() == 1
    hit = store.search(embedder.embed_query("original version"), k=1)[0]
    assert "rewritten" in hit.chunk.text


def test_same_file_through_different_paths_is_one_source(setup, tmp_path):
    pipeline, _, store = setup
    (tmp_path / "sub").mkdir()
    path = write(tmp_path / "notes.txt", "Version one.")
    pipeline.ingest_file(path)

    write(path, "Version two.")
    pipeline.ingest_file(tmp_path / "sub" / ".." / "notes.txt")  # same file, other spelling

    assert store.count() == 1


def test_broken_file_is_reported_not_raised(setup, tmp_path):
    pipeline, _, _ = setup
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"this is not a pdf")

    result = pipeline.ingest_file(path)

    assert result.status == "failed"
    assert result.error


def test_embedding_service_failure_stops_the_run(tmp_path):
    client = QdrantClient(":memory:")
    try:
        pipeline = IngestPipeline(
            loader=DocumentLoader(FakeOCR()),
            chunker=LangChainChunker(),
            embedder=FailingEmbedder(),
            store=QdrantVectorStore(
                client, collection="t", model_name="fake-embedder", dimension=64
            ),
        )
        write(tmp_path / "a.txt", "first file")
        write(tmp_path / "b.txt", "second file")
        with pytest.raises(EmbeddingError):
            pipeline.ingest_folder(tmp_path)
    finally:
        client.close()


# --- Folders and reports -------------------------------------------------------


def test_folder_ingestion_and_rerun(setup, tmp_path):
    pipeline, _, _ = setup
    write(tmp_path / "a.txt", "first file")
    write(tmp_path / "b.md", "second file")
    (tmp_path / "picture.svg").write_text("<svg/>")  # unsupported: ignored

    first = pipeline.ingest_folder(tmp_path)
    second = pipeline.ingest_folder(tmp_path)

    assert (first.indexed, first.unchanged, first.failed) == (2, 0, 0)
    assert (second.indexed, second.unchanged, second.failed) == (0, 2, 0)


def test_format_report_lists_failures():
    report = IngestReport(
        files=(
            FileResult("/docs/good.pdf", "indexed", chunks=12, seconds=2.0),
            FileResult("/docs/same.pdf", "unchanged"),
            FileResult("/docs/broken.pdf", "failed", error="FileDataError: cannot open"),
        ),
        seconds=2.5,
    )
    text = format_report(report)

    assert "1 indexed, 1 unchanged, 1 failed, 12 chunks" in text
    assert "FAILED broken.pdf: FileDataError: cannot open" in text


def test_cli_help_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    assert "ingest" in capsys.readouterr().out
