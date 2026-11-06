"""Command-line entry point: `litrarag ingest <folder>`."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="litrarag", description="LitraRag: local RAG over your own documents."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest", help="index every supported file in a folder")
    ingest.add_argument("folder", type=Path, help="folder with PDFs, images or text files")
    args = parser.parse_args(argv)

    # Imported here, not at the top: `litrarag --help` stays instant,
    # without loading PyMuPDF, Qdrant and LangChain first.
    from litrarag.config import Settings
    from litrarag.embedder import EmbeddingError
    from litrarag.logging_setup import setup_logging
    from litrarag.pipeline import format_report, open_pipeline
    from litrarag.vector_store import EmbeddingMismatchError, VectorStoreLockedError

    settings = Settings.from_env()
    setup_logging(settings.log_level, settings.log_dir)

    try:
        with open_pipeline(settings) as pipeline:
            report = pipeline.ingest_folder(args.folder)
    except (
        EmbeddingError,
        VectorStoreLockedError,
        EmbeddingMismatchError,
        FileNotFoundError,
    ) as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1

    print(format_report(report))
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
