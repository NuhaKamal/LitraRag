"""Chat session: retrieval + generation for the interactive app.

start_chat() is the chat's composition root. It reuses open_retrievers(), so the
stores are built in exactly one place.
"""

import re
import threading
from collections.abc import Iterator, Sequence
from contextlib import ExitStack

from litrarag.config import Settings
from litrarag.generator import LangChainGenerator
from litrarag.protocols import Generator, Retriever
from litrarag.schemas import SearchHit
from litrarag.search import open_retrievers

_CITATION = re.compile(r"\[(\d+)\]")


def cited_hits(text: str, hits: Sequence[SearchHit]) -> list[tuple[int, SearchHit]]:
    """The passages an answer actually cites, in citation order, without repeats. Pure."""
    numbers: list[int] = []
    for match in _CITATION.finditer(text):
        n = int(match.group(1))
        if 1 <= n <= len(hits) and n not in numbers:  # ignore invented numbers like [9]
            numbers.append(n)
    return [(n, hits[n - 1]) for n in numbers]


class Chat:
    """A retriever and a generator for one app. Safe to share between threads."""

    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
        *,
        top_k: int,
        resources: ExitStack | None = None,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k
        self._resources = resources or ExitStack()
        self._lock = threading.Lock()

    def retrieve(self, question: str) -> tuple[SearchHit, ...]:
        with self._lock:  # one search at a time: the stores share single connections
            return self.retriever.retrieve(question, self.top_k)

    def stream(self, question: str, hits: Sequence[SearchHit]) -> Iterator[str]:
        return self.generator.stream(question, hits)

    def close(self) -> None:
        self._resources.close()


def start_chat(settings: Settings) -> Chat:
    """Build the chat from settings. The stores stay open until close() is called."""
    resources = ExitStack()
    try:
        retrievers = resources.enter_context(open_retrievers(settings))
        generator = LangChainGenerator(
            settings.llm_model,
            base_url=settings.llm_url,
            num_ctx=settings.limits.llm_num_ctx,
            keep_alive=settings.limits.ollama_keep_alive,
        )
    except BaseException:
        resources.close()  # if anything fails, release what was already opened
        raise
    return Chat(retrievers["hybrid"], generator, top_k=settings.top_k, resources=resources)
