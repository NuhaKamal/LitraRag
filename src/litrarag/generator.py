"""Generator stage: question + retrieved chunks -> answer, via an LLM.

LangChain builds the chain INSIDE this adapter: prompt | chat model | plain text.
Callers get LitraRag's Answer, or text pieces, and never see LangChain types.
"""

import time
from collections.abc import Iterator, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from litrarag.logging_setup import get_logger
from litrarag.prompts import DEFAULT_PROMPT, NO_CONTEXT_REPLY, format_context
from litrarag.schemas import Answer, SearchHit

log = get_logger(__name__)


class GenerationError(RuntimeError):
    """The LLM service failed. The message says what to check."""


class LangChainGenerator:
    """Generator backed by a LangChain chat model (Ollama by default, local or Colab)."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://localhost:11434",
        num_ctx: int = 2048,
        keep_alive: str = "2m",
        temperature: float = 0.1,
        prompt: ChatPromptTemplate = DEFAULT_PROMPT,
        llm: BaseChatModel | None = None,
    ) -> None:
        self.model_name = model
        self.base_url = base_url
        self._llm = llm or ChatOllama(
            model=model,
            base_url=base_url,
            num_ctx=num_ctx,  # context window: more = more RAM
            keep_alive=keep_alive,  # unload the model from RAM when idle
            temperature=temperature,  # low = stick closely to the context
        )
        self._chain = prompt | self._llm | StrOutputParser()

    def answer(self, question: str, hits: Sequence[SearchHit]) -> Answer:
        start = time.perf_counter()
        text = "".join(self.stream(question, hits))
        return Answer(
            question, text, tuple(hits), self.model_name, (time.perf_counter() - start) * 1000
        )

    def stream(self, question: str, hits: Sequence[SearchHit]) -> Iterator[str]:
        if not hits:  # nothing retrieved: answer at once, and nothing to invent from
            yield NO_CONTEXT_REPLY
            return

        start = time.perf_counter()
        first_piece_ms: float | None = None
        try:
            for piece in self._chain.stream(
                {"context": format_context(hits), "question": question}
            ):
                if first_piece_ms is None:
                    first_piece_ms = (time.perf_counter() - start) * 1000
                yield piece
        except Exception as err:
            raise GenerationError(
                f"Answering with {self.model_name!r} at {self.base_url} failed: {err}. "
                "Is Ollama running, and is the model pulled?"
            ) from err

        log.info(
            "Answered with %s: first words after %.0f ms, finished after %.0f ms",
            self.model_name,
            first_piece_ms or 0,
            (time.perf_counter() - start) * 1000,
        )


def _demo() -> None:
    """Ask real questions over your documents, streaming the answers."""
    from litrarag.config import settings
    from litrarag.logging_setup import setup_logging
    from litrarag.search import open_retrievers

    setup_logging("INFO")
    generator = LangChainGenerator(
        settings.llm_model,
        base_url=settings.llm_url,
        num_ctx=settings.limits.llm_num_ctx,
        keep_alive=settings.limits.ollama_keep_alive,
    )
    with open_retrievers(settings) as retrievers:
        for question in (
            "Why does the author avoid back squats?",
            "Which exercises can isolate the hamstrings?",
            "What is the capital of France?",  # not in your documents
        ):
            hits = retrievers["hybrid"].retrieve(question, settings.top_k)
            print(f"\nQ: {question}\nA: ", end="", flush=True)
            for piece in generator.stream(question, hits):
                print(piece, end="", flush=True)  # appears word by word
            print("\nSources:")
            for n, hit in enumerate(hits, start=1):
                print(f"  [{n}] {hit.chunk.metadata.get('source')}, page {hit.chunk.page_number}")


if __name__ == "__main__":
    _demo()
