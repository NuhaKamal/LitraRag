"""Prompt templates for answering questions from retrieved chunks.

Prompts are versioned like code: to improve one, add RAG_PROMPT_V2 and keep V1 to compare.
"""

from collections.abc import Sequence

from langchain_core.prompts import ChatPromptTemplate

from litrarag.schemas import SearchHit

RAG_PROMPT_V1 = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer questions using ONLY the numbered context passages below.\n"
            "Write a short, direct answer. Cite passages inline at the end of a sentence, "
            "for example: 'Squats work the quads [1].' Do not copy the passage labels.\n"
            "If the context does not contain the answer, reply only: "
            "'I could not find this in your documents.' Do not guess.\n\n"
            "Context:\n{context}",
        ),
        ("human", "{question}"),
    ]
)

NO_CONTEXT_REPLY = "I couldn't find anything about that in your documents."


def format_context(hits: Sequence[SearchHit]) -> str:
    """Number each passage and label it with its source file and page. Pure."""
    blocks = []
    for n, hit in enumerate(hits, start=1):
        source = hit.chunk.metadata.get("source", "unknown")
        blocks.append(f"[{n}] ({source}, page {hit.chunk.page_number})\n{hit.chunk.text}")
    return "\n\n".join(blocks)
