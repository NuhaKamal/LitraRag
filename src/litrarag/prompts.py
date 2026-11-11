"""Prompt templates for answering questions from retrieved chunks.

Prompts are versioned like code: improve one by adding a new version,
keep the old one to compare, and point DEFAULT_PROMPT at the best.
"""

from collections.abc import Sequence

from langchain_core.prompts import ChatPromptTemplate

from litrarag.schemas import SearchHit

# V1: rules and context in the system message. Grounded, but cites loosely.
RAG_PROMPT_V1 = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer questions using ONLY the numbered context passages below.\n"
            "Cite the passages you use, like [1] or [2].\n"
            "If the context does not contain the answer, say that you could not find it "
            "in the documents. Do not guess.\n\n"
            "Context:\n{context}",
        ),
        ("human", "{question}"),
    ]
)

# V2: context next to the question in the user message; small models attend to it better.
RAG_PROMPT_V2 = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer questions about the user's documents, using only the context "
            "they provide. Do not use outside knowledge. Keep answers short and direct. "
            "Cite the passages you use inline, for example: 'Squats work the quads [1].' "
            "If the context does not contain the answer, say: "
            "'I could not find this in your documents.'",
        ),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)

DEFAULT_PROMPT = RAG_PROMPT_V2

NO_CONTEXT_REPLY = "I couldn't find anything about that in your documents."


def format_context(hits: Sequence[SearchHit]) -> str:
    """Number each passage and label it with its source file and page. Pure."""
    blocks = []
    for n, hit in enumerate(hits, start=1):
        source = hit.chunk.metadata.get("source", "unknown")
        blocks.append(f"[{n}] ({source}, page {hit.chunk.page_number})\n{hit.chunk.text}")
    return "\n\n".join(blocks)
