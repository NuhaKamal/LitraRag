"""Compare prompt versions on the two behaviors we need: answering and declining.

Run with:  uv run python scripts/compare_prompts.py   (Ollama running)
"""

from litrarag.generator import LangChainGenerator
from litrarag.prompts import RAG_PROMPT_V2, RAG_PROMPT_V3, format_context, format_context_plain
from litrarag.schemas import Chunk, SearchHit


def hit(text: str) -> SearchHit:
    return SearchHit(Chunk("c1", "doc", text, 3, 0, {"source": "manual.pdf"}), 1.0, "hybrid")


CASES = [
    # (behavior, question, the only passage, text the answer must contain)
    (
        "answer",
        "What does error code E-4417 mean?",
        "Error code E-4417 on the panel means the backup battery needs replacing.",
        "battery",
    ),
    (
        "decline",
        "What is the capital of France?",
        "Squats train the quadriceps and glutes.",
        "could not find",
    ),
]
RUNS = 3

VARIANTS = [
    ("V2 + labels", RAG_PROMPT_V2, format_context),
    ("V2 + plain ", RAG_PROMPT_V2, format_context_plain),
    ("V3 + labels", RAG_PROMPT_V3, format_context),
    ("V3 + plain ", RAG_PROMPT_V3, format_context_plain),
]

for name, prompt, formatter in VARIANTS:
    generator = LangChainGenerator("llama3.2:1b", prompt=prompt, context_formatter=formatter)
    for behavior, question, passage, expected in CASES:
        passed = 0
        text = ""
        for _ in range(RUNS):
            text = generator.answer(question, [hit(passage)]).text
            passed += expected in text.lower()
        print(f"{name}  {behavior:<8} {passed}/{RUNS}   last answer: {text[:60]!r}")
