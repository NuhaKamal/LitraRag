"""Tests for prompts.py and generator.py. LangChain's fake chat model replaces Ollama."""

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from litrarag.generator import LangChainGenerator
from litrarag.prompts import NO_CONTEXT_REPLY, RAG_PROMPT_V1, format_context
from litrarag.schemas import Chunk, SearchHit
from tests.helpers import needs_ollama_llm


def make_hit(text: str, source: str = "book.pdf", page: int = 3) -> SearchHit:
    return SearchHit(
        Chunk(f"{source}-{page}", "doc", text, page, 0, {"source": source}), 1.0, "hybrid"
    )


def fake_generator(*replies: str) -> tuple[LangChainGenerator, FakeListChatModel]:
    llm = FakeListChatModel(responses=list(replies))
    return LangChainGenerator("fake-model", llm=llm), llm


# --- Prompts: pure, no LLM ------------------------------------------------------


def test_format_context_numbers_and_labels_passages():
    context = format_context([make_hit("First text.", page=3), make_hit("Second text.", page=7)])
    assert "[1] (book.pdf, page 3)\nFirst text." in context
    assert "[2] (book.pdf, page 7)\nSecond text." in context


def test_prompt_contains_rules_context_and_question():
    messages = RAG_PROMPT_V1.format_messages(context="[1] some passage", question="What is it?")
    assert "Do not guess" in messages[0].content
    assert "[1] some passage" in messages[0].content
    assert messages[1].content == "What is it?"


def test_curly_braces_in_chunk_text_are_safe():
    context = format_context([make_hit("Use {placeholder} syntax.")])
    messages = RAG_PROMPT_V1.format_messages(context=context, question="q")
    assert "{placeholder}" in messages[0].content


# --- Generator ----------------------------------------------------------------------


def test_answer_returns_text_sources_and_model():
    generator, _ = fake_generator("Squats work the quads [1].")
    hits = [make_hit("Squats train the quadriceps.")]

    answer = generator.answer("What do squats work?", hits)

    assert answer.text == "Squats work the quads [1]."
    assert answer.hits == tuple(hits)
    assert answer.model == "fake-model"
    assert answer.latency_ms >= 0


def test_stream_yields_the_whole_answer_in_pieces():
    generator, _ = fake_generator("Hip thrusts.")
    pieces = list(generator.stream("q", [make_hit("text")]))
    assert len(pieces) > 1  # streamed, not one block
    assert "".join(pieces) == "Hip thrusts."


def test_no_hits_skips_the_llm():
    generator, llm = fake_generator("this reply must not be used")
    answer = generator.answer("Anything about rockets?", [])
    assert answer.text == NO_CONTEXT_REPLY
    assert llm.i == 0  # the fake model was never called


# --- Real Ollama ----------------------------------------------------------------------


@pytest.mark.integration
@needs_ollama_llm
def test_real_llama_answers_from_the_context():
    generator = LangChainGenerator("llama3.2:1b")
    hits = [make_hit("The capital of Testland is Bloomtown.")]
    answer = generator.answer("What is the capital of Testland?", hits)
    assert "Bloomtown" in answer.text
