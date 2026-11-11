"""Tests for chat.py: citation parsing and the Chat session object."""

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from litrarag.chat import Chat, cited_hits, clean_question, sources_for
from litrarag.generator import LangChainGenerator
from litrarag.schemas import Chunk, SearchHit


def make_hits(n: int) -> tuple[SearchHit, ...]:
    return tuple(
        SearchHit(Chunk(f"c{i}", "doc", f"text {i}", i, i, {"source": "book.pdf"}), 1.0, "hybrid")
        for i in range(1, n + 1)
    )


# --- cited_hits: pure -----------------------------------------------------------


def test_cited_hits_follow_citation_order():
    cited = cited_hits("Second point [2], then the first [1].", make_hits(3))
    assert [n for n, _ in cited] == [2, 1]


def test_cited_hits_pair_each_number_with_its_hit():
    hits = make_hits(3)
    assert cited_hits("See [2].", hits)[0][1] is hits[1]


def test_cited_hits_ignore_repeats():
    assert [n for n, _ in cited_hits("[1] and again [1].", make_hits(2))] == [1]


def test_cited_hits_ignore_numbers_out_of_range():
    assert cited_hits("Made up [9] and [0].", make_hits(3)) == []


def test_answer_without_citations_has_no_sources():
    assert cited_hits("I could not find this in your documents.", make_hits(4)) == []


def test_clean_question_strips_wrapping_quotes():
    assert clean_question('  "Why does the author avoid back squats?" ') == (
        "Why does the author avoid back squats?"
    )
    assert clean_question("“Curly quotes?”") == "Curly quotes?"


def test_clean_question_keeps_inner_quotes():
    assert clean_question('What does "RDL" mean?') == 'What does "RDL" mean?'


def test_sources_for_prefers_cited_passages():
    label, shown = sources_for("Squats work the quads [2].", make_hits(3))
    assert label == "Cited sources"
    assert [n for n, _ in shown] == [2]


def test_sources_for_refusal_shows_nothing():
    assert sources_for("I could not find this in your documents.", make_hits(3)) == ("", [])


def test_sources_for_uncited_answer_shows_retrieved_passages():
    label, shown = sources_for("Squats work the quads.", make_hits(3))
    assert label == "Retrieved passages"
    assert [n for n, _ in shown] == [1, 2, 3]


# --- Chat ---------------------------------------------------------------------------


class RecordingRetriever:
    def __init__(self) -> None:
        self.asked_k: list[int] = []

    def retrieve(self, question: str, k: int = 5) -> tuple[SearchHit, ...]:
        self.asked_k.append(k)
        return make_hits(2)


def test_chat_retrieves_top_k_and_streams_the_answer():
    retriever = RecordingRetriever()
    generator = LangChainGenerator("fake", llm=FakeListChatModel(responses=["Answer [1]."]))
    chat = Chat(retriever, generator, top_k=3)

    hits = chat.retrieve("question")

    assert retriever.asked_k == [3]
    assert "".join(chat.stream("question", hits)) == "Answer [1]."
    chat.close()
