"""Tests for chat.py: citation parsing and the Chat session object."""

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from litrarag.chat import Chat, cited_hits
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
