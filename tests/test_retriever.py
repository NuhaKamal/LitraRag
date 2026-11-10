"""Tests for retriever.py: rank fusion, retriever wrappers and composition."""

import pytest
from qdrant_client import QdrantClient

from litrarag.keyword_index import SQLiteKeywordIndex
from litrarag.retriever import (
    DenseRetriever,
    HybridRetriever,
    KeywordRetriever,
    reciprocal_rank_fusion,
)
from litrarag.schemas import Chunk, HitSource, SearchHit
from litrarag.vector_store import QdrantVectorStore
from tests.fakes import FakeEmbedder


def make_chunk(name: str) -> Chunk:
    return Chunk(name, "doc1", f"text of {name}", 1, 0)


def hits(*names: str, source: HitSource = "dense") -> list[SearchHit]:
    return [SearchHit(make_chunk(name), 1.0, source) for name in names]


class StaticRetriever:
    """Returns a fixed ranking, and records which k it was asked for."""

    def __init__(self, *names: str) -> None:
        self.ranking = tuple(hits(*names))
        self.asked_k: list[int] = []

    def retrieve(self, question: str, k: int = 5) -> tuple[SearchHit, ...]:
        self.asked_k.append(k)
        return self.ranking[:k]


def ids(results) -> list[str]:
    return [hit.chunk.chunk_id for hit in results]


# --- reciprocal_rank_fusion: pure ------------------------------------------------


def test_rrf_keeps_the_order_of_a_single_ranking():
    assert ids(reciprocal_rank_fusion([hits("a", "b", "c")])) == ["a", "b", "c"]


def test_rrf_agreement_beats_one_top_rank():
    fused = reciprocal_rank_fusion([hits("a", "b"), hits("c", "b")])
    assert ids(fused)[0] == "b"  # 2nd in both rankings beats 1st in one


def test_rrf_merges_duplicates():
    fused = reciprocal_rank_fusion([hits("a", "b"), hits("b", "a")])
    assert sorted(ids(fused)) == ["a", "b"]


def test_rrf_score_formula():
    fused = reciprocal_rank_fusion([hits("a")], rrf_k=60)
    assert fused[0].score == pytest.approx(1 / 61)


def test_rrf_scores_descend_and_are_marked_hybrid():
    fused = reciprocal_rank_fusion([hits("a", "b", "c"), hits("c", "d")])
    scores = [hit.score for hit in fused]
    assert scores == sorted(scores, reverse=True)
    assert all(hit.source == "hybrid" for hit in fused)


def test_rrf_of_nothing_is_empty():
    assert reciprocal_rank_fusion([[], []]) == []


# --- HybridRetriever: composition ----------------------------------------------


def test_hybrid_fuses_its_retrievers():
    hybrid = HybridRetriever([StaticRetriever("a", "b"), StaticRetriever("c", "b")], candidates=10)
    assert ids(hybrid.retrieve("q", k=1)) == ["b"]


def test_hybrid_asks_each_retriever_for_candidates():
    first, second = StaticRetriever("a"), StaticRetriever("b")
    HybridRetriever([first, second], candidates=20).retrieve("q", k=3)
    assert first.asked_k == [20]
    assert second.asked_k == [20]


def test_hybrid_returns_at_most_k():
    hybrid = HybridRetriever(
        [StaticRetriever("a", "b", "c"), StaticRetriever("d", "e")], candidates=10
    )
    assert len(hybrid.retrieve("q", k=2)) == 2


# --- Wrappers around the real stores ---------------------------------------------


def test_dense_retriever_embeds_the_question_as_a_query():
    embedder = FakeEmbedder(dimension=256)
    client = QdrantClient(":memory:")
    try:
        store = QdrantVectorStore(
            client, collection="t", model_name=embedder.model_name, dimension=256
        )
        chunk = Chunk("c1", "doc1", "tesseract reads scanned pages", 1, 0)
        store.replace_source("a.pdf", [chunk], embedder.embed_documents([chunk.text]))
        calls_before = embedder.calls

        top = DenseRetriever(embedder, store).retrieve("scanned pages", k=1)

        assert top[0].chunk == chunk
        assert embedder.calls == calls_before + 1  # one embed_query call
    finally:
        client.close()


def test_keyword_retriever_searches_the_index():
    with SQLiteKeywordIndex(":memory:") as index:
        chunk = Chunk("c1", "doc1", "Error AIQ001440 means the battery is low", 1, 0)
        index.replace_source("a.pdf", [chunk])
        assert KeywordRetriever(index).retrieve("AIQ001440", k=1)[0].chunk == chunk
