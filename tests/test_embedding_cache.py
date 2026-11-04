"""Tests for embedding_cache.py, using FakeEmbedder and temporary SQLite files."""

import numpy as np
import pytest

from litrarag.embedding_cache import CachedEmbedder, SQLiteVectorCache, cache_key
from tests.fakes import FakeEmbedder


@pytest.fixture
def cache(tmp_path):
    store = SQLiteVectorCache(tmp_path / "cache.db")
    yield store
    store.close()  # Windows: release the file so the temp folder can be removed


# --- cache_key ---------------------------------------------------------------


def test_cache_key_is_deterministic():
    assert cache_key("m", "document", "text") == cache_key("m", "document", "text")


def test_cache_key_depends_on_model_kind_and_text():
    keys = {
        cache_key("m", "document", "text"),
        cache_key("other-model", "document", "text"),
        cache_key("m", "query", "text"),
        cache_key("m", "document", "different text"),
    }
    assert len(keys) == 4


# --- SQLiteVectorCache -------------------------------------------------------


def test_store_round_trip(cache):
    vector = np.array([0.5, -1.0, 2.0], dtype=np.float32)
    cache.put_many([("k1", "m", vector)])

    found = cache.get_many(["k1", "missing"])

    assert list(found) == ["k1"]
    assert np.array_equal(found["k1"], vector)
    assert found["k1"].dtype == np.float32


def test_store_survives_reopening(tmp_path):
    path = tmp_path / "cache.db"
    with SQLiteVectorCache(path) as store:
        store.put_many([("k", "m", np.ones(3, dtype=np.float32))])
    with SQLiteVectorCache(path) as store:
        assert "k" in store.get_many(["k"])


def test_store_looks_up_many_keys(cache):
    items = [(f"k{i}", "m", np.full(2, i, dtype=np.float32)) for i in range(1200)]
    cache.put_many(items)
    assert len(cache.get_many([key for key, _, _ in items])) == 1200


# --- CachedEmbedder ----------------------------------------------------------


def test_second_run_embeds_nothing(cache):
    inner = FakeEmbedder()
    embedder = CachedEmbedder(inner, cache)
    texts = ["alpha text", "beta text", "gamma text"]

    first = embedder.embed_documents(texts)
    second = embedder.embed_documents(texts)

    assert inner.texts_embedded == 3  # only the first run reached the model
    assert np.array_equal(first, second)


def test_only_new_texts_are_embedded(cache):
    inner = FakeEmbedder()
    embedder = CachedEmbedder(inner, cache)
    embedder.embed_documents(["a", "b"])
    embedder.embed_documents(["a", "b", "c"])
    assert inner.texts_embedded == 3


def test_duplicates_in_one_call_are_embedded_once(cache):
    inner = FakeEmbedder()
    vectors = CachedEmbedder(inner, cache).embed_documents(["same", "same", "other"])
    assert inner.texts_embedded == 2
    assert np.array_equal(vectors[0], vectors[1])


def test_cached_result_matches_uncached(cache):
    texts = ["one fish", "two fish", "red fish"]
    expected = FakeEmbedder().embed_documents(texts)
    embedder = CachedEmbedder(FakeEmbedder(), cache)
    embedder.embed_documents(texts[:1])  # warm the cache partially

    assert np.array_equal(embedder.embed_documents(texts), expected)


def test_models_do_not_share_vectors(cache):
    model_a = FakeEmbedder(model_name="model-a")
    model_b = FakeEmbedder(model_name="model-b")
    CachedEmbedder(model_a, cache).embed_documents(["text"])
    CachedEmbedder(model_b, cache).embed_documents(["text"])
    assert model_b.texts_embedded == 1


def test_documents_and_queries_are_cached_separately(cache):
    inner = FakeEmbedder()
    embedder = CachedEmbedder(inner, cache)
    embedder.embed_documents(["text"])
    embedder.embed_query("text")  # different kind: must be embedded
    embedder.embed_query("text")  # same kind again: cached
    assert inner.texts_embedded == 2


def test_cache_survives_restart(tmp_path):
    path = tmp_path / "cache.db"
    with SQLiteVectorCache(path) as store:
        CachedEmbedder(FakeEmbedder(), store).embed_documents(["persisted text"])

    inner = FakeEmbedder()  # a "new run" of LitraRag
    with SQLiteVectorCache(path) as store:
        CachedEmbedder(inner, store).embed_documents(["persisted text"])

    assert inner.texts_embedded == 0
