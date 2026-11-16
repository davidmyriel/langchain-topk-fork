from __future__ import annotations

from typing import Any

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from topk_sdk.schema import (
    f32_vector,
    keyword_index,
    semantic_index,
    text,
    vector_index,
)

from langchain_topk import (
    MultiVectorEmbeddings,
    RetrievalMode,
    SparseEmbeddings,
    TextIndex,
    TopKVectorStore,
)
from tests.unit_tests.fake_topk import FakeAsyncClient, FakeClient


class FakeSparse(SparseEmbeddings):
    def embed_documents(self, texts: list[str]) -> list[dict[int, float]]:
        return [{len(t): 1.0, 7: 0.5} for t in texts]

    def embed_query(self, text: str) -> dict[int, float]:
        return {7: 1.0}


class FakeMulti(MultiVectorEmbeddings):
    def embed_documents(self, texts: list[str]) -> list[list[list[float]]]:
        return [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]] for _ in texts]

    def embed_query(self, text: str) -> list[list[float]]:
        return [[1.0, 0.0, 0.0]]


def make_store(**kwargs: Any) -> tuple[TopKVectorStore, FakeClient]:
    client = FakeClient()
    store = TopKVectorStore(
        "test",
        client=client,  # type: ignore[arg-type]
        async_client=FakeAsyncClient(client),  # type: ignore[arg-type]
        **kwargs,
    )
    return store, client


def schema_repr(client: FakeClient) -> dict[str, str]:
    return {k: repr(v) for k, v in client.last("create")[2].items()}


# --- construction --------------------------------------------------------------


def test_default_is_semantic_with_server_side_embeddings() -> None:
    store, client = make_store()
    assert store.retrieval_mode == RetrievalMode.SEMANTIC
    assert store.embeddings is None
    schema = schema_repr(client)
    assert set(schema) == {"text"}
    assert "Semantic" in schema["text"]


def test_dense_schema_uses_embedding_dimension() -> None:
    store, client = make_store(embedding=DeterministicFakeEmbedding(size=6))
    assert store.retrieval_mode == RetrievalMode.DENSE
    schema = schema_repr(client)
    assert "6" in schema["dense_vector"]
    assert "Cosine" in schema["dense_vector"]


def test_full_schema() -> None:
    _, client = make_store(
        embedding=DeterministicFakeEmbedding(size=4),
        sparse_embedding=FakeSparse(),
        multi_vector_embedding=FakeMulti(),
        distance_metric="dot_product",
    )
    schema = schema_repr(client)
    assert set(schema) == {"text", "dense_vector", "sparse_vector", "multi_vector"}
    assert "DotProduct" in schema["sparse_vector"]
    assert "Maxsim" in schema["multi_vector"] or "MaxSim" in schema["multi_vector"]
    assert "3" in schema["multi_vector"]


def test_existing_collection_is_not_recreated() -> None:
    client = FakeClient()
    client.schemas["test"] = {"text": text().index(semantic_index())}
    TopKVectorStore("test", client=client)  # type: ignore[arg-type]
    assert not [c for c in client.calls if c[0] == "create"]


def test_existing_collection_settings_come_from_schema() -> None:
    client = FakeClient()
    client.schemas["test"] = {
        "text": text().index(keyword_index()),
        "dense_vector": f32_vector(3).index(vector_index(metric="euclidean")),
    }
    store = TopKVectorStore(
        "test",
        DeterministicFakeEmbedding(size=3),
        client=client,  # type: ignore[arg-type]
    )
    assert store.distance_metric == "euclidean"
    assert store.text_index == TextIndex.KEYWORD


@pytest.mark.parametrize(
    ("schema", "kwargs", "message"),
    [
        (
            {"text": text().index(semantic_index())},
            {"embedding": DeterministicFakeEmbedding(size=3)},
            "indexed field 'dense_vector'",
        ),
        (
            {
                "text": text().index(semantic_index()),
                "dense_vector": f32_vector(3).index(vector_index(metric="cosine")),
            },
            {
                "embedding": DeterministicFakeEmbedding(size=3),
                "distance_metric": "euclidean",
            },
            "does not match",
        ),
        (
            {"text": text().index(keyword_index())},
            {"text_index": "semantic"},
            "does not match",
        ),
        ({"text": text()}, {}, "Semantic search needs"),
        (
            {"text": text().index(semantic_index())},
            {"sparse_embedding": FakeSparse()},
            "sparse_vector",
        ),
    ],
)
def test_existing_collection_mismatch(schema: dict, kwargs: dict, message: str) -> None:
    client = FakeClient()
    client.schemas["test"] = schema
    with pytest.raises(ValueError, match=message):
        TopKVectorStore("test", client=client, **kwargs)  # type: ignore[arg-type]


def test_hybrid_weights_validated_at_init() -> None:
    with pytest.raises(ValueError, match="not configured"):
        make_store(hybrid_weights={"dense": 1.0})


def test_keyword_text_index() -> None:
    store, client = make_store(text_index="keyword", retrieval_mode="keyword")
    assert "Keyword" in schema_repr(client)["text"]
    with pytest.raises(ValueError, match="Semantic search needs"):
        store.similarity_search("x", mode="semantic")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"retrieval_mode": "dense"}, "embedding"),
        ({"retrieval_mode": "sparse"}, "sparse_embedding"),
        ({"retrieval_mode": "multi_vector"}, "multi_vector_embedding"),
        ({"distance_metric": "manhattan"}, "distance_metric"),
        ({"hybrid_weights": {"bogus": 1.0}}, "Unknown hybrid weight"),
    ],
)
def test_invalid_configuration(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        make_store(**kwargs)


def test_missing_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TOPK_API_KEY", raising=False)
    monkeypatch.delenv("TOPK_REGION", raising=False)
    with pytest.raises(ValueError, match="TOPK_API_KEY"):
        TopKVectorStore("x")
    with pytest.raises(ValueError, match="TOPK_REGION"):
        TopKVectorStore("x", api_key="k")


# --- writes --------------------------------------------------------------------


def test_add_texts_semantic_sends_no_vectors() -> None:
    store, client = make_store()
    ids = store.add_texts(["a", "b"], [{"lang": "rust"}, {}], ids=["1", "2"])
    assert ids == ["1", "2"]
    _, key, rows = client.last("upsert")
    assert key == ("test", None)
    assert rows == [
        {"_id": "1", "text": "a", "metadata": {"lang": "rust"}},
        {"_id": "2", "text": "b"},
    ]
    assert store.last_lsn == "1"


def test_add_texts_generates_ids_and_batches() -> None:
    store, client = make_store()
    ids = store.add_texts([str(i) for i in range(5)], batch_size=2)
    assert len(set(ids)) == 5
    assert len([c for c in client.calls if c[0] == "upsert"]) == 3


def test_add_texts_with_all_vector_types() -> None:
    store, client = make_store(
        embedding=DeterministicFakeEmbedding(size=3),
        sparse_embedding=FakeSparse(),
        multi_vector_embedding=FakeMulti(),
    )
    store.add_texts(["hello"], ids=["h"])
    row = client.last("upsert")[2][0]
    assert len(row["dense_vector"]) == 3
    assert "SparseVector" in type(row["sparse_vector"]).__name__ or row["sparse_vector"]
    assert row["multi_vector"] is not None


def test_add_texts_length_mismatch() -> None:
    store, _ = make_store()
    with pytest.raises(ValueError, match="metadatas"):
        store.add_texts(["a"], [{}, {}])
    with pytest.raises(ValueError, match="ids"):
        store.add_texts(["a"], ids=["1", "2"])


def test_add_documents_uses_document_ids() -> None:
    store, client = make_store()
    store.add_documents([Document(page_content="x", id="doc-1", metadata={"a": 1})])
    assert client.last("upsert")[2][0]["_id"] == "doc-1"


def test_delete_by_ids_and_filter() -> None:
    store, client = make_store()
    store.add_texts(["a"], ids=["1"])
    assert store.delete(["1"]) is True
    assert client.last("delete")[2] == ["1"]
    store.delete(filter={"lang": "rust"})
    assert "metadata.lang" in repr(client.last("delete")[2])


def test_delete_requires_target() -> None:
    store, _ = make_store()
    with pytest.raises(ValueError, match="ids"):
        store.delete()
    with pytest.raises(ValueError, match="not both"):
        store.delete(["1"], filter={"a": 1})
    assert store.delete([]) is True


def test_delete_in_missing_partition_is_noop() -> None:
    store, _ = make_store(partition="nobody")
    assert store.delete(["1"]) is True


# --- reads ---------------------------------------------------------------------


def test_get_by_ids_keeps_order_and_skips_missing() -> None:
    store, client = make_store()
    store.add_texts(["a", "b"], [{"n": 1}, {"n": 2}], ids=["1", "2"])
    docs = store.get_by_ids(["2", "missing", "1"])
    assert [d.id for d in docs] == ["2", "1"]
    assert docs[0] == Document(id="2", page_content="b", metadata={"n": 2})
    _, _, _, fields, opts = client.last("get")
    assert fields == ["text", "metadata"]
    assert opts == {"lsn": store.last_lsn}


def test_reads_in_missing_partition_return_empty() -> None:
    store, _ = make_store(partition="tenant-x")
    assert store.get_by_ids(["1"]) == []
    assert store.similarity_search("anything") == []


def test_read_options_precedence() -> None:
    store, _ = make_store(consistency="indexed")
    assert store._read_options() == {"consistency": "indexed"}
    store.add_texts(["a"])
    assert store._read_options() == {"lsn": store.last_lsn}
    assert store._read_options(consistency="strong") == {"consistency": "strong"}
    assert store._read_options(lsn="42") == {"lsn": "42"}
    store.read_your_writes = False
    assert store._read_options() == {"consistency": "indexed"}


def test_with_partition_shares_client() -> None:
    store, client = make_store()
    store.add_texts(["a"])
    tenant = store.with_partition("acme")
    assert tenant.partition == "acme"
    assert tenant.client is client
    assert tenant.last_lsn is None
    tenant.add_texts(["b"], ids=["b"])
    assert client.last("upsert")[1] == ("test", "acme")
    assert store.partition is None


def test_semantic_search_query() -> None:
    store, client = make_store()
    store.add_texts(["a"], ids=["1"])
    client.query_results = [
        {"_id": "1", "text": "a", "metadata": {"n": 1}, "semantic_score": 2.5}
    ]
    results = store.similarity_search_with_score("query", k=3, filter={"n": 1})
    assert results == [(Document(id="1", page_content="a", metadata={"n": 1}), 2.5)]
    q = repr(client.last("query")[2])
    assert 'SemanticSimilarity { field: "text", query: "query" }' in q
    assert "field(metadata.n)" in q
    assert "Limit { k: 3 }" in q
    assert "order: Desc" in q


def test_keyword_search_query() -> None:
    store, client = make_store()
    store.add_texts(["a"])
    store.keyword_search("rust memory", k=2, keyword_match_all=True)
    q = repr(client.last("query")[2])
    assert "KeywordScore" in q or "Bm25" in q
    assert "all: true, should: false" in q


def test_hybrid_search_query_and_score() -> None:
    store, client = make_store(embedding=DeterministicFakeEmbedding(size=3))
    store.add_texts(["a"], ids=["1"])
    client.query_results = [
        {
            "_id": "1",
            "text": "a",
            "semantic_score": 2.0,
            "keyword_score": 4.0,
            "dense_score": 0.5,
        }
    ]
    results = store.similarity_search_with_score(
        "q", mode="hybrid", weights={"semantic": 1.0, "keyword": 0.5, "dense": 2.0}
    )
    assert results[0][1] == pytest.approx(2.0 + 2.0 + 1.0)
    q = repr(client.last("query")[2])
    assert "SemanticSimilarity" in q
    assert "VectorScore" in q or "VectorDistance" in q
    assert "should: true" in q  # BM25 boosts without excluding documents
    assert "Coalesce" in q


def test_hybrid_keyword_required_uses_match() -> None:
    store, client = make_store()
    store.add_texts(["a"])
    store.hybrid_search("q", keyword_required=True)
    assert "should: false" in repr(client.last("query")[2])


def test_hybrid_rejects_unconfigured_signal() -> None:
    store, _ = make_store()
    with pytest.raises(ValueError, match="not configured"):
        store.hybrid_search("q", weights={"dense": 1.0})


def test_hybrid_defaults_to_available_signals() -> None:
    store, _ = make_store(sparse_embedding=FakeSparse())
    assert store._signals(RetrievalMode.HYBRID, None) == {
        "semantic": 1.0,
        "keyword": 1.0,
        "sparse": 1.0,
    }


def test_custom_ranking_callable() -> None:
    store, client = make_store()
    store.add_texts(["a"])
    store.similarity_search(
        "q",
        ranking=lambda s: s.semantic * 0.8 + s.keyword.coalesce(0.0) * 0.2,
        mode="hybrid",
    )
    q = repr(client.last("query")[2])
    assert "field(semantic_score)" in q and "literal(Float(0.8))" in q


def test_ranking_must_be_expression() -> None:
    store, _ = make_store()
    store.add_texts(["a"])
    with pytest.raises(ValueError, match="ranking"):
        store.similarity_search("q", ranking=42)


def test_euclidean_dense_sorts_ascending() -> None:
    store, client = make_store(
        embedding=DeterministicFakeEmbedding(size=3), distance_metric="euclidean"
    )
    store.add_texts(["a"])
    store.similarity_search("q")
    assert "order: Asc" in repr(client.last("query")[2])


def test_sparse_and_multi_vector_queries() -> None:
    store, client = make_store(
        sparse_embedding=FakeSparse(), multi_vector_embedding=FakeMulti()
    )
    store.add_texts(["a"])
    store.similarity_search("q", mode="sparse")
    assert "sparse_vector" in repr(client.last("query")[2])
    store.similarity_search("q", mode="multi_vector")
    assert "multi_vector" in repr(client.last("query")[2])


def test_unknown_search_kwarg() -> None:
    store, _ = make_store()
    store.add_texts(["a"])
    with pytest.raises(TypeError, match="Unexpected"):
        store.similarity_search("q", bogus=1)


@pytest.mark.parametrize("k", [0, 10_001])
def test_k_bounds(k: int) -> None:
    store, _ = make_store()
    with pytest.raises(ValueError, match="between 1 and 10000"):
        store.similarity_search("q", k=k)


def test_relevance_scores_only_for_dense() -> None:
    store, _ = make_store()
    with pytest.raises(NotImplementedError):
        store._select_relevance_score_fn()
    dense, _ = make_store(embedding=DeterministicFakeEmbedding(size=3))
    assert dense._select_relevance_score_fn()(1.0) == 1.0
    assert dense._select_relevance_score_fn()(-1.0) == 0.0


@pytest.mark.parametrize("metric", ["cosine", "dot_product", "euclidean"])
def test_relevance_scores_keep_ranking_order(metric: str) -> None:
    store, client = make_store(
        embedding=DeterministicFakeEmbedding(size=3), distance_metric=metric
    )
    store.add_texts(["a", "b"], ids=["1", "2"])
    best, worst = (0.1, 0.9) if metric == "euclidean" else (0.95, -0.5)
    client.query_results = [
        {"_id": "1", "text": "a", "dense_score": best},
        {"_id": "2", "text": "b", "dense_score": worst},
    ]
    scored = store.similarity_search_with_relevance_scores("q", k=2)
    assert [d.id for d, _ in scored] == ["1", "2"]
    assert scored[0][1] > scored[1][1]
    assert all(0.0 <= s <= 1.0 for _, s in scored)
    kept = store.similarity_search_with_relevance_scores("q", k=2, score_threshold=0.5)
    assert [d.id for d, _ in kept] == ["1"]


def test_relevance_scores_follow_per_call_mode() -> None:
    store, client = make_store(
        embedding=DeterministicFakeEmbedding(size=3), retrieval_mode="semantic"
    )
    store.add_texts(["a"], ids=["1"])
    client.query_results = [{"_id": "1", "text": "a", "dense_score": 1.0}]
    assert store.similarity_search_with_relevance_scores("q", mode="dense")[0][1] == 1
    with pytest.raises(NotImplementedError, match="hybrid"):
        store.similarity_search_with_relevance_scores("q", mode="hybrid")
    with pytest.raises(NotImplementedError):
        store.similarity_search_with_relevance_scores("q")


def test_large_upserts_are_split_by_size() -> None:
    class BigMulti(MultiVectorEmbeddings):
        def embed_documents(self, texts: list[str]) -> list[list[list[float]]]:
            return [[[0.1] * 128 for _ in range(300)] for _ in texts]

        def embed_query(self, text: str) -> list[list[float]]:
            return [[0.1] * 128]

    store, client = make_store(multi_vector_embedding=BigMulti())
    ids = store.add_texts([f"doc {i}" for i in range(100)])
    upserts = [c for c in client.calls if c[0] == "upsert"]
    assert len(upserts) > 1
    assert sum(len(c[2]) for c in upserts) == 100 == len(ids)


def test_from_texts_accepts_batch_size() -> None:
    client = FakeClient()
    TopKVectorStore.from_texts(
        ["a", "b", "c"],
        collection_name="c",
        client=client,  # type: ignore[arg-type]
        batch_size=2,
    )
    assert len([c for c in client.calls if c[0] == "upsert"]) == 2


def test_lsn_reset_after_partition_delete() -> None:
    store, _ = make_store(partition="acme")
    store.add_texts(["a"])
    assert store.last_lsn is not None
    store.delete_partition()
    assert store.last_lsn is None


def test_ranking_rejects_text_expression_and_supports_ascending() -> None:
    from topk_sdk.query import field, match

    store, client = make_store()
    store.add_texts(["a"])
    with pytest.raises(ValueError, match="ranking"):
        store.similarity_search("q", ranking=match("x"))
    store.similarity_search("q", ranking=field("metadata.year"), ranking_ascending=True)
    assert "order: Asc" in repr(client.last("query")[2])


def test_delete_rejects_text_filter() -> None:
    from topk_sdk.query import match

    store, _ = make_store()
    with pytest.raises(ValueError):
        store.delete(filter=match("x"))


def test_mmr_fetches_vectors_by_id() -> None:
    store, client = make_store(embedding=DeterministicFakeEmbedding(size=3))
    store.add_texts(["a", "b", "c"], ids=["1", "2", "3"])
    client.query_results = [
        {"_id": i, "text": t, "dense_score": 1.0} for i, t in [("1", "a"), ("2", "b")]
    ]
    docs = store.max_marginal_relevance_search("q", k=1, fetch_k=2)
    assert len(docs) == 1
    assert client.last("get")[3] == ["dense_vector"]


def test_mmr_requires_embedding() -> None:
    store, _ = make_store()
    with pytest.raises(ValueError, match="embedding"):
        store.max_marginal_relevance_search("q")


def test_search_with_query_adds_extra_fields_to_metadata() -> None:
    from topk_sdk.query import select

    store, client = make_store()
    store.add_texts(["a"])
    client.query_results = [{"_id": "1", "text": "a", "metadata": {"x": 1}, "s": 3.0}]
    docs = store.search_with_query(select("text").limit(1))
    assert docs[0].metadata == {"x": 1, "s": 3.0}


def test_retriever_passes_mode() -> None:
    store, client = make_store()
    store.add_texts(["a"])
    store.as_retriever(search_kwargs={"k": 2, "mode": "keyword"}).invoke("rust")
    assert "should: false" in repr(client.last("query")[2])


def test_from_texts_without_embedding() -> None:
    client = FakeClient()
    store = TopKVectorStore.from_texts(
        ["a", "b"], collection_name="c", client=client, ids=["1", "2"]
    )
    assert store.retrieval_mode == RetrievalMode.SEMANTIC
    assert len(client.last("upsert")[2]) == 2
    with pytest.raises(ValueError, match="collection_name"):
        TopKVectorStore.from_texts(["a"], client=client)


def test_from_documents_without_embedding() -> None:
    client = FakeClient()
    store = TopKVectorStore.from_documents(
        [Document(page_content="a", id="x")], collection_name="c", client=client
    )
    assert store.get_by_ids(["x"])[0].page_content == "a"


def test_partition_management() -> None:
    store, client = make_store(partition="acme")
    store.add_texts(["a"])
    assert store.list_partitions() == ["acme"]
    store.delete_partition()
    assert client.last("delete_partition")[2] == "acme"
    unscoped, _ = make_store()
    with pytest.raises(ValueError, match="partition"):
        unscoped.delete_partition()


# --- async ---------------------------------------------------------------------


async def test_afrom_texts_uses_only_the_async_client() -> None:
    client = FakeClient()

    class NoSyncClient:
        def __getattr__(self, name: str) -> Any:
            raise AssertionError(f"sync client used: {name}")

    store = await TopKVectorStore.afrom_texts(
        ["a"],
        collection_name="c",
        client=NoSyncClient(),  # type: ignore[arg-type]
        async_client=FakeAsyncClient(client),  # type: ignore[arg-type]
    )
    assert client.last("create")[1] == "c"
    assert store.retrieval_mode == RetrievalMode.SEMANTIC


async def test_async_keyword_and_hybrid() -> None:
    client = FakeClient()
    store = await TopKVectorStore.afrom_texts(
        ["a"],
        collection_name="c",
        client=client,  # type: ignore[arg-type]
        async_client=FakeAsyncClient(client),  # type: ignore[arg-type]
    )
    await store.asimilarity_search("q", mode="keyword")
    assert "should: false" in repr(client.last("query")[2])
    await store.asimilarity_search("q", mode="hybrid")
    assert "should: true" in repr(client.last("query")[2])


async def test_async_roundtrip() -> None:
    client = FakeClient()
    store = await TopKVectorStore.afrom_texts(
        ["a"],
        DeterministicFakeEmbedding(size=3),
        [{"k": "v"}],
        ids=["1"],
        collection_name="c",
        client=client,
        async_client=FakeAsyncClient(client),
    )
    assert (await store.aget_by_ids(["1"]))[0].metadata == {"k": "v"}
    client.query_results = [{"_id": "1", "text": "a", "dense_score": 0.9}]
    assert (await store.asimilarity_search("a"))[0].id == "1"
    assert (await store.asimilarity_search_by_vector([0.1, 0.2, 0.3]))[0].id == "1"
    docs = await store.amax_marginal_relevance_search("a", k=1)
    assert docs[0].id == "1"
    assert await store.adelete(["1"]) is True
