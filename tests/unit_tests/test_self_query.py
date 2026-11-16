"""SelfQueryRetriever wiring with `TopKTranslator`, using a fake chat model."""

from __future__ import annotations

import json

import pytest
from langchain_core.language_models import FakeListChatModel

from langchain_topk import TopKTranslator, TopKVectorStore
from tests.unit_tests.fake_topk import FakeAsyncClient, FakeClient

pytest.importorskip("langchain_classic")
pytest.importorskip("lark")


def test_self_query_retriever_builds_topk_filter() -> None:
    from langchain_classic.chains.query_constructor.schema import AttributeInfo
    from langchain_classic.retrievers.self_query.base import SelfQueryRetriever

    client = FakeClient()
    store = TopKVectorStore(
        "test",
        client=client,  # type: ignore[arg-type]
        async_client=FakeAsyncClient(client),  # type: ignore[arg-type]
    )
    store.add_texts(["a"])
    answer = json.dumps(
        {
            "query": "garbage collector",
            "filter": 'and(gt("year", 2000), ne("lang", "java"))',
        }
    )
    retriever = SelfQueryRetriever.from_llm(
        llm=FakeListChatModel(responses=[f"```json\n{answer}\n```"]),
        vectorstore=store,
        document_contents="A short description of a programming language",
        metadata_field_info=[
            AttributeInfo(name="lang", description="Language name", type="string"),
            AttributeInfo(name="year", description="Year released", type="integer"),
        ],
        structured_query_translator=TopKTranslator(),
    )
    retriever.invoke("languages with a garbage collector released after 2000")
    q = repr(client.last("query")[2])
    assert 'query: "garbage collector"' in q
    assert "field(metadata.year), op=Gt" in q
    assert "field(metadata.lang), op=NotEq" in q
