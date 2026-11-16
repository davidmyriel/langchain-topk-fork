"""LangChain standard vector store tests, run against live TopK.

Each test gets an empty store by writing to a fresh partition of one shared
collection, which is how TopK isolates tenants.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Iterator

import pytest
from langchain_core.vectorstores import VectorStore
from langchain_tests.integration_tests import VectorStoreIntegrationTests

from langchain_topk import TopKVectorStore


@pytest.fixture(scope="module")
def dense_base_store(
    new_collection_name: Callable[[str], str], region: str
) -> TopKVectorStore:
    return TopKVectorStore(
        new_collection_name("std-dense"),
        embedding=VectorStoreIntegrationTests.get_embeddings(),
        region=region,
    )


@pytest.fixture(scope="module")
def semantic_base_store(
    new_collection_name: Callable[[str], str], region: str
) -> TopKVectorStore:
    return TopKVectorStore(new_collection_name("std-semantic"), region=region)


class TestTopKDenseStandard(VectorStoreIntegrationTests):
    """Standard suite with client-side (dense) embeddings."""

    @pytest.fixture
    def vectorstore(self, dense_base_store: TopKVectorStore) -> Iterator[VectorStore]:
        yield dense_base_store.with_partition(f"t-{uuid.uuid4().hex[:12]}")


class TestTopKSemanticStandard(VectorStoreIntegrationTests):
    """Standard suite with server-side embeddings: no embedding model at all."""

    @pytest.fixture
    def vectorstore(
        self, semantic_base_store: TopKVectorStore
    ) -> Iterator[VectorStore]:
        yield semantic_base_store.with_partition(f"t-{uuid.uuid4().hex[:12]}")
