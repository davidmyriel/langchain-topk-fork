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


class TestTopKDenseStandard(VectorStoreIntegrationTests):
    """Standard suite with client-side (dense) embeddings."""

    @pytest.fixture(scope="class")
    def base_store(
        self, new_collection_name: Callable[[str], str], region: str
    ) -> TopKVectorStore:
        return TopKVectorStore(
            new_collection_name("std-dense"),
            embedding=self.get_embeddings(),
            region=region,
        )

    @pytest.fixture
    def vectorstore(self, base_store: TopKVectorStore) -> Iterator[VectorStore]:
        yield base_store.with_partition(f"t-{uuid.uuid4().hex[:12]}")


class TestTopKSemanticStandard(VectorStoreIntegrationTests):
    """Standard suite with server-side embeddings: no embedding model at all."""

    @pytest.fixture(scope="class")
    def base_store(
        self, new_collection_name: Callable[[str], str], region: str
    ) -> TopKVectorStore:
        return TopKVectorStore(new_collection_name("std-semantic"), region=region)

    @pytest.fixture
    def vectorstore(self, base_store: TopKVectorStore) -> Iterator[VectorStore]:
        yield base_store.with_partition(f"t-{uuid.uuid4().hex[:12]}")
