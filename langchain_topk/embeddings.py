"""Sparse and multi-vector embedding interfaces for `TopKVectorStore`.

LangChain's `Embeddings` interface only describes dense vectors. TopK also
indexes sparse vectors (for example SPLADE) with exact, 100% recall search, and
multi-vectors (one vector per token, ColBERT-style) scored with MaxSim. These
small interfaces let you plug such models into `TopKVectorStore`.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

SparseVector = dict[int, float]
"""A sparse vector as a mapping from dimension index to value."""

MultiVector = list[list[float]]
"""A multi-vector (matrix): one row per token, all rows of the same width."""


class SparseEmbeddings(ABC):
    """Interface for sparse embedding models such as SPLADE or BM42."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        """Embed documents into sparse vectors."""

    @abstractmethod
    def embed_query(self, text: str) -> SparseVector:
        """Embed a query into a sparse vector."""

    async def aembed_documents(self, texts: list[str]) -> list[SparseVector]:
        """Asynchronously embed documents into sparse vectors."""
        return await asyncio.get_running_loop().run_in_executor(
            None, self.embed_documents, texts
        )

    async def aembed_query(self, text: str) -> SparseVector:
        """Asynchronously embed a query into a sparse vector."""
        return await asyncio.get_running_loop().run_in_executor(
            None, self.embed_query, text
        )


class MultiVectorEmbeddings(ABC):
    """Interface for late-interaction (multi-vector) models such as ColBERT."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[MultiVector]:
        """Embed documents into multi-vectors (one row per token)."""

    @abstractmethod
    def embed_query(self, text: str) -> MultiVector:
        """Embed a query into a multi-vector (one row per token)."""

    async def aembed_documents(self, texts: list[str]) -> list[MultiVector]:
        """Asynchronously embed documents into multi-vectors."""
        return await asyncio.get_running_loop().run_in_executor(
            None, self.embed_documents, texts
        )

    async def aembed_query(self, text: str) -> MultiVector:
        """Asynchronously embed a query into a multi-vector."""
        return await asyncio.get_running_loop().run_in_executor(
            None, self.embed_query, text
        )


def _import_fastembed() -> Any:
    try:
        import fastembed
    except ImportError as e:  # pragma: no cover - depends on optional extra
        msg = (
            "This class requires the `fastembed` package. Install it with "
            "`pip install 'langchain-topk[fastembed]'`."
        )
        raise ImportError(msg) from e
    return fastembed


class FastEmbedSparse(SparseEmbeddings):
    """Sparse embeddings computed locally with FastEmbed.

    The default model is SPLADE++ (`prithivida/Splade_PP_en_v1`). Requires the
    `fastembed` extra: `pip install "langchain-topk[fastembed]"`.

    Example:
        ```python
        from langchain_topk import FastEmbedSparse

        sparse = FastEmbedSparse()
        vector = sparse.embed_query("memory safe systems programming")
        ```
    """

    def __init__(
        self,
        model_name: str = "prithivida/Splade_PP_en_v1",
        batch_size: int = 256,
        **kwargs: Any,
    ) -> None:
        fastembed = _import_fastembed()
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = fastembed.SparseTextEmbedding(model_name=model_name, **kwargs)

    @staticmethod
    def _to_dict(embedding: Any) -> SparseVector:
        return {
            int(i): float(v)
            for i, v in zip(embedding.indices, embedding.values, strict=True)
        }

    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        """Embed documents into sparse vectors."""
        return [
            self._to_dict(e)
            for e in self._model.embed(texts, batch_size=self.batch_size)
        ]

    def embed_query(self, text: str) -> SparseVector:
        """Embed a query into a sparse vector."""
        return self._to_dict(next(iter(self._model.query_embed(text))))


class FastEmbedLateInteraction(MultiVectorEmbeddings):
    """Multi-vector (late-interaction) embeddings computed locally with FastEmbed.

    The default model is `answerdotai/answerai-colbert-small-v1` (96 dimensions).
    Requires the `fastembed` extra: `pip install "langchain-topk[fastembed]"`.

    Example:
        ```python
        from langchain_topk import FastEmbedLateInteraction

        colbert = FastEmbedLateInteraction()
        matrix = colbert.embed_query("memory safe systems programming")
        ```
    """

    def __init__(
        self,
        model_name: str = "answerdotai/answerai-colbert-small-v1",
        batch_size: int = 64,
        **kwargs: Any,
    ) -> None:
        fastembed = _import_fastembed()
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = fastembed.LateInteractionTextEmbedding(
            model_name=model_name, **kwargs
        )

    @staticmethod
    def _to_rows(matrix: Any) -> MultiVector:
        return [[float(x) for x in row] for row in matrix]

    def embed_documents(self, texts: list[str]) -> list[MultiVector]:
        """Embed documents into multi-vectors."""
        return [
            self._to_rows(m)
            for m in self._model.embed(texts, batch_size=self.batch_size)
        ]

    def embed_query(self, text: str) -> MultiVector:
        """Embed a query into a multi-vector."""
        return self._to_rows(next(iter(self._model.query_embed(text))))


def normalize_sparse(vector: Any) -> SparseVector:
    """Coerce a sparse vector into a `{index: value}` dict.

    Accepts a dict, or any object with `indices` and `values` attributes (the
    shape used by FastEmbed and several other libraries).
    """
    if isinstance(vector, dict):
        return {int(k): float(v) for k, v in vector.items()}
    indices = getattr(vector, "indices", None)
    values = getattr(vector, "values", None)
    if indices is not None and values is not None:
        return {int(i): float(v) for i, v in zip(indices, values, strict=True)}
    msg = (
        "Sparse embeddings must be dicts of {index: value} or objects with "
        f"`indices` and `values`, got {type(vector).__name__}."
    )
    raise TypeError(msg)


def normalize_multi_vector(matrix: Any) -> MultiVector:
    """Coerce a multi-vector (list of rows or 2-D array) into a list of rows."""
    rows: Sequence[Any] = matrix.tolist() if hasattr(matrix, "tolist") else matrix
    return [[float(x) for x in row] for row in rows]
