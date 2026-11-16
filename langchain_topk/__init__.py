"""LangChain integration for TopK."""

from importlib import metadata

from langchain_topk._filters import TopKTranslator
from langchain_topk.embeddings import (
    FastEmbedLateInteraction,
    FastEmbedSparse,
    MultiVectorEmbeddings,
    SparseEmbeddings,
)
from langchain_topk.vectorstores import RetrievalMode, TextIndex, TopKVectorStore

try:
    __version__ = metadata.version(__package__ or "langchain_topk")
except metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = ""
del metadata

__all__ = [
    "FastEmbedLateInteraction",
    "FastEmbedSparse",
    "MultiVectorEmbeddings",
    "RetrievalMode",
    "SparseEmbeddings",
    "TextIndex",
    "TopKTranslator",
    "TopKVectorStore",
    "__version__",
]
