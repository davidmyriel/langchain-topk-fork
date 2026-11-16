# langchain-topk

[![PyPI - Version](https://img.shields.io/pypi/v/langchain-topk?label=%20)](https://pypi.org/project/langchain-topk/)
[![PyPI - License](https://img.shields.io/pypi/l/langchain-topk)](https://opensource.org/licenses/MIT)

LangChain integration for [TopK](https://topk.io), a search database with built-in semantic search, BM25 keyword search, dense, sparse and multi-vector search, metadata filtering and custom scoring, all available in one query.

- **No embedding model needed.** TopK embeds and reranks on the server, so `from_texts` and `similarity_search` work without an `Embeddings` object or an embedding API key.
- **True hybrid search.** Semantic, BM25, dense, sparse and multi-vector scores and metadata filters in a single query, with your own weights or scoring expression.
- **Keyword search (BM25)** as a first-class retrieval mode.
- **Sparse vectors** (SPLADE) with exact search and 100% recall, and **multi-vector** late interaction (ColBERT-style MaxSim).
- **Multi-tenancy** with partitions, and **read consistency** you choose per store or per call (balanced, indexed, strong, or LSN).
- LangChain metadata filters, a self-query translator, MMR, async, and the LangChain standard test suite.

## Installation

```bash
pip install -U langchain-topk
```

Create an API key in the [TopK console](https://console.topk.io) and set it with your [region](https://docs.topk.io/regions):

```bash
export TOPK_API_KEY="your-api-key"
export TOPK_REGION="aws-us-east-1-elastica"
```

## Quick start

```python
from langchain_core.documents import Document
from langchain_topk import TopKVectorStore

vector_store = TopKVectorStore.from_documents(
    [
        Document(page_content="Rust guarantees memory safety without a garbage collector.", metadata={"year": 2015}),
        Document(page_content="Python is known for readable syntax and scientific libraries.", metadata={"year": 1991}),
        Document(page_content="Go has a garbage collector and built-in goroutines for concurrency.", metadata={"year": 2012}),
    ],
    collection_name="langchain-quickstart",
)

# Semantic search, embedded on the server
vector_store.similarity_search("memory safe systems programming", k=1)

# BM25 keyword search
vector_store.keyword_search("garbage collector", k=2)

# Hybrid search with a metadata filter, in one query
vector_store.similarity_search(
    "garbage collection",
    k=2,
    mode="hybrid",
    weights={"semantic": 1.0, "keyword": 0.2},
    filter={"year": {"$gte": 2010}},
)

# As a retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 2, "mode": "hybrid"})
retriever.invoke("garbage collection")
```

## Retrieval modes

Set a default with `retrieval_mode=` or pass `mode=` to any search call (also through `as_retriever(search_kwargs=...)`):

| Mode | What it scores | Needs |
|------|----------------|-------|
| `"semantic"` | TopK's server-side embeddings and reranking | Nothing (default without `embedding`) |
| `"keyword"` | BM25 | Nothing |
| `"dense"` | Your own embeddings | `embedding=` any LangChain `Embeddings` |
| `"sparse"` | Sparse vectors such as SPLADE (exact, 100% recall) | `sparse_embedding=`, for example `FastEmbedSparse()` |
| `"multi_vector"` | Late-interaction multi-vectors, scored with MaxSim | `multi_vector_embedding=`, for example `FastEmbedLateInteraction()` |
| `"hybrid"` | Any of the above plus BM25, combined in one query | Nothing extra |

`FastEmbedSparse` and `FastEmbedLateInteraction` run models locally and need the `fastembed` extra: `pip install "langchain-topk[fastembed]"`.

## More

```python
from topk_sdk.query import field

# Custom scoring: combine search scores with metadata
vector_store.similarity_search(
    "programming language",
    mode="hybrid",
    ranking=lambda s: s.semantic + s.keyword.coalesce(0.0) * 0.1 + (field("metadata.year") > 2000).choose(0.5, 0.0),
)

# Filters: dict operators ($eq, $ne, $gt, $gte, $lt, $lte, $in, $nin, $contains,
# $starts_with, $exists, $and, $or, $not) or any TopK expression
vector_store.similarity_search("language", filter={"$or": [{"year": {"$lt": 1995}}, {"year": 2015}]})
vector_store.similarity_search("language", filter=field("metadata.year") > 2010)

# Multi-tenancy: one partition per tenant
acme = vector_store.with_partition("acme")
acme.add_texts(["Acme sells rocket-powered roller skates."])
acme.delete_partition()

# Consistency: balanced (default), "indexed", "strong", or an LSN
vector_store.similarity_search("garbage collector", consistency="strong")
vector_store.get_by_ids(["some-id"], lsn=vector_store.last_lsn)
```

`TopKTranslator` plugs into LangChain's self-querying retriever (`structured_query_translator=TopKTranslator()`), and dense stores support maximal marginal relevance (`max_marginal_relevance_search`).

## Documentation

- [TopK documentation](https://docs.topk.io): [semantic search](https://docs.topk.io/guides/semantic-search), [keyword search](https://docs.topk.io/guides/keyword-search), [true hybrid search](https://docs.topk.io/guides/true-hybrid-search), [sparse vectors](https://docs.topk.io/guides/sparse-vector-search), [multi-vector search](https://docs.topk.io/guides/multi-vector-search)
- [TopK console](https://console.topk.io): API keys and usage
- [LangChain vector stores](https://docs.langchain.com/oss/python/integrations/vectorstores)

## Development

```bash
uv sync --all-groups
make test              # unit tests, no network
make integration_test  # live tests; needs TOPK_API_KEY (and optionally TOPK_REGION)
make lint
```

Integration tests create collections named `langchain-topk-it-<run id>-...` and delete only the collections they created.
