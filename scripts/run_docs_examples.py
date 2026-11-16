"""Run every Python example in a docs page against live TopK, in order.

Usage: python scripts/run_docs_examples.py docs/langchain/topk.mdx

* Install commands and credential / tracing setup blocks are skipped.
* `<EmbeddingTabs/>` is replaced by a local FastEmbed model (no API key).
* `<ChatModelTabs />` is replaced by a fake chat model that answers with a
  fixed structured query, so the self-query example runs without an LLM key.
* Collection names get a unique prefix, and those collections are deleted at
  the end. Nothing else in the project is touched.
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys
import uuid

from topk_sdk import Client

# Setup blocks: installs, credentials, and third-party model providers that
# need their own API keys (replaced below by local or fake models).
SKIP_MARKERS = (
    "pip install",
    "uv add",
    "getpass",
    "LANGSMITH",
    "OpenAIEmbeddings(",
    "init_chat_model(",
)


def blocks(path: str) -> list[tuple[str, str]]:
    """Return (title, code) for every Python code block in the page."""
    text = open(path).read()
    out = []
    for m in re.finditer(r"^( *)```python([^\n]*)\n(.*?)^\1```", text, re.S | re.M):
        indent, title, body = m.group(1), m.group(2).strip(), m.group(3)
        body = "\n".join(line[len(indent) :] for line in body.splitlines())
        out.append((title, body))
    return out


def run_block(code: str, ns: dict) -> None:
    tree = ast.parse(code)
    last = tree.body[-1] if tree.body else None
    if isinstance(last, ast.Expr):
        exec(compile(ast.Module(tree.body[:-1], []), "<doc>", "exec"), ns)
        value = eval(compile(ast.Expression(last.value), "<doc>", "eval"), ns)
        if value is not None:
            print(repr(value))
    else:
        exec(compile(tree, "<doc>", "exec"), ns)


def setup_embeddings(ns: dict) -> None:
    from fastembed import TextEmbedding
    from langchain_core.embeddings import Embeddings

    class LocalEmbeddings(Embeddings):
        def __init__(self) -> None:
            self.model = TextEmbedding("BAAI/bge-small-en-v1.5")

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [v.tolist() for v in self.model.embed(texts)]

        def embed_query(self, text: str) -> list[float]:
            return next(iter(self.model.query_embed(text))).tolist()

    ns["embeddings"] = LocalEmbeddings()


def setup_chat_model(ns: dict) -> None:
    from langchain_core.language_models import FakeListChatModel

    answer = json.dumps({"query": "garbage collector", "filter": 'gt("year", 2000)'})
    ns["model"] = FakeListChatModel(responses=[f"```json\n{answer}\n```"])


def main(path: str) -> int:
    prefix = f"docs-run-{uuid.uuid4().hex[:6]}-"
    created: set[str] = set()
    found = blocks(path)
    ns: dict = {"__name__": "__docs__"}
    failures = 0
    for i, (title, code) in enumerate(found, 1):
        if any(m in code for m in SKIP_MARKERS):
            print(f"\n### [{i}] {title} -- skipped (setup)")
            continue
        if "embeddings," in code and "embeddings" not in ns:
            setup_embeddings(ns)
        if "llm=model" in code and "model" not in ns:
            setup_chat_model(ns)

        def rename(m: re.Match) -> str:
            name = prefix + m.group(1)
            created.add(name)
            return f'collection_name="{name}"'

        code = re.sub(r'collection_name="([^"]+)"', rename, code)
        print(f"\n### [{i}] {title}")
        try:
            run_block(code, ns)
        except Exception as e:
            failures += 1
            print(f"!!! FAILED: {type(e).__name__}: {e}")
    client = Client(
        api_key=os.environ["TOPK_API_KEY"], region=os.environ["TOPK_REGION"]
    )
    for name in created:
        try:
            client.collections().delete(name)
        except Exception:
            pass
    print(f"\n{len(found)} blocks, {failures} failed; deleted {sorted(created)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
