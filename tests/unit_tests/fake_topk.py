"""In-memory stand-in for `topk_sdk.Client`, recording every call."""

from __future__ import annotations

from typing import Any

from topk_sdk.error import CollectionNotFoundError, PartitionNotFoundError


class FakeCollection:
    def __init__(self, owner: FakeClient, name: str, partition: str | None) -> None:
        self.owner = owner
        self.name = name
        self.partition = partition

    def _key(self) -> tuple[str, str | None]:
        return (self.name, self.partition)

    def _store(self) -> dict[str, dict[str, Any]]:
        if self._key() not in self.owner.data:
            raise PartitionNotFoundError("partition not found")
        return self.owner.data[self._key()]

    def upsert(self, docs: list[dict[str, Any]]) -> str:
        self.owner.calls.append(("upsert", self._key(), docs))
        store = self.owner.data.setdefault(self._key(), {})
        for d in docs:
            store[d["_id"]] = dict(d)
        self.owner.lsn += 1
        return str(self.owner.lsn)

    def delete(self, target: Any) -> str:
        self.owner.calls.append(("delete", self._key(), target))
        store = self._store()
        if isinstance(target, list):
            for i in target:
                store.pop(i, None)
        self.owner.lsn += 1
        return str(self.owner.lsn)

    def get(self, ids: list[str], fields: list[str] | None = None, **opts: Any) -> dict:
        self.owner.calls.append(("get", self._key(), ids, fields, opts))
        store = self._store()
        out = {}
        for i in ids:
            if i in store:
                doc = store[i]
                out[i] = {k: v for k, v in doc.items() if fields is None or k in fields}
        return out

    def query(self, q: Any, **opts: Any) -> list[dict[str, Any]]:
        self.owner.calls.append(("query", self._key(), q, opts))
        self._store()
        return list(self.owner.query_results)

    def delete_partition(self, name: str) -> None:
        self.owner.calls.append(("delete_partition", self.name, name))
        self.owner.data.pop((self.name, name), None)

    def list_partitions(self) -> list[Any]:
        class P:
            def __init__(self, name: str) -> None:
                self.name = name

        return [P(p) for (c, p) in self.owner.data if c == self.name and p]


class FakeAsyncCollection:
    def __init__(self, sync: FakeCollection) -> None:
        self.sync = sync

    async def upsert(self, docs: list[dict[str, Any]]) -> str:
        return self.sync.upsert(docs)

    async def delete(self, target: Any) -> str:
        return self.sync.delete(target)

    async def get(
        self, ids: list[str], fields: list[str] | None = None, **o: Any
    ) -> dict:
        return self.sync.get(ids, fields, **o)

    async def query(self, q: Any, **opts: Any) -> list[dict[str, Any]]:
        return self.sync.query(q, **opts)


class FakeCollections:
    def __init__(self, owner: FakeClient) -> None:
        self.owner = owner

    def get(self, name: str) -> Any:
        if name not in self.owner.schemas:
            raise CollectionNotFoundError("collection not found")
        return self.owner.schemas[name]

    def create(self, name: str, schema: dict[str, Any]) -> Any:
        self.owner.calls.append(("create", name, schema))
        self.owner.schemas[name] = schema
        self.owner.data.setdefault((name, None), {})
        return schema

    def delete(self, name: str) -> None:
        self.owner.calls.append(("delete_collection", name))
        self.owner.schemas.pop(name, None)


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.schemas: dict[str, Any] = {}
        self.data: dict[tuple[str, str | None], dict[str, dict[str, Any]]] = {}
        self.query_results: list[dict[str, Any]] = []
        self.lsn = 0

    def collection(self, name: str, partition: str | None = None) -> FakeCollection:
        return FakeCollection(self, name, partition)

    def collections(self) -> FakeCollections:
        return FakeCollections(self)

    def last(self, kind: str) -> tuple:
        return next(c for c in reversed(self.calls) if c[0] == kind)


class FakeAsyncClient:
    def __init__(self, sync: FakeClient) -> None:
        self.sync = sync

    def collection(
        self, name: str, partition: str | None = None
    ) -> FakeAsyncCollection:
        return FakeAsyncCollection(self.sync.collection(name, partition))
