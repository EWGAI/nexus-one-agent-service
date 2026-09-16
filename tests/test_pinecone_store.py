"""Pinecone adapter tests.

The unit tests drive the adapter through an in-memory fake that implements the
slice of the Pinecone SDK the adapter uses, so both the ``namespace`` and the
``index`` isolation strategies are covered without network access. The
``integration`` test at the bottom runs the same flow against the real service
and self-skips unless ``PINECONE_API_KEY`` is exported.
"""

from __future__ import annotations

from collections.abc import Iterator
import math
import os
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
import pytest

from agent_service.core.exceptions import ConfigurationError
from agent_service.kb.embeddings import build_embeddings
from agent_service.kb.filters import build_predicate
from agent_service.kb.ingestion import content_hash
from agent_service.kb.pinecone_store import PineconeStore

from .conftest import make_settings

HR_TEXT = "Employees accrue 1.75 days of paid leave per completed month of service."
FIN_TEXT = "Accounts payable invoices above 50000 require two approvals before payment."


def cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return dot / norm if norm else 0.0


class FakeIndex:
    """Minimal in-memory stand-in for ``pinecone.Index``."""

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.namespaces: dict[str, dict[str, dict[str, Any]]] = {}

    def _ns(self, namespace: str | None) -> dict[str, dict[str, Any]]:
        return self.namespaces.setdefault(namespace or "", {})

    def upsert(self, *, vectors: list[dict[str, Any]], namespace: str | None = None) -> None:
        bucket = self._ns(namespace)
        for vector in vectors:
            assert len(vector["values"]) == self.dimension
            bucket[vector["id"]] = vector

    def query(
        self,
        *,
        vector: list[float],
        top_k: int,
        namespace: str | None = None,
        filter: dict[str, Any] | None = None,
        include_metadata: bool = True,
    ) -> dict[str, Any]:
        predicate = build_predicate(filter)
        matches = [
            {
                "id": item["id"],
                "score": cosine(vector, item["values"]),
                "metadata": item["metadata"],
            }
            for item in self._ns(namespace).values()
            if predicate is None or predicate(item["metadata"])
        ]
        matches.sort(key=lambda item: item["score"], reverse=True)
        return {"matches": matches[:top_k]}

    def fetch(self, *, ids: list[str], namespace: str | None = None) -> dict[str, Any]:
        bucket = self._ns(namespace)
        return {"vectors": {key: bucket[key] for key in ids if key in bucket}}

    def delete(
        self,
        *,
        ids: list[str] | None = None,
        namespace: str | None = None,
        delete_all: bool = False,
    ) -> None:
        if delete_all:
            self.namespaces.pop(namespace or "", None)
            return
        bucket = self._ns(namespace)
        for key in ids or []:
            bucket.pop(key, None)

    def list(self, *, namespace: str | None = None) -> Iterator[list[str]]:
        yield list(self._ns(namespace))

    def describe_index_stats(self) -> dict[str, Any]:
        return {
            "namespaces": {
                name: {"vector_count": len(items)} for name, items in self.namespaces.items()
            },
            "total_vector_count": sum(len(items) for items in self.namespaces.values()),
        }


class FakePinecone:
    """Minimal in-memory stand-in for the Pinecone control plane."""

    def __init__(self, dimension: int, *, existing: tuple[str, ...] = ()) -> None:
        self.dimension = dimension
        self.indexes: dict[str, FakeIndex] = {name: FakeIndex(dimension) for name in existing}
        self.created: list[str] = []

    def Index(self, name: str) -> FakeIndex:
        return self.indexes.setdefault(name, FakeIndex(self.dimension))

    def list_indexes(self) -> dict[str, Any]:
        return {"indexes": [{"name": name} for name in self.indexes]}

    def describe_index(self, name: str) -> dict[str, Any]:
        return {"dimension": self.dimension, "status": {"ready": True}}

    def create_index(self, **kwargs: Any) -> None:
        self.created.append(kwargs["name"])
        self.indexes[kwargs["name"]] = FakeIndex(kwargs["dimension"])

    def delete_index(self, name: str) -> None:
        self.indexes.pop(name, None)


def make_doc(domain: str, text: str, tenant: str | None = None, **meta: object) -> Document:
    digest = content_hash(domain, tenant, text)
    return Document(
        page_content=text,
        metadata={"domain": domain, "tenant_id": tenant, "content_hash": digest, **meta},
    )


async def build_store(tmp_path: Path, **overrides: Any) -> tuple[PineconeStore, FakePinecone]:
    settings = make_settings(
        tmp_path,
        vector_store="pinecone",
        pinecone_api_key="pc-fake",
        pinecone_index="nexusone",
        pinecone_auto_create=True,
        **overrides,
    )
    client = FakePinecone(128, existing=("nexusone",))
    store = PineconeStore(settings, build_embeddings(settings), client=client)
    await store.initialize()
    return store, client


# ------------------------------------------------------- namespace strategy --
async def test_namespace_strategy_isolates_domains(tmp_path: Path) -> None:
    store, client = await build_store(tmp_path)
    await store.add_documents("hr", [make_doc("hr", HR_TEXT, module="leave")])
    await store.add_documents("finance", [make_doc("finance", FIN_TEXT, module="ap")])

    index = client.Index("nexusone")
    assert set(index.namespaces) == {"kb_hr", "kb_finance"}

    hr_hits = await store.similarity_search("hr", "paid leave accrual", k=5)
    assert hr_hits[0].content == HR_TEXT
    assert hr_hits[0].domain == "hr"

    finance_hits = await store.similarity_search("finance", "paid leave accrual", k=5)
    assert all(hit.content != HR_TEXT for hit in finance_hits)

    assert sorted(await store.list_domains()) == ["finance", "hr"]
    assert (await store.domain_stats("hr")).vector_count == 1


async def test_namespace_strategy_filters_dedupes_and_deletes(tmp_path: Path) -> None:
    store, _ = await build_store(tmp_path)
    document = make_doc("finance", FIN_TEXT, module="ap", fiscal_year=2026)
    digest = str(document.metadata["content_hash"])

    assert await store.existing_ids("finance", [digest]) == set()
    await store.add_documents("finance", [document])
    assert await store.existing_ids("finance", [digest]) == {digest}

    filtered = await store.similarity_search("finance", "invoices", k=5, filters={"module": "gl"})
    assert filtered == []

    assert await store.delete("finance", ids=[digest]) == 1
    assert await store.similarity_search("finance", "invoices", k=5) == []


async def test_namespace_strategy_reindexes_and_drops(tmp_path: Path) -> None:
    store, client = await build_store(tmp_path)
    await store.add_documents(
        "hr",
        [
            make_doc("hr", HR_TEXT, source="a"),
            make_doc("hr", "Timesheets are due Friday.", source="b"),
        ],
    )
    assert await store.reindex("hr") == 2
    assert (await store.domain_stats("hr")).vector_count == 2
    assert "kb_hr__staging" not in client.Index("nexusone").namespaces

    await store.drop_domain("hr")
    assert await store.list_domains() == []


async def test_namespace_prefix_is_applied(tmp_path: Path) -> None:
    store, client = await build_store(tmp_path, pinecone_namespace="prod")
    await store.add_documents("hr", [make_doc("hr", HR_TEXT)])
    assert set(client.Index("nexusone").namespaces) == {"prod_kb_hr"}
    assert await store.list_domains() == ["hr"]


async def test_tenant_metadata_mode_filters_results(tmp_path: Path) -> None:
    store, _ = await build_store(tmp_path, tenant_mode="metadata")
    await store.add_documents(
        "hr", [make_doc("hr", "Acme pays overtime at 1.5x.", tenant="acme")], tenant_id="acme"
    )
    await store.add_documents(
        "hr", [make_doc("hr", "Globex pays overtime at 2x.", tenant="globex")], tenant_id="globex"
    )
    acme = await store.similarity_search("hr", "overtime", k=10, tenant_id="acme")
    assert [hit.content for hit in acme] == ["Acme pays overtime at 1.5x."]


# ----------------------------------------------------------- index strategy --
async def test_index_strategy_creates_one_index_per_domain(tmp_path: Path) -> None:
    store, client = await build_store(tmp_path, pinecone_strategy="index")
    await store.add_documents("hr", [make_doc("hr", HR_TEXT)])
    await store.add_documents("finance", [make_doc("finance", FIN_TEXT)])

    assert "kb-hr" in client.indexes
    assert "kb-finance" in client.indexes
    assert set(client.created) == {"kb-hr", "kb-finance"}

    hits = await store.similarity_search("finance", "paid leave accrual", k=5)
    assert all(hit.content != HR_TEXT for hit in hits)
    assert sorted(await store.list_domains()) == ["finance", "hr"]

    await store.drop_domain("hr")
    assert "kb-hr" not in client.indexes


async def test_missing_index_without_auto_create_fails_fast(tmp_path: Path) -> None:
    settings = make_settings(
        tmp_path,
        vector_store="pinecone",
        pinecone_api_key="pc-fake",
        pinecone_index="missing",
        pinecone_auto_create=False,
    )
    store = PineconeStore(settings, build_embeddings(settings), client=FakePinecone(128))
    with pytest.raises(ConfigurationError, match="does not exist"):
        await store.initialize()


async def test_dimension_mismatch_fails_fast(tmp_path: Path) -> None:
    settings = make_settings(
        tmp_path,
        vector_store="pinecone",
        pinecone_api_key="pc-fake",
        pinecone_index="nexusone",
        embedding_dimension=128,
    )
    client = FakePinecone(1536, existing=("nexusone",))
    store = PineconeStore(settings, build_embeddings(settings), client=client)
    with pytest.raises(ConfigurationError, match="dimension mismatch"):
        await store.initialize()


# ------------------------------------------------------------- integration --
@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("PINECONE_API_KEY") or not os.environ.get("PINECONE_INDEX"),
    reason="set PINECONE_API_KEY and PINECONE_INDEX to run the Pinecone integration test",
)
async def test_real_pinecone_round_trip(tmp_path: Path) -> None:  # pragma: no cover - needs creds
    from agent_service.kb.factory import get_vector_store

    settings = make_settings(
        tmp_path,
        vector_store="pinecone",
        pinecone_api_key=os.environ["PINECONE_API_KEY"],
        pinecone_index=os.environ["PINECONE_INDEX"],
        pinecone_namespace="citest",
        embedding_dimension=None,
    )
    store = get_vector_store(settings)
    await store.initialize()
    try:
        await store.add_documents("hr", [make_doc("hr", HR_TEXT)])
        hits = await store.similarity_search("hr", "paid leave accrual", k=3)
        assert any(HR_TEXT in hit.content for hit in hits)
        assert all(hit.domain == "hr" for hit in hits)
    finally:
        await store.drop_domain("hr")
