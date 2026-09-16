"""Contract suite every vector store adapter must satisfy.

Parametrised over Chroma and FAISS (see the ``adapter`` fixture). Pinecone runs
the same assertions in ``test_pinecone_store.py`` against a fake SDK client, and
against the real service when ``PINECONE_API_KEY`` is available.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
import pytest

from agent_service.core.exceptions import VectorStoreError
from agent_service.kb.base import VectorStoreAdapter
from agent_service.kb.factory import (
    available_stores,
    get_vector_store,
    register_store,
)
from agent_service.kb.ingestion import content_hash

from .conftest import make_settings

HR_TEXT = "Employees accrue 1.75 days of paid leave per completed month of service."
FINANCE_TEXT = "Accounts payable invoices above 50000 require two approvals before payment."


def make_doc(domain: str, text: str, tenant: str | None = None, **meta: object) -> Document:
    """A chunk stamped the way the ingestion pipeline stamps them."""
    digest = content_hash(domain, tenant, text)
    return Document(
        page_content=text,
        metadata={"domain": domain, "tenant_id": tenant, "content_hash": digest, **meta},
    )


async def seed(store: VectorStoreAdapter) -> None:
    await store.add_documents("hr", [make_doc("hr", HR_TEXT, module="leave")])
    await store.add_documents("finance", [make_doc("finance", FINANCE_TEXT, module="ap")])


async def test_add_and_search(adapter: VectorStoreAdapter) -> None:
    await seed(adapter)
    hits = await adapter.similarity_search("hr", "how much paid leave do employees accrue", k=3)
    assert hits
    assert hits[0].content == HR_TEXT
    assert hits[0].domain == "hr"
    assert hits[0].metadata["module"] == "leave"
    assert -1.01 <= hits[0].score <= 1.01


async def test_a_query_against_finance_never_returns_hr_chunks(
    adapter: VectorStoreAdapter,
) -> None:
    await seed(adapter)
    hits = await adapter.similarity_search("finance", "paid leave accrual per month", k=10)
    assert all(hit.domain == "finance" for hit in hits)
    assert all(HR_TEXT not in hit.content for hit in hits)


async def test_empty_domain_returns_nothing(adapter: VectorStoreAdapter) -> None:
    await seed(adapter)
    assert await adapter.similarity_search("engineering", "anything at all", k=5) == []


async def test_existing_ids_enables_dedupe(adapter: VectorStoreAdapter) -> None:
    document = make_doc("hr", HR_TEXT)
    digest = str(document.metadata["content_hash"])
    assert await adapter.existing_ids("hr", [digest]) == set()

    await adapter.add_documents("hr", [document])
    assert await adapter.existing_ids("hr", [digest]) == {digest}
    # the same content hashes differently in another domain
    assert await adapter.existing_ids("finance", [digest]) == set()


async def test_re_adding_the_same_document_is_idempotent(adapter: VectorStoreAdapter) -> None:
    document = make_doc("hr", HR_TEXT)
    await adapter.add_documents("hr", [document])
    await adapter.add_documents("hr", [document])
    assert (await adapter.domain_stats("hr")).vector_count == 1


async def test_metadata_filters(adapter: VectorStoreAdapter) -> None:
    await adapter.add_documents(
        "finance",
        [
            make_doc("finance", FINANCE_TEXT, module="ap", fiscal_year=2026),
            make_doc(
                "finance",
                "General ledger closes on the fifth working day.",
                module="gl",
                fiscal_year=2025,
            ),
        ],
    )
    only_ap = await adapter.similarity_search(
        "finance", "invoice approvals", k=10, filters={"module": "ap"}
    )
    assert [hit.metadata["module"] for hit in only_ap] == ["ap"]

    membership = await adapter.similarity_search(
        "finance", "ledger", k=10, filters={"module": ["gl", "ap"]}
    )
    assert len(membership) == 2

    operator = await adapter.similarity_search(
        "finance", "ledger", k=10, filters={"fiscal_year": {"$gte": 2026}}
    )
    assert [hit.metadata["fiscal_year"] for hit in operator] == [2026]


async def test_delete_by_ids(adapter: VectorStoreAdapter) -> None:
    document = make_doc("hr", HR_TEXT)
    digest = str(document.metadata["content_hash"])
    await adapter.add_documents("hr", [document])

    assert await adapter.delete("hr", ids=[digest]) == 1
    assert await adapter.delete("hr", ids=[digest]) == 0
    assert await adapter.similarity_search("hr", HR_TEXT, k=5) == []


async def test_delete_by_filter(adapter: VectorStoreAdapter) -> None:
    await adapter.add_documents(
        "hr",
        [
            make_doc("hr", HR_TEXT, source="policy-a"),
            make_doc("hr", "Timesheets must be submitted every Friday.", source="policy-b"),
        ],
    )
    assert await adapter.delete("hr", filters={"source": "policy-a"}) == 1
    remaining = await adapter.iter_documents("hr")
    assert [doc.metadata["source"] for doc in remaining] == ["policy-b"]


async def test_delete_without_criteria_is_an_error(adapter: VectorStoreAdapter) -> None:
    await seed(adapter)
    with pytest.raises(VectorStoreError):
        await adapter.delete("hr")


async def test_list_domains_and_stats(adapter: VectorStoreAdapter) -> None:
    await seed(adapter)
    assert set(await adapter.list_domains()) == {"hr", "finance"}

    stats = await adapter.domain_stats("hr")
    assert stats.domain == "hr"
    assert stats.store == adapter.store_type
    assert stats.vector_count == 1
    assert stats.embedding_dimension == 128
    assert stats.physical_name.startswith("kb")

    empty = await adapter.domain_stats("it")
    assert empty.vector_count == 0


async def test_drop_domain_leaves_other_domains_intact(adapter: VectorStoreAdapter) -> None:
    await seed(adapter)
    await adapter.drop_domain("hr")
    assert await adapter.list_domains() == ["finance"]
    assert (await adapter.domain_stats("hr")).vector_count == 0
    assert await adapter.similarity_search("finance", "invoice approvals", k=3)


async def test_dropping_a_missing_domain_is_a_noop(adapter: VectorStoreAdapter) -> None:
    await adapter.drop_domain("engineering")
    assert await adapter.list_domains() == []


async def test_reindex_preserves_content_and_swaps_atomically(
    adapter: VectorStoreAdapter,
) -> None:
    await adapter.add_documents(
        "hr",
        [
            make_doc("hr", HR_TEXT, source="policy-a"),
            make_doc("hr", "Timesheets must be submitted every Friday.", source="policy-b"),
        ],
    )
    assert await adapter.reindex("hr") == 2

    stats = await adapter.domain_stats("hr")
    assert stats.vector_count == 2
    assert stats.physical_name == adapter.physical_name("hr")
    assert await adapter.list_domains() == ["hr"]  # staging must not linger

    hits = await adapter.similarity_search("hr", "paid leave accrual", k=2)
    assert hits[0].content == HR_TEXT


async def test_retriever_is_bound_to_one_domain(adapter: VectorStoreAdapter) -> None:
    await seed(adapter)
    retriever = adapter.as_retriever("finance", k=2)
    documents = await retriever.ainvoke("invoice approval limits")
    assert documents
    assert all(document.metadata["domain"] == "finance" for document in documents)


async def test_iter_documents_returns_everything(adapter: VectorStoreAdapter) -> None:
    await seed(adapter)
    documents = await adapter.iter_documents("hr")
    assert [document.page_content for document in documents] == [HR_TEXT]


# ------------------------------------------------------------- multi-tenant --
@pytest.mark.parametrize("store", ["chroma", "faiss"])
@pytest.mark.parametrize("mode", ["metadata", "prefix"])
async def test_tenants_are_isolated(tmp_path: Path, store: str, mode: str) -> None:
    settings = make_settings(tmp_path, vector_store=store, tenant_mode=mode)
    adapter = get_vector_store(settings)
    await adapter.initialize()

    await adapter.add_documents(
        "hr", [make_doc("hr", "Acme pays overtime at 1.5x.", tenant="acme")], tenant_id="acme"
    )
    await adapter.add_documents(
        "hr", [make_doc("hr", "Globex pays overtime at 2x.", tenant="globex")], tenant_id="globex"
    )

    acme = await adapter.similarity_search("hr", "overtime rate", k=10, tenant_id="acme")
    assert [hit.content for hit in acme] == ["Acme pays overtime at 1.5x."]

    globex = await adapter.similarity_search("hr", "overtime rate", k=10, tenant_id="globex")
    assert [hit.content for hit in globex] == ["Globex pays overtime at 2x."]

    if mode == "prefix":
        assert adapter.physical_name("hr", "acme") == "kb_acme_hr"
        assert await adapter.list_domains(tenant_id="acme") == ["hr"]
    else:
        assert adapter.physical_name("hr", "acme") == "kb_hr"


# ------------------------------------------------------------------ factory --
def test_factory_knows_the_builtin_stores() -> None:
    assert {"chroma", "faiss", "pinecone"} <= set(available_stores())


def test_unknown_store_is_reported_clearly(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    object.__setattr__(settings, "vector_store", "qdrant")
    with pytest.raises(Exception, match="Unknown VECTOR_STORE"):
        get_vector_store(settings)


def test_out_of_tree_stores_can_be_registered() -> None:
    @register_store("dummy-store")
    def _build(settings: object, embeddings: object) -> object:  # pragma: no cover - registry only
        return object()

    assert "dummy-store" in available_stores()
