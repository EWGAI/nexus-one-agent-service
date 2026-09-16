"""The single interface every vector store backend implements.

Callers (API routes, agent tools) only ever see :class:`VectorStoreAdapter`.
Domain isolation, tenancy and filter translation are the adapter's job, so
switching ``VECTOR_STORE`` never changes calling code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
import re
from typing import TYPE_CHECKING, Any, ClassVar

from langchain_core.callbacks import AsyncCallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel, ConfigDict, Field

from agent_service.config import Settings, TenantMode

if TYPE_CHECKING:  # pragma: no cover
    from agent_service.kb.embeddings import EmbeddingBundle

#: Metadata filter expressed in a small store-agnostic dialect::
#:
#:     {"module": "payroll"}                 -> equality
#:     {"module": ["payroll", "leave"]}      -> membership ("in")
#:     {"fiscal_year": {"$gte": 2025}}       -> operator form
MetadataFilter = dict[str, Any]

_SLUG_RE = re.compile(r"[^a-z0-9]+")

STAGING_SUFFIX = "__staging"


def slugify(value: str) -> str:
    """Lowercase, hyphen-free slug safe for collection / namespace names."""
    return _SLUG_RE.sub("_", value.strip().lower()).strip("_") or "default"


class SearchHit(BaseModel):
    """A single ranked chunk returned by a similarity search."""

    id: str = Field(description="Deterministic chunk id (its content hash).")
    content: str = Field(description="Chunk text.")
    score: float = Field(description="Similarity score, higher is better (cosine).")
    domain: str = Field(description="Knowledge base domain the chunk belongs to.")
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_document(self) -> Document:
        """Convert back into a LangChain document."""
        return Document(page_content=self.content, metadata={**self.metadata, "id": self.id})


class DomainStats(BaseModel):
    """Per-domain inventory information."""

    domain: str
    store: str
    physical_name: str
    vector_count: int
    embedding_model: str
    embedding_dimension: int
    extra: dict[str, Any] = Field(default_factory=dict)


class VectorStoreAdapter(ABC):
    """Uniform, domain-aware vector store facade."""

    store_type: ClassVar[str] = "base"

    def __init__(self, settings: Settings, embeddings: EmbeddingBundle) -> None:
        self.settings = settings
        self.embeddings = embeddings

    # -- naming / filtering helpers ---------------------------------------
    def physical_name(self, domain: str, tenant_id: str | None = None) -> str:
        """Map a logical domain (+ tenant) onto a backend collection/namespace name."""
        prefix = self.settings.collection_prefix
        if self.settings.tenant_mode is TenantMode.PREFIX and tenant_id:
            return f"{prefix}_{slugify(tenant_id)}_{slugify(domain)}"
        return f"{prefix}_{slugify(domain)}"

    def name_prefix(self, tenant_id: str | None = None) -> str:
        """Prefix shared by every physical name visible to this tenant."""
        prefix = self.settings.collection_prefix
        if self.settings.tenant_mode is TenantMode.PREFIX and tenant_id:
            return f"{prefix}_{slugify(tenant_id)}_"
        return f"{prefix}_"

    def logical_name(self, physical: str, tenant_id: str | None = None) -> str | None:
        """Inverse of :meth:`physical_name`; ``None`` when the name is foreign."""
        prefix = self.name_prefix(tenant_id)
        if not physical.startswith(prefix):
            return None
        domain = physical[len(prefix) :]
        if not domain or domain.endswith(STAGING_SUFFIX):
            return None
        return domain

    def tenant_filter(self, tenant_id: str | None) -> MetadataFilter | None:
        """Metadata predicate enforcing tenant isolation in ``metadata`` mode."""
        if self.settings.tenant_mode is TenantMode.METADATA and tenant_id:
            return {"tenant_id": tenant_id}
        return None

    def effective_filter(
        self, filters: MetadataFilter | None, tenant_id: str | None
    ) -> MetadataFilter | None:
        """Merge caller supplied filters with the mandatory tenant predicate."""
        tenant = self.tenant_filter(tenant_id)
        if not filters:
            return tenant
        if not tenant:
            return dict(filters)
        return {**filters, **tenant}

    @staticmethod
    def staging_domain(domain: str) -> str:
        """Name of the shadow domain used while re-indexing."""
        return f"{domain}{STAGING_SUFFIX}"

    # -- lifecycle ---------------------------------------------------------
    # These three are optional hooks with a working no-op default, not abstract
    # members: a server backed store has nothing to open, close or flush.
    async def initialize(self) -> None:  # noqa: B027
        """Open connections / warm caches. Called once during app startup."""

    async def aclose(self) -> None:  # noqa: B027
        """Release resources. Called during app shutdown."""

    async def health(self) -> bool:
        """Cheap readiness probe."""
        await self.list_domains()
        return True

    # -- required operations ----------------------------------------------
    @abstractmethod
    async def add_documents(
        self,
        domain: str,
        documents: Sequence[Document],
        *,
        ids: Sequence[str] | None = None,
        tenant_id: str | None = None,
    ) -> list[str]:
        """Embed and upsert ``documents`` into ``domain``. Returns the stored ids."""

    @abstractmethod
    async def existing_ids(
        self, domain: str, ids: Sequence[str], *, tenant_id: str | None = None
    ) -> set[str]:
        """Subset of ``ids`` already present in ``domain`` (used for dedupe)."""

    @abstractmethod
    async def similarity_search(
        self,
        domain: str,
        query: str,
        *,
        k: int = 5,
        filters: MetadataFilter | None = None,
        tenant_id: str | None = None,
    ) -> list[SearchHit]:
        """Rank chunks of ``domain`` against ``query``."""

    @abstractmethod
    async def delete(
        self,
        domain: str,
        *,
        ids: Sequence[str] | None = None,
        filters: MetadataFilter | None = None,
        tenant_id: str | None = None,
    ) -> int:
        """Delete by ids and/or metadata filter. Returns the number removed."""

    @abstractmethod
    async def iter_documents(self, domain: str, *, tenant_id: str | None = None) -> list[Document]:
        """Return every stored chunk of ``domain`` (used by re-index)."""

    @abstractmethod
    async def list_domains(self, *, tenant_id: str | None = None) -> list[str]:
        """Logical domains that physically exist in the backend."""

    @abstractmethod
    async def domain_stats(self, domain: str, *, tenant_id: str | None = None) -> DomainStats:
        """Inventory for a single domain."""

    @abstractmethod
    async def drop_domain(self, domain: str, *, tenant_id: str | None = None) -> None:
        """Delete the whole domain (collection / namespace / index)."""

    @abstractmethod
    async def swap_domain(self, target: str, staging: str, *, tenant_id: str | None = None) -> None:
        """Atomically replace ``target`` with the contents of ``staging``."""

    async def persist(self) -> None:  # noqa: B027
        """Flush to durable storage. No-op for server backed stores."""

    # -- derived operations ------------------------------------------------
    async def reindex(self, domain: str, *, tenant_id: str | None = None) -> int:
        """Re-embed every chunk with the current model, then swap atomically."""
        documents = await self.iter_documents(domain, tenant_id=tenant_id)
        staging = self.staging_domain(domain)
        await self.drop_domain(staging, tenant_id=tenant_id)
        if documents:
            ids = [str(doc.metadata.get("content_hash") or doc.metadata["id"]) for doc in documents]
            await self.add_documents(staging, documents, ids=ids, tenant_id=tenant_id)
        await self.swap_domain(domain, staging, tenant_id=tenant_id)
        await self.persist()
        return len(documents)

    def as_retriever(
        self,
        domain: str,
        *,
        k: int = 5,
        filters: MetadataFilter | None = None,
        tenant_id: str | None = None,
    ) -> BaseRetriever:
        """A LangChain retriever bound to one domain of this store."""
        return AdapterRetriever(
            adapter=self, domain=domain, k=k, filters=filters, tenant_id=tenant_id
        )


class AdapterRetriever(BaseRetriever):
    """Generic retriever that works with any :class:`VectorStoreAdapter`."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    adapter: VectorStoreAdapter
    domain: str
    k: int = 5
    filters: MetadataFilter | None = None
    tenant_id: str | None = None

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[Document]:
        hits = await self.adapter.similarity_search(
            self.domain, query, k=self.k, filters=self.filters, tenant_id=self.tenant_id
        )
        return [hit.to_document() for hit in hits]

    def _get_relevant_documents(self, query: str, *, run_manager: Any) -> list[Document]:
        raise NotImplementedError("AdapterRetriever is async-only; use ainvoke()")
