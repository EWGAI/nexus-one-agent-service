"""Pinecone adapter - domain isolation via namespaces (default) or dedicated indexes.

``PINECONE_STRATEGY=namespace`` (default)
    One shared index, one namespace per ``(tenant, domain)``. Cheapest and the
    usual choice.

``PINECONE_STRATEGY=index``
    One Pinecone index per ``(tenant, domain)`` for hard isolation and separate
    quotas. Requires ``PINECONE_AUTO_CREATE=true`` unless every index already
    exists.

The Pinecone SDK is synchronous, so all calls run in a worker thread. A client
can be injected for testing.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Iterator, Sequence
import time
from typing import Any, ClassVar, Protocol, cast

from langchain_core.documents import Document

from agent_service.config import PineconeStrategy, Settings
from agent_service.core.exceptions import ConfigurationError, VectorStoreError
from agent_service.core.logging import get_logger
from agent_service.kb.base import DomainStats, MetadataFilter, SearchHit, VectorStoreAdapter
from agent_service.kb.embeddings import EmbeddingBundle
from agent_service.kb.filters import build_predicate, to_pinecone_filter

logger = get_logger(__name__)

TEXT_KEY = "text"
_UPSERT_BATCH = 100
_FETCH_BATCH = 100


class PineconeIndexLike(Protocol):
    """Subset of the Pinecone ``Index`` API this adapter relies on."""

    def upsert(self, *, vectors: list[dict[str, Any]], namespace: str | None = None) -> Any: ...

    def query(
        self,
        *,
        vector: list[float],
        top_k: int,
        namespace: str | None = None,
        filter: dict[str, Any] | None = None,
        include_metadata: bool = True,
    ) -> Any: ...

    def fetch(self, *, ids: list[str], namespace: str | None = None) -> Any: ...

    def delete(
        self,
        *,
        ids: list[str] | None = None,
        namespace: str | None = None,
        delete_all: bool = False,
    ) -> Any: ...

    def list(self, *, namespace: str | None = None) -> Iterator[Any]: ...

    def describe_index_stats(self) -> Any: ...


class PineconeClientLike(Protocol):
    """Subset of the Pinecone control-plane API this adapter relies on."""

    def Index(self, name: str) -> PineconeIndexLike: ...

    def list_indexes(self) -> Any: ...

    def describe_index(self, name: str) -> Any: ...

    def create_index(self, **kwargs: Any) -> Any: ...

    def delete_index(self, name: str) -> Any: ...


def _attr(source: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from a dict-like or attribute-like SDK response object."""
    if source is None:
        return default
    if isinstance(source, dict):
        return source.get(key, default)
    return getattr(source, key, default)


class PineconeStore(VectorStoreAdapter):
    """:class:`~agent_service.kb.base.VectorStoreAdapter` backed by Pinecone."""

    store_type: ClassVar[str] = "pinecone"

    def __init__(
        self,
        settings: Settings,
        embeddings: EmbeddingBundle,
        *,
        client: PineconeClientLike | None = None,
    ) -> None:
        super().__init__(settings, embeddings)
        self.strategy = settings.pinecone_strategy
        self._client = client or self._build_client(settings)
        self._indexes: dict[str, PineconeIndexLike] = {}

    @staticmethod
    def _build_client(settings: Settings) -> PineconeClientLike:
        if not settings.pinecone_api_key:
            raise ConfigurationError("VECTOR_STORE=pinecone requires PINECONE_API_KEY")
        try:
            from pinecone import Pinecone
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise ConfigurationError(
                'VECTOR_STORE=pinecone requires the pinecone extra: pip install -e ".[pinecone]"'
            ) from exc
        client: PineconeClientLike = cast(
            "PineconeClientLike", Pinecone(api_key=settings.pinecone_api_key)
        )
        return client

    # -- resource naming ----------------------------------------------------
    def _index_name(self, domain: str, tenant_id: str | None) -> str:
        if self.strategy is PineconeStrategy.NAMESPACE:
            if not self.settings.pinecone_index:
                raise ConfigurationError("PINECONE_STRATEGY=namespace requires PINECONE_INDEX")
            return self.settings.pinecone_index
        return self.physical_name(domain, tenant_id).replace("_", "-")

    def _namespace(self, domain: str, tenant_id: str | None) -> str:
        if self.strategy is PineconeStrategy.INDEX:
            return self.settings.pinecone_namespace
        physical = self.physical_name(domain, tenant_id)
        prefix = self.settings.pinecone_namespace
        return f"{prefix}_{physical}" if prefix else physical

    def _namespace_to_domain(self, namespace: str, tenant_id: str | None) -> str | None:
        prefix = self.settings.pinecone_namespace
        if prefix:
            if not namespace.startswith(f"{prefix}_"):
                return None
            namespace = namespace[len(prefix) + 1 :]
        return self.logical_name(namespace, tenant_id)

    def _index(self, name: str) -> PineconeIndexLike:
        index = self._indexes.get(name)
        if index is None:
            index = self._client.Index(name)
            self._indexes[name] = index
        return index

    def _serverless_spec(self) -> Any:
        """``ServerlessSpec`` when the SDK is installed, else its dict equivalent."""
        cloud, region = self.settings.pinecone_cloud, self.settings.pinecone_region
        try:
            from pinecone import ServerlessSpec
        except ImportError:  # pragma: no cover - exercised only without the extra
            return {"serverless": {"cloud": cloud, "region": region}}
        return ServerlessSpec(cloud=cloud, region=region)

    def _handle(self, domain: str, tenant_id: str | None) -> tuple[PineconeIndexLike, str]:
        return self._index(self._index_name(domain, tenant_id)), self._namespace(domain, tenant_id)

    # -- lifecycle ----------------------------------------------------------
    def _existing_index_names(self) -> list[str]:
        listing = self._client.list_indexes()
        names = _attr(listing, "names", None)
        if callable(names):
            return [str(item) for item in names()]
        indexes = _attr(listing, "indexes", listing)
        return [str(_attr(item, "name", item)) for item in (indexes or [])]

    def _ensure_index(self, name: str) -> None:
        """Create the index if allowed, otherwise validate its dimension."""
        if name in self._existing_index_names():
            description = self._client.describe_index(name)
            dimension = _attr(description, "dimension")
            if dimension is not None and int(dimension) != self.embeddings.dimension:
                raise ConfigurationError(
                    "Embedding dimension mismatch for Pinecone index",
                    details={
                        "index": name,
                        "index_dimension": int(dimension),
                        "embedding_dimension": self.embeddings.dimension,
                        "hint": "point EMBEDDING_MODEL at the original model or re-create the index",
                    },
                )
            return

        if not self.settings.pinecone_auto_create:
            raise ConfigurationError(
                f"Pinecone index {name!r} does not exist",
                details={"hint": "create it manually or set PINECONE_AUTO_CREATE=true"},
            )

        logger.info("creating serverless pinecone index", extra={"index": name})
        self._client.create_index(
            name=name,
            dimension=self.embeddings.dimension,
            metric=self.settings.pinecone_metric,
            spec=self._serverless_spec(),
        )
        for _ in range(60):
            description = self._client.describe_index(name)
            if bool(_attr(_attr(description, "status", {}), "ready", False)):
                return
            time.sleep(1.0)
        raise VectorStoreError(f"Pinecone index {name!r} did not become ready in time")

    async def initialize(self) -> None:
        await self.embeddings.ensure_dimension()
        if self.strategy is PineconeStrategy.NAMESPACE:
            await asyncio.to_thread(self._ensure_index, self._index_name("_", None))
        logger.info(
            "pinecone ready",
            extra={"strategy": self.strategy.value, "index": self.settings.pinecone_index},
        )

    async def health(self) -> bool:
        await asyncio.to_thread(self._existing_index_names)
        return True

    # -- operations ---------------------------------------------------------
    async def add_documents(
        self,
        domain: str,
        documents: Sequence[Document],
        *,
        ids: Sequence[str] | None = None,
        tenant_id: str | None = None,
    ) -> list[str]:
        if not documents:
            return []
        doc_ids = (
            list(ids) if ids is not None else [str(d.metadata["content_hash"]) for d in documents]
        )
        if len(doc_ids) != len(documents):
            raise VectorStoreError("ids length does not match documents length")

        index_name = self._index_name(domain, tenant_id)
        await asyncio.to_thread(self._ensure_index, index_name)
        vectors = await self.embeddings.aembed_documents([d.page_content for d in documents])
        namespace = self._namespace(domain, tenant_id)
        payload: list[dict[str, Any]] = []
        for position, document in enumerate(documents):
            metadata = _scalar_metadata(document.metadata)
            metadata[TEXT_KEY] = document.page_content
            payload.append(
                {"id": doc_ids[position], "values": vectors[position], "metadata": metadata}
            )
        index = self._index(index_name)
        for start in range(0, len(payload), _UPSERT_BATCH):
            batch = payload[start : start + _UPSERT_BATCH]
            await asyncio.to_thread(index.upsert, vectors=batch, namespace=namespace)
        return doc_ids

    async def existing_ids(
        self, domain: str, ids: Sequence[str], *, tenant_id: str | None = None
    ) -> set[str]:
        if not ids:
            return set()
        index, namespace = self._handle(domain, tenant_id)
        found: set[str] = set()
        id_list = list(ids)
        for start in range(0, len(id_list), _FETCH_BATCH):
            batch = id_list[start : start + _FETCH_BATCH]
            try:
                response = await asyncio.to_thread(index.fetch, ids=batch, namespace=namespace)
            except Exception:
                return found
            found.update(_attr(response, "vectors", {}) or {})
        return found

    async def similarity_search(
        self,
        domain: str,
        query: str,
        *,
        k: int = 5,
        filters: MetadataFilter | None = None,
        tenant_id: str | None = None,
    ) -> list[SearchHit]:
        index, namespace = self._handle(domain, tenant_id)
        vector = await self.embeddings.aembed_query(query)
        native_filter = to_pinecone_filter(self.effective_filter(filters, tenant_id))
        response = await asyncio.to_thread(
            index.query,
            vector=vector,
            top_k=max(1, k),
            namespace=namespace,
            filter=native_filter,
            include_metadata=True,
        )
        hits: list[SearchHit] = []
        for match in _attr(response, "matches", []) or []:
            metadata = dict(_attr(match, "metadata", {}) or {})
            content = str(metadata.pop(TEXT_KEY, ""))
            hits.append(
                SearchHit(
                    id=str(_attr(match, "id", "")),
                    content=content,
                    score=round(float(_attr(match, "score", 0.0) or 0.0), 6),
                    domain=domain,
                    metadata=metadata,
                )
            )
        return hits

    async def delete(
        self,
        domain: str,
        *,
        ids: Sequence[str] | None = None,
        filters: MetadataFilter | None = None,
        tenant_id: str | None = None,
    ) -> int:
        index, namespace = self._handle(domain, tenant_id)
        if ids:
            target = sorted(await self.existing_ids(domain, ids, tenant_id=tenant_id))
        elif filters or self.tenant_filter(tenant_id):
            predicate = build_predicate(self.effective_filter(filters, tenant_id))
            assert predicate is not None  # noqa: S101 - guarded by the branch condition
            target = [
                str(document.metadata["id"])
                for document in await self.iter_documents(domain, tenant_id=tenant_id)
                if predicate(document.metadata)
            ]
        else:
            raise VectorStoreError("delete requires either ids or a metadata filter")
        if not target:
            return 0
        for start in range(0, len(target), _UPSERT_BATCH):
            await asyncio.to_thread(
                index.delete, ids=target[start : start + _UPSERT_BATCH], namespace=namespace
            )
        return len(target)

    def _all_ids(self, index: PineconeIndexLike, namespace: str) -> list[str]:
        collected: list[str] = []
        try:
            pages: Iterable[Any] = index.list(namespace=namespace)
        except Exception:
            return collected
        for page in pages:
            if isinstance(page, str):
                collected.append(page)
            else:
                collected.extend(str(_attr(item, "id", item)) for item in page)
        return collected

    async def iter_documents(self, domain: str, *, tenant_id: str | None = None) -> list[Document]:
        index, namespace = self._handle(domain, tenant_id)
        ids = await asyncio.to_thread(self._all_ids, index, namespace)
        documents: list[Document] = []
        for start in range(0, len(ids), _FETCH_BATCH):
            batch = ids[start : start + _FETCH_BATCH]
            response = await asyncio.to_thread(index.fetch, ids=batch, namespace=namespace)
            for vector_id, record in (_attr(response, "vectors", {}) or {}).items():
                metadata = dict(_attr(record, "metadata", {}) or {})
                content = str(metadata.pop(TEXT_KEY, ""))
                metadata.setdefault("id", vector_id)
                documents.append(Document(page_content=content, metadata=metadata))
        return documents

    async def list_domains(self, *, tenant_id: str | None = None) -> list[str]:
        if self.strategy is PineconeStrategy.INDEX:
            names = await asyncio.to_thread(self._existing_index_names)
            domains = [self.logical_name(name.replace("-", "_"), tenant_id) for name in names]
        else:
            index = self._index(self._index_name("_", tenant_id))
            stats = await asyncio.to_thread(index.describe_index_stats)
            namespaces = _attr(stats, "namespaces", {}) or {}
            domains = [self._namespace_to_domain(str(ns), tenant_id) for ns in namespaces]
        return sorted({domain for domain in domains if domain})

    async def domain_stats(self, domain: str, *, tenant_id: str | None = None) -> DomainStats:
        index, namespace = self._handle(domain, tenant_id)
        try:
            stats = await asyncio.to_thread(index.describe_index_stats)
        except Exception:
            stats = None
        if self.strategy is PineconeStrategy.INDEX:
            count = int(_attr(stats, "total_vector_count", 0) or 0)
        else:
            namespaces = _attr(stats, "namespaces", {}) or {}
            count = int(_attr(namespaces.get(namespace), "vector_count", 0) or 0)
        return DomainStats(
            domain=domain,
            store=self.store_type,
            physical_name=f"{self._index_name(domain, tenant_id)}/{namespace or '(default)'}",
            vector_count=count,
            embedding_model=self.embeddings.model_name,
            embedding_dimension=await self.embeddings.ensure_dimension(),
            extra={"strategy": self.strategy.value, "namespace": namespace},
        )

    async def drop_domain(self, domain: str, *, tenant_id: str | None = None) -> None:
        if self.strategy is PineconeStrategy.INDEX:
            name = self._index_name(domain, tenant_id)
            if name in await asyncio.to_thread(self._existing_index_names):
                await asyncio.to_thread(self._client.delete_index, name)
                self._indexes.pop(name, None)
            return
        index, namespace = self._handle(domain, tenant_id)
        try:
            await asyncio.to_thread(index.delete, namespace=namespace, delete_all=True)
        except Exception:
            logger.debug("namespace absent on drop", extra={"namespace": namespace})

    async def swap_domain(self, target: str, staging: str, *, tenant_id: str | None = None) -> None:
        """Pinecone cannot rename, so copy staging -> target then drop staging."""
        documents = await self.iter_documents(staging, tenant_id=tenant_id)
        await self.drop_domain(target, tenant_id=tenant_id)
        if documents:
            await self.add_documents(
                target,
                documents,
                ids=[str(document.metadata["id"]) for document in documents],
                tenant_id=tenant_id,
            )
        await self.drop_domain(staging, tenant_id=tenant_id)


def _scalar_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Pinecone metadata accepts strings, numbers, booleans and string lists only."""
    clean: dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            clean[key] = value
        elif isinstance(value, (list, tuple, set)):
            clean[key] = [str(item) for item in value]
        else:
            clean[key] = str(value)
    return clean
