"""Chroma adapter - one collection per (tenant, domain).

Supports both the embedded ``PersistentClient`` (``CHROMA_PERSIST_DIR``) and a
remote ``HttpClient`` (``CHROMA_HOST`` / ``CHROMA_PORT``). Embeddings are always
computed by the configured provider, never by Chroma's bundled model.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
import json
from typing import Any, ClassVar

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings as ChromaSettings
from langchain_core.documents import Document

from agent_service.config import Settings
from agent_service.core.exceptions import ConfigurationError, VectorStoreError
from agent_service.core.logging import get_logger
from agent_service.kb.base import (
    DomainStats,
    MetadataFilter,
    SearchHit,
    VectorStoreAdapter,
)
from agent_service.kb.embeddings import EmbeddingBundle
from agent_service.kb.filters import to_chroma_where

logger = get_logger(__name__)

_SCALARS = (str, int, float, bool)


def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Chroma only stores scalar metadata values; encode anything else as JSON."""
    clean: dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        clean[key] = value if isinstance(value, _SCALARS) else json.dumps(value, default=str)
    return clean


class ChromaStore(VectorStoreAdapter):
    """:class:`~agent_service.kb.base.VectorStoreAdapter` backed by Chroma."""

    store_type: ClassVar[str] = "chroma"

    def __init__(
        self,
        settings: Settings,
        embeddings: EmbeddingBundle,
        *,
        client: ClientAPI | None = None,
    ) -> None:
        super().__init__(settings, embeddings)
        self._client = client or self._build_client(settings)

    @staticmethod
    def _build_client(settings: Settings) -> ClientAPI:
        chroma_settings = ChromaSettings(anonymized_telemetry=False, allow_reset=True)
        if settings.chroma_host:
            logger.info(
                "connecting to chroma server",
                extra={"host": settings.chroma_host, "port": settings.chroma_port},
            )
            return chromadb.HttpClient(
                host=settings.chroma_host,
                port=settings.chroma_port,
                ssl=settings.chroma_ssl,
                settings=chroma_settings,
            )
        settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "using embedded chroma", extra={"path": str(settings.chroma_persist_dir.resolve())}
        )
        return chromadb.PersistentClient(
            path=str(settings.chroma_persist_dir), settings=chroma_settings
        )

    # -- lifecycle ---------------------------------------------------------
    async def initialize(self) -> None:
        dimension = await self.embeddings.ensure_dimension()
        await asyncio.to_thread(self._client.heartbeat)
        logger.info(
            "chroma ready",
            extra={"embedding_model": self.embeddings.model_name, "dimension": dimension},
        )

    async def health(self) -> bool:
        await asyncio.to_thread(self._client.heartbeat)
        return True

    # -- collection helpers -------------------------------------------------
    def _collection(self, domain: str, tenant_id: str | None, *, create: bool) -> Collection | None:
        name = self.physical_name(domain, tenant_id)
        if not create:
            try:
                return self._client.get_collection(name=name, embedding_function=None)
            except Exception:
                return None
        collection = self._client.get_or_create_collection(
            name=name,
            embedding_function=None,
            metadata={
                "hnsw:space": "cosine",
                "logical_domain": domain,
                "embedding_model": self.embeddings.model_name,
                "embedding_dimension": self.embeddings.dimension,
            },
        )
        existing_dimension = (collection.metadata or {}).get("embedding_dimension")
        if existing_dimension is not None and int(existing_dimension) != self.embeddings.dimension:
            raise ConfigurationError(
                "Embedding dimension mismatch for Chroma collection",
                details={
                    "collection": name,
                    "collection_dimension": int(existing_dimension),
                    "embedding_dimension": self.embeddings.dimension,
                    "hint": "run POST /kb/{domain}/reindex or drop the collection",
                },
            )
        return collection

    async def _get_or_create(self, domain: str, tenant_id: str | None) -> Collection:
        collection = await asyncio.to_thread(self._collection, domain, tenant_id, create=True)
        assert collection is not None  # noqa: S101 - create=True always returns
        return collection

    async def _get(self, domain: str, tenant_id: str | None) -> Collection | None:
        return await asyncio.to_thread(self._collection, domain, tenant_id, create=False)

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

        texts = [doc.page_content for doc in documents]
        vectors = await self.embeddings.aembed_documents(texts)
        metadatas = [_sanitize_metadata(dict(doc.metadata)) for doc in documents]

        collection = await self._get_or_create(domain, tenant_id)
        await asyncio.to_thread(
            collection.upsert,
            ids=doc_ids,
            embeddings=vectors,  # type: ignore[arg-type]
            documents=texts,
            metadatas=metadatas,  # type: ignore[arg-type]
        )
        return doc_ids

    async def existing_ids(
        self, domain: str, ids: Sequence[str], *, tenant_id: str | None = None
    ) -> set[str]:
        if not ids:
            return set()
        collection = await self._get(domain, tenant_id)
        if collection is None:
            return set()
        result = await asyncio.to_thread(collection.get, ids=list(ids), include=[])
        return set(result.get("ids") or [])

    async def similarity_search(
        self,
        domain: str,
        query: str,
        *,
        k: int = 5,
        filters: MetadataFilter | None = None,
        tenant_id: str | None = None,
    ) -> list[SearchHit]:
        collection = await self._get(domain, tenant_id)
        if collection is None:
            return []
        vector = await self.embeddings.aembed_query(query)
        where = to_chroma_where(self.effective_filter(filters, tenant_id))
        result = await asyncio.to_thread(
            collection.query,
            query_embeddings=[vector],  # type: ignore[arg-type]
            n_results=max(1, k),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        hits: list[SearchHit] = []
        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        for index, chunk_id in enumerate(ids):
            distance = float(distances[index]) if index < len(distances) else 1.0
            hits.append(
                SearchHit(
                    id=str(chunk_id),
                    content=str(docs[index] if index < len(docs) else ""),
                    score=round(1.0 - distance, 6),
                    domain=domain,
                    metadata=dict(metas[index] or {}) if index < len(metas) else {},
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
        collection = await self._get(domain, tenant_id)
        if collection is None:
            return 0
        where = to_chroma_where(self.effective_filter(filters, tenant_id))
        target_ids: list[str]
        if ids:
            existing = await self.existing_ids(domain, ids, tenant_id=tenant_id)
            target_ids = [item for item in ids if item in existing]
        elif where is not None:
            found = await asyncio.to_thread(collection.get, where=where, include=[])
            target_ids = list(found.get("ids") or [])
        else:
            raise VectorStoreError("delete requires either ids or a metadata filter")
        if not target_ids:
            return 0
        await asyncio.to_thread(collection.delete, ids=target_ids)
        return len(target_ids)

    async def iter_documents(self, domain: str, *, tenant_id: str | None = None) -> list[Document]:
        collection = await self._get(domain, tenant_id)
        if collection is None:
            return []
        documents: list[Document] = []
        offset = 0
        page = 500
        while True:
            batch = await asyncio.to_thread(
                collection.get,
                limit=page,
                offset=offset,
                include=["documents", "metadatas"],
            )
            ids = list(batch.get("ids") or [])
            if not ids:
                break
            texts = list(batch.get("documents") or [])
            metas = list(batch.get("metadatas") or [])
            for index, chunk_id in enumerate(ids):
                metadata = dict(metas[index] or {}) if index < len(metas) else {}
                metadata.setdefault("id", chunk_id)
                documents.append(
                    Document(
                        page_content=str(texts[index] if index < len(texts) else ""),
                        metadata=metadata,
                    )
                )
            offset += len(ids)
            if len(ids) < page:
                break
        return documents

    async def list_domains(self, *, tenant_id: str | None = None) -> list[str]:
        collections = await asyncio.to_thread(self._client.list_collections)
        names = [getattr(item, "name", item) for item in collections]
        domains = [self.logical_name(str(name), tenant_id) for name in names]
        return sorted({domain for domain in domains if domain})

    async def domain_stats(self, domain: str, *, tenant_id: str | None = None) -> DomainStats:
        collection = await self._get(domain, tenant_id)
        count = await asyncio.to_thread(collection.count) if collection is not None else 0
        return DomainStats(
            domain=domain,
            store=self.store_type,
            physical_name=self.physical_name(domain, tenant_id),
            vector_count=int(count),
            embedding_model=self.embeddings.model_name,
            embedding_dimension=await self.embeddings.ensure_dimension(),
            extra={
                "exists": collection is not None,
                "mode": "http" if self.settings.chroma_host else "embedded",
            },
        )

    async def drop_domain(self, domain: str, *, tenant_id: str | None = None) -> None:
        name = self.physical_name(domain, tenant_id)
        try:
            await asyncio.to_thread(self._client.delete_collection, name=name)
        except Exception:
            logger.debug("collection absent on drop", extra={"collection": name})

    async def swap_domain(self, target: str, staging: str, *, tenant_id: str | None = None) -> None:
        staging_collection = await self._get(staging, tenant_id)
        if staging_collection is None:
            staging_collection = await self._get_or_create(staging, tenant_id)
        await self.drop_domain(target, tenant_id=tenant_id)
        # Only the name may change: Chroma refuses to alter the distance function
        # of an existing collection.
        await asyncio.to_thread(
            staging_collection.modify, name=self.physical_name(target, tenant_id)
        )
