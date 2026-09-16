"""FAISS adapter - one on-disk index per (tenant, domain).

Layout::

    FAISS_INDEX_PATH/
      kb_hr/{index.faiss,records.json,meta.json}
      kb_finance/{index.faiss,records.json,meta.json}

Built directly on ``faiss`` so there is no pickle involved: chunk text and
metadata are persisted as JSON, which removes the "allow dangerous
deserialization" footgun that wrapper libraries need.

Vectors are L2-normalised and stored in an inner-product index, so the raw FAISS
score *is* the cosine similarity and is directly comparable with the other
adapters. Indexes are loaded lazily, cached in memory, and written after every
mutation.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
import json
from pathlib import Path
import shutil
from typing import Any, ClassVar

import faiss
from langchain_core.documents import Document
import numpy as np

from agent_service.config import Settings
from agent_service.core.exceptions import ConfigurationError, VectorStoreError
from agent_service.core.logging import get_logger
from agent_service.kb.base import DomainStats, MetadataFilter, SearchHit, VectorStoreAdapter
from agent_service.kb.embeddings import EmbeddingBundle
from agent_service.kb.filters import build_predicate

logger = get_logger(__name__)

INDEX_FILE = "index.faiss"
RECORDS_FILE = "records.json"
META_FILE = "meta.json"

Predicate = Callable[[dict[str, Any]], bool]


def _as_matrix(vectors: Sequence[Sequence[float]]) -> Any:
    """Float32 matrix with L2-normalised rows (so inner product == cosine)."""
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    faiss.normalize_L2(matrix)
    return matrix


class DomainIndex:
    """A single FAISS index plus the chunk payloads that belong to it."""

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.index: faiss.Index = faiss.IndexFlatIP(dimension)
        self.ids: list[str] = []
        self.records: dict[str, dict[str, Any]] = {}

    @property
    def count(self) -> int:
        """Number of stored vectors."""
        return len(self.ids)

    def add(
        self,
        ids: Sequence[str],
        vectors: Sequence[Sequence[float]],
        documents: Sequence[Document],
    ) -> None:
        """Upsert: existing ids are removed first, so re-adding is idempotent."""
        duplicates = [item for item in ids if item in self.records]
        if duplicates:
            self.remove(duplicates)
        self.index.add(_as_matrix(vectors))
        self.ids.extend(ids)
        for identifier, document in zip(ids, documents, strict=True):
            self.records[identifier] = {
                "content": document.page_content,
                "metadata": dict(document.metadata),
            }

    def remove(self, ids: Sequence[str]) -> int:
        """Delete ids, rebuilding the flat index around the survivors."""
        targets = {item for item in ids if item in self.records}
        if not targets:
            return 0
        survivors = [position for position, item in enumerate(self.ids) if item not in targets]
        rebuilt = faiss.IndexFlatIP(self.dimension)
        if survivors:
            existing = np.asarray(self.index.reconstruct_n(0, self.index.ntotal), dtype=np.float32)
            rebuilt.add(existing[survivors])
        self.index = rebuilt
        self.ids = [self.ids[position] for position in survivors]
        for item in targets:
            self.records.pop(item, None)
        return len(targets)

    def search(
        self, vector: Sequence[float], k: int, predicate: Predicate | None
    ) -> list[tuple[str, dict[str, Any], float]]:
        """Top ``k`` matches after applying the metadata post-filter."""
        if not self.ids:
            return []
        # Over-fetch so post-filtering still has candidates left.
        limit = min(len(self.ids), max(k * 10, k))
        scores, positions = self.index.search(_as_matrix([vector]), limit)
        results: list[tuple[str, dict[str, Any], float]] = []
        for score, position in zip(scores[0], positions[0], strict=True):
            if position < 0:
                continue
            identifier = self.ids[int(position)]
            record = self.records[identifier]
            if predicate is not None and not predicate(record["metadata"]):
                continue
            results.append((identifier, record, float(score)))
            if len(results) >= k:
                break
        return results

    def matching_ids(self, predicate: Predicate) -> list[str]:
        """Ids whose metadata satisfies ``predicate``."""
        return [item for item in self.ids if predicate(self.records[item]["metadata"])]

    # -- persistence -------------------------------------------------------
    def save(self, folder: Path, *, embedding_model: str) -> None:
        """Write the index and its JSON sidecars."""
        folder.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(folder / INDEX_FILE))
        (folder / RECORDS_FILE).write_text(
            json.dumps({"ids": self.ids, "records": self.records}, default=str),
            encoding="utf-8",
        )
        (folder / META_FILE).write_text(
            json.dumps(
                {
                    "embedding_model": embedding_model,
                    "embedding_dimension": self.dimension,
                    "vector_count": self.count,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, folder: Path, expected_dimension: int) -> DomainIndex:
        """Read an index back from disk, validating the embedding dimension."""
        index = faiss.read_index(str(folder / INDEX_FILE))
        if index.d != expected_dimension:
            raise ConfigurationError(
                "Embedding dimension mismatch for FAISS index",
                details={
                    "index": folder.name,
                    "index_dimension": int(index.d),
                    "embedding_dimension": expected_dimension,
                    "hint": "run POST /kb/{domain}/reindex or delete the index directory",
                },
            )
        payload = json.loads((folder / RECORDS_FILE).read_text(encoding="utf-8"))
        loaded = cls(expected_dimension)
        loaded.index = index
        loaded.ids = list(payload.get("ids", []))
        loaded.records = dict(payload.get("records", {}))
        return loaded


class FaissStore(VectorStoreAdapter):
    """:class:`~agent_service.kb.base.VectorStoreAdapter` backed by local FAISS indexes."""

    store_type: ClassVar[str] = "faiss"

    def __init__(self, settings: Settings, embeddings: EmbeddingBundle) -> None:
        super().__init__(settings, embeddings)
        self.root: Path = settings.faiss_index_path
        self._cache: dict[str, DomainIndex] = {}
        self._lock = asyncio.Lock()

    def _folder(self, physical: str) -> Path:
        return self.root / physical

    # -- lifecycle ---------------------------------------------------------
    async def initialize(self) -> None:
        dimension = await self.embeddings.ensure_dimension()
        self.root.mkdir(parents=True, exist_ok=True)
        loaded = []
        for domain in await self.list_domains():
            await self._load(domain, None, create=False)
            loaded.append(domain)
        logger.info(
            "faiss ready",
            extra={"root": str(self.root.resolve()), "dimension": dimension, "domains": loaded},
        )

    async def health(self) -> bool:
        return self.root.exists() or self.root.parent.exists()

    async def persist(self) -> None:
        async with self._lock:
            entries = list(self._cache.items())
        for physical, index in entries:
            await asyncio.to_thread(self._save, physical, index)

    def _save(self, physical: str, index: DomainIndex) -> None:
        index.save(self._folder(physical), embedding_model=self.embeddings.model_name)

    # -- index management --------------------------------------------------
    def _load_sync(self, physical: str, create: bool) -> DomainIndex | None:
        folder = self._folder(physical)
        if (folder / INDEX_FILE).is_file():
            return DomainIndex.load(folder, self.embeddings.dimension)
        return DomainIndex(self.embeddings.dimension) if create else None

    async def _load(
        self, domain: str, tenant_id: str | None, *, create: bool
    ) -> DomainIndex | None:
        physical = self.physical_name(domain, tenant_id)
        async with self._lock:
            cached = self._cache.get(physical)
            if cached is not None:
                return cached
            index = await asyncio.to_thread(self._load_sync, physical, create)
            if index is not None:
                self._cache[physical] = index
            return index

    async def _persist_one(self, domain: str, tenant_id: str | None) -> None:
        physical = self.physical_name(domain, tenant_id)
        index = self._cache.get(physical)
        if index is not None:
            await asyncio.to_thread(self._save, physical, index)

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

        vectors = await self.embeddings.aembed_documents([d.page_content for d in documents])
        index = await self._load(domain, tenant_id, create=True)
        assert index is not None  # noqa: S101 - create=True always returns
        await asyncio.to_thread(index.add, doc_ids, vectors, documents)
        await self._persist_one(domain, tenant_id)
        return doc_ids

    async def existing_ids(
        self, domain: str, ids: Sequence[str], *, tenant_id: str | None = None
    ) -> set[str]:
        if not ids:
            return set()
        index = await self._load(domain, tenant_id, create=False)
        if index is None:
            return set()
        return {item for item in ids if item in index.records}

    async def similarity_search(
        self,
        domain: str,
        query: str,
        *,
        k: int = 5,
        filters: MetadataFilter | None = None,
        tenant_id: str | None = None,
    ) -> list[SearchHit]:
        index = await self._load(domain, tenant_id, create=False)
        if index is None:
            return []
        vector = await self.embeddings.aembed_query(query)
        predicate = build_predicate(self.effective_filter(filters, tenant_id))
        results = await asyncio.to_thread(index.search, vector, max(1, k), predicate)
        return [
            SearchHit(
                id=identifier,
                content=str(record["content"]),
                score=round(score, 6),
                domain=domain,
                metadata=dict(record["metadata"]),
            )
            for identifier, record, score in results
        ]

    async def delete(
        self,
        domain: str,
        *,
        ids: Sequence[str] | None = None,
        filters: MetadataFilter | None = None,
        tenant_id: str | None = None,
    ) -> int:
        index = await self._load(domain, tenant_id, create=False)
        if index is None:
            return 0
        predicate = build_predicate(self.effective_filter(filters, tenant_id))
        if ids:
            targets = list(ids)
        elif predicate is not None:
            targets = index.matching_ids(predicate)
        else:
            raise VectorStoreError("delete requires either ids or a metadata filter")

        removed = await asyncio.to_thread(index.remove, targets)
        if removed:
            await self._persist_one(domain, tenant_id)
        return removed

    async def iter_documents(self, domain: str, *, tenant_id: str | None = None) -> list[Document]:
        index = await self._load(domain, tenant_id, create=False)
        if index is None:
            return []
        documents = []
        for identifier in index.ids:
            record = index.records[identifier]
            metadata = dict(record["metadata"])
            metadata.setdefault("id", identifier)
            documents.append(Document(page_content=str(record["content"]), metadata=metadata))
        return documents

    async def list_domains(self, *, tenant_id: str | None = None) -> list[str]:
        if not self.root.is_dir():
            return []
        domains = []
        for child in self.root.iterdir():
            if not child.is_dir():
                continue
            logical = self.logical_name(child.name, tenant_id)
            if logical:
                domains.append(logical)
        return sorted(set(domains))

    async def domain_stats(self, domain: str, *, tenant_id: str | None = None) -> DomainStats:
        index = await self._load(domain, tenant_id, create=False)
        physical = self.physical_name(domain, tenant_id)
        return DomainStats(
            domain=domain,
            store=self.store_type,
            physical_name=physical,
            vector_count=index.count if index is not None else 0,
            embedding_model=self.embeddings.model_name,
            embedding_dimension=await self.embeddings.ensure_dimension(),
            extra={"exists": index is not None, "path": str(self._folder(physical).resolve())},
        )

    async def drop_domain(self, domain: str, *, tenant_id: str | None = None) -> None:
        physical = self.physical_name(domain, tenant_id)
        async with self._lock:
            self._cache.pop(physical, None)
        folder = self._folder(physical)
        if folder.is_dir():
            await asyncio.to_thread(shutil.rmtree, folder)

    async def swap_domain(self, target: str, staging: str, *, tenant_id: str | None = None) -> None:
        staging_physical = self.physical_name(staging, tenant_id)
        target_physical = self.physical_name(target, tenant_id)

        staging_index = self._cache.get(staging_physical) or DomainIndex(self.embeddings.dimension)
        await asyncio.to_thread(self._save, staging_physical, staging_index)

        async with self._lock:
            self._cache.pop(target_physical, None)
            self._cache.pop(staging_physical, None)

        def _swap() -> None:
            target_dir = self._folder(target_physical)
            staging_dir = self._folder(staging_physical)
            if target_dir.is_dir():
                shutil.rmtree(target_dir)
            staging_dir.replace(target_dir)

        await asyncio.to_thread(_swap)
