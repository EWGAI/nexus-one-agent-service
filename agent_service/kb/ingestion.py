"""Document ingestion: load -> chunk -> stamp metadata -> dedupe -> upsert.

Every chunk is stamped with the provenance metadata the ERP needs:
``domain``, ``tenant_id``, ``source``, ``doc_id``, ``chunk_index``,
``content_hash``, ``ingested_at``, ``uploader`` plus any caller supplied tags
(``module=payroll``, ``fiscal_year=2026``, ...).

De-duplication is per domain and based on ``content_hash``, which is also used
as the vector id, so re-ingesting a document is idempotent.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import datetime as dt
import hashlib
import io
import ipaddress
from pathlib import Path
import re
import socket
import tempfile
from typing import Any
from urllib.parse import urlparse

import httpx
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

from agent_service.config import Settings
from agent_service.core.exceptions import IngestionError
from agent_service.core.logging import get_logger
from agent_service.kb.base import VectorStoreAdapter

logger = get_logger(__name__)

TEXT_EXTENSIONS = frozenset(
    {".txt", ".md", ".markdown", ".rst", ".csv", ".json", ".log", ".yaml", ".yml"}
)
HTML_EXTENSIONS = frozenset({".html", ".htm"})
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | HTML_EXTENSIONS | {".pdf", ".docx"}

_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class UploadPayload:
    """An uploaded file already read into memory."""

    filename: str
    content: bytes


class IngestResult(BaseModel):
    """Summary of one ingestion run."""

    domain: str = Field(description="Target knowledge base domain.")
    documents_processed: int = Field(description="Number of source documents loaded.")
    chunks_added: int = Field(description="Chunks embedded and stored.")
    duplicates_skipped: int = Field(description="Chunks already present (same content hash).")
    sources: list[str] = Field(default_factory=list, description="Source identifiers processed.")
    chunk_ids: list[str] = Field(default_factory=list, description="Ids of the stored chunks.")
    errors: list[str] = Field(default_factory=list, description="Per-source failures, if any.")

    def merge(self, other: IngestResult) -> IngestResult:
        """Combine two results (used when several sources are ingested together)."""
        return IngestResult(
            domain=self.domain,
            documents_processed=self.documents_processed + other.documents_processed,
            chunks_added=self.chunks_added + other.chunks_added,
            duplicates_skipped=self.duplicates_skipped + other.duplicates_skipped,
            sources=[*self.sources, *other.sources],
            chunk_ids=[*self.chunk_ids, *other.chunk_ids],
            errors=[*self.errors, *other.errors],
        )


def normalise(text: str) -> str:
    """Collapse whitespace so trivially different copies hash identically."""
    return _WHITESPACE_RE.sub(" ", text).strip()


def content_hash(domain: str, tenant_id: str | None, text: str) -> str:
    """Deterministic chunk id: unique per (domain, tenant, normalised content)."""
    digest = hashlib.sha256()
    digest.update(domain.encode("utf-8"))
    digest.update(b"\x00")
    digest.update((tenant_id or "").encode("utf-8"))
    digest.update(b"\x00")
    digest.update(normalise(text).encode("utf-8"))
    return digest.hexdigest()


def document_id(domain: str, tenant_id: str | None, source: str) -> str:
    """Stable id for a logical source document within a domain."""
    digest = hashlib.sha256(f"{domain}\x00{tenant_id or ''}\x00{source}".encode())
    return digest.hexdigest()[:32]


def _is_public_url(url: str) -> tuple[bool, str]:
    """Reject non-http(s) schemes and addresses inside private ranges (SSRF guard)."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False, f"unsupported scheme {parsed.scheme!r}"
    if not parsed.hostname:
        return False, "missing host"
    try:
        infos = socket.getaddrinfo(parsed.hostname, parsed.port or 0, proto=socket.IPPROTO_TCP)
    except OSError as exc:
        return False, f"dns resolution failed: {exc}"
    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_reserved
            or address.is_multicast
        ):
            return False, f"resolves to non-public address {address}"
    return True, ""


class IngestionService:
    """Turns raw inputs into embedded, de-duplicated chunks in a domain index."""

    def __init__(self, store: VectorStoreAdapter, settings: Settings) -> None:
        self.store = store
        self.settings = settings
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            add_start_index=True,
        )

    # -- loaders -----------------------------------------------------------
    def _load_file_sync(self, payload: UploadPayload) -> list[Document]:
        suffix = Path(payload.filename).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise IngestionError(
                f"Unsupported file type {suffix or '(none)'} for {payload.filename!r}",
                details={"supported": sorted(SUPPORTED_EXTENSIONS)},
            )
        if len(payload.content) > self.settings.max_upload_bytes:
            raise IngestionError(
                f"{payload.filename!r} exceeds MAX_UPLOAD_BYTES",
                details={"size": len(payload.content), "limit": self.settings.max_upload_bytes},
            )

        if suffix in TEXT_EXTENSIONS:
            return [Document(page_content=payload.content.decode("utf-8", errors="replace"))]
        if suffix in HTML_EXTENSIONS:
            return [Document(page_content=html_to_text(payload.content.decode("utf-8", "replace")))]
        if suffix == ".pdf":
            return _pdf_to_documents(payload.content)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"upload{suffix}"
            path.write_bytes(payload.content)
            import docx2txt

            return [Document(page_content=docx2txt.process(str(path)) or "")]

    async def load_file(self, payload: UploadPayload) -> list[Document]:
        """Parse an uploaded file into documents (runs in a worker thread)."""
        return await asyncio.to_thread(self._load_file_sync, payload)

    async def load_url(self, url: str) -> list[Document]:
        """Fetch a URL and extract its readable text."""
        allowed, reason = await asyncio.to_thread(_is_public_url, url)
        if not allowed:
            raise IngestionError(f"Refusing to fetch {url!r}: {reason}", code="url_not_allowed")
        try:
            async with httpx.AsyncClient(
                timeout=30.0, follow_redirects=True, max_redirects=5
            ) as client:
                response = await client.get(url, headers={"User-Agent": "agent-service/1.0"})
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise IngestionError(f"Failed to fetch {url!r}: {exc}") from exc

        content_type = response.headers.get("content-type", "")
        if "html" in content_type:
            text = html_to_text(response.text)
        elif "pdf" in content_type:
            return await self.load_file(
                UploadPayload(filename="download.pdf", content=response.content)
            )
        else:
            text = response.text
        if not text.strip():
            raise IngestionError(f"{url!r} returned no extractable text")
        return [Document(page_content=text, metadata={"content_type": content_type})]

    # -- ingestion ----------------------------------------------------------
    async def ingest_documents(
        self,
        domain: str,
        documents: list[Document],
        *,
        source: str,
        tenant_id: str | None,
        uploader: str,
        tags: dict[str, Any] | None = None,
    ) -> IngestResult:
        """Chunk, stamp, de-duplicate and upsert ``documents`` into ``domain``."""
        if not documents:
            return IngestResult(
                domain=domain, documents_processed=0, chunks_added=0, duplicates_skipped=0
            )

        chunks = self.splitter.split_documents(documents)
        doc_id = document_id(domain, tenant_id, source)
        ingested_at = dt.datetime.now(tz=dt.UTC).isoformat()

        stamped: list[Document] = []
        ids: list[str] = []
        seen: set[str] = set()
        duplicates = 0

        for index, chunk in enumerate(chunks):
            text = chunk.page_content.strip()
            if not text:
                continue
            chunk_hash = content_hash(domain, tenant_id, text)
            if chunk_hash in seen:
                duplicates += 1
                continue
            seen.add(chunk_hash)
            metadata: dict[str, Any] = {
                **{key: value for key, value in chunk.metadata.items() if key != "source"},
                **(tags or {}),
                "domain": domain,
                "tenant_id": tenant_id,
                "source": source,
                "doc_id": doc_id,
                "chunk_index": index,
                "content_hash": chunk_hash,
                "ingested_at": ingested_at,
                "uploader": uploader,
            }
            stamped.append(Document(page_content=text, metadata=metadata))
            ids.append(chunk_hash)

        already = await self.store.existing_ids(domain, ids, tenant_id=tenant_id)
        fresh_documents = [doc for doc, cid in zip(stamped, ids, strict=True) if cid not in already]
        fresh_ids = [cid for cid in ids if cid not in already]
        duplicates += len(already)

        if fresh_documents:
            await self.store.add_documents(
                domain, fresh_documents, ids=fresh_ids, tenant_id=tenant_id
            )
            await self.store.persist()

        logger.info(
            "ingested source",
            extra={
                "domain": domain,
                "source": source,
                "chunks_added": len(fresh_ids),
                "duplicates_skipped": duplicates,
            },
        )
        return IngestResult(
            domain=domain,
            documents_processed=len(documents),
            chunks_added=len(fresh_ids),
            duplicates_skipped=duplicates,
            sources=[source],
            chunk_ids=fresh_ids,
        )

    async def ingest_text(
        self,
        domain: str,
        text: str,
        *,
        source: str,
        tenant_id: str | None,
        uploader: str,
        tags: dict[str, Any] | None = None,
    ) -> IngestResult:
        """Ingest a raw string."""
        if not text.strip():
            raise IngestionError("'text' must not be empty")
        return await self.ingest_documents(
            domain,
            [Document(page_content=text)],
            source=source,
            tenant_id=tenant_id,
            uploader=uploader,
            tags=tags,
        )

    async def ingest_files(
        self,
        domain: str,
        payloads: list[UploadPayload],
        *,
        tenant_id: str | None,
        uploader: str,
        tags: dict[str, Any] | None = None,
    ) -> IngestResult:
        """Ingest a batch of uploaded files, isolating per-file failures."""
        result = IngestResult(
            domain=domain, documents_processed=0, chunks_added=0, duplicates_skipped=0
        )
        for payload in payloads:
            try:
                documents = await self.load_file(payload)
                partial = await self.ingest_documents(
                    domain,
                    documents,
                    source=payload.filename,
                    tenant_id=tenant_id,
                    uploader=uploader,
                    tags=tags,
                )
            except Exception as exc:
                logger.warning(
                    "file ingestion failed",
                    extra={"domain": domain, "source": payload.filename, "error": str(exc)},
                )
                result = result.merge(
                    IngestResult(
                        domain=domain,
                        documents_processed=0,
                        chunks_added=0,
                        duplicates_skipped=0,
                        errors=[f"{payload.filename}: {exc}"],
                    )
                )
                continue
            result = result.merge(partial)
        return result

    async def ingest_urls(
        self,
        domain: str,
        urls: list[str],
        *,
        tenant_id: str | None,
        uploader: str,
        tags: dict[str, Any] | None = None,
    ) -> IngestResult:
        """Ingest a batch of URLs, isolating per-URL failures."""
        result = IngestResult(
            domain=domain, documents_processed=0, chunks_added=0, duplicates_skipped=0
        )
        for url in urls:
            try:
                documents = await self.load_url(url)
                partial = await self.ingest_documents(
                    domain,
                    documents,
                    source=url,
                    tenant_id=tenant_id,
                    uploader=uploader,
                    tags=tags,
                )
            except Exception as exc:
                logger.warning(
                    "url ingestion failed",
                    extra={"domain": domain, "source": url, "error": str(exc)},
                )
                result = result.merge(
                    IngestResult(
                        domain=domain,
                        documents_processed=0,
                        chunks_added=0,
                        duplicates_skipped=0,
                        errors=[f"{url}: {exc}"],
                    )
                )
                continue
            result = result.merge(partial)
        return result


def html_to_text(markup: str) -> str:
    """Strip scripts/styles and return readable text."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(markup, "html.parser")
    for element in soup(["script", "style", "noscript"]):
        element.decompose()
    return "\n".join(line.strip() for line in soup.get_text("\n").splitlines() if line.strip())


def _pdf_to_documents(content: bytes) -> list[Document]:
    """One document per PDF page, so page numbers survive into chunk metadata."""
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(content))
    documents = [
        Document(page_content=page.extract_text() or "", metadata={"page": number})
        for number, page in enumerate(reader.pages, start=1)
    ]
    kept = [document for document in documents if document.page_content.strip()]
    if not kept:
        raise IngestionError("the PDF contains no extractable text (is it a scan?)")
    return kept
