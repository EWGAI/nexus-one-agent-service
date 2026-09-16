"""Request/response models with OpenAPI examples."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from agent_service.kb.base import SearchHit
from agent_service.kb.ingestion import IngestResult
from agent_service.kb.jobs import Job

MetadataTags = dict[str, Any]


class ErrorBody(BaseModel):
    """Inner payload of every error response."""

    code: str = Field(description="Stable machine readable error code.")
    message: str = Field(description="Human readable explanation.")
    details: dict[str, Any] = Field(default_factory=dict)
    request_id: str | None = Field(
        default=None, description="Correlation id, also in X-Request-ID."
    )


class ErrorResponse(BaseModel):
    """Consistent error envelope used by every non-2xx response."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "error": {
                        "code": "forbidden",
                        "message": "Insufficient scope for domain 'hr'",
                        "details": {"required_scope": "kb:hr:write"},
                        "request_id": "3f1c1a0e9f4b4c0a8f2a",
                    }
                }
            ]
        }
    )

    error: ErrorBody


class IngestTextRequest(BaseModel):
    """Ingest a raw string into a domain."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "text": "Employees accrue 1.75 days of paid leave per completed month.",
                    "source": "hr-policy-2026",
                    "tags": {"module": "leave", "fiscal_year": 2026},
                }
            ]
        }
    )

    text: str = Field(min_length=1, description="Raw text to chunk, embed and store.")
    source: str = Field(
        default="inline-text", max_length=512, description="Provenance label stored on every chunk."
    )
    tags: MetadataTags = Field(
        default_factory=dict, description="Extra metadata stamped on every chunk."
    )


class IngestUrlsRequest(BaseModel):
    """Ingest one or more public URLs."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"urls": ["https://example.com/handbook"], "tags": {"module": "policies"}}]
        }
    )

    urls: list[str] = Field(min_length=1, max_length=50, description="HTTP(S) URLs to fetch.")
    tags: MetadataTags = Field(default_factory=dict)


class IngestResponse(BaseModel):
    """Either an inline summary or a background job handle."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "mode": "sync",
                    "job_id": None,
                    "result": {
                        "domain": "hr",
                        "documents_processed": 1,
                        "chunks_added": 3,
                        "duplicates_skipped": 0,
                        "sources": ["hr-policy-2026"],
                        "chunk_ids": ["6f1b..."],
                        "errors": [],
                    },
                }
            ]
        }
    )

    mode: Literal["sync", "async"] = Field(description="Whether ingestion ran inline or queued.")
    job_id: str | None = Field(default=None, description="Poll GET /kb/jobs/{id} when mode=async.")
    result: IngestResult | None = Field(default=None, description="Present when mode=sync.")


class JobResponse(BaseModel):
    """Status of a background ingestion job."""

    job: Job


class SearchResponse(BaseModel):
    """Ranked chunks for a query."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "domain": "hr",
                    "query": "how much leave do I accrue",
                    "k": 5,
                    "hits": [
                        {
                            "id": "6f1b9c...",
                            "content": "Employees accrue 1.75 days of paid leave per month.",
                            "score": 0.8123,
                            "domain": "hr",
                            "metadata": {"source": "hr-policy-2026", "module": "leave"},
                        }
                    ],
                }
            ]
        }
    )

    domain: str
    query: str
    k: int
    hits: list[SearchHit]


class DeleteDocumentsRequest(BaseModel):
    """Delete by explicit ids or by metadata filter (at least one is required)."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"ids": None, "filter": {"doc_id": "9b1f0c2a", "module": "leave"}},
                {"ids": ["6f1b9c..."], "filter": None},
            ]
        }
    )

    ids: list[str] | None = Field(default=None, description="Chunk ids (content hashes).")
    filter: dict[str, Any] | None = Field(
        default=None,
        description="Metadata filter, e.g. {'doc_id': '...'} or {'module': ['a','b']}.",
    )


class DeleteResponse(BaseModel):
    """How many chunks were removed."""

    domain: str
    deleted: int


class DomainInfo(BaseModel):
    """A configured domain and whether it currently holds data."""

    domain: str
    configured: bool = True
    exists: bool = Field(description="Whether a physical collection/namespace exists.")


class DomainsResponse(BaseModel):
    """All configured domains plus their physical presence."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "store": "chroma",
                    "tenant_mode": "off",
                    "domains": [{"domain": "hr", "configured": True, "exists": True}],
                }
            ]
        }
    )

    store: str
    tenant_mode: str
    domains: list[DomainInfo]


class ReindexResponse(BaseModel):
    """Result of an atomic re-embed + swap."""

    domain: str
    documents_reindexed: int
    embedding_model: str
    embedding_dimension: int


class ChatRequest(BaseModel):
    """A single agent turn."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "message": "How many leave days do I accrue each month?",
                    "thread_id": None,
                    "domains": ["hr"],
                    "filters": {"module": "leave"},
                }
            ]
        }
    )

    message: str = Field(min_length=1, max_length=8000, description="User message.")
    thread_id: str | None = Field(
        default=None, description="Continue an existing conversation; omit to start a new one."
    )
    domains: list[str] | None = Field(
        default=None,
        description=(
            "Restrict retrieval to these domains. Omit to let the router choose from the "
            "domains the token may read."
        ),
    )
    filters: dict[str, Any] | None = Field(
        default=None, description="Metadata filter applied to every retrieval."
    )


class ChatResponse(BaseModel):
    """Agent answer with its sources."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "answer": "You accrue 1.75 days of paid leave per completed month. [1]",
                    "thread_id": "a3c9f0...",
                    "domains_searched": ["hr", "company"],
                    "routing_reason": "question mentions leave policy",
                    "sources": [],
                    "tool_calls": [],
                }
            ]
        }
    )

    answer: str
    thread_id: str
    domains_searched: list[str]
    routing_reason: str | None = None
    sources: list[SearchHit] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Liveness payload."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"status": "ok", "service": "agent-service", "version": "0.1.0"}]
        }
    )

    status: Literal["ok"] = "ok"
    service: str
    version: str


class ReadyResponse(BaseModel):
    """Readiness payload including backend checks."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "status": "ready",
                    "store": "chroma",
                    "store_healthy": True,
                    "embedding_model": "text-embedding-3-small",
                    "embedding_dimension": 1536,
                    "domains": ["hr", "finance", "engineering", "it", "company"],
                }
            ]
        }
    )

    status: Literal["ready", "degraded"]
    store: str
    store_healthy: bool
    embedding_model: str
    embedding_dimension: int
    domains: list[str]
