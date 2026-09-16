"""Knowledge base endpoints.

Every route is domain scoped (``/kb/{domain}/...``) and protected by
``verify_jwt`` at the **router** level, so a new route cannot accidentally be
added unauthenticated.
"""

from __future__ import annotations

import json
from typing import Annotated, Any

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, Query, UploadFile, status

from agent_service.api.schemas import (
    DeleteDocumentsRequest,
    DeleteResponse,
    DomainInfo,
    DomainsResponse,
    ErrorResponse,
    IngestResponse,
    IngestTextRequest,
    IngestUrlsRequest,
    JobResponse,
    ReindexResponse,
    SearchResponse,
)
from agent_service.core.deps import (
    ClaimsDep,
    IngestionDep,
    JobsDep,
    ReadableDomainDep,
    SettingsDep,
    StoreDep,
    TenantDep,
    WritableDomainDep,
)
from agent_service.core.exceptions import IngestionError
from agent_service.core.logging import get_logger
from agent_service.core.security import ADMIN_SCOPE, require_scopes, verify_jwt
from agent_service.kb.base import DomainStats
from agent_service.kb.ingestion import IngestResult, UploadPayload

logger = get_logger(__name__)

PROTECTED_RESPONSES: dict[int | str, dict[str, Any]] = {
    401: {"model": ErrorResponse, "description": "Missing, malformed or expired bearer token."},
    403: {"model": ErrorResponse, "description": "Token lacks the required scope."},
    422: {"model": ErrorResponse, "description": "Unknown domain or invalid request body."},
}

router = APIRouter(
    prefix="/kb",
    tags=["kb"],
    dependencies=[Depends(verify_jwt)],
    responses=PROTECTED_RESPONSES,
)

AdminDep = Annotated[object, Depends(require_scopes(ADMIN_SCOPE))]


def _parse_json_object(raw: str | None, field: str) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise IngestionError(f"{field!r} must be a JSON object: {exc}") from exc
    if not isinstance(value, dict):
        raise IngestionError(f"{field!r} must be a JSON object")
    return value


# ---------------------------------------------------------------- admin ----
@router.get(
    "/domains",
    response_model=DomainsResponse,
    summary="List knowledge base domains",
    description=(
        "Returns every domain configured through `KB_DOMAINS` together with whether a "
        "physical collection/namespace currently exists for it. Requires `kb:admin`."
    ),
)
async def list_domains(
    settings: SettingsDep, store: StoreDep, tenant: TenantDep, _: AdminDep
) -> DomainsResponse:
    """Inventory of configured versus materialised domains."""
    existing = set(await store.list_domains(tenant_id=tenant))
    return DomainsResponse(
        store=settings.vector_store.value,
        tenant_mode=settings.tenant_mode.value,
        domains=[
            DomainInfo(domain=domain, configured=True, exists=domain in existing)
            for domain in settings.domains
        ],
    )


@router.get(
    "/jobs/{job_id}",
    response_model=JobResponse,
    summary="Get a background ingestion job",
    description="Poll the status of an ingestion job returned by an `mode=async` ingest call.",
    responses={**PROTECTED_RESPONSES, 404: {"model": ErrorResponse, "description": "Unknown job."}},
)
async def get_job(job_id: str, jobs: JobsDep, claims: ClaimsDep) -> JobResponse:
    """Return one job record."""
    job = await jobs.get(job_id)
    if job.submitted_by and job.submitted_by != claims.sub and not claims.has_scope(ADMIN_SCOPE):
        # Same shape as "not found" so job ids cannot be probed.
        from agent_service.core.exceptions import NotFoundError

        raise NotFoundError(f"Job {job_id!r} was not found")
    return JobResponse(job=job)


# --------------------------------------------------------------- ingest ----
@router.post(
    "/{domain}/ingest/files",
    response_model=IngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingest uploaded files into a domain",
    description=(
        "Accepts a multipart upload with one or more files (`pdf`, `docx`, `txt`, `md`, "
        "`csv`, `json`, `html`). Files are chunked, embedded and de-duplicated by content "
        "hash. Small batches run inline; anything above `BACKGROUND_INGEST_THRESHOLD` "
        "returns a job id to poll at `GET /kb/jobs/{id}`.\n\n"
        "Requires scope `kb:<domain>:write`."
    ),
)
async def ingest_files(
    domain: WritableDomainDep,
    background: BackgroundTasks,
    ingestion: IngestionDep,
    jobs: JobsDep,
    claims: ClaimsDep,
    tenant: TenantDep,
    settings: SettingsDep,
    files: Annotated[list[UploadFile], File(description="Documents to ingest.")],
    tags: Annotated[str | None, Form(description='JSON object, e.g. {"module":"payroll"}')] = None,
) -> IngestResponse:
    """Chunk, embed and upsert uploaded documents."""
    parsed_tags = _parse_json_object(tags, "tags")
    payloads = [
        UploadPayload(filename=item.filename or "upload.bin", content=await item.read())
        for item in files
    ]
    if not payloads:
        raise IngestionError("at least one file is required")

    async def work() -> dict[str, Any]:
        result = await ingestion.ingest_files(
            domain, payloads, tenant_id=tenant, uploader=claims.sub, tags=parsed_tags
        )
        return result.model_dump()

    if len(payloads) > settings.background_ingest_threshold:
        job = await jobs.create(kind="ingest_files", domain=domain, submitted_by=claims.sub)
        background.add_task(jobs.run, job.id, work)
        return IngestResponse(mode="async", job_id=job.id)

    return IngestResponse(mode="sync", result=IngestResult(**await work()))


@router.post(
    "/{domain}/ingest/text",
    response_model=IngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingest raw text into a domain",
    description=(
        "Chunks and embeds a string. Useful for ERP records, generated summaries or "
        "policy snippets. Requires scope `kb:<domain>:write`."
    ),
)
async def ingest_text(
    domain: WritableDomainDep,
    payload: IngestTextRequest,
    ingestion: IngestionDep,
    claims: ClaimsDep,
    tenant: TenantDep,
) -> IngestResponse:
    """Chunk, embed and upsert a raw string."""
    result = await ingestion.ingest_text(
        domain,
        payload.text,
        source=payload.source,
        tenant_id=tenant,
        uploader=claims.sub,
        tags=payload.tags,
    )
    return IngestResponse(mode="sync", result=result)


@router.post(
    "/{domain}/ingest/urls",
    response_model=IngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingest web pages into a domain",
    description=(
        "Fetches each URL, extracts readable text and ingests it. URLs resolving to "
        "private, loopback or link-local addresses are rejected (SSRF guard). "
        "Requires scope `kb:<domain>:write`."
    ),
)
async def ingest_urls(
    domain: WritableDomainDep,
    payload: IngestUrlsRequest,
    background: BackgroundTasks,
    ingestion: IngestionDep,
    jobs: JobsDep,
    claims: ClaimsDep,
    tenant: TenantDep,
    settings: SettingsDep,
) -> IngestResponse:
    """Fetch and ingest a list of URLs."""

    async def work() -> dict[str, Any]:
        result = await ingestion.ingest_urls(
            domain, payload.urls, tenant_id=tenant, uploader=claims.sub, tags=payload.tags
        )
        return result.model_dump()

    if len(payload.urls) > settings.background_ingest_threshold:
        job = await jobs.create(kind="ingest_urls", domain=domain, submitted_by=claims.sub)
        background.add_task(jobs.run, job.id, work)
        return IngestResponse(mode="async", job_id=job.id)

    return IngestResponse(mode="sync", result=IngestResult(**await work()))


# --------------------------------------------------------------- search ----
@router.get(
    "/{domain}/search",
    response_model=SearchResponse,
    summary="Search one domain",
    description=(
        "Returns the top `k` chunks for `q`, ranked by cosine similarity, with their "
        "metadata. An optional `filter` query parameter carries a JSON metadata filter, "
        'for example `{"module":"payroll"}`. Requires scope `kb:<domain>:read`.'
    ),
)
async def search(
    domain: ReadableDomainDep,
    store: StoreDep,
    tenant: TenantDep,
    q: Annotated[str, Query(min_length=1, description="Query text.", examples=["leave accrual"])],
    k: Annotated[int, Query(ge=1, le=50, description="Number of chunks to return.")] = 5,
    filter: Annotated[
        str | None, Query(description='JSON metadata filter, e.g. {"module":"payroll"}')
    ] = None,
) -> SearchResponse:
    """Rank chunks of a single domain."""
    hits = await store.similarity_search(
        domain, q, k=k, filters=_parse_json_object(filter, "filter") or None, tenant_id=tenant
    )
    return SearchResponse(domain=domain, query=q, k=k, hits=hits)


# --------------------------------------------------------------- delete ----
@router.delete(
    "/{domain}/documents",
    response_model=DeleteResponse,
    summary="Delete chunks from a domain",
    description=(
        "Removes chunks by explicit id or by metadata filter (for example every chunk of "
        "one `doc_id`). Requires scope `kb:<domain>:write`."
    ),
)
async def delete_documents(
    domain: WritableDomainDep,
    payload: DeleteDocumentsRequest,
    store: StoreDep,
    tenant: TenantDep,
) -> DeleteResponse:
    """Delete by ids and/or metadata filter."""
    if not payload.ids and not payload.filter:
        raise IngestionError("provide 'ids' and/or 'filter'")
    deleted = await store.delete(domain, ids=payload.ids, filters=payload.filter, tenant_id=tenant)
    await store.persist()
    return DeleteResponse(domain=domain, deleted=deleted)


@router.delete(
    "/{domain}",
    response_model=DeleteResponse,
    summary="Drop an entire domain",
    description="Deletes the whole collection/namespace for a domain. Requires `kb:admin`.",
)
async def drop_domain(
    domain: ReadableDomainDep, store: StoreDep, tenant: TenantDep, _: AdminDep
) -> DeleteResponse:
    """Remove every vector in a domain."""
    stats = await store.domain_stats(domain, tenant_id=tenant)
    await store.drop_domain(domain, tenant_id=tenant)
    logger.warning("domain dropped", extra={"domain": domain, "vectors": stats.vector_count})
    return DeleteResponse(domain=domain, deleted=stats.vector_count)


# ---------------------------------------------------------------- stats ----
@router.get(
    "/{domain}/stats",
    response_model=DomainStats,
    summary="Domain statistics",
    description=(
        "Store type, physical collection/namespace name, vector count, embedding model "
        "and dimension. Requires `kb:admin`."
    ),
)
async def domain_stats(
    domain: ReadableDomainDep, store: StoreDep, tenant: TenantDep, _: AdminDep
) -> DomainStats:
    """Inventory for one domain."""
    return await store.domain_stats(domain, tenant_id=tenant)


@router.post(
    "/{domain}/reindex",
    response_model=ReindexResponse,
    summary="Re-embed a domain and swap it in atomically",
    description=(
        "Reads every chunk, re-embeds it with the currently configured embedding model "
        "into a staging collection, then swaps the staging collection over the live one. "
        "Use after changing `EMBEDDING_MODEL`. Requires `kb:admin`."
    ),
)
async def reindex(
    domain: ReadableDomainDep, store: StoreDep, tenant: TenantDep, _: AdminDep
) -> ReindexResponse:
    """Rebuild a domain with the current embedding model."""
    count = await store.reindex(domain, tenant_id=tenant)
    logger.info("domain reindexed", extra={"domain": domain, "documents": count})
    return ReindexResponse(
        domain=domain,
        documents_reindexed=count,
        embedding_model=store.embeddings.model_name,
        embedding_dimension=await store.embeddings.ensure_dimension(),
    )
