"""Unit tests for ingestion helpers, jobs, filters, logging and health."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import httpx
from langchain_core.documents import Document
import pytest

from agent_service.core.exceptions import AppError, IngestionError, NotFoundError, VectorStoreError
from agent_service.core.logging import (
    JSONFormatter,
    PlainFormatter,
    configure_logging,
    request_id_var,
)
from agent_service.kb.factory import get_vector_store
from agent_service.kb.filters import build_predicate, to_chroma_where, to_pinecone_filter
from agent_service.kb.ingestion import (
    IngestionService,
    UploadPayload,
    _is_public_url,
    content_hash,
    document_id,
    html_to_text,
    normalise,
)
from agent_service.kb.jobs import JobStatus, JobStore
from agent_service.main import create_app

from .conftest import make_settings


# ------------------------------------------------------------------ hashing --
def test_normalise_collapses_whitespace() -> None:
    assert normalise("  a \n\t b  ") == "a b"


def test_content_hash_is_stable_and_scoped() -> None:
    assert content_hash("hr", None, "hello  world") == content_hash("hr", None, "hello world")
    assert content_hash("hr", None, "x") != content_hash("finance", None, "x")
    assert content_hash("hr", "acme", "x") != content_hash("hr", "globex", "x")


def test_document_id_is_stable_per_source() -> None:
    assert document_id("hr", None, "a.pdf") == document_id("hr", None, "a.pdf")
    assert document_id("hr", None, "a.pdf") != document_id("hr", None, "b.pdf")


# ------------------------------------------------------------------ loaders --
def test_html_to_text_drops_scripts_and_styles() -> None:
    text = html_to_text("<html><style>b{}</style><script>x()</script><p>Hello</p></html>")
    assert text == "Hello"


@pytest.mark.parametrize(
    "url",
    ["http://127.0.0.1/x", "http://localhost/x", "ftp://example.com/x", "http:///x"],
)
def test_private_and_non_http_urls_are_refused(url: str) -> None:
    allowed, reason = _is_public_url(url)
    assert allowed is False
    assert reason


async def test_unsupported_and_oversized_files_are_rejected(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, max_upload_bytes=10)
    store = get_vector_store(settings)
    service = IngestionService(store, settings)

    with pytest.raises(IngestionError, match="Unsupported file type"):
        await service.load_file(UploadPayload(filename="a.exe", content=b"x"))
    with pytest.raises(IngestionError, match="MAX_UPLOAD_BYTES"):
        await service.load_file(UploadPayload(filename="a.txt", content=b"x" * 11))


async def test_text_and_html_uploads_are_parsed(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    service = IngestionService(get_vector_store(settings), settings)

    text_docs = await service.load_file(UploadPayload(filename="a.md", content=b"# Title"))
    assert text_docs[0].page_content == "# Title"

    html_docs = await service.load_file(
        UploadPayload(filename="a.html", content=b"<p>Hello <b>there</b></p>")
    )
    assert "Hello" in html_docs[0].page_content


async def test_empty_text_ingestion_is_rejected(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    service = IngestionService(get_vector_store(settings), settings)
    with pytest.raises(IngestionError):
        await service.ingest_text("hr", "  ", source="x", tenant_id=None, uploader="u")


async def test_ingesting_no_documents_is_a_noop(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    service = IngestionService(get_vector_store(settings), settings)
    result = await service.ingest_documents("hr", [], source="x", tenant_id=None, uploader="u")
    assert result.chunks_added == 0


async def test_chunking_stamps_every_chunk(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, chunk_size=80, chunk_overlap=10)
    store = get_vector_store(settings)
    await store.initialize()
    service = IngestionService(store, settings)

    long_text = " ".join(f"sentence number {index} about payroll policy." for index in range(30))
    result = await service.ingest_documents(
        "hr",
        [Document(page_content=long_text)],
        source="handbook.pdf",
        tenant_id="acme",
        uploader="alice",
        tags={"module": "payroll", "fiscal_year": 2026},
    )
    assert result.chunks_added > 1

    documents = await store.iter_documents("hr")
    metadata = documents[0].metadata
    for key in (
        "domain",
        "tenant_id",
        "source",
        "doc_id",
        "chunk_index",
        "content_hash",
        "ingested_at",
        "uploader",
        "module",
        "fiscal_year",
    ):
        assert key in metadata, key
    assert metadata["uploader"] == "alice"
    assert metadata["module"] == "payroll"


# ------------------------------------------------------------------ filters --
def test_filter_dialect_translations() -> None:
    assert to_chroma_where({"a": 1}) == {"a": {"$eq": 1}}
    assert to_chroma_where({"a": [1, 2]}) == {"a": {"$in": [1, 2]}}
    assert to_chroma_where({"a": 1, "b": 2}) == {"$and": [{"a": {"$eq": 1}}, {"b": {"$eq": 2}}]}
    assert to_chroma_where(None) is None

    assert to_pinecone_filter({"a": {"$gte": 3}}) == {"a": {"$gte": 3}}
    assert to_pinecone_filter(None) is None


def test_predicate_supports_every_operator() -> None:
    predicate = build_predicate(
        {"year": {"$gte": 2025, "$lt": 2027}, "module": ["ap", "gl"], "source": {"$ne": "x"}}
    )
    assert predicate is not None
    assert predicate({"year": 2026, "module": "ap", "source": "y"}) is True
    assert predicate({"year": 2024, "module": "ap", "source": "y"}) is False
    assert predicate({"year": 2026, "module": "hr", "source": "y"}) is False
    assert build_predicate(None) is None
    assert build_predicate({"a": {"$contains": "b"}})({"a": "abc"}) is True  # type: ignore[misc]


def test_unsupported_operators_are_reported() -> None:
    with pytest.raises(VectorStoreError, match="Unsupported filter operator"):
        build_predicate({"a": {"$regex": "x"}})
    with pytest.raises(VectorStoreError, match="do not support"):
        to_chroma_where({"a": {"$contains": "x"}})
    with pytest.raises(VectorStoreError, match="do not support"):
        to_pinecone_filter({"a": {"$contains": "x"}})


# --------------------------------------------------------------------- jobs --
async def test_job_lifecycle_success_and_failure() -> None:
    store = JobStore()
    job = await store.create(kind="ingest_files", domain="hr", submitted_by="alice")
    assert job.status is JobStatus.QUEUED

    async def ok() -> dict[str, Any]:
        return {"chunks_added": 3}

    await store.run(job.id, ok)
    finished = await store.get(job.id)
    assert finished.status is JobStatus.SUCCEEDED
    assert finished.result == {"chunks_added": 3}

    async def boom() -> dict[str, Any]:
        raise RuntimeError("nope")

    failing = await store.create(kind="ingest_urls", domain="it", submitted_by="bob")
    await store.run(failing.id, boom)
    assert (await store.get(failing.id)).status is JobStatus.FAILED
    assert "nope" in str((await store.get(failing.id)).error)

    assert [item.domain for item in await store.list(domain="hr")] == ["hr"]
    with pytest.raises(NotFoundError):
        await store.get("missing")


async def test_job_store_evicts_the_oldest_entry() -> None:
    store = JobStore(max_jobs=2)
    first = await store.create(kind="k", domain="hr", submitted_by=None)
    await store.create(kind="k", domain="hr", submitted_by=None)
    await store.create(kind="k", domain="hr", submitted_by=None)
    with pytest.raises(NotFoundError):
        await store.get(first.id)


# ------------------------------------------------------------------ logging --
def test_json_formatter_includes_context_and_extras() -> None:
    token = request_id_var.set("req-1")
    try:
        record = logging.LogRecord("t", logging.INFO, __file__, 1, "hello", None, None)
        record.domain = "hr"
        payload = json.loads(JSONFormatter().format(record))
    finally:
        request_id_var.reset(token)
    assert payload["message"] == "hello"
    assert payload["request_id"] == "req-1"
    assert payload["domain"] == "hr"
    assert payload["level"] == "INFO"


def test_plain_formatter_appends_the_request_id() -> None:
    token = request_id_var.set("req-2")
    try:
        record = logging.LogRecord("t", logging.INFO, __file__, 1, "hello", None, None)
        assert "request_id=req-2" in PlainFormatter().format(record)
    finally:
        request_id_var.reset(token)


def test_configure_logging_is_idempotent() -> None:
    configure_logging("DEBUG", json_logs=True)
    configure_logging("INFO", json_logs=False)
    assert len(logging.getLogger().handlers) == 1


# --------------------------------------------------------------- exceptions --
def test_error_payload_shape() -> None:
    error = AppError("boom", details={"a": 1}, status_code=418, code="teapot")
    assert error.to_payload("rid") == {
        "error": {"code": "teapot", "message": "boom", "details": {"a": 1}, "request_id": "rid"}
    }


# ------------------------------------------------------------------- health --
async def test_health_and_ready(client: httpx.AsyncClient) -> None:
    health = await client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    ready = await client.get("/ready")
    body = ready.json()
    assert body["status"] == "ready"
    assert body["store"] == "chroma"
    assert body["embedding_dimension"] == 128
    assert body["domains"] == ["hr", "finance", "engineering", "it", "company"]


async def test_ready_reports_degraded_when_the_store_is_down(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def broken() -> bool:
        raise RuntimeError("store is gone")

    monkeypatch.setattr(client.app.state.store, "health", broken)  # type: ignore[attr-defined]
    body = (await client.get("/ready")).json()
    assert body["status"] == "degraded"
    assert body["store_healthy"] is False


async def test_unhandled_errors_are_wrapped_in_the_error_envelope(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    app = create_app(settings)

    @app.get("/boom")
    async def boom() -> None:
        raise RuntimeError("kaboom")

    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://testserver") as client,
    ):
        response = await client.get("/boom")
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"


async def test_body_validation_errors_use_the_envelope(
    client: httpx.AsyncClient, auth: dict[str, str]
) -> None:
    response = await client.post("/agent/chat", headers=auth, json={})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


async def test_not_found_uses_the_envelope(client: httpx.AsyncClient) -> None:
    response = await client.get("/nope")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"
