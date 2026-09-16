"""End-to-end knowledge base API tests (Chroma backed)."""

from __future__ import annotations

from collections.abc import Callable
import io
import json
from pathlib import Path

import httpx
import pytest

from .conftest import make_settings

HR_TEXT = "Employees accrue 1.75 days of paid leave per completed month of service."
FIN_TEXT = "Accounts payable invoices above 50000 require two approvals before payment."


async def ingest_text(
    client: httpx.AsyncClient,
    auth: dict[str, str],
    domain: str,
    text: str,
    *,
    source: str = "policy",
    tags: dict[str, object] | None = None,
) -> dict[str, object]:
    response = await client.post(
        f"/kb/{domain}/ingest/text",
        headers=auth,
        json={"text": text, "source": source, "tags": tags or {}},
    )
    assert response.status_code == 202, response.text
    return response.json()


async def test_ingest_search_delete_round_trip(
    client: httpx.AsyncClient, auth: dict[str, str]
) -> None:
    body = await ingest_text(
        client, auth, "hr", HR_TEXT, source="hr-leave-policy", tags={"module": "leave"}
    )
    assert body["mode"] == "sync"
    assert body["result"]["chunks_added"] == 1
    assert body["result"]["duplicates_skipped"] == 0

    search = await client.get(
        "/kb/hr/search", headers=auth, params={"q": "paid leave accrual", "k": 3}
    )
    assert search.status_code == 200
    hits = search.json()["hits"]
    assert hits[0]["content"] == HR_TEXT
    metadata = hits[0]["metadata"]
    assert metadata["domain"] == "hr"
    assert metadata["source"] == "hr-leave-policy"
    assert metadata["module"] == "leave"
    assert metadata["uploader"] == "tester@example.com"
    assert metadata["content_hash"] == hits[0]["id"]
    assert "ingested_at" in metadata and "doc_id" in metadata and "chunk_index" in metadata

    deleted = await client.request(
        "DELETE",
        "/kb/hr/documents",
        headers=auth,
        json={"filter": {"source": "hr-leave-policy"}},
    )
    assert deleted.status_code == 200
    assert deleted.json() == {"domain": "hr", "deleted": 1}

    after = await client.get("/kb/hr/search", headers=auth, params={"q": "paid leave accrual"})
    assert after.json()["hits"] == []


async def test_duplicate_ingestion_is_skipped(
    client: httpx.AsyncClient, auth: dict[str, str]
) -> None:
    await ingest_text(client, auth, "hr", HR_TEXT)
    second = await ingest_text(client, auth, "hr", HR_TEXT)
    assert second["result"]["chunks_added"] == 0
    assert second["result"]["duplicates_skipped"] == 1


async def test_search_never_crosses_domains(
    client: httpx.AsyncClient, auth: dict[str, str]
) -> None:
    await ingest_text(client, auth, "hr", HR_TEXT, source="hr-policy")
    await ingest_text(client, auth, "finance", FIN_TEXT, source="ap-policy")

    finance = await client.get(
        "/kb/finance/search", headers=auth, params={"q": "paid leave accrual", "k": 10}
    )
    contents = [hit["content"] for hit in finance.json()["hits"]]
    assert HR_TEXT not in contents
    assert all(hit["domain"] == "finance" for hit in finance.json()["hits"])


async def test_search_supports_metadata_filters(
    client: httpx.AsyncClient, auth: dict[str, str]
) -> None:
    await ingest_text(client, auth, "finance", FIN_TEXT, source="ap", tags={"module": "ap"})
    await ingest_text(
        client,
        auth,
        "finance",
        "The general ledger closes on the fifth working day.",
        source="gl",
        tags={"module": "gl"},
    )
    response = await client.get(
        "/kb/finance/search",
        headers=auth,
        params={"q": "ledger close", "k": 10, "filter": json.dumps({"module": "gl"})},
    )
    modules = {hit["metadata"]["module"] for hit in response.json()["hits"]}
    assert modules == {"gl"}


async def test_invalid_filter_json_is_rejected(
    client: httpx.AsyncClient, auth: dict[str, str]
) -> None:
    response = await client.get(
        "/kb/hr/search", headers=auth, params={"q": "x", "filter": "{not json"}
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "ingestion_error"


async def test_delete_requires_criteria(client: httpx.AsyncClient, auth: dict[str, str]) -> None:
    response = await client.request("DELETE", "/kb/hr/documents", headers=auth, json={})
    assert response.status_code == 400


async def test_file_upload_and_background_job(
    client: httpx.AsyncClient, auth: dict[str, str]
) -> None:
    files = [
        ("files", ("policy.txt", io.BytesIO(HR_TEXT.encode()), "text/plain")),
        (
            "files",
            (
                "handbook.md",
                io.BytesIO(b"# Handbook\n\nDress code is smart casual."),
                "text/markdown",
            ),
        ),
        (
            "files",
            ("notes.txt", io.BytesIO(b"Quarterly town hall happens in March."), "text/plain"),
        ),
    ]
    response = await client.post(
        "/kb/hr/ingest/files",
        headers=auth,
        files=files,
        data={"tags": json.dumps({"module": "policies"})},
    )
    assert response.status_code == 202
    body = response.json()
    assert body["mode"] == "async"
    job_id = body["job_id"]

    job = await client.get(f"/kb/jobs/{job_id}", headers=auth)
    assert job.status_code == 200
    assert job.json()["job"]["status"] in {"queued", "running", "succeeded"}


async def test_small_upload_runs_inline_and_reports_bad_files(
    client: httpx.AsyncClient, auth: dict[str, str]
) -> None:
    files = [
        ("files", ("policy.txt", io.BytesIO(HR_TEXT.encode()), "text/plain")),
        ("files", ("virus.exe", io.BytesIO(b"MZ"), "application/octet-stream")),
    ]
    response = await client.post("/kb/hr/ingest/files", headers=auth, files=files)
    body = response.json()
    assert body["mode"] == "sync"
    assert body["result"]["chunks_added"] == 1
    assert any("virus.exe" in error for error in body["result"]["errors"])


async def test_unknown_job_is_404(client: httpx.AsyncClient, auth: dict[str, str]) -> None:
    response = await client.get("/kb/jobs/does-not-exist", headers=auth)
    assert response.status_code == 404


async def test_jobs_are_not_readable_by_other_subjects(
    client: httpx.AsyncClient, auth: dict[str, str], make_token: Callable[..., str]
) -> None:
    files = [
        ("files", (f"f{index}.txt", io.BytesIO(f"note number {index}".encode()), "text/plain"))
        for index in range(3)
    ]
    created = await client.post("/kb/hr/ingest/files", headers=auth, files=files)
    job_id = created.json()["job_id"]

    other = {"Authorization": f"Bearer {make_token(sub='mallory', scopes='kb:*:read')}"}
    response = await client.get(f"/kb/jobs/{job_id}", headers=other)
    assert response.status_code == 404


async def test_url_ingestion_refuses_private_addresses(
    client: httpx.AsyncClient, auth: dict[str, str]
) -> None:
    response = await client.post(
        "/kb/it/ingest/urls", headers=auth, json={"urls": ["http://127.0.0.1:9/secret"]}
    )
    assert response.status_code == 202
    assert response.json()["result"]["chunks_added"] == 0
    assert "Refusing to fetch" in response.json()["result"]["errors"][0]


async def test_domains_stats_and_reindex(client: httpx.AsyncClient, auth: dict[str, str]) -> None:
    await ingest_text(client, auth, "hr", HR_TEXT)

    domains = await client.get("/kb/domains", headers=auth)
    assert domains.status_code == 200
    body = domains.json()
    assert body["store"] == "chroma"
    assert {item["domain"] for item in body["domains"]} == {
        "hr",
        "finance",
        "engineering",
        "it",
        "company",
    }
    assert next(item for item in body["domains"] if item["domain"] == "hr")["exists"] is True

    stats = await client.get("/kb/hr/stats", headers=auth)
    assert stats.json()["vector_count"] == 1
    assert stats.json()["embedding_dimension"] == 128

    reindexed = await client.post("/kb/hr/reindex", headers=auth)
    assert reindexed.status_code == 200
    assert reindexed.json()["documents_reindexed"] == 1

    still_there = await client.get("/kb/hr/search", headers=auth, params={"q": "leave"})
    assert still_there.json()["hits"]


async def test_drop_domain(client: httpx.AsyncClient, auth: dict[str, str]) -> None:
    await ingest_text(client, auth, "hr", HR_TEXT)
    dropped = await client.delete("/kb/hr", headers=auth)
    assert dropped.status_code == 200
    assert dropped.json()["deleted"] == 1
    assert (await client.get("/kb/hr/stats", headers=auth)).json()["vector_count"] == 0


async def test_empty_text_is_rejected(client: httpx.AsyncClient, auth: dict[str, str]) -> None:
    response = await client.post("/kb/hr/ingest/text", headers=auth, json={"text": "   "})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "ingestion_error"


async def test_request_id_is_echoed(client: httpx.AsyncClient, auth: dict[str, str]) -> None:
    response = await client.get(
        "/kb/hr/search", headers={**auth, "X-Request-ID": "abc123"}, params={"q": "x"}
    )
    assert response.headers["X-Request-ID"] == "abc123"


@pytest.mark.parametrize("store", ["chroma", "faiss"])
async def test_the_same_api_flow_works_on_every_store(
    tmp_path: Path, client_factory: Callable[..., object], auth: dict[str, str], store: str
) -> None:
    settings = make_settings(tmp_path, vector_store=store)
    agen = client_factory(settings)
    client = await agen.__anext__()  # type: ignore[attr-defined]
    try:
        await ingest_text(client, auth, "hr", HR_TEXT)
        await ingest_text(client, auth, "finance", FIN_TEXT)

        hr_hits = (
            await client.get("/kb/hr/search", headers=auth, params={"q": "leave accrual"})
        ).json()["hits"]
        assert hr_hits[0]["content"] == HR_TEXT

        finance_hits = (
            await client.get("/kb/finance/search", headers=auth, params={"q": "leave accrual"})
        ).json()["hits"]
        assert all(hit["content"] != HR_TEXT for hit in finance_hits)
    finally:
        with pytest.raises(StopAsyncIteration):
            await agen.__anext__()  # type: ignore[attr-defined]
