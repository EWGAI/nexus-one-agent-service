"""OpenAPI document, Swagger security wiring and docs gating."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from agent_service.main import create_app

from .conftest import make_settings

PROTECTED_PREFIXES = ("/kb", "/agent")


@pytest.fixture
def schema(settings: object) -> dict[str, Any]:
    return create_app(settings).openapi()  # type: ignore[arg-type]


def test_metadata_is_populated(schema: dict[str, Any]) -> None:
    info = schema["info"]
    assert info["title"] == "NexusOne Agent Service"
    assert info["version"].count(".") >= 1
    assert "domain" in info["description"].lower()
    assert info["contact"]["email"]
    assert info["license"]["name"] == "MIT"


def test_tags_are_described(schema: dict[str, Any]) -> None:
    tags = {tag["name"]: tag["description"] for tag in schema["tags"]}
    assert set(tags) == {"kb", "agent", "health"}
    assert all(len(description) > 30 for description in tags.values())


def test_bearer_security_scheme_is_registered(schema: dict[str, Any]) -> None:
    scheme = schema["components"]["securitySchemes"]["BearerJWT"]
    assert scheme == {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "JWT",
        "description": scheme["description"],
    }
    assert "JWT_SECRET_KEY" in scheme["description"]


def test_every_protected_operation_requires_the_bearer_scheme(schema: dict[str, Any]) -> None:
    for path, operations in schema["paths"].items():
        for method, operation in operations.items():
            if method not in {"get", "post", "put", "patch", "delete"}:
                continue
            requires = any(path.startswith(prefix) for prefix in PROTECTED_PREFIXES)
            has_security = bool(operation.get("security"))
            assert has_security is requires, f"{method.upper()} {path} security={has_security}"


def test_health_routes_are_public(schema: dict[str, Any]) -> None:
    assert "security" not in schema["paths"]["/health"]["get"]
    assert "security" not in schema["paths"]["/ready"]["get"]


def test_every_operation_is_documented(schema: dict[str, Any]) -> None:
    for path, operations in schema["paths"].items():
        for method, operation in operations.items():
            if method not in {"get", "post", "put", "patch", "delete"}:
                continue
            where = f"{method.upper()} {path}"
            assert operation.get("summary"), f"{where} has no summary"
            assert len(operation.get("description", "")) > 20, f"{where} has no description"
            assert "200" in operation["responses"] or "202" in operation["responses"], where


def test_protected_operations_document_error_responses(schema: dict[str, Any]) -> None:
    for path, operations in schema["paths"].items():
        if not path.startswith(PROTECTED_PREFIXES):
            continue
        for method, operation in operations.items():
            if method not in {"get", "post", "put", "patch", "delete"}:
                continue
            responses = operation["responses"]
            for code in ("401", "403", "422"):
                assert code in responses, f"{method.upper()} {path} does not document {code}"


def test_request_and_response_examples_are_present(schema: dict[str, Any]) -> None:
    schemas = schema["components"]["schemas"]
    assert schemas["ChatRequest"]["examples"]
    assert schemas["ChatResponse"]["examples"]
    assert schemas["IngestTextRequest"]["examples"]
    assert schemas["SearchResponse"]["examples"]
    assert schemas["ErrorResponse"]["examples"]


async def test_docs_are_served_when_enabled(client: httpx.AsyncClient) -> None:
    assert (await client.get("/docs")).status_code == 200
    assert (await client.get("/redoc")).status_code == 200
    assert (await client.get("/openapi.json")).status_code == 200


async def test_docs_are_hidden_when_disabled(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, environment="production", enable_docs=False)
    app = create_app(settings)
    transport = httpx.ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://testserver") as client,
    ):
        assert (await client.get("/docs")).status_code == 404
        assert (await client.get("/redoc")).status_code == 404
        assert (await client.get("/openapi.json")).status_code == 404
        assert (await client.get("/health")).status_code == 200


def test_committed_openapi_json_matches_the_code() -> None:
    """Mirrors the `openapi-drift` CI job so drift is caught locally too."""
    import subprocess
    import sys

    root = Path(__file__).resolve().parent.parent
    if not (root / "openapi.json").is_file():
        pytest.skip("openapi.json has not been generated yet")
    result = subprocess.run(  # noqa: S603 - fixed argv
        [sys.executable, str(root / "scripts" / "export_openapi.py"), "--check"],
        capture_output=True,
        text=True,
        cwd=root,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
