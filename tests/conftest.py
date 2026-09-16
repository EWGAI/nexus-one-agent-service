"""Shared fixtures.

Everything runs offline: ``LLM_PROVIDER=fake`` and ``EMBEDDING_PROVIDER=fake``
give deterministic, credential-free behaviour, and each test gets its own
temporary store directory.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator
import datetime as dt
from pathlib import Path
from typing import Any

import httpx
import jwt
import pytest

from agent_service.config import Settings, build_settings
from agent_service.kb.base import VectorStoreAdapter
from agent_service.kb.factory import get_vector_store
from agent_service.main import create_app

SECRET = "test-secret-key-that-is-long-enough-32ch"
ALL_SCOPES = "kb:*:read kb:*:write kb:admin agent:chat"


def make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    """Offline settings pointed at a throwaway directory."""
    defaults: dict[str, Any] = {
        "_env_file": None,
        "jwt_secret_key": SECRET,
        "environment": "local",
        "llm_provider": "fake",
        "embedding_provider": "fake",
        "embedding_dimension": 128,
        "vector_store": "chroma",
        "chroma_persist_dir": tmp_path / "chroma",
        "faiss_index_path": tmp_path / "faiss",
        "kb_domains": "hr,finance,engineering,it,company",
        "log_json": False,
        "chunk_size": 400,
        "chunk_overlap": 40,
        "background_ingest_threshold": 2,
    }
    defaults.update(overrides)
    return build_settings(**defaults)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Default (Chroma backed) settings."""
    return make_settings(tmp_path)


@pytest.fixture
def make_token() -> Callable[..., str]:
    """Factory producing signed tokens for the test secret."""

    def _make(
        *,
        sub: str = "tester@example.com",
        scopes: str = ALL_SCOPES,
        secret: str = SECRET,
        algorithm: str = "HS256",
        expires_in: int = 900,
        not_before: int = 0,
        tenant_id: str | None = None,
        issuer: str | None = None,
        audience: str | None = None,
        omit_sub: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> str:
        now = dt.datetime.now(tz=dt.UTC)
        payload: dict[str, Any] = {
            "iat": int(now.timestamp()),
            "nbf": int((now + dt.timedelta(seconds=not_before)).timestamp()),
            "exp": int((now + dt.timedelta(seconds=expires_in)).timestamp()),
            "scopes": scopes.split() if scopes else [],
        }
        if not omit_sub:
            payload["sub"] = sub
        if tenant_id:
            payload["tenant_id"] = tenant_id
        if issuer:
            payload["iss"] = issuer
        if audience:
            payload["aud"] = audience
        payload.update(extra or {})
        return jwt.encode(payload, secret, algorithm=algorithm)

    return _make


@pytest.fixture
def auth(make_token: Callable[..., str]) -> dict[str, str]:
    """Authorization header granting every scope."""
    return {"Authorization": f"Bearer {make_token()}"}


async def _client_for(settings: Settings) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            client.app = app  # type: ignore[attr-defined]
            yield client


@pytest.fixture
async def client(settings: Settings) -> AsyncIterator[httpx.AsyncClient]:
    """HTTP client bound to an app whose lifespan has actually run."""
    async for item in _client_for(settings):
        yield item


@pytest.fixture
def client_factory() -> Callable[[Settings], Any]:
    """Build a client for custom settings inside a test."""
    return _client_for


@pytest.fixture(params=["chroma", "faiss"])
async def adapter(
    request: pytest.FixtureRequest, tmp_path: Path
) -> AsyncIterator[VectorStoreAdapter]:
    """Every local store, so the contract tests run against all of them."""
    store = get_vector_store(make_settings(tmp_path, vector_store=request.param))
    await store.initialize()
    yield store
    await store.aclose()


@pytest.fixture(autouse=True)
def _quiet_logging() -> Iterator[None]:
    import logging

    logging.getLogger("agent_service.access").setLevel(logging.WARNING)
    yield
