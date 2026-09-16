"""JWT verification and scope enforcement."""

from __future__ import annotations

from collections.abc import Callable
import datetime as dt
from pathlib import Path

import httpx
import jwt
import pytest

from agent_service.core.exceptions import AuthenticationError
from agent_service.core.security import TokenClaims, decode_token

from .conftest import SECRET, make_settings


def test_valid_token_is_decoded(tmp_path: Path, make_token: Callable[..., str]) -> None:
    claims = decode_token(make_token(sub="alice", scopes="kb:hr:read"), make_settings(tmp_path))
    assert claims.sub == "alice"
    assert claims.scopes == ("kb:hr:read",)


def test_space_delimited_scope_claim_is_supported(tmp_path: Path) -> None:
    token = jwt.encode(
        {
            "sub": "svc",
            "exp": int((dt.datetime.now(tz=dt.UTC) + dt.timedelta(minutes=5)).timestamp()),
            "scope": "kb:hr:read agent:chat",
        },
        SECRET,
        algorithm="HS256",
    )
    claims = decode_token(token, make_settings(tmp_path))
    assert claims.scopes == ("kb:hr:read", "agent:chat")


def test_roles_are_collected_from_both_shapes(tmp_path: Path) -> None:
    token = jwt.encode(
        {
            "sub": "svc",
            "exp": int((dt.datetime.now(tz=dt.UTC) + dt.timedelta(minutes=5)).timestamp()),
            "roles": ["hr-admin"],
            "realm_access": {"roles": ["kb-reader"]},
        },
        SECRET,
        algorithm="HS256",
    )
    assert decode_token(token, make_settings(tmp_path)).roles == ("hr-admin", "kb-reader")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"expires_in": -60}, "expired"),
        ({"secret": "another-secret-that-is-long-enough-32ch"}, "signature"),
        ({"omit_sub": True}, "missing required claim"),
        ({"not_before": 3600}, "not valid yet"),
    ],
)
def test_invalid_tokens_are_rejected(
    tmp_path: Path, make_token: Callable[..., str], kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(AuthenticationError) as excinfo:
        decode_token(make_token(**kwargs), make_settings(tmp_path))
    assert message.lower() in str(excinfo.value).lower()


def test_alg_none_is_rejected(tmp_path: Path) -> None:
    unsigned = jwt.encode({"sub": "mallory", "exp": 9999999999}, key="", algorithm="none")
    with pytest.raises(AuthenticationError, match="Unsigned"):
        decode_token(unsigned, make_settings(tmp_path))


def test_other_algorithms_are_rejected(tmp_path: Path) -> None:
    token = jwt.encode({"sub": "mallory", "exp": 9999999999}, SECRET, algorithm="HS512")
    with pytest.raises(AuthenticationError, match="algorithm is not allowed"):
        decode_token(token, make_settings(tmp_path, jwt_algorithm="HS256"))


def test_malformed_token_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(AuthenticationError, match="Malformed"):
        decode_token("not-a-jwt", make_settings(tmp_path))


def test_issuer_and_audience_are_enforced_when_configured(
    tmp_path: Path, make_token: Callable[..., str]
) -> None:
    settings = make_settings(tmp_path, jwt_issuer="nexusone", jwt_audience="agent-service")
    good = make_token(issuer="nexusone", audience="agent-service")
    assert decode_token(good, settings).issuer == "nexusone"

    with pytest.raises(AuthenticationError, match="issuer"):
        decode_token(make_token(issuer="evil", audience="agent-service"), settings)
    with pytest.raises(AuthenticationError, match="audience"):
        decode_token(make_token(issuer="nexusone", audience="other"), settings)


# ------------------------------------------------------------------ scopes --
def _claims(*scopes: str) -> TokenClaims:
    return TokenClaims(sub="u", scopes=tuple(scopes))


@pytest.mark.parametrize(
    ("scopes", "domain", "action", "allowed"),
    [
        (("kb:hr:read",), "hr", "read", True),
        (("kb:hr:read",), "hr", "write", False),
        (("kb:hr:read",), "finance", "read", False),
        (("kb:*:read",), "finance", "read", True),
        (("kb:*:*",), "finance", "write", True),
        (("kb:admin",), "engineering", "write", True),
        ((), "hr", "read", False),
    ],
)
def test_domain_scope_matrix(
    scopes: tuple[str, ...], domain: str, action: str, allowed: bool
) -> None:
    assert _claims(*scopes).can_access_domain(domain, action) is allowed  # type: ignore[arg-type]


def test_readable_domains_filters_the_configured_set() -> None:
    claims = _claims("kb:hr:read", "kb:company:read")
    assert claims.readable_domains(("hr", "finance", "company")) == ["hr", "company"]


def test_admin_is_a_super_scope_for_kb_only() -> None:
    claims = _claims("kb:admin")
    assert claims.has_scope("kb:anything") is True
    assert claims.has_scope("agent:chat") is False


# ------------------------------------------------------------- end to end --
async def test_protected_routes_require_a_token(client: httpx.AsyncClient) -> None:
    response = await client.get("/kb/hr/search", params={"q": "x"})
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"
    assert response.headers["www-authenticate"] == "Bearer"


async def test_wrong_scheme_is_rejected(
    client: httpx.AsyncClient, make_token: Callable[..., str]
) -> None:
    response = await client.get(
        "/kb/hr/search", params={"q": "x"}, headers={"Authorization": f"Basic {make_token()}"}
    )
    assert response.status_code == 401


async def test_insufficient_domain_scope_returns_403(
    client: httpx.AsyncClient, make_token: Callable[..., str]
) -> None:
    token = make_token(scopes="kb:finance:read agent:chat")
    response = await client.get(
        "/kb/hr/search", params={"q": "x"}, headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 403
    body = response.json()["error"]
    assert body["code"] == "forbidden"
    assert body["details"]["required_scope"] == "kb:hr:read"


async def test_unknown_domain_returns_422(client: httpx.AsyncClient, auth: dict[str, str]) -> None:
    response = await client.get("/kb/marketing/search", params={"q": "x"}, headers=auth)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "unknown_domain"


async def test_admin_routes_require_admin_scope(
    client: httpx.AsyncClient, make_token: Callable[..., str]
) -> None:
    token = make_token(scopes="kb:*:read kb:*:write agent:chat")
    response = await client.get("/kb/domains", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 403
    assert "kb:admin" in response.json()["error"]["details"]["required"]


async def test_tenant_must_come_from_the_token(
    tmp_path: Path, client_factory: Callable[..., object], make_token: Callable[..., str]
) -> None:
    settings = make_settings(tmp_path, tenant_mode="metadata")
    agen = client_factory(settings)
    client = await agen.__anext__()  # type: ignore[attr-defined]
    try:
        no_tenant = {"Authorization": f"Bearer {make_token()}"}
        response = await client.get("/kb/hr/search", params={"q": "x"}, headers=no_tenant)
        assert response.status_code == 403
        assert response.json()["error"]["code"] == "missing_tenant"

        with_tenant = {"Authorization": f"Bearer {make_token(tenant_id='acme')}"}
        response = await client.get("/kb/hr/search", params={"q": "x"}, headers=with_tenant)
        assert response.status_code == 200
    finally:
        with pytest.raises(StopAsyncIteration):
            await agen.__anext__()  # type: ignore[attr-defined]
