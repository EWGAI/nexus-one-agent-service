"""JWT authentication and scope based authorization.

Design notes
------------
* Tokens are minted by the calling backend and signed with a **shared HMAC
  secret**; this service only verifies them (HS256 by default).
* ``alg: none`` - or any algorithm other than the configured one - is rejected
  by passing an explicit allow-list to :func:`jwt.decode` *and* by inspecting
  the unverified header first so we can return a precise error.
* The verified :class:`TokenClaims` are attached to ``request.state.claims`` so
  middleware (logging) and route handlers can use them.
* The tenant is **always** taken from the token, never from the request body.
"""

from __future__ import annotations

import datetime as dt
from typing import Annotated, Any, Literal

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import jwt
from pydantic import BaseModel, ConfigDict, Field

from agent_service.config import Settings, TenantMode
from agent_service.core.exceptions import AuthenticationError, AuthorizationError
from agent_service.core.logging import get_logger, subject_var, tenant_var

logger = get_logger(__name__)

Action = Literal["read", "write"]

ADMIN_SCOPE = "kb:admin"
AGENT_CHAT_SCOPE = "agent:chat"

bearer_scheme = HTTPBearer(
    scheme_name="BearerJWT",
    bearerFormat="JWT",
    description=(
        "JWT signed by the calling backend with the shared secret. "
        "Send as `Authorization: Bearer <token>`."
    ),
    auto_error=False,
)


class TokenClaims(BaseModel):
    """Validated token payload."""

    model_config = ConfigDict(frozen=True)

    sub: str = Field(description="Subject - the acting user or service account id.")
    scopes: tuple[str, ...] = Field(default=(), description="Granted scopes.")
    roles: tuple[str, ...] = Field(default=(), description="Granted roles.")
    tenant_id: str | None = Field(default=None, description="Tenant taken from the token.")
    issuer: str | None = None
    audience: str | None = None
    issued_at: dt.datetime | None = None
    expires_at: dt.datetime | None = None
    claims: dict[str, Any] = Field(default_factory=dict, description="All remaining claims.")

    # -- scope helpers ----------------------------------------------------
    def has_scope(self, scope: str) -> bool:
        """Exact scope match, with ``kb:admin`` acting as a super-scope for ``kb:*``."""
        if scope in self.scopes:
            return True
        return scope.startswith("kb:") and ADMIN_SCOPE in self.scopes

    def can_access_domain(self, domain: str, action: Action) -> bool:
        """Domain scoped check: ``kb:<domain>:<action>``, ``kb:*:<action>`` or ``kb:admin``."""
        candidates = (
            ADMIN_SCOPE,
            f"kb:{domain}:{action}",
            f"kb:*:{action}",
            "kb:*:*",
            f"kb:{action}",
        )
        return any(candidate in self.scopes for candidate in candidates)

    def readable_domains(self, domains: tuple[str, ...]) -> list[str]:
        """Subset of ``domains`` this token may read."""
        return [domain for domain in domains if self.can_access_domain(domain, "read")]


def _extract_scopes(payload: dict[str, Any]) -> tuple[str, ...]:
    collected: list[str] = []
    for key in ("scopes", "scope", "permissions"):
        value = payload.get(key)
        if isinstance(value, str):
            collected.extend(part for part in value.replace(",", " ").split() if part)
        elif isinstance(value, (list, tuple)):
            collected.extend(str(item) for item in value)
    # preserve order, drop duplicates
    return tuple(dict.fromkeys(collected))


def _extract_roles(payload: dict[str, Any]) -> tuple[str, ...]:
    roles: list[str] = []
    value = payload.get("roles")
    if isinstance(value, str):
        roles.extend(part for part in value.replace(",", " ").split() if part)
    elif isinstance(value, (list, tuple)):
        roles.extend(str(item) for item in value)
    realm = payload.get("realm_access")
    if isinstance(realm, dict) and isinstance(realm.get("roles"), list):
        roles.extend(str(item) for item in realm["roles"])
    return tuple(dict.fromkeys(roles))


def _as_datetime(value: Any) -> dt.datetime | None:
    if isinstance(value, (int, float)):
        return dt.datetime.fromtimestamp(float(value), tz=dt.UTC)
    return None


def decode_token(token: str, settings: Settings) -> TokenClaims:
    """Verify ``token`` and return typed claims. Raises :class:`AuthenticationError`."""
    try:
        header = jwt.get_unverified_header(token)
    except jwt.PyJWTError as exc:
        raise AuthenticationError("Malformed token header") from exc

    algorithm = str(header.get("alg", "")).upper()
    if algorithm in {"", "NONE"}:
        raise AuthenticationError("Unsigned tokens are not accepted")
    if algorithm != settings.jwt_algorithm.upper():
        raise AuthenticationError(
            "Token algorithm is not allowed",
            details={"expected": settings.jwt_algorithm, "received": header.get("alg")},
        )

    # PyJWT types this as a TypedDict with many optional members; a plain mapping
    # is what it actually accepts.
    options: Any = {
        "require": ["exp", "sub"],
        "verify_signature": True,
        "verify_exp": True,
        "verify_nbf": True,
        "verify_iat": True,
        "verify_aud": settings.jwt_audience is not None,
        "verify_iss": settings.jwt_issuer is not None,
    }
    try:
        payload: dict[str, Any] = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
            audience=settings.jwt_audience,
            issuer=settings.jwt_issuer,
            leeway=settings.jwt_leeway_seconds,
            options=options,
        )
    except jwt.ExpiredSignatureError as exc:
        raise AuthenticationError("Token has expired", code="token_expired") from exc
    except jwt.ImmatureSignatureError as exc:
        raise AuthenticationError(
            "Token is not valid yet (nbf)", code="token_not_yet_valid"
        ) from exc
    except jwt.InvalidAudienceError as exc:
        raise AuthenticationError("Token audience is invalid") from exc
    except jwt.InvalidIssuerError as exc:
        raise AuthenticationError("Token issuer is invalid") from exc
    except jwt.MissingRequiredClaimError as exc:
        raise AuthenticationError(f"Token is missing required claim: {exc.claim}") from exc
    except jwt.PyJWTError as exc:
        raise AuthenticationError("Token signature verification failed") from exc

    subject = payload.get("sub")
    if not isinstance(subject, str) or not subject:
        raise AuthenticationError("Token is missing a usable 'sub' claim")

    audience = payload.get("aud")
    return TokenClaims(
        sub=subject,
        scopes=_extract_scopes(payload),
        roles=_extract_roles(payload),
        tenant_id=payload.get("tenant_id") or payload.get("tid"),
        issuer=payload.get("iss"),
        audience=audience if isinstance(audience, str) else None,
        issued_at=_as_datetime(payload.get("iat")),
        expires_at=_as_datetime(payload.get("exp")),
        claims=payload,
    )


async def verify_jwt(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
) -> TokenClaims:
    """FastAPI dependency enforcing a valid bearer token on the whole router."""
    settings: Settings = request.app.state.settings

    if credentials is None or not credentials.credentials:
        raise AuthenticationError("Missing bearer token")
    if credentials.scheme.lower() != "bearer":
        raise AuthenticationError("Authorization scheme must be 'Bearer'")

    claims = decode_token(credentials.credentials, settings)

    if settings.tenant_mode is not TenantMode.OFF and not (
        claims.tenant_id or settings.default_tenant_id
    ):
        raise AuthorizationError(
            "Token is missing the 'tenant_id' claim required by this deployment",
            code="missing_tenant",
        )

    request.state.claims = claims
    subject_var.set(claims.sub)
    tenant_var.set(resolve_tenant(claims, settings))
    return claims


def resolve_tenant(claims: TokenClaims, settings: Settings) -> str | None:
    """Effective tenant for the request, or ``None`` when tenancy is disabled."""
    if settings.tenant_mode is TenantMode.OFF:
        return None
    return claims.tenant_id or settings.default_tenant_id


def require_scopes(*required: str) -> Any:
    """Dependency factory asserting the token carries **all** ``required`` scopes."""

    async def _dependency(request: Request) -> TokenClaims:
        claims: TokenClaims | None = getattr(request.state, "claims", None)
        if claims is None:  # pragma: no cover - router dependency guarantees this
            raise AuthenticationError("Missing bearer token")
        missing = [scope for scope in required if not claims.has_scope(scope)]
        if missing:
            raise AuthorizationError(
                "Insufficient scope",
                details={"required": list(required), "missing": missing},
            )
        return claims

    return _dependency
