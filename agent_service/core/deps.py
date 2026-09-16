"""FastAPI dependency providers.

Singletons (vector store, agent graph, job store) live on ``app.state`` and are
created once in the lifespan handler; these helpers expose them as typed
dependencies.
"""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import Depends, Path, Request

from agent_service.agent.graph import AgentRuntime
from agent_service.config import Settings
from agent_service.core.exceptions import (
    AuthenticationError,
    AuthorizationError,
    UnknownDomainError,
)
from agent_service.core.security import TokenClaims, resolve_tenant
from agent_service.kb.base import VectorStoreAdapter
from agent_service.kb.ingestion import IngestionService
from agent_service.kb.jobs import JobStore


def get_settings_dep(request: Request) -> Settings:
    """Settings bound to this application instance."""
    settings: Settings = request.app.state.settings
    return settings


def get_store(request: Request) -> VectorStoreAdapter:
    """The active vector store adapter."""
    store: VectorStoreAdapter = request.app.state.store
    return store


def get_agent(request: Request) -> AgentRuntime:
    """The compiled agent graph."""
    agent: AgentRuntime = request.app.state.agent
    return agent


def get_ingestion(request: Request) -> IngestionService:
    """The ingestion pipeline."""
    service: IngestionService = request.app.state.ingestion
    return service


def get_jobs(request: Request) -> JobStore:
    """The background job registry."""
    jobs: JobStore = request.app.state.jobs
    return jobs


def get_claims(request: Request) -> TokenClaims:
    """Verified claims attached by ``verify_jwt`` (a router level dependency)."""
    claims: TokenClaims | None = getattr(request.state, "claims", None)
    if claims is None:  # pragma: no cover - unreachable on protected routers
        raise AuthenticationError("Missing bearer token")
    return claims


def get_tenant(request: Request) -> str | None:
    """Effective tenant for this request (from the token, never the body)."""
    return resolve_tenant(get_claims(request), get_settings_dep(request))


SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
StoreDep = Annotated[VectorStoreAdapter, Depends(get_store)]
AgentDep = Annotated[AgentRuntime, Depends(get_agent)]
IngestionDep = Annotated[IngestionService, Depends(get_ingestion)]
JobsDep = Annotated[JobStore, Depends(get_jobs)]
ClaimsDep = Annotated[TokenClaims, Depends(get_claims)]
TenantDep = Annotated[str | None, Depends(get_tenant)]


def valid_domain(
    settings: SettingsDep,
    domain: Annotated[
        str,
        Path(
            description="Business domain, must be one of KB_DOMAINS.",
            examples=["hr", "finance"],
        ),
    ],
) -> str:
    """Reject unknown domains with 422 before anything else runs."""
    if domain not in settings.domains:
        raise UnknownDomainError(
            f"Unknown knowledge base domain {domain!r}",
            details={"known_domains": list(settings.domains)},
        )
    return domain


DomainDep = Annotated[str, Depends(valid_domain)]


def domain_access(action: Literal["read", "write"]) -> object:
    """Dependency factory enforcing ``kb:<domain>:<action>`` (or a wildcard/admin)."""

    def _dependency(domain: DomainDep, claims: ClaimsDep) -> str:
        if not claims.can_access_domain(domain, action):
            # Deliberately identical for "no such data" and "not allowed".
            raise AuthorizationError(
                f"Insufficient scope for domain {domain!r}",
                details={"required_scope": f"kb:{domain}:{action}"},
            )
        return domain

    return Depends(_dependency)


ReadableDomainDep = Annotated[str, domain_access("read")]
WritableDomainDep = Annotated[str, domain_access("write")]
