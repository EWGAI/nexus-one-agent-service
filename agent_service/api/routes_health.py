"""Liveness and readiness probes. These are the only unauthenticated routes."""

from __future__ import annotations

from fastapi import APIRouter

from agent_service.api.schemas import HealthResponse, ReadyResponse
from agent_service.core.deps import SettingsDep, StoreDep
from agent_service.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    description=(
        "Returns 200 as soon as the process is serving traffic. Unauthenticated so "
        "orchestrators can use it without credentials."
    ),
)
async def health(settings: SettingsDep) -> HealthResponse:
    """Report that the process is alive."""
    return HealthResponse(service=settings.app_name, version=settings.app_version)


@router.get(
    "/ready",
    response_model=ReadyResponse,
    summary="Readiness probe",
    description=(
        "Verifies that the configured vector store answers and that the embedding "
        "model dimension is known. Returns `status=degraded` when the store is "
        "unreachable so load balancers can drain the instance."
    ),
)
async def ready(settings: SettingsDep, store: StoreDep) -> ReadyResponse:
    """Check the backing store before declaring the instance ready."""
    try:
        healthy = await store.health()
    except Exception as exc:
        logger.warning("readiness check failed", extra={"error": str(exc)})
        healthy = False
    return ReadyResponse(
        status="ready" if healthy else "degraded",
        store=settings.vector_store.value,
        store_healthy=healthy,
        embedding_model=store.embeddings.model_name,
        embedding_dimension=await store.embeddings.ensure_dimension(),
        domains=list(settings.domains),
    )
