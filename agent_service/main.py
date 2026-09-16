"""Application factory, lifespan wiring and OpenAPI customisation."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from agent_service.agent.graph import build_agent
from agent_service.api import routes_agent, routes_health, routes_kb
from agent_service.config import Settings, get_settings
from agent_service.core.exceptions import AppError
from agent_service.core.logging import configure_logging, get_logger, request_id_var
from agent_service.core.middleware import RequestContextMiddleware
from agent_service.kb.embeddings import build_embeddings
from agent_service.kb.factory import get_vector_store
from agent_service.kb.ingestion import IngestionService
from agent_service.kb.jobs import JobStore

logger = get_logger(__name__)

DESCRIPTION = """
Retrieval-augmented agent service for the NexusOne ERP.

**Domain separated knowledge bases.** Each business domain (`hr`, `finance`,
`engineering`, `it`, `company`, ...) is an isolated index. A query never crosses
domains unless the caller asks for it, and access is granted per domain through
JWT scopes of the form `kb:<domain>:read` / `kb:<domain>:write`.

**Pluggable everything.** The vector store (`chroma` | `faiss` | `pinecone`),
the chat model (`openai` | `anthropic`) and the embedding model
(`openai` | `huggingface`) are selected purely through environment variables.

**Authentication.** Every endpoint except `/health`, `/ready` and the docs
requires `Authorization: Bearer <jwt>`; tokens are signed by the calling backend
with a shared HMAC secret. Click **Authorize** to try the endpoints below.
""".strip()

TAGS_METADATA: list[dict[str, Any]] = [
    {
        "name": "kb",
        "description": (
            "Ingest, search, inspect and delete knowledge base content. All routes are "
            "domain scoped and require `kb:<domain>:read` or `kb:<domain>:write`; "
            "administrative routes require `kb:admin`."
        ),
    },
    {
        "name": "agent",
        "description": (
            "Conversational retrieval agent (LangGraph). Supports multi-domain routing, "
            "reciprocal-rank-fusion retrieval, tool calling and SSE streaming. "
            "Requires `agent:chat`."
        ),
    },
    {
        "name": "health",
        "description": "Unauthenticated liveness and readiness probes for orchestrators.",
    },
]


def _error_response(
    status_code: int, payload: dict[str, Any], headers: dict[str, str] | None = None
) -> JSONResponse:
    return JSONResponse(status_code=status_code, content=payload, headers=headers)


def register_exception_handlers(app: FastAPI) -> None:
    """Render every failure with the same error envelope."""

    @app.exception_handler(AppError)
    async def _app_error(request: Request, exc: AppError) -> JSONResponse:
        headers = {"WWW-Authenticate": "Bearer"} if exc.status_code == 401 else None
        if exc.status_code >= 500:
            logger.error("application error", extra={"code": exc.code, "message": exc.message})
        return _error_response(exc.status_code, exc.to_payload(request_id_var.get()), headers)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        return _error_response(
            422,
            {
                "error": {
                    "code": "validation_error",
                    "message": "Request validation failed",
                    "details": {"errors": exc.errors()},
                    "request_id": request_id_var.get(),
                }
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = {401: "unauthorized", 403: "forbidden", 404: "not_found"}.get(
            exc.status_code, "http_error"
        )
        return _error_response(
            exc.status_code,
            {
                "error": {
                    "code": code,
                    "message": str(exc.detail),
                    "details": {},
                    "request_id": request_id_var.get(),
                }
            },
            dict(exc.headers) if exc.headers else None,
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error")
        return _error_response(
            500,
            {
                "error": {
                    "code": "internal_error",
                    "message": "An unexpected error occurred",
                    "details": {},
                    "request_id": request_id_var.get(),
                }
            },
        )


def custom_openapi(app: FastAPI, settings: Settings) -> dict[str, Any]:
    """Build the OpenAPI document, guaranteeing the Bearer scheme is registered."""
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title="NexusOne Agent Service",
        version=settings.app_version,
        description=DESCRIPTION,
        routes=app.routes,
        tags=TAGS_METADATA,
        contact={"name": "Platform Engineering", "email": "platform@example.com"},
        license_info={"name": "MIT", "identifier": "MIT"},
        servers=[{"url": "/", "description": "This deployment"}],
    )
    components = schema.setdefault("components", {})
    security_schemes = components.setdefault("securitySchemes", {})
    security_schemes["BearerJWT"] = {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "JWT",
        "description": (
            "JWT signed with the shared secret (`JWT_SECRET_KEY`, HS256 by default). "
            "Mint a development token with `python scripts/mint_token.py`."
        ),
    }
    app.openapi_schema = schema
    return schema


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create the store, ingestion pipeline and agent graph once per process."""
    settings: Settings = app.state.settings
    embeddings = build_embeddings(settings)
    store = get_vector_store(settings, embeddings)
    await store.initialize()

    app.state.embeddings = embeddings
    app.state.store = store
    app.state.ingestion = IngestionService(store, settings)
    app.state.jobs = JobStore()
    app.state.agent = build_agent(store, settings)

    logger.info(
        "service started",
        extra={
            "environment": settings.environment,
            "vector_store": settings.vector_store.value,
            "domains": list(settings.domains),
            "docs_enabled": settings.docs_enabled,
        },
    )
    try:
        yield
    finally:
        await store.persist()
        await store.aclose()
        logger.info("service stopped")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build a configured FastAPI application."""
    resolved = settings or get_settings()
    configure_logging(resolved.log_level, json_logs=resolved.log_json)

    app = FastAPI(
        title="NexusOne Agent Service",
        version=resolved.app_version,
        description=DESCRIPTION,
        openapi_tags=TAGS_METADATA,
        docs_url="/docs" if resolved.docs_enabled else None,
        redoc_url="/redoc" if resolved.docs_enabled else None,
        openapi_url="/openapi.json" if resolved.docs_enabled else None,
        lifespan=lifespan,
        swagger_ui_parameters={"persistAuthorization": True},
    )
    app.state.settings = resolved

    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=resolved.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )

    register_exception_handlers(app)
    app.include_router(routes_health.router)
    app.include_router(routes_kb.router)
    app.include_router(routes_agent.router)

    app.openapi = lambda: custom_openapi(app, resolved)  # type: ignore[method-assign]
    return app


app = None  # populated by `run()` / uvicorn factory below


def get_app() -> FastAPI:
    """ASGI factory: ``uvicorn agent_service.main:get_app --factory``."""
    return create_app()


def run() -> None:  # pragma: no cover - console entry point
    """Console script entry point (``agent-service``)."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "agent_service.main:get_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        log_config=None,
    )
