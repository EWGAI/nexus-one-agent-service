"""Agent endpoints: synchronous chat and SSE token streaming."""

from __future__ import annotations

from collections.abc import AsyncIterator
import json
from typing import Annotated, Any

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from agent_service.api.routes_kb import PROTECTED_RESPONSES
from agent_service.api.schemas import ChatRequest, ChatResponse, ErrorResponse
from agent_service.config import Settings
from agent_service.core.deps import AgentDep, ClaimsDep, SettingsDep, TenantDep
from agent_service.core.exceptions import AppError, AuthorizationError, UnknownDomainError
from agent_service.core.logging import get_logger, request_id_var
from agent_service.core.security import AGENT_CHAT_SCOPE, TokenClaims, require_scopes, verify_jwt

logger = get_logger(__name__)

router = APIRouter(
    prefix="/agent",
    tags=["agent"],
    dependencies=[Depends(verify_jwt), Depends(require_scopes(AGENT_CHAT_SCOPE))],
    responses=PROTECTED_RESPONSES,
)

ChatScopeDep = Annotated[object, Depends(require_scopes(AGENT_CHAT_SCOPE))]

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _resolve_domains(
    payload: ChatRequest, claims: TokenClaims, settings: Settings
) -> tuple[list[str], list[str]]:
    """Return ``(allowed, requested)`` domains, rejecting unknown or unreadable ones."""
    allowed = claims.readable_domains(settings.domains)
    if not allowed:
        raise AuthorizationError(
            "Token cannot read any knowledge base domain",
            details={"required_scope_example": f"kb:{settings.domains[0]}:read"},
        )
    requested = payload.domains or []
    unknown = [domain for domain in requested if domain not in settings.domains]
    if unknown:
        raise UnknownDomainError(
            "Unknown knowledge base domain(s)",
            details={"unknown": unknown, "known_domains": list(settings.domains)},
        )
    forbidden = [domain for domain in requested if domain not in allowed]
    if forbidden:
        raise AuthorizationError(
            "Insufficient scope for the requested domain(s)",
            details={"required_scopes": [f"kb:{domain}:read" for domain in forbidden]},
        )
    return allowed, requested


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Ask the agent a question",
    description=(
        "Runs the LangGraph agent: route to the relevant domain(s), retrieve, then answer "
        "with tool calls as needed.\n\n"
        "* `domains` is optional - when omitted the router picks from the domains the token "
        "may read, and the `company` domain stays eligible as a general fallback.\n"
        "* `thread_id` continues an existing conversation (state is kept by the checkpointer).\n"
        "* Requires scope `agent:chat` plus `kb:<domain>:read` for each domain used."
    ),
)
async def chat(
    payload: ChatRequest,
    agent: AgentDep,
    claims: ClaimsDep,
    settings: SettingsDep,
    tenant: TenantDep,
) -> ChatResponse:
    """Execute one agent turn."""
    allowed, requested = _resolve_domains(payload, claims, settings)
    outcome = await agent.chat(
        message=payload.message,
        allowed_domains=allowed,
        thread_id=payload.thread_id,
        requested_domains=requested,
        tenant_id=tenant,
        filters=payload.filters,
    )
    logger.info(
        "agent turn completed",
        extra={
            "thread_id": outcome.thread_id,
            "domains_searched": outcome.domains_searched,
            "sources": len(outcome.sources),
            "tool_calls": len(outcome.tool_calls),
        },
    )
    return ChatResponse(**outcome.model_dump())


@router.post(
    "/chat/stream",
    summary="Ask the agent a question (SSE stream)",
    description=(
        "Server-Sent Events stream of the same run as `POST /agent/chat`.\n\n"
        "Event payloads are JSON objects with a `type` field:\n"
        "`start`, `routing`, `sources`, `tool_start`, `tool_end`, `token`, `done`, `error`.\n\n"
        "Requires scope `agent:chat`."
    ),
    response_class=StreamingResponse,
    responses={
        **PROTECTED_RESPONSES,
        200: {
            "description": "SSE stream.",
            "content": {
                "text/event-stream": {
                    "schema": {"type": "string"},
                    "example": (
                        'data: {"type":"start","thread_id":"a3c9f0"}\n\n'
                        'data: {"type":"token","content":"You "}\n\n'
                        'data: {"type":"done","thread_id":"a3c9f0","answer":"You accrue ..."}\n\n'
                    ),
                }
            },
        },
        500: {"model": ErrorResponse, "description": "Unexpected failure."},
    },
)
async def chat_stream(
    payload: ChatRequest,
    agent: AgentDep,
    claims: ClaimsDep,
    settings: SettingsDep,
    tenant: TenantDep,
) -> StreamingResponse:
    """Stream tokens and tool events for one agent turn."""
    allowed, requested = _resolve_domains(payload, claims, settings)
    request_id = request_id_var.get()

    async def event_source() -> AsyncIterator[str]:
        try:
            async for event in agent.stream(
                message=payload.message,
                allowed_domains=allowed,
                thread_id=payload.thread_id,
                requested_domains=requested,
                tenant_id=tenant,
                filters=payload.filters,
            ):
                yield _sse(event)
        except AppError as exc:
            yield _sse({"type": "error", **exc.to_payload(request_id)["error"]})
        except Exception as exc:
            logger.exception("streaming chat failed")
            yield _sse(
                {
                    "type": "error",
                    "code": "internal_error",
                    "message": f"{type(exc).__name__}: {exc}",
                    "request_id": request_id,
                }
            )

    return StreamingResponse(event_source(), media_type="text/event-stream", headers=SSE_HEADERS)


def _sse(event: dict[str, Any]) -> str:
    """Render one Server-Sent Event frame."""
    return f"event: {event.get('type', 'message')}\ndata: {json.dumps(event, default=str)}\n\n"
