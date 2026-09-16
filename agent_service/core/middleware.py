"""Request scoped middleware: correlation ids and access logs."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from agent_service.core.logging import get_logger, request_id_var, subject_var, tenant_var

if TYPE_CHECKING:  # pragma: no cover
    from agent_service.core.security import TokenClaims

logger = get_logger("agent_service.access")

REQUEST_ID_HEADER = "X-Request-ID"


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assign a request id, expose it on the response and emit an access log."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
        request_id_token = request_id_var.set(request_id)
        subject_token = subject_var.set(None)
        tenant_token = tenant_var.set(None)
        request.state.request_id = request_id

        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            logger.exception(
                "request failed",
                extra={
                    "http_method": request.method,
                    "path": request.url.path,
                    "duration_ms": duration_ms,
                },
            )
            raise
        finally:
            claims: TokenClaims | None = getattr(request.state, "claims", None)
            if claims is not None:
                subject_var.set(claims.sub)

        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        response.headers[REQUEST_ID_HEADER] = request_id
        logger.info(
            "request completed",
            extra={
                "http_method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            },
        )
        request_id_var.reset(request_id_token)
        subject_var.reset(subject_token)
        tenant_var.reset(tenant_token)
        return response
