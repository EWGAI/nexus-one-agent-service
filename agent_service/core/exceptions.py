"""Application error hierarchy.

Every error carries a stable machine-readable ``code`` and an HTTP status so the
API can render a single consistent error envelope:

``{"error": {"code": ..., "message": ..., "details": {...}, "request_id": ...}}``
"""

from __future__ import annotations

from typing import Any


class AppError(Exception):
    """Base class for all errors raised deliberately by the service."""

    status_code: int = 500
    code: str = "internal_error"

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        status_code: int | None = None,
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = details or {}
        if status_code is not None:
            self.status_code = status_code
        if code is not None:
            self.code = code

    def to_payload(self, request_id: str | None = None) -> dict[str, Any]:
        """Render the canonical error envelope."""
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
                "request_id": request_id,
            }
        }


class ConfigurationError(AppError):
    """Invalid or incomplete configuration; raised at startup (fail fast)."""

    status_code = 500
    code = "configuration_error"


class AuthenticationError(AppError):
    """Missing, malformed, expired or otherwise unverifiable credentials."""

    status_code = 401
    code = "unauthorized"


class AuthorizationError(AppError):
    """Authenticated but lacking the required scope."""

    status_code = 403
    code = "forbidden"


class NotFoundError(AppError):
    """Requested resource does not exist."""

    status_code = 404
    code = "not_found"


class UnknownDomainError(AppError):
    """The requested knowledge-base domain is not registered."""

    status_code = 422
    code = "unknown_domain"


class IngestionError(AppError):
    """A document could not be loaded, parsed or chunked."""

    status_code = 400
    code = "ingestion_error"


class VectorStoreError(AppError):
    """The backing vector store rejected or failed an operation."""

    status_code = 502
    code = "vector_store_error"


class LLMError(AppError):
    """The LLM provider failed after all retries."""

    status_code = 502
    code = "llm_error"
