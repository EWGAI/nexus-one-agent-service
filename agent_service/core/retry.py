"""Retry helpers built on tenacity.

Used for every outbound call that can fail transiently: embeddings and LLM
completions.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from tenacity import (
    AsyncRetrying,
    RetryCallState,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from agent_service.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


def _log_retry(operation: str) -> Callable[[RetryCallState], None]:
    def _hook(state: RetryCallState) -> None:
        logger.warning(
            "retrying operation",
            extra={
                "operation": operation,
                "attempt": state.attempt_number,
                "error": repr(state.outcome.exception()) if state.outcome else None,
            },
        )

    return _hook


def _kwargs(operation: str, attempts: int, initial: float, maximum: float) -> dict[str, Any]:
    return {
        "stop": stop_after_attempt(max(1, attempts)),
        "wait": wait_exponential_jitter(initial=initial, max=maximum),
        "retry": retry_if_exception_type(Exception),
        "before_sleep": _log_retry(operation),
        "reraise": True,
    }


def sync_retrying(operation: str, *, attempts: int, initial: float, maximum: float) -> Retrying:
    """Blocking retry controller."""
    return Retrying(**_kwargs(operation, attempts, initial, maximum))


def async_retrying(
    operation: str, *, attempts: int, initial: float, maximum: float
) -> AsyncRetrying:
    """Awaitable retry controller."""
    return AsyncRetrying(**_kwargs(operation, attempts, initial, maximum))
