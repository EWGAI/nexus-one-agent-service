"""Structured JSON logging with request-scoped context.

``request_id`` and ``subject`` (the JWT ``sub``) are propagated through
:mod:`contextvars` so every log line emitted while handling a request is
automatically correlated. Raw tokens are never logged.
"""

from __future__ import annotations

from contextvars import ContextVar
import datetime as dt
import json
import logging
import sys
from typing import Any

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
subject_var: ContextVar[str | None] = ContextVar("subject", default=None)
tenant_var: ContextVar[str | None] = ContextVar("tenant_id", default=None)

_RESERVED = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
    }
)


class JSONFormatter(logging.Formatter):
    """Render log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": dt.datetime.fromtimestamp(record.created, tz=dt.UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if (rid := request_id_var.get()) is not None:
            payload["request_id"] = rid
        if (sub := subject_var.get()) is not None:
            payload["subject"] = sub
        if (tenant := tenant_var.get()) is not None:
            payload["tenant_id"] = tenant
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


class PlainFormatter(logging.Formatter):
    """Human friendly formatter used when ``LOG_JSON=false``."""

    def __init__(self) -> None:
        super().__init__("%(asctime)s %(levelname)-8s %(name)s :: %(message)s")

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        rid = request_id_var.get()
        return f"{base} [request_id={rid}]" if rid else base


def configure_logging(level: str = "INFO", *, json_logs: bool = True) -> None:
    """Install the root logging handler. Safe to call more than once."""
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(JSONFormatter() if json_logs else PlainFormatter())

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level.upper())

    for noisy in ("uvicorn.access", "httpx", "httpcore", "chromadb", "urllib3"):
        logging.getLogger(noisy).setLevel(max(logging.WARNING, root.level))


def get_logger(name: str) -> logging.Logger:
    """Return a module logger."""
    return logging.getLogger(name)
