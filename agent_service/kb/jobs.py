"""In-memory background job registry for long running ingestion runs.

Deliberately simple: a single process store with an asyncio lock. Swap the
implementation for Redis/Postgres if you need jobs to survive a restart or to
be shared across replicas - the API surface is what matters.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import datetime as dt
from enum import StrEnum
from typing import Any
import uuid

from pydantic import BaseModel

from agent_service.core.exceptions import NotFoundError
from agent_service.core.logging import get_logger

logger = get_logger(__name__)


class JobStatus(StrEnum):
    """Lifecycle of an ingestion job."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class Job(BaseModel):
    """A unit of background work and its outcome."""

    id: str
    kind: str
    domain: str
    status: JobStatus = JobStatus.QUEUED
    created_at: dt.datetime
    updated_at: dt.datetime
    submitted_by: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


class JobStore:
    """Thread-safe (asyncio) registry of :class:`Job` objects."""

    def __init__(self, max_jobs: int = 1000) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = asyncio.Lock()
        self._max_jobs = max_jobs

    async def create(self, *, kind: str, domain: str, submitted_by: str | None) -> Job:
        """Register a new queued job."""
        now = dt.datetime.now(tz=dt.UTC)
        job = Job(
            id=uuid.uuid4().hex,
            kind=kind,
            domain=domain,
            created_at=now,
            updated_at=now,
            submitted_by=submitted_by,
        )
        async with self._lock:
            if len(self._jobs) >= self._max_jobs:
                oldest = min(self._jobs.values(), key=lambda item: item.created_at)
                self._jobs.pop(oldest.id, None)
            self._jobs[job.id] = job
        return job

    async def get(self, job_id: str) -> Job:
        """Fetch a job or raise :class:`NotFoundError`."""
        async with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise NotFoundError(f"Job {job_id!r} was not found")
        return job

    async def list(self, *, domain: str | None = None) -> list[Job]:
        """All known jobs, newest first."""
        async with self._lock:
            jobs = list(self._jobs.values())
        if domain:
            jobs = [job for job in jobs if job.domain == domain]
        return sorted(jobs, key=lambda item: item.created_at, reverse=True)

    async def _update(self, job_id: str, **changes: Any) -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:  # pragma: no cover - job ids are internal
                return
            self._jobs[job_id] = job.model_copy(
                update={**changes, "updated_at": dt.datetime.now(tz=dt.UTC)}
            )

    async def run(self, job_id: str, work: Callable[[], Awaitable[dict[str, Any]]]) -> None:
        """Execute ``work``, recording success or failure against the job."""
        await self._update(job_id, status=JobStatus.RUNNING)
        try:
            result = await work()
        except Exception as exc:
            logger.exception("ingestion job failed", extra={"job_id": job_id})
            await self._update(
                job_id, status=JobStatus.FAILED, error=f"{type(exc).__name__}: {exc}"
            )
            return
        await self._update(job_id, status=JobStatus.SUCCEEDED, result=result)
