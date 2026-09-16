"""Tools exposed to the agent.

Request scoped data (which domains the caller may read, the tenant, extra
metadata filters) is delivered through the ``configurable`` section of the
``RunnableConfig``, so the graph can be compiled once at startup and still
enforce per-request authorization.
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from agent_service.config import Settings
from agent_service.core.logging import get_logger
from agent_service.kb.base import SearchHit, VectorStoreAdapter

logger = get_logger(__name__)

# LangChain injects the RunnableConfig only when a parameter is annotated with
# exactly ``RunnableConfig`` (``RunnableConfig | None`` is not recognised), so the
# default has to be a real empty config rather than ``None``.
EMPTY_CONFIG: RunnableConfig = {}


class KBSearchInput(BaseModel):
    """Arguments for the ``kb_search`` tool."""

    query: str = Field(description="Natural language search query.")
    domain: str | None = Field(
        default=None,
        description="Restrict the search to one domain. Omit to search every allowed domain.",
    )
    k: int = Field(default=4, ge=1, le=20, description="Number of chunks to return.")


class NoArgs(BaseModel):
    """Marker schema for tools that take no arguments."""


def _configurable(config: RunnableConfig | None) -> dict[str, Any]:
    return dict((config or {}).get("configurable", {}) or {})


def format_hits(hits: list[SearchHit]) -> str:
    """Render hits as compact, citable text for the model."""
    if not hits:
        return "No matching passages were found."
    lines = []
    for position, hit in enumerate(hits, start=1):
        source = hit.metadata.get("source", "unknown")
        lines.append(
            f"[{position}] (domain={hit.domain}, source={source}, score={hit.score:.4f})\n"
            f"{hit.content.strip()}"
        )
    return "\n\n".join(lines)


def build_tools(store: VectorStoreAdapter, settings: Settings) -> list[BaseTool]:
    """Create the tool set bound to the active vector store."""

    async def kb_search(
        query: str,
        domain: str | None = None,
        k: int = 4,
        config: RunnableConfig = EMPTY_CONFIG,
    ) -> str:
        configurable = _configurable(config)
        allowed: list[str] = list(configurable.get("allowed_domains") or [])
        tenant_id: str | None = configurable.get("tenant_id")
        filters: dict[str, Any] | None = configurable.get("filters")

        if domain is not None and domain not in allowed:
            # Never reveal whether the domain exists or holds data.
            return f"You are not allowed to search the {domain!r} knowledge base."
        targets = [domain] if domain else allowed
        if not targets:
            return "No knowledge base domains are available to this caller."

        collected: list[SearchHit] = []
        for target in targets:
            collected.extend(
                await store.similarity_search(
                    target, query, k=k, filters=filters, tenant_id=tenant_id
                )
            )
        collected.sort(key=lambda hit: hit.score, reverse=True)
        logger.info(
            "kb_search tool executed",
            extra={"domains": targets, "hits": len(collected), "k": k},
        )
        return format_hits(collected[:k])

    async def list_kb_domains(config: RunnableConfig = EMPTY_CONFIG) -> str:
        allowed = list(_configurable(config).get("allowed_domains") or [])
        if not allowed:
            return "No knowledge base domains are available to this caller."
        return "Readable domains: " + ", ".join(sorted(allowed))

    return [
        StructuredTool.from_function(
            coroutine=kb_search,
            name="kb_search",
            description=(
                "Search the ERP knowledge base for passages relevant to a question. "
                "Use it whenever the answer may depend on company documents, policies, "
                "ledgers, runbooks or specifications."
            ),
            args_schema=KBSearchInput,
        ),
        StructuredTool.from_function(
            coroutine=list_kb_domains,
            name="list_kb_domains",
            description="List the knowledge base domains this caller is allowed to read.",
            args_schema=NoArgs,
        ),
    ]
