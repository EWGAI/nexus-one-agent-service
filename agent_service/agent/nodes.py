"""Graph node implementations.

Flow::

    START -> route_domain -> retrieve -> generate -> [tools -> generate]* -> END
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END

from agent_service.agent.llm import ROUTING_MARKER, bind_tools
from agent_service.agent.state import AgentState
from agent_service.config import Settings
from agent_service.core.exceptions import LLMError
from agent_service.core.logging import get_logger
from agent_service.core.retry import async_retrying
from agent_service.kb.base import SearchHit, VectorStoreAdapter
from agent_service.kb.fusion import reciprocal_rank_fusion

logger = get_logger(__name__)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class AgentDeps:
    """Everything the nodes need, injected once when the graph is built."""

    store: VectorStoreAdapter
    llm: BaseChatModel
    settings: Settings
    tools: list[BaseTool]


async def _invoke(
    runnable: Any, messages: list[BaseMessage], settings: Settings, operation: str
) -> BaseMessage:
    """Call a chat model with retry/backoff, normalising provider failures."""
    try:
        async for attempt in async_retrying(
            operation,
            attempts=settings.retry_attempts,
            initial=settings.retry_initial_seconds,
            maximum=settings.retry_max_seconds,
        ):
            with attempt:
                result: BaseMessage = await runnable.ainvoke(messages)
        return result
    except Exception as exc:
        raise LLMError(f"LLM call {operation!r} failed: {exc}") from exc


async def route_domain_node(state: AgentState, deps: AgentDeps) -> dict[str, Any]:
    """Pick the domains to search, honouring an explicit caller choice."""
    allowed = list(state.get("allowed_domains") or [])
    requested = [domain for domain in (state.get("requested_domains") or []) if domain in allowed]

    if requested:
        logger.info(
            "domain routing skipped (explicit)",
            extra={"selected_domains": requested, "allowed_domains": allowed},
        )
        return {"selected_domains": requested, "routing_reason": "explicitly requested by caller"}

    if len(allowed) <= 1:
        return {
            "selected_domains": allowed,
            "routing_reason": "only one readable domain" if allowed else "no readable domains",
        }

    fallback = deps.settings.kb_fallback_domain
    prompt = (
        f"{ROUTING_MARKER}\n"
        "You route ERP questions to knowledge base domains.\n"
        f"Available domains: {', '.join(allowed)}\n"
        f"The '{fallback}' domain holds general company-wide information.\n"
        'Reply ONLY with JSON: {"domains": ["<domain>", ...], "reason": "<short reason>"}\n'
        "Pick between one and three domains."
    )
    response = await _invoke(
        deps.llm,
        [SystemMessage(content=prompt), HumanMessage(content=state.get("question", ""))],
        deps.settings,
        "route_domain",
    )

    selected: list[str] = []
    reason = "llm routing"
    match = _JSON_RE.search(str(response.content))
    if match:
        try:
            payload = json.loads(match.group(0))
            selected = [domain for domain in payload.get("domains", []) if domain in allowed]
            reason = str(payload.get("reason") or reason)
        except json.JSONDecodeError:
            logger.warning(
                "router returned malformed JSON", extra={"raw": str(response.content)[:200]}
            )

    if not selected:
        selected = allowed[:]
        reason = "router produced no usable choice; searching all readable domains"
    elif fallback in allowed and fallback not in selected:
        selected.append(fallback)
        reason = f"{reason} (+ '{fallback}' fallback)"

    logger.info(
        "domain routing decided",
        extra={"selected_domains": selected, "allowed_domains": allowed, "reason": reason},
    )
    return {"selected_domains": selected, "routing_reason": reason}


async def retrieve_node(state: AgentState, deps: AgentDeps) -> dict[str, Any]:
    """Fan out per domain and merge the ranked lists with reciprocal rank fusion."""
    domains = list(state.get("selected_domains") or [])
    question = state.get("question", "")
    if not domains or not question:
        return {"retrieved_docs": []}

    k = deps.settings.retrieval_k
    per_domain: dict[str, list[SearchHit]] = {}
    for domain in domains:
        per_domain[domain] = await deps.store.similarity_search(
            domain,
            question,
            k=k,
            filters=state.get("filters"),
            tenant_id=state.get("tenant_id"),
        )

    merged = (
        reciprocal_rank_fusion(per_domain, k=deps.settings.rrf_k, top_k=k)
        if len(per_domain) > 1
        else sorted(next(iter(per_domain.values()), []), key=lambda hit: hit.score, reverse=True)[
            :k
        ]
    )
    logger.info(
        "retrieval complete",
        extra={
            "domains": domains,
            "per_domain_hits": {domain: len(hits) for domain, hits in per_domain.items()},
            "merged_hits": len(merged),
        },
    )
    return {"retrieved_docs": [hit.model_dump() for hit in merged]}


def _context_block(state: AgentState) -> str:
    lines = []
    for position, raw in enumerate(state.get("retrieved_docs") or [], start=1):
        metadata = raw.get("metadata") or {}
        lines.append(
            f"[{position}] domain={raw.get('domain')} source={metadata.get('source', 'unknown')}\n"
            f"{raw.get('content', '').strip()}"
        )
    return "\n\n".join(lines)


async def generate_node(state: AgentState, deps: AgentDeps) -> dict[str, Any]:
    """Answer using the retrieved context, optionally requesting tools."""
    context = _context_block(state)
    system = deps.settings.agent_system_prompt
    if context:
        system = f"{system}\n\nContext:\n{context}"
    else:
        system = (
            f"{system}\n\nContext:\n(no passages retrieved - say you could not find "
            "anything in the knowledge base, or call a tool)"
        )

    runnable = bind_tools(deps.llm, deps.tools)
    messages: list[BaseMessage] = [SystemMessage(content=system), *state.get("messages", [])]
    response = await _invoke(runnable, messages, deps.settings, "generate")
    if not isinstance(response, AIMessage):  # pragma: no cover - providers return AIMessage
        response = AIMessage(content=str(response.content))
    return {"messages": [response], "iterations": int(state.get("iterations", 0)) + 1}


def should_continue(state: AgentState, settings: Settings) -> str:
    """Route to the tool node while the model keeps asking for tools."""
    messages = state.get("messages") or []
    last = messages[-1] if messages else None
    tool_calls = getattr(last, "tool_calls", None) if last is not None else None
    if tool_calls and int(state.get("iterations", 0)) < settings.agent_max_tool_iterations:
        return "tools"
    return END
