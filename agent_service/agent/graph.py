"""LangGraph assembly and the runtime facade used by the API layer."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
import uuid

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from agent_service.agent.llm import build_llm
from agent_service.agent.nodes import (
    AgentDeps,
    generate_node,
    retrieve_node,
    route_domain_node,
    should_continue,
)
from agent_service.agent.state import AgentState
from agent_service.agent.tools import build_tools
from agent_service.config import Settings
from agent_service.core.logging import get_logger
from agent_service.kb.base import SearchHit, VectorStoreAdapter

logger = get_logger(__name__)


class ChatOutcome(BaseModel):
    """Result of one agent turn."""

    answer: str = Field(description="The assistant's reply.")
    sources: list[SearchHit] = Field(
        default_factory=list, description="Passages used, with domain."
    )
    thread_id: str = Field(description="Conversation id; pass it back to continue the thread.")
    domains_searched: list[str] = Field(default_factory=list)
    routing_reason: str | None = Field(default=None, description="Why those domains were chosen.")
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)


class AgentRuntime:
    """Compiled graph plus the request plumbing around it."""

    def __init__(self, store: VectorStoreAdapter, settings: Settings) -> None:
        self.store = store
        self.settings = settings
        self.deps = AgentDeps(
            store=store,
            llm=build_llm(settings),
            settings=settings,
            tools=build_tools(store, settings),
        )
        self.checkpointer = MemorySaver() if settings.agent_checkpointer == "memory" else None
        self.graph = self._build()

    # -- graph -------------------------------------------------------------
    def _build(self) -> Any:
        async def _route(state: AgentState) -> dict[str, Any]:
            return await route_domain_node(state, self.deps)

        async def _retrieve(state: AgentState) -> dict[str, Any]:
            return await retrieve_node(state, self.deps)

        async def _generate(state: AgentState) -> dict[str, Any]:
            return await generate_node(state, self.deps)

        def _branch(state: AgentState) -> str:
            return should_continue(state, self.settings)

        builder = StateGraph(AgentState)
        builder.add_node("route_domain", _route)
        builder.add_node("retrieve", _retrieve)
        builder.add_node("generate", _generate)
        builder.add_node("tools", ToolNode(self.deps.tools))

        builder.add_edge(START, "route_domain")
        builder.add_edge("route_domain", "retrieve")
        builder.add_edge("retrieve", "generate")
        builder.add_conditional_edges("generate", _branch, {"tools": "tools", END: END})
        builder.add_edge("tools", "generate")

        return builder.compile(checkpointer=self.checkpointer)

    # -- invocation --------------------------------------------------------
    def _config(
        self,
        thread_id: str,
        allowed_domains: list[str],
        tenant_id: str | None,
        filters: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return {
            "configurable": {
                "thread_id": thread_id,
                "allowed_domains": allowed_domains,
                "tenant_id": tenant_id,
                "filters": filters,
            },
            "recursion_limit": 6 + self.settings.agent_max_tool_iterations * 2,
        }

    @staticmethod
    def _initial_state(
        message: str,
        thread_id: str,
        allowed_domains: list[str],
        requested_domains: list[str],
        tenant_id: str | None,
        filters: dict[str, Any] | None,
    ) -> AgentState:
        return AgentState(
            messages=[HumanMessage(content=message)],
            question=message,
            thread_id=thread_id,
            tenant_id=tenant_id,
            allowed_domains=allowed_domains,
            requested_domains=requested_domains,
            selected_domains=[],
            routing_reason="",
            filters=filters,
            retrieved_docs=[],
            iterations=0,
            metadata={},
        )

    async def chat(
        self,
        *,
        message: str,
        allowed_domains: list[str],
        thread_id: str | None = None,
        requested_domains: list[str] | None = None,
        tenant_id: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> ChatOutcome:
        """Run one full turn and return the answer with its sources."""
        thread = thread_id or uuid.uuid4().hex
        state = self._initial_state(
            message, thread, allowed_domains, requested_domains or [], tenant_id, filters
        )
        config = self._config(thread, allowed_domains, tenant_id, filters)
        final: dict[str, Any] = await self.graph.ainvoke(state, config=config)

        answer = ""
        tool_calls: list[dict[str, Any]] = []
        for item in final.get("messages", []):
            if isinstance(item, AIMessage):
                if item.tool_calls:
                    tool_calls.extend(
                        {"name": call["name"], "args": call["args"]} for call in item.tool_calls
                    )
                if item.content:
                    answer = str(item.content)
        return ChatOutcome(
            answer=answer,
            sources=[SearchHit(**raw) for raw in final.get("retrieved_docs", [])],
            thread_id=thread,
            domains_searched=list(final.get("selected_domains") or []),
            routing_reason=final.get("routing_reason") or None,
            tool_calls=tool_calls,
        )

    async def stream(
        self,
        *,
        message: str,
        allowed_domains: list[str],
        thread_id: str | None = None,
        requested_domains: list[str] | None = None,
        tenant_id: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield SSE-shaped events: ``token``, ``tool_start``, ``tool_end``, ``sources``, ``done``."""
        thread = thread_id or uuid.uuid4().hex
        state = self._initial_state(
            message, thread, allowed_domains, requested_domains or [], tenant_id, filters
        )
        config = self._config(thread, allowed_domains, tenant_id, filters)

        yield {"type": "start", "thread_id": thread}
        buffer: list[str] = []
        async for event in self.graph.astream_events(state, config=config, version="v2"):
            kind = event.get("event")
            node = (event.get("metadata") or {}).get("langgraph_node")

            if kind == "on_chat_model_stream" and node == "generate":
                chunk = event["data"].get("chunk")
                text = str(getattr(chunk, "content", "") or "")
                if text:
                    buffer.append(text)
                    yield {"type": "token", "content": text}
            elif kind == "on_tool_start":
                yield {
                    "type": "tool_start",
                    "name": event.get("name"),
                    "args": event["data"].get("input"),
                }
            elif kind == "on_tool_end":
                yield {
                    "type": "tool_end",
                    "name": event.get("name"),
                    "output": str(event["data"].get("output"))[:2000],
                }
            elif kind == "on_chain_end" and event.get("name") == "retrieve":
                output = event["data"].get("output") or {}
                docs = output.get("retrieved_docs") if isinstance(output, dict) else None
                yield {"type": "sources", "sources": docs or []}
            elif kind == "on_chain_end" and event.get("name") == "route_domain":
                output = event["data"].get("output") or {}
                if isinstance(output, dict):
                    yield {
                        "type": "routing",
                        "domains": output.get("selected_domains") or [],
                        "reason": output.get("routing_reason"),
                    }

        yield {"type": "done", "thread_id": thread, "answer": "".join(buffer)}


def build_agent(store: VectorStoreAdapter, settings: Settings) -> AgentRuntime:
    """Factory used by the application lifespan."""
    runtime = AgentRuntime(store, settings)
    logger.info(
        "agent graph compiled",
        extra={
            "llm_provider": settings.llm_provider.value,
            "model": settings.llm_model,
            "tools": [tool.name for tool in runtime.deps.tools],
            "checkpointer": settings.agent_checkpointer,
        },
    )
    return runtime
