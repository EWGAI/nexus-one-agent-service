"""Graph state shared by every node."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    """State threaded through the LangGraph run.

    ``messages`` uses the ``add_messages`` reducer so nodes only ever return the
    *new* messages they produced.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    question: str
    thread_id: str
    tenant_id: str | None
    allowed_domains: list[str]
    requested_domains: list[str]
    selected_domains: list[str]
    routing_reason: str
    filters: dict[str, Any] | None
    retrieved_docs: list[dict[str, Any]]
    iterations: int
    metadata: dict[str, Any]
