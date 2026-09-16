"""Agent graph, routing, retrieval fusion, tools and streaming."""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path

import httpx
import pytest

from agent_service.agent.graph import build_agent
from agent_service.agent.llm import FakeChatModel
from agent_service.agent.tools import build_tools, format_hits
from agent_service.kb.base import SearchHit
from agent_service.kb.factory import get_vector_store
from agent_service.kb.fusion import reciprocal_rank_fusion
from agent_service.kb.ingestion import IngestionService

from .conftest import make_settings

HR_TEXT = "Employees accrue 1.75 days of paid leave per completed month of service."
FIN_TEXT = "Accounts payable invoices above 50000 require two approvals before payment."


async def build_runtime(tmp_path: Path, **overrides: object) -> tuple[object, object]:
    settings = make_settings(tmp_path, **overrides)
    store = get_vector_store(settings)
    await store.initialize()
    ingestion = IngestionService(store, settings)
    await ingestion.ingest_text(
        "hr", HR_TEXT, source="hr-policy", tenant_id=None, uploader="seed", tags={"module": "leave"}
    )
    await ingestion.ingest_text(
        "finance",
        FIN_TEXT,
        source="ap-policy",
        tenant_id=None,
        uploader="seed",
        tags={"module": "ap"},
    )
    return build_agent(store, settings), store


# ------------------------------------------------------------------- graph --
async def test_chat_answers_from_the_requested_domain(tmp_path: Path) -> None:
    runtime, _ = await build_runtime(tmp_path)
    outcome = await runtime.chat(  # type: ignore[attr-defined]
        message="How much paid leave do employees accrue?",
        allowed_domains=["hr", "finance", "company"],
        requested_domains=["hr"],
    )
    assert outcome.domains_searched == ["hr"]
    assert outcome.routing_reason == "explicitly requested by caller"
    assert outcome.sources and all(source.domain == "hr" for source in outcome.sources)
    assert "1.75" in outcome.answer
    assert outcome.thread_id


async def test_router_picks_a_domain_when_none_is_given(tmp_path: Path) -> None:
    runtime, _ = await build_runtime(tmp_path)
    outcome = await runtime.chat(  # type: ignore[attr-defined]
        message="What do the finance rules say about invoice approvals?",
        allowed_domains=["hr", "finance", "company"],
    )
    assert "finance" in outcome.domains_searched
    # the fallback domain stays eligible for general questions
    assert "company" in outcome.domains_searched
    assert outcome.routing_reason


async def test_retrieval_is_restricted_to_readable_domains(tmp_path: Path) -> None:
    runtime, _ = await build_runtime(tmp_path)
    outcome = await runtime.chat(  # type: ignore[attr-defined]
        message="How much paid leave do employees accrue?", allowed_domains=["finance"]
    )
    assert outcome.domains_searched == ["finance"]
    assert all(source.domain == "finance" for source in outcome.sources)
    assert all(HR_TEXT not in source.content for source in outcome.sources)


async def test_threads_are_remembered_by_the_checkpointer(tmp_path: Path) -> None:
    runtime, _ = await build_runtime(tmp_path)
    first = await runtime.chat(  # type: ignore[attr-defined]
        message="How much paid leave do employees accrue?",
        allowed_domains=["hr"],
        requested_domains=["hr"],
    )
    second = await runtime.chat(  # type: ignore[attr-defined]
        message="And when is it credited?",
        allowed_domains=["hr"],
        requested_domains=["hr"],
        thread_id=first.thread_id,
    )
    assert second.thread_id == first.thread_id
    state = await runtime.graph.aget_state(  # type: ignore[attr-defined]
        {"configurable": {"thread_id": first.thread_id}}
    )
    assert len(state.values["messages"]) >= 4


async def test_tool_loop_runs_and_terminates(tmp_path: Path) -> None:
    runtime, _ = await build_runtime(tmp_path)
    outcome = await runtime.chat(  # type: ignore[attr-defined]
        message="/tool paid leave accrual",
        allowed_domains=["hr"],
        requested_domains=["hr"],
    )
    assert outcome.tool_calls
    assert outcome.tool_calls[0]["name"] == "kb_search"
    assert outcome.answer


async def test_streaming_emits_tokens_sources_and_done(tmp_path: Path) -> None:
    runtime, _ = await build_runtime(tmp_path)
    events = [
        event
        async for event in runtime.stream(  # type: ignore[attr-defined]
            message="How much paid leave do employees accrue?",
            allowed_domains=["hr"],
            requested_domains=["hr"],
        )
    ]
    kinds = [event["type"] for event in events]
    assert kinds[0] == "start"
    assert kinds[-1] == "done"
    assert "token" in kinds
    assert "sources" in kinds
    assert "routing" in kinds
    assert "1.75" in events[-1]["answer"]


# ------------------------------------------------------------------- tools --
async def test_kb_search_tool_enforces_allowed_domains(tmp_path: Path) -> None:
    _, store = await build_runtime(tmp_path)
    settings = make_settings(tmp_path)
    tools = {tool.name: tool for tool in build_tools(store, settings)}  # type: ignore[arg-type]

    allowed = await tools["kb_search"].ainvoke(
        {"query": "leave accrual", "domain": "hr"},
        config={"configurable": {"allowed_domains": ["hr"], "tenant_id": None, "filters": None}},
    )
    assert "1.75" in allowed

    denied = await tools["kb_search"].ainvoke(
        {"query": "invoice approvals", "domain": "finance"},
        config={"configurable": {"allowed_domains": ["hr"], "tenant_id": None, "filters": None}},
    )
    assert "not allowed" in denied

    listed = await tools["list_kb_domains"].ainvoke(
        {}, config={"configurable": {"allowed_domains": ["hr", "company"]}}
    )
    assert "company, hr" in listed

    nothing = await tools["kb_search"].ainvoke(
        {"query": "anything"}, config={"configurable": {"allowed_domains": []}}
    )
    assert "No knowledge base domains" in nothing


def test_format_hits_renders_citations() -> None:
    hit = SearchHit(id="1", content="text", score=0.5, domain="hr", metadata={"source": "p"})
    rendered = format_hits([hit])
    assert "[1] (domain=hr, source=p" in rendered
    assert format_hits([]) == "No matching passages were found."


# --------------------------------------------------------------- fake model --
def test_fake_model_routes_by_keyword_overlap() -> None:
    from langchain_core.messages import HumanMessage, SystemMessage

    model = FakeChatModel()
    response = model.invoke(
        [
            SystemMessage(content="ROUTING_TASK\nAvailable domains: hr, finance, company\n"),
            HumanMessage(content="a question about finance invoices"),
        ]
    )
    assert json.loads(str(response.content))["domains"] == ["finance"]


def test_fake_model_without_context_says_so() -> None:
    from langchain_core.messages import HumanMessage

    assert "no context" in str(FakeChatModel().invoke([HumanMessage(content="hi")]).content)


# --------------------------------------------------------------------- RRF --
def test_reciprocal_rank_fusion_merges_and_reranks() -> None:
    def hit(identifier: str, domain: str, score: float) -> SearchHit:
        return SearchHit(id=identifier, content=identifier, score=score, domain=domain)

    results = {
        "hr": [hit("a", "hr", 0.9), hit("b", "hr", 0.4)],
        "finance": [hit("c", "finance", 0.95), hit("d", "finance", 0.2)],
    }
    # Rank, not raw score, decides: the top hit of each domain outranks the
    # second hit of the other, even though 0.95 > 0.9.
    merged = reciprocal_rank_fusion(results, k=60, top_k=3)
    assert [item.id for item in merged] == ["a", "c", "b"]
    assert merged[0].score == merged[1].score
    assert merged[1].score > merged[2].score
    assert merged[0].metadata["raw_score"] == 0.9
    assert reciprocal_rank_fusion({}, top_k=3) == []


# ---------------------------------------------------------------- HTTP API --
async def test_chat_endpoint(client: httpx.AsyncClient, auth: dict[str, str]) -> None:
    await client.post(
        "/kb/hr/ingest/text", headers=auth, json={"text": HR_TEXT, "source": "hr-policy"}
    )
    response = await client.post(
        "/agent/chat",
        headers=auth,
        json={"message": "How much paid leave do employees accrue?", "domains": ["hr"]},
    )
    assert response.status_code == 200
    body = response.json()
    assert "1.75" in body["answer"]
    assert body["domains_searched"] == ["hr"]
    assert body["sources"][0]["domain"] == "hr"
    assert body["thread_id"]


async def test_chat_requires_the_agent_scope(
    client: httpx.AsyncClient, make_token: Callable[..., str]
) -> None:
    headers = {"Authorization": f"Bearer {make_token(scopes='kb:*:read')}"}
    response = await client.post("/agent/chat", headers=headers, json={"message": "hi"})
    assert response.status_code == 403


async def test_chat_rejects_unknown_and_forbidden_domains(
    client: httpx.AsyncClient, auth: dict[str, str], make_token: Callable[..., str]
) -> None:
    unknown = await client.post(
        "/agent/chat", headers=auth, json={"message": "hi", "domains": ["marketing"]}
    )
    assert unknown.status_code == 422
    assert unknown.json()["error"]["code"] == "unknown_domain"

    limited = {"Authorization": f"Bearer {make_token(scopes='kb:hr:read agent:chat')}"}
    forbidden = await client.post(
        "/agent/chat", headers=limited, json={"message": "hi", "domains": ["finance"]}
    )
    assert forbidden.status_code == 403


async def test_chat_without_any_readable_domain_is_forbidden(
    client: httpx.AsyncClient, make_token: Callable[..., str]
) -> None:
    headers = {"Authorization": f"Bearer {make_token(scopes='agent:chat')}"}
    response = await client.post("/agent/chat", headers=headers, json={"message": "hi"})
    assert response.status_code == 403


async def test_chat_stream_endpoint(client: httpx.AsyncClient, auth: dict[str, str]) -> None:
    await client.post(
        "/kb/hr/ingest/text", headers=auth, json={"text": HR_TEXT, "source": "hr-policy"}
    )
    async with client.stream(
        "POST",
        "/agent/chat/stream",
        headers=auth,
        json={"message": "How much paid leave do employees accrue?", "domains": ["hr"]},
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        payloads = [
            json.loads(line[len("data: ") :])
            async for line in response.aiter_lines()
            if line.startswith("data: ")
        ]

    kinds = [item["type"] for item in payloads]
    assert kinds[0] == "start"
    assert kinds[-1] == "done"
    assert "1.75" in payloads[-1]["answer"]


async def test_chat_stream_reports_errors_inside_the_stream(
    client: httpx.AsyncClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = client.app.state.agent  # type: ignore[attr-defined]

    async def boom(**_: object) -> object:
        raise RuntimeError("retrieval exploded")
        yield  # pragma: no cover

    monkeypatch.setattr(runtime, "stream", boom)
    async with client.stream(
        "POST", "/agent/chat/stream", headers=auth, json={"message": "hi", "domains": ["hr"]}
    ) as response:
        body = "".join([line async for line in response.aiter_lines()])
    assert '"type": "error"' in body
    assert "retrieval exploded" in body
