"""Chat model factory plus an offline deterministic model for tests and CI."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
import json
import re
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool

from agent_service.config import LLMProvider, Settings
from agent_service.core.exceptions import ConfigurationError
from agent_service.core.logging import get_logger

logger = get_logger(__name__)

ROUTING_MARKER = "ROUTING_TASK"
TOOL_PREFIX = "/tool"
_DOMAIN_LIST_RE = re.compile(r"Available domains:\s*(.+)")
_WORD_RE = re.compile(r"\S+\s*")


class FakeChatModel(BaseChatModel):
    """Deterministic chat model used when ``LLM_PROVIDER=fake``.

    It is intentionally *not* random: it echoes the retrieved context so an
    end-to-end ingest -> search -> chat assertion actually proves retrieval
    worked, and it emits a tool call when a message starts with ``/tool``.
    """

    context_preview_chars: int = 240

    @property
    def _llm_type(self) -> str:
        return "fake-chat"

    # Deliberately looser than the base signature: this stub accepts any tool spec.
    def bind_tools(  # type: ignore[override]
        self, tools: Sequence[Any], **kwargs: Any
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Mimic provider tool binding by stashing OpenAI-style tool schemas."""
        formatted = [convert_to_openai_tool(tool) for tool in tools]
        return self.bind(tools=formatted, **kwargs)

    # -- response construction ---------------------------------------------
    @staticmethod
    def _system_text(messages: Sequence[BaseMessage]) -> str:
        return "\n".join(str(message.content) for message in messages if message.type == "system")

    @staticmethod
    def _last_human(messages: Sequence[BaseMessage]) -> str:
        for message in reversed(messages):
            if message.type == "human":
                return str(message.content)
        return ""

    def _build_message(self, messages: Sequence[BaseMessage], **kwargs: Any) -> AIMessage:
        system_text = self._system_text(messages)
        question = self._last_human(messages)

        if ROUTING_MARKER in system_text:
            match = _DOMAIN_LIST_RE.search(system_text)
            available = (
                [item.strip() for item in match.group(1).split(",") if item.strip()]
                if match
                else []
            )
            words = set(re.findall(r"[a-z0-9_-]+", question.lower()))
            chosen = [domain for domain in available if domain in words] or available
            return AIMessage(
                content=json.dumps(
                    {"domains": chosen, "reason": "keyword overlap with the question"}
                )
            )

        tools = kwargs.get("tools") or []
        already_used = any(isinstance(message, ToolMessage) for message in messages)
        if tools and question.startswith(TOOL_PREFIX) and not already_used:
            name = str(tools[0].get("function", {}).get("name", "kb_search"))
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": name,
                        "args": {"query": question[len(TOOL_PREFIX) :].strip() or question},
                        "id": "call_fake_1",
                        "type": "tool_call",
                    }
                ],
            )

        context = system_text.split("Context:", 1)[-1].strip() if "Context:" in system_text else ""
        if context:
            preview = " ".join(context.split())[: self.context_preview_chars]
            return AIMessage(content=f"{question.strip()} Based on the knowledge base: {preview}")
        return AIMessage(content=f"{question.strip()} I have no context for that.")

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(
            generations=[ChatGeneration(message=self._build_message(messages, **kwargs))]
        )

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        message = self._build_message(messages, **kwargs)
        if message.tool_calls:
            call = message.tool_calls[0]
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {
                            "name": call["name"],
                            "args": json.dumps(call["args"]),
                            "id": call.get("id"),
                            "index": 0,
                            "type": "tool_call_chunk",
                        }
                    ],
                )
            )
            return
        for token in _WORD_RE.findall(str(message.content)):
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=token))
            if run_manager is not None:
                run_manager.on_llm_new_token(token, chunk=chunk)
            yield chunk


def build_llm(settings: Settings) -> BaseChatModel:
    """Instantiate the chat model selected by ``LLM_PROVIDER``."""
    provider = settings.llm_provider

    if provider is LLMProvider.FAKE:
        return FakeChatModel()

    if provider is LLMProvider.OPENAI:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise ConfigurationError(
                'LLM_PROVIDER=openai requires the openai extra: pip install -e ".[openai]"'
            ) from exc
        kwargs: dict[str, Any] = {
            "model": settings.llm_model,
            "temperature": settings.llm_temperature,
            "max_tokens": settings.llm_max_tokens,
            "api_key": settings.openai_api_key,
            "timeout": 60,
            "streaming": True,
        }
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url
        openai_model: BaseChatModel = ChatOpenAI(**kwargs)
        return openai_model

    if provider is LLMProvider.ANTHROPIC:
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise ConfigurationError(
                'LLM_PROVIDER=anthropic requires the anthropic extra: pip install -e ".[anthropic]"'
            ) from exc
        anthropic_model: BaseChatModel = ChatAnthropic(
            model_name=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens_to_sample=settings.llm_max_tokens,
            api_key=settings.anthropic_api_key,
            timeout=60,
            stop=None,
        )
        return anthropic_model

    raise ConfigurationError(f"Unsupported LLM_PROVIDER={provider!r}")  # pragma: no cover


def bind_tools(llm: BaseChatModel, tools: Sequence[BaseTool]) -> Runnable[Any, BaseMessage]:
    """Bind tools when the provider supports it, otherwise return the bare model."""
    if not tools:
        return llm
    try:
        return llm.bind_tools(tools)
    except NotImplementedError:  # pragma: no cover - provider without tool support
        logger.warning("chat model does not support tool calling; running without tools")
        return llm
