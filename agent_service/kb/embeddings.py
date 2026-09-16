"""Embedding provider factory.

Every provider is wrapped in the same retrying facade so the rest of the code
only deals with :class:`EmbeddingBundle`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import hashlib
import math
import re
from typing import Any

from langchain_core.embeddings import Embeddings

from agent_service.config import EmbeddingProvider, Settings
from agent_service.core.exceptions import ConfigurationError
from agent_service.core.logging import get_logger
from agent_service.core.retry import async_retrying, sync_retrying

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class DeterministicFakeEmbeddings(Embeddings):
    """Hashing-trick embeddings used by tests, CI and offline smoke runs.

    Unlike random fakes these are *lexically meaningful*: documents sharing
    tokens get similar vectors, so ranking assertions are stable and real.
    """

    def __init__(self, dimension: int = 256) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.dimension = dimension

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = _TOKEN_RE.findall(text.lower()) or ["\x00empty"]
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:8], "big") % self.dimension
            sign = 1.0 if digest[8] & 1 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            vector[0] = 1.0
            return vector
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


@dataclass
class EmbeddingBundle:
    """An embedding model plus the metadata the vector stores need."""

    embeddings: Embeddings
    model_name: str
    provider: str
    settings: Settings
    _dimension: int | None = field(default=None, repr=False)

    # -- dimension ---------------------------------------------------------
    async def ensure_dimension(self) -> int:
        """Resolve (and cache) the embedding dimension, probing only if needed."""
        if self._dimension is None:
            vector = await self.aembed_query("dimension probe")
            self._dimension = len(vector)
            logger.info(
                "resolved embedding dimension",
                extra={"provider": self.provider, "model": self.model_name, "dim": self._dimension},
            )
        return self._dimension

    @property
    def dimension(self) -> int:
        """Cached dimension; falls back to a blocking probe when cold."""
        if self._dimension is None:
            self._dimension = len(self.embeddings.embed_query("dimension probe"))
        return self._dimension

    # -- embedding calls ---------------------------------------------------
    async def aembed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of documents with retry/backoff and batching."""
        if not texts:
            return []
        batch_size = max(1, self.settings.embed_batch_size)
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            chunk = list(texts[start : start + batch_size])
            async for attempt in self._retrying("embed_documents"):
                with attempt:
                    vectors.extend(await self.embeddings.aembed_documents(chunk))
        return vectors

    async def aembed_query(self, text: str) -> list[float]:
        """Embed a single query with retry/backoff."""
        result: list[float] = []
        async for attempt in self._retrying("embed_query"):
            with attempt:
                result = await self.embeddings.aembed_query(text)
        return result

    def embed_query_sync(self, text: str) -> list[float]:
        """Blocking variant, used from sync code paths such as FAISS loading."""
        for attempt in sync_retrying(
            "embed_query",
            attempts=self.settings.retry_attempts,
            initial=self.settings.retry_initial_seconds,
            maximum=self.settings.retry_max_seconds,
        ):
            with attempt:
                return self.embeddings.embed_query(text)
        raise RuntimeError("unreachable")  # pragma: no cover

    def _retrying(self, operation: str) -> Any:
        return async_retrying(
            operation,
            attempts=self.settings.retry_attempts,
            initial=self.settings.retry_initial_seconds,
            maximum=self.settings.retry_max_seconds,
        )


def build_embeddings(settings: Settings) -> EmbeddingBundle:
    """Create the embedding bundle selected by ``EMBEDDING_PROVIDER``."""
    provider = settings.embedding_provider

    if provider is EmbeddingProvider.FAKE:
        dimension = settings.embedding_dimension or 256
        return EmbeddingBundle(
            embeddings=DeterministicFakeEmbeddings(dimension),
            model_name=f"fake-{dimension}d",
            provider=provider.value,
            settings=settings,
            _dimension=dimension,
        )

    if provider is EmbeddingProvider.OPENAI:
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise ConfigurationError(
                'EMBEDDING_PROVIDER=openai requires the openai extra: pip install -e ".[openai]"'
            ) from exc
        kwargs: dict[str, Any] = {
            "model": settings.embedding_model,
            "api_key": settings.openai_api_key,
        }
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url
        if settings.embedding_dimension:
            kwargs["dimensions"] = settings.embedding_dimension
        return EmbeddingBundle(
            embeddings=OpenAIEmbeddings(**kwargs),
            model_name=settings.embedding_model,
            provider=provider.value,
            settings=settings,
            _dimension=settings.resolve_embedding_dimension(),
        )

    if provider is EmbeddingProvider.HUGGINGFACE:
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise ConfigurationError(
                "EMBEDDING_PROVIDER=huggingface requires the huggingface extra: "
                'pip install -e ".[huggingface]"'
            ) from exc
        return EmbeddingBundle(
            embeddings=HuggingFaceEmbeddings(
                model_name=settings.embedding_model,
                model_kwargs={"device": settings.huggingface_device},
                encode_kwargs={"normalize_embeddings": True},
            ),
            model_name=settings.embedding_model,
            provider=provider.value,
            settings=settings,
            _dimension=settings.resolve_embedding_dimension(),
        )

    raise ConfigurationError(  # pragma: no cover - exhaustive enum
        f"Unsupported EMBEDDING_PROVIDER={provider!r}"
    )
