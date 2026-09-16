"""Vector store registry.

Adding a new backend is three steps:

1. implement :class:`~agent_service.kb.base.VectorStoreAdapter` in
   ``agent_service/kb/<name>_store.py``;
2. add ``"<name>"`` to :data:`BUILTIN_STORES` (or call :func:`register_store`
   from your own package at import time);
3. add the value to ``VectorStoreType`` in :mod:`agent_service.config`.

No calling code changes - ``VECTOR_STORE=<name>`` is enough.
"""

from __future__ import annotations

from collections.abc import Callable
import importlib
from typing import TypeAlias

from agent_service.config import EmbeddingProvider, PineconeStrategy, Settings, VectorStoreType
from agent_service.core.exceptions import ConfigurationError
from agent_service.core.logging import get_logger
from agent_service.kb.base import VectorStoreAdapter
from agent_service.kb.embeddings import EmbeddingBundle, build_embeddings

logger = get_logger(__name__)

StoreBuilder: TypeAlias = Callable[[Settings, EmbeddingBundle], VectorStoreAdapter]

#: name -> (module, attribute, pip extra)
BUILTIN_STORES: dict[str, tuple[str, str, str]] = {
    VectorStoreType.CHROMA.value: ("agent_service.kb.chroma_store", "ChromaStore", "chroma"),
    VectorStoreType.FAISS.value: ("agent_service.kb.faiss_store", "FaissStore", "faiss"),
    VectorStoreType.PINECONE.value: (
        "agent_service.kb.pinecone_store",
        "PineconeStore",
        "pinecone",
    ),
}

_REGISTRY: dict[str, StoreBuilder] = {}


def register_store(name: str) -> Callable[[StoreBuilder], StoreBuilder]:
    """Decorator registering an out-of-tree adapter under ``name``."""

    def _decorate(builder: StoreBuilder) -> StoreBuilder:
        _REGISTRY[name.lower()] = builder
        return builder

    return _decorate


def available_stores() -> list[str]:
    """Every store name the factory knows about."""
    return sorted(set(BUILTIN_STORES) | set(_REGISTRY))


def _require(settings: Settings, pairs: list[tuple[str, object]]) -> None:
    missing = [name for name, value in pairs if not value]
    if missing:
        raise ConfigurationError(
            f"VECTOR_STORE={settings.vector_store} is missing required configuration",
            details={"missing": missing},
        )


def validate_store_config(settings: Settings) -> None:
    """Fail fast with an actionable message when store specific env vars are absent."""
    if settings.vector_store is VectorStoreType.PINECONE:
        required: list[tuple[str, object]] = [("PINECONE_API_KEY", settings.pinecone_api_key)]
        if settings.pinecone_strategy is PineconeStrategy.NAMESPACE:
            required.append(("PINECONE_INDEX", settings.pinecone_index))
        _require(settings, required)
    elif settings.vector_store is VectorStoreType.CHROMA:
        if not settings.chroma_host:
            _require(settings, [("CHROMA_PERSIST_DIR", settings.chroma_persist_dir)])
    elif settings.vector_store is VectorStoreType.FAISS:
        _require(settings, [("FAISS_INDEX_PATH", settings.faiss_index_path)])


def _resolve(name: str) -> StoreBuilder:
    key = name.lower()
    if key in _REGISTRY:
        return _REGISTRY[key]
    if key not in BUILTIN_STORES:
        raise ConfigurationError(
            f"Unknown VECTOR_STORE={name!r}", details={"available": available_stores()}
        )
    module_path, attribute, extra = BUILTIN_STORES[key]
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ConfigurationError(
            f"VECTOR_STORE={key} requires the '{extra}' extra",
            details={"install": f'pip install -e ".[{extra}]"', "import_error": str(exc)},
        ) from exc
    builder: StoreBuilder = getattr(module, attribute)
    _REGISTRY[key] = builder
    return builder


def get_vector_store(
    settings: Settings, embeddings: EmbeddingBundle | None = None
) -> VectorStoreAdapter:
    """Build the adapter selected by ``VECTOR_STORE``."""
    store_name = str(settings.vector_store)
    validate_store_config(settings)
    bundle = embeddings or build_embeddings(settings)
    adapter = _resolve(store_name)(settings, bundle)
    logger.info(
        "vector store selected",
        extra={
            "store": store_name,
            "embedding_provider": str(settings.embedding_provider),
            "embedding_model": bundle.model_name,
            "tenant_mode": str(settings.tenant_mode),
            "domains": list(settings.domains),
        },
    )
    if (
        settings.embedding_provider is EmbeddingProvider.FAKE
        and settings.environment == "production"
    ):
        logger.warning("EMBEDDING_PROVIDER=fake is running in production")
    return adapter
