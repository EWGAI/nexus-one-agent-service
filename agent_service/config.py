"""Typed application settings loaded from environment / ``.env``.

Everything that changes behaviour at runtime lives here. Switching the vector
store or the LLM provider is a pure environment change - no code edits.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
import re
import tomllib
from typing import Any, Literal

from pydantic import Field, ValidationError, computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from agent_service.core.exceptions import ConfigurationError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DOMAIN_RE = re.compile(r"^[a-z][a-z0-9_-]{0,39}$")

#: Known embedding dimensions, used to avoid a network probe when possible.
KNOWN_EMBEDDING_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
    "BAAI/bge-small-en-v1.5": 384,
}


def read_project_version() -> str:
    """Read the package version from installed metadata, falling back to pyproject."""
    try:
        from importlib.metadata import version

        return version("agent-service")
    except Exception:
        pyproject = PROJECT_ROOT / "pyproject.toml"
        if pyproject.is_file():
            with pyproject.open("rb") as handle:
                data: dict[str, Any] = tomllib.load(handle)
            candidate = data.get("project", {}).get("version")
            if isinstance(candidate, str):
                return candidate
        return "0.0.0"


class VectorStoreType(StrEnum):
    """Supported vector store backends (``VECTOR_STORE``)."""

    CHROMA = "chroma"
    FAISS = "faiss"
    PINECONE = "pinecone"


class LLMProvider(StrEnum):
    """Supported chat model providers (``LLM_PROVIDER``)."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    FAKE = "fake"


class EmbeddingProvider(StrEnum):
    """Supported embedding providers (``EMBEDDING_PROVIDER``)."""

    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    FAKE = "fake"


class TenantMode(StrEnum):
    """Multi-tenancy isolation strategy (``TENANT_MODE``)."""

    OFF = "off"
    METADATA = "metadata"
    PREFIX = "prefix"


class PineconeStrategy(StrEnum):
    """How domains map onto Pinecone resources (``PINECONE_STRATEGY``)."""

    NAMESPACE = "namespace"
    INDEX = "index"


class Settings(BaseSettings):
    """All runtime configuration for the service."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # -- application ------------------------------------------------------
    app_name: str = "agent-service"
    app_version: str = Field(default_factory=read_project_version)
    environment: Literal["local", "dev", "staging", "production"] = "local"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    log_json: bool = True
    enable_docs: bool | None = Field(
        default=None,
        description="Serve /docs and /redoc. Defaults to true unless ENVIRONMENT=production.",
    )
    cors_origins: str = Field(
        default="*", description="Comma separated list of allowed CORS origins."
    )

    # -- knowledge base domains -------------------------------------------
    kb_domains: str = Field(
        default="hr,finance,engineering,it,company",
        description="Comma separated business domains. Adding one is config-only.",
    )
    kb_fallback_domain: str = Field(
        default="company",
        description="Domain always eligible for general questions when readable.",
    )

    # -- vector store ------------------------------------------------------
    vector_store: VectorStoreType = VectorStoreType.CHROMA
    collection_prefix: str = "kb"

    chroma_persist_dir: Path = Path("./data/chroma")
    chroma_host: str | None = None
    chroma_port: int = 8001
    chroma_ssl: bool = False

    faiss_index_path: Path = Path("./data/faiss")

    pinecone_api_key: str | None = None
    pinecone_index: str | None = None
    pinecone_namespace: str = ""
    pinecone_strategy: PineconeStrategy = PineconeStrategy.NAMESPACE
    pinecone_auto_create: bool = False
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    pinecone_metric: str = "cosine"

    # -- providers ---------------------------------------------------------
    llm_provider: LLMProvider = LLMProvider.OPENAI
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    anthropic_api_key: str | None = None

    embedding_provider: EmbeddingProvider = EmbeddingProvider.OPENAI
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int | None = Field(
        default=None, description="Override / expected dimension. Validated against the store."
    )
    huggingface_device: str = "cpu"

    # -- ingestion / retrieval --------------------------------------------
    chunk_size: int = 1000
    chunk_overlap: int = 150
    embed_batch_size: int = 64
    max_upload_bytes: int = 25 * 1024 * 1024
    retrieval_k: int = 5
    rrf_k: int = 60
    background_ingest_threshold: int = 2

    # -- agent -------------------------------------------------------------
    agent_max_tool_iterations: int = 4
    agent_checkpointer: Literal["memory", "none"] = "memory"
    agent_system_prompt: str = (
        "You are the NexusOne ERP assistant. Answer strictly from the provided context. "
        "Cite sources by their [n] marker. If the context is insufficient, say so plainly."
    )

    # -- resilience --------------------------------------------------------
    retry_attempts: int = 3
    retry_initial_seconds: float = 0.5
    retry_max_seconds: float = 8.0

    # -- security ----------------------------------------------------------
    jwt_secret_key: str = Field(
        ..., min_length=32, description="Shared HMAC secret used by the calling backend."
    )
    jwt_algorithm: str = "HS256"
    jwt_issuer: str | None = None
    jwt_audience: str | None = None
    jwt_leeway_seconds: int = 30

    tenant_mode: TenantMode = TenantMode.OFF
    default_tenant_id: str | None = None

    # -- derived -----------------------------------------------------------
    @computed_field  # type: ignore[prop-decorator]
    @property
    def domains(self) -> tuple[str, ...]:
        """Registered knowledge-base domains, normalised and de-duplicated."""
        seen: list[str] = []
        for raw in self.kb_domains.split(","):
            item = raw.strip().lower()
            if item and item not in seen:
                seen.append(item)
        return tuple(seen)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cors_origin_list(self) -> list[str]:
        """Parsed CORS origins."""
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]

    @property
    def docs_enabled(self) -> bool:
        """Whether Swagger UI / ReDoc should be served."""
        if self.enable_docs is None:
            return self.environment != "production"
        return self.enable_docs

    @model_validator(mode="after")
    def _validate(self) -> Settings:
        if not self.domains:
            raise ValueError("KB_DOMAINS must list at least one domain")
        for domain in self.domains:
            if not _DOMAIN_RE.match(domain):
                raise ValueError(
                    f"invalid domain {domain!r}: must match {_DOMAIN_RE.pattern} "
                    "(lowercase letters, digits, '-' and '_')"
                )
        if self.kb_fallback_domain and self.kb_fallback_domain not in self.domains:
            raise ValueError(
                f"KB_FALLBACK_DOMAIN={self.kb_fallback_domain!r} is not part of KB_DOMAINS"
            )
        if self.jwt_algorithm.lower() == "none":
            raise ValueError("JWT_ALGORITHM='none' is never accepted")
        if not self.jwt_algorithm.startswith("HS"):
            raise ValueError("only HMAC algorithms (HS256/HS384/HS512) are supported")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")

        if self.llm_provider is LLMProvider.OPENAI and not self.openai_api_key:
            raise ValueError("LLM_PROVIDER=openai requires OPENAI_API_KEY")
        if self.llm_provider is LLMProvider.ANTHROPIC and not self.anthropic_api_key:
            raise ValueError("LLM_PROVIDER=anthropic requires ANTHROPIC_API_KEY")
        if self.embedding_provider is EmbeddingProvider.OPENAI and not self.openai_api_key:
            raise ValueError("EMBEDDING_PROVIDER=openai requires OPENAI_API_KEY")

        if self.vector_store is VectorStoreType.PINECONE:
            missing = [
                name
                for name, value in (
                    ("PINECONE_API_KEY", self.pinecone_api_key),
                    ("PINECONE_INDEX", self.pinecone_index),
                )
                if not value
            ]
            if missing and not (
                self.pinecone_strategy is PineconeStrategy.INDEX and missing == ["PINECONE_INDEX"]
            ):
                raise ValueError(f"VECTOR_STORE=pinecone requires {', '.join(missing)} to be set")
        if (
            self.vector_store is VectorStoreType.PINECONE
            and self.tenant_mode is TenantMode.PREFIX
            and not self.pinecone_auto_create
        ):
            raise ValueError(
                "TENANT_MODE=prefix with Pinecone requires PINECONE_AUTO_CREATE=true "
                "so tenant scoped namespaces/indexes can be provisioned"
            )
        return self

    def resolve_embedding_dimension(self) -> int | None:
        """Best-effort known dimension without contacting the provider."""
        return self.embedding_dimension or KNOWN_EMBEDDING_DIMENSIONS.get(self.embedding_model)


def build_settings(**overrides: Any) -> Settings:
    """Instantiate :class:`Settings`, converting pydantic errors into config errors."""
    try:
        return Settings(**overrides)
    except ValidationError as exc:
        problems = [
            f"{'.'.join(str(part) for part in err['loc']) or '<root>'}: {err['msg']}"
            for err in exc.errors()
        ]
        raise ConfigurationError(
            "Invalid configuration - fix your environment/.env file",
            details={"problems": problems},
        ) from exc


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide cached settings instance."""
    return build_settings()
