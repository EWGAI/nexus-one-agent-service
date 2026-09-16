"""Settings parsing, validation and fail-fast behaviour."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_service.config import Settings, build_settings, read_project_version
from agent_service.core.exceptions import ConfigurationError

from .conftest import SECRET, make_settings


def test_domains_are_parsed_normalised_and_deduplicated(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, kb_domains=" HR , finance,hr ,company ")
    assert settings.domains == ("hr", "finance", "company")


def test_cors_origins_are_parsed(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, cors_origins="https://a.example, https://b.example")
    assert settings.cors_origin_list == ["https://a.example", "https://b.example"]


def test_docs_are_disabled_in_production_by_default(tmp_path: Path) -> None:
    assert make_settings(tmp_path, environment="production").docs_enabled is False
    assert make_settings(tmp_path, environment="local").docs_enabled is True
    assert make_settings(tmp_path, environment="production", enable_docs=True).docs_enabled is True


def test_short_jwt_secret_is_rejected() -> None:
    with pytest.raises(ConfigurationError) as excinfo:
        build_settings(_env_file=None, jwt_secret_key="too-short")
    assert any("jwt_secret_key" in problem for problem in excinfo.value.details["problems"])


def test_missing_jwt_secret_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JWT_SECRET_KEY", raising=False)
    with pytest.raises(ConfigurationError):
        build_settings(_env_file=None)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"kb_domains": "HR Payroll"}, "invalid domain"),
        ({"kb_domains": ""}, "at least one domain"),
        ({"kb_fallback_domain": "nope"}, "KB_FALLBACK_DOMAIN"),
        ({"jwt_algorithm": "none"}, "never accepted"),
        ({"jwt_algorithm": "RS256"}, "HMAC"),
        ({"chunk_size": 100, "chunk_overlap": 200}, "CHUNK_OVERLAP"),
        ({"llm_provider": "openai"}, "OPENAI_API_KEY"),
        ({"llm_provider": "anthropic"}, "ANTHROPIC_API_KEY"),
        ({"embedding_provider": "openai"}, "OPENAI_API_KEY"),
        ({"vector_store": "pinecone"}, "PINECONE_API_KEY"),
    ],
)
def test_invalid_combinations_fail_fast(
    tmp_path: Path, overrides: dict[str, object], expected: str
) -> None:
    with pytest.raises(ConfigurationError) as excinfo:
        make_settings(tmp_path, **overrides)
    assert expected in str(excinfo.value.details["problems"])


def test_valid_provider_combination_is_accepted(tmp_path: Path) -> None:
    settings = make_settings(
        tmp_path,
        llm_provider="openai",
        embedding_provider="openai",
        openai_api_key="sk-test",
        embedding_model="text-embedding-3-small",
        embedding_dimension=None,
    )
    assert settings.resolve_embedding_dimension() == 1536


def test_unknown_embedding_model_has_no_known_dimension(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, embedding_model="custom/model", embedding_dimension=None)
    assert settings.resolve_embedding_dimension() is None


def test_version_is_readable() -> None:
    assert read_project_version().count(".") >= 1


def test_settings_type(tmp_path: Path) -> None:
    assert isinstance(make_settings(tmp_path), Settings)
    assert make_settings(tmp_path).jwt_secret_key == SECRET
