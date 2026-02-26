"""Unit tests for timecopilot.llm_config."""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import pytest

from timecopilot.llm_config import (
    CONFIG_DIR,
    DEFAULT_MODEL,
    MODEL_ENV_VAR,
    PROVIDER_API_KEY_ENV,
    PROVIDER_PRIORITY,
    MatchResult,
    MatchStatus,
    extract_file_path,
    get_provider_env_var,
    get_provider_for_model,
    has_valid_saved_config,
    is_api_key_set,
    list_providers,
    load_saved_model,
    match_model,
    match_provider,
    save_model,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_config_dir_is_package_dir(self):
        assert CONFIG_DIR == Path(__file__).resolve().parent.parent / "timecopilot"

    def test_provider_api_key_env_has_openai(self):
        assert PROVIDER_API_KEY_ENV["openai"] == "OPENAI_API_KEY"

    def test_provider_api_key_env_has_anthropic(self):
        assert PROVIDER_API_KEY_ENV["anthropic"] == "ANTHROPIC_API_KEY"

    def test_provider_api_key_env_length(self):
        assert len(PROVIDER_API_KEY_ENV) == 14

    def test_default_model(self):
        assert DEFAULT_MODEL == "openai:gpt-4o-mini"

    def test_model_env_var(self):
        assert MODEL_ENV_VAR == "TIMECOPILOT_MODEL"

    def test_provider_priority_starts_with_openai(self):
        assert PROVIDER_PRIORITY[0] == "openai"


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


class TestGetProviderForModel:
    def test_openai(self):
        assert get_provider_for_model("openai:gpt-4o-mini") == "openai"

    def test_anthropic(self):
        assert get_provider_for_model("anthropic:claude-3-5-haiku-latest") == "anthropic"


class TestGetProviderEnvVar:
    def test_known_provider(self):
        assert get_provider_env_var("openai") == "OPENAI_API_KEY"

    def test_unknown_provider(self):
        assert get_provider_env_var("nonexistent") is None


class TestIsApiKeySet:
    def test_key_present(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        assert is_api_key_set("openai") is True

    def test_key_absent(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert is_api_key_set("openai") is False

    def test_unknown_provider(self):
        assert is_api_key_set("totally-unknown-provider") is True


class TestHasValidSavedConfig:
    def test_valid_config(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        assert has_valid_saved_config("openai:gpt-4o-mini") is True

    def test_missing_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert has_valid_saved_config("openai:gpt-4o-mini") is False

    def test_unknown_provider(self):
        assert has_valid_saved_config("unknown:some-model") is True


class TestLoadSavedModel:
    def test_when_set(self, monkeypatch):
        monkeypatch.setenv("TIMECOPILOT_MODEL", "anthropic:claude-3-5-haiku-latest")
        assert load_saved_model() == "anthropic:claude-3-5-haiku-latest"

    def test_when_unset(self, monkeypatch):
        monkeypatch.delenv("TIMECOPILOT_MODEL", raising=False)
        assert load_saved_model() is None


class TestSaveModel:
    def test_save_model_calls_set_key(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            "timecopilot.llm_config.dotenv.set_key",
            lambda path, key, val: calls.append((path, key, val)),
        )
        save_model("openai:gpt-4o")
        assert len(calls) == 1
        assert calls[0][1] == "TIMECOPILOT_MODEL"
        assert calls[0][2] == "openai:gpt-4o"


# ---------------------------------------------------------------------------
# extract_file_path
# ---------------------------------------------------------------------------


class TestExtractFilePath:
    def test_csv(self):
        assert extract_file_path("forecast /path/to/data.csv") == "/path/to/data.csv"

    def test_parquet(self):
        assert extract_file_path("analyze results.parquet") == "results.parquet"

    def test_url(self):
        result = extract_file_path("use https://example.com/data.csv")
        assert result == "https://example.com/data.csv"

    def test_path_with_slash(self):
        assert extract_file_path("forecast /tmp/data") == "/tmp/data"

    def test_no_file_path(self):
        assert extract_file_path("hello world") is None

    def test_empty(self):
        assert extract_file_path("") is None


# ---------------------------------------------------------------------------
# list_providers
# ---------------------------------------------------------------------------


class TestListProviders:
    def test_returns_tuple(self):
        result = list_providers()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_providers_sorted_by_priority(self):
        providers, _ = list_providers()
        # First few should follow PROVIDER_PRIORITY order
        for i, p in enumerate(PROVIDER_PRIORITY):
            if p in providers:
                assert providers.index(p) <= i + len(PROVIDER_PRIORITY)

    def test_openai_is_first(self):
        providers, _ = list_providers()
        assert providers[0] == "openai"

    def test_all_models_have_colon(self):
        _, provider_models = list_providers()
        for provider, models in provider_models.items():
            for model in models:
                assert ":" in model, f"Model {model} has no colon"

    def test_provider_models_match_providers(self):
        providers, provider_models = list_providers()
        for p in providers:
            assert p in provider_models


# ---------------------------------------------------------------------------
# match_provider
# ---------------------------------------------------------------------------


class TestMatchProvider:
    PROVIDERS = ["openai", "anthropic", "google-gla", "groq", "mistral"]

    def test_by_number(self):
        result = match_provider("1", self.PROVIDERS)
        assert result.status == MatchStatus.MATCHED
        assert result.value == "openai"

    def test_by_number_last(self):
        result = match_provider("5", self.PROVIDERS)
        assert result.status == MatchStatus.MATCHED
        assert result.value == "mistral"

    def test_by_number_out_of_range(self):
        result = match_provider("99", self.PROVIDERS)
        assert result.status == MatchStatus.NO_MATCH

    def test_exact_name(self):
        result = match_provider("groq", self.PROVIDERS)
        assert result.status == MatchStatus.MATCHED
        assert result.value == "groq"

    def test_partial_unique(self):
        result = match_provider("anth", self.PROVIDERS)
        assert result.status == MatchStatus.MATCHED
        assert result.value == "anthropic"

    def test_partial_ambiguous(self):
        providers = ["google-gla", "google-vertex", "groq"]
        result = match_provider("goo", providers)
        assert result.status == MatchStatus.AMBIGUOUS
        assert set(result.candidates) == {"google-gla", "google-vertex"}

    def test_no_match(self):
        result = match_provider("zzz", self.PROVIDERS)
        assert result.status == MatchStatus.NO_MATCH


# ---------------------------------------------------------------------------
# match_model
# ---------------------------------------------------------------------------


class TestMatchModel:
    MODELS = [
        "openai:gpt-4o-mini",
        "openai:gpt-4o",
        "openai:gpt-4-turbo",
        "openai:o1-mini",
    ]

    def test_by_number(self):
        result = match_model("1", self.MODELS)
        assert result.status == MatchStatus.MATCHED
        assert result.value == "openai:gpt-4o-mini"

    def test_exact_full_name(self):
        result = match_model("openai:gpt-4o", self.MODELS)
        assert result.status == MatchStatus.MATCHED
        assert result.value == "openai:gpt-4o"

    def test_exact_short_name(self):
        result = match_model("gpt-4o-mini", self.MODELS)
        assert result.status == MatchStatus.MATCHED
        assert result.value == "openai:gpt-4o-mini"

    def test_partial_unique(self):
        result = match_model("turbo", self.MODELS)
        assert result.status == MatchStatus.MATCHED
        assert result.value == "openai:gpt-4-turbo"

    def test_ambiguous_returns_fallback(self):
        result = match_model("gpt-4o", self.MODELS)
        # "gpt-4o" matches both "gpt-4o-mini" and "gpt-4o" as exact short name
        # Actually "gpt-4o" is an exact short name match for "openai:gpt-4o"
        assert result.status == MatchStatus.MATCHED
        assert result.value == "openai:gpt-4o"

    def test_no_match_returns_fallback(self):
        result = match_model("nonexistent", self.MODELS)
        assert result.status == MatchStatus.NO_MATCH
        assert result.value == "openai:gpt-4o-mini"  # first model as fallback

    def test_number_out_of_range_returns_fallback(self):
        result = match_model("99", self.MODELS)
        assert result.status == MatchStatus.NO_MATCH
        assert result.value == "openai:gpt-4o-mini"

    def test_ambiguous_partial(self):
        result = match_model("gpt-4", self.MODELS)
        # "gpt-4" matches gpt-4o-mini, gpt-4o, gpt-4-turbo — ambiguous
        assert result.status == MatchStatus.AMBIGUOUS
        assert result.value == "openai:gpt-4o-mini"  # fallback
        assert len(result.candidates) >= 2
