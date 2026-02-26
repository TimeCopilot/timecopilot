"""Shared LLM configuration, discovery, and matching logic.

Used by both ``_cli.py`` (one-shot forecast) and ``_tui.py`` (interactive TUI).
Has **no** dependency on the forecasting core, TUI, CLI, or worker modules.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import dotenv

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONFIG_DIR: Path = Path(__file__).resolve().parent

PROVIDER_API_KEY_ENV: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google-gla": "GOOGLE_API_KEY",
    "google-vertex": "GOOGLE_CLOUD_PROJECT",
    "groq": "GROQ_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "cohere": "CO_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "grok": "GROK_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
    "huggingface": "HF_TOKEN",
    "moonshotai": "MOONSHOTAI_API_KEY",
    "heroku": "HEROKU_INFERENCE_KEY",
    "bedrock": "AWS_ACCESS_KEY_ID",
}

MODEL_ENV_VAR: str = "TIMECOPILOT_MODEL"
DEFAULT_MODEL: str = "openai:gpt-4o-mini"

PROVIDER_PRIORITY: list[str] = [
    "openai",
    "anthropic",
    "google-gla",
    "google-vertex",
    "groq",
    "mistral",
]

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class MatchStatus(Enum):
    MATCHED = "matched"
    AMBIGUOUS = "ambiguous"
    NO_MATCH = "no_match"


@dataclass
class MatchResult:
    status: MatchStatus
    value: str | None = None
    candidates: list[str] | None = None


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def load_env() -> None:
    """Load the package-level ``.env`` file."""
    dotenv.load_dotenv(CONFIG_DIR / ".env")


def save_model(model_name: str) -> None:
    """Persist the selected model to the package ``.env``."""
    dotenv.set_key(str(CONFIG_DIR / ".env"), MODEL_ENV_VAR, model_name)


def load_saved_model() -> str | None:
    """Return the previously-saved model name, or *None*."""
    return os.environ.get(MODEL_ENV_VAR)


def get_provider_for_model(model_name: str) -> str:
    """Extract the provider prefix from a qualified model name.

    >>> get_provider_for_model("openai:gpt-4o-mini")
    'openai'
    """
    return model_name.split(":")[0]


def get_provider_env_var(provider: str) -> str | None:
    """Return the environment-variable name for *provider*, or *None*."""
    return PROVIDER_API_KEY_ENV.get(provider)


def is_api_key_set(provider: str) -> bool:
    """Return *True* if the API key for *provider* is present in the env."""
    env_var = PROVIDER_API_KEY_ENV.get(provider)
    if env_var is None:
        return True  # unknown provider — let pydantic-ai handle it
    return bool(os.environ.get(env_var))


def has_valid_saved_config(saved_model: str) -> bool:
    """Return *True* if *saved_model*'s provider API key is available."""
    provider = get_provider_for_model(saved_model)
    return is_api_key_set(provider)


def save_api_key(provider: str, key: str) -> None:
    """Set the API key in the environment and persist it to ``.env``."""
    env_var = PROVIDER_API_KEY_ENV.get(provider, "")
    if not env_var:
        return
    os.environ[env_var] = key
    dotenv.set_key(str(CONFIG_DIR / ".env"), env_var, key)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def extract_file_path(user_input: str) -> str | None:
    """Scan *user_input* for a CSV/Parquet path or HTTP URL."""
    for word in user_input.split():
        if (
            word.endswith(".csv")
            or word.endswith(".parquet")
            or word.startswith("http")
            or "/" in word
            or "\\" in word
        ):
            return word
    return None


def list_providers() -> tuple[list[str], dict[str, list[str]]]:
    """Query ``pydantic_ai`` for known models, grouped by provider.

    Returns ``(sorted_providers, provider_models)`` where
    *sorted_providers* is ordered by :data:`PROVIDER_PRIORITY` then
    alphabetically, and *provider_models* maps each provider to its
    sorted model list.
    """
    from collections import defaultdict

    from pydantic_ai.models import KnownModelName
    from typing_inspection.introspection import get_literal_values

    all_models = sorted(
        n for n in get_literal_values(KnownModelName.__value__) if ":" in str(n)
    )

    providers: dict[str, list[str]] = defaultdict(list)
    for model in all_models:
        provider = str(model).split(":")[0]
        providers[provider].append(str(model))

    sorted_providers = [p for p in PROVIDER_PRIORITY if p in providers]
    sorted_providers += sorted(p for p in providers if p not in PROVIDER_PRIORITY)

    return sorted_providers, dict(providers)


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def match_provider(query: str, providers: list[str]) -> MatchResult:
    """Match *query* (number, exact name, or partial) against *providers*."""
    # Try number
    try:
        idx = int(query) - 1
        if 0 <= idx < len(providers):
            return MatchResult(MatchStatus.MATCHED, providers[idx])
        return MatchResult(MatchStatus.NO_MATCH)
    except ValueError:
        pass

    q = query.lower()
    exact = [p for p in providers if p == q]
    if exact:
        return MatchResult(MatchStatus.MATCHED, exact[0])

    partial = [p for p in providers if q in p]
    if len(partial) == 1:
        return MatchResult(MatchStatus.MATCHED, partial[0])
    if len(partial) > 1:
        return MatchResult(MatchStatus.AMBIGUOUS, candidates=partial)

    return MatchResult(MatchStatus.NO_MATCH)


def match_model(query: str, models: list[str]) -> MatchResult:
    """Match *query* against a list of fully-qualified model names.

    Supports matching by number, exact full name, exact short name
    (without provider prefix), or partial substring.  When ambiguous or
    unmatched, ``value`` is set to ``models[0]`` as a fallback so callers
    can use it directly.
    """
    fallback = models[0] if models else None

    # Try number
    try:
        idx = int(query) - 1
        if 0 <= idx < len(models):
            return MatchResult(MatchStatus.MATCHED, models[idx])
        return MatchResult(MatchStatus.NO_MATCH, value=fallback)
    except ValueError:
        pass

    q = query.lower()
    exact = [m for m in models if m.lower() == q or m.split(":", 1)[1].lower() == q]
    if exact:
        return MatchResult(MatchStatus.MATCHED, exact[0])

    partial = [m for m in models if q in m.lower()]
    if len(partial) == 1:
        return MatchResult(MatchStatus.MATCHED, partial[0])
    if len(partial) > 1:
        return MatchResult(MatchStatus.AMBIGUOUS, value=fallback, candidates=partial)

    return MatchResult(MatchStatus.NO_MATCH, value=fallback)
