"""Runtime configuration, read from environment variables.

Nothing here talks to the network. `Settings.mode` decides whether the LLM
gateway uses the real Anthropic API ("live") or the deterministic offline
mock ("mock"). Mock mode is the default whenever no credential is present so
the whole factory can run, be tested and be demoed without spending money.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass
class Settings:
    """All tunables in one place. Construct with `Settings.from_env()`."""

    # "live" uses the Anthropic API, "mock" runs fully offline.
    mode: str = "mock"
    model: str = "claude-opus-5"
    # Optional: a different model for reviewers and the gate, so judges do not share the drafter's blind spots.
    review_model: str = ""
    # Effort floor for every LLM call; emotions can raise it (see emotions.py).
    base_effort: str = "high"
    max_tokens: int = 16000
    db_path: Path = field(default_factory=lambda: Path("workhouse.db"))
    # Wall-clock seconds between ticks when running the daemon loop. Lifecycle
    # rules are written in days; see ticks_per_day.
    tick_seconds: float = 86400.0
    # Hard ceiling on estimated LLM spend per tick (USD). The engine refuses to
    # start new LLM work in a tick once this is crossed.
    max_tick_cost_usd: float = 2.0
    # Hard ceiling on cumulative LLM spend (USD) before the engine pauses and
    # raises an Airlock request asking the human to lift the cap.
    max_total_cost_usd: float = 50.0
    # Whether agents may use the Anthropic web-search server tool.
    web_search: bool = True
    # Whether the gate queries GitHub/HN for numeric saturation (live mode only).
    saturation_probes: bool = True
    # Money the company may commit without asking the human (cents).
    autonomous_spend_limit_cents: int = 0
    # Dashboard
    host: str = "127.0.0.1"
    port: int = 8787
    # Where the operator's unique assets live (skills, audiences, accounts).
    operator_profile_path: Path = field(default_factory=lambda: Path("operator.json"))
    random_seed: int = 7

    @property
    def ticks_per_day(self) -> int:
        """All lifecycle rules are written in days and converted once, here."""
        if self.tick_seconds <= 0:
            return 1
        return max(1, int(round(86400.0 / self.tick_seconds)))

    def days_to_ticks(self, days: float) -> int:
        return max(1, int(round(days * self.ticks_per_day)))

    @classmethod
    def from_env(cls) -> "Settings":
        has_key = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
        mode = os.environ.get("WORKHOUSE_MODE") or ("live" if has_key else "mock")
        if mode not in {"live", "mock"}:
            mode = "mock"
        # Offline: one tick is one simulated day. Live: ten minutes, mostly idle.
        default_tick = 86400.0 if mode == "mock" else 600.0
        return cls(
            mode=mode,
            model=validate_model(os.environ.get("WORKHOUSE_MODEL", "claude-opus-5"), "WORKHOUSE_MODEL"),
            review_model=validate_model(os.environ.get("WORKHOUSE_REVIEW_MODEL", ""), "WORKHOUSE_REVIEW_MODEL"),
            base_effort=os.environ.get("WORKHOUSE_EFFORT", "high"),
            max_tokens=_env_int("WORKHOUSE_MAX_TOKENS", 16000),
            db_path=Path(os.environ.get("WORKHOUSE_DB", "workhouse.db")),
            tick_seconds=_env_float("WORKHOUSE_TICK_SECONDS", default_tick),
            max_tick_cost_usd=_env_float("WORKHOUSE_MAX_TICK_COST_USD", 2.0),
            max_total_cost_usd=_env_float("WORKHOUSE_MAX_TOTAL_COST_USD", 50.0),
            web_search=_env_bool("WORKHOUSE_WEB_SEARCH", True),
            saturation_probes=_env_bool("WORKHOUSE_SATURATION_PROBES", True),
            autonomous_spend_limit_cents=_env_int("WORKHOUSE_AUTONOMOUS_SPEND_CENTS", 0),
            host=os.environ.get("WORKHOUSE_HOST", "127.0.0.1"),
            port=_env_int("WORKHOUSE_PORT", 8787),
            operator_profile_path=Path(os.environ.get("WORKHOUSE_OPERATOR_PROFILE", "operator.json")),
            random_seed=_env_int("WORKHOUSE_SEED", 7),
        )


# Models this gateway knows how to call (adaptive thinking, effort, structured
# outputs, the 2026-02-09 web-search tool). Others are rejected at startup.
SUPPORTED_MODELS = {"claude-opus-5", "claude-sonnet-5", "claude-opus-4-8", "claude-opus-4-7", "claude-fable-5-1", "claude-fable-5"}
# Models that accept the server-side refusal fallback parameter.
FALLBACK_MODELS = {"claude-opus-5", "claude-fable-5-1", "claude-fable-5"}


def validate_model(model: str, what: str) -> str:
    if model and model not in SUPPORTED_MODELS:
        raise ValueError(f"{what} '{model}' is not supported; choose one of {sorted(SUPPORTED_MODELS)}")
    return model


# Approximate first-party list prices (USD per million tokens) used to estimate
# spend. Kept here so the ledger can charge LLM usage to ventures.
MODEL_PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-fable-5-1": (10.00, 50.00),
}


WEB_SEARCH_USD_PER_REQUEST = 0.01


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int, cache_read_tokens: int = 0, cache_write_tokens: int = 0, web_searches: int = 0) -> float:
    inp, out = MODEL_PRICES_PER_MTOK.get(model, (5.00, 25.00))
    # Cache reads are billed at roughly a tenth of the input price, writes at 1.25x.
    tokens = input_tokens * inp + cache_read_tokens * inp * 0.1 + cache_write_tokens * inp * 1.25 + output_tokens * out
    return tokens / 1_000_000 + web_searches * WEB_SEARCH_USD_PER_REQUEST
