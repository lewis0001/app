"""LLM gateway.

Two implementations of the same interface:

* `AnthropicLLM` calls the Claude API through the official SDK with adaptive
  thinking, structured outputs (pydantic schema), server-side refusal
  fallbacks, prompt caching on the stable system prefix, and optional web
  search. It records usage so the ledger can charge LLM spend to ventures.
* `MockLLM` runs fully offline and deterministically. It fills any pydantic
  schema with plausible values derived from a hash of the prompt so the whole
  factory (tasks, reviews, signals, emotions, ventures, airlock) exercises end
  to end without a key. Rooms can register richer canned examples.

Both expose `complete(system, user, schema, ...) -> schema instance`.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import types
import typing
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Literal, TypeVar, get_args, get_origin

from pydantic import BaseModel, ValidationError

from .config import Settings, estimate_cost_usd

log = logging.getLogger("workhouse.llm")

T = TypeVar("T", bound=BaseModel)

EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")


class LLMRefusal(RuntimeError):
    """Raised when the model (and its fallbacks) declined the request."""


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cost_usd: float = 0.0
    # Per-call cost records: (label, venture_id, cost) so the engine can attribute.
    records: list[tuple[str, str | None, float]] = field(default_factory=list)

    def add(self, label: str, inp: int, out: int, cache_read: int, cost: float, venture_id: str | None = None) -> None:
        self.calls += 1
        self.input_tokens += inp
        self.output_tokens += out
        self.cache_read_tokens += cache_read
        self.cost_usd += cost
        self.records.append((label, venture_id, cost))

    def drain(self) -> list[tuple[str, str | None, float]]:
        recs, self.records = self.records, []
        return recs


class LLM:
    """Interface. Subclasses implement `_complete`."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.usage = Usage()

    def complete(
        self,
        system: str,
        user: str,
        schema: type[T],
        *,
        effort: str = "high",
        web_search: bool = False,
        label: str = "agent",
        max_tokens: int | None = None,
        model: str | None = None,
        venture_id: str | None = None,
    ) -> T:
        if effort not in EFFORT_LEVELS:
            effort = "high"
        return self._complete(system, user, schema, effort=effort, web_search=web_search, label=label, max_tokens=max_tokens, model=model, venture_id=venture_id)

    def _complete(self, system: str, user: str, schema: type[T], *, effort: str, web_search: bool, label: str, max_tokens: int | None, model: str | None = None, venture_id: str | None = None) -> T:  # pragma: no cover - abstract
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Live implementation
# --------------------------------------------------------------------------- #


class AnthropicLLM(LLM):
    """Live gateway.

    Uses `beta.messages.create` with a JSON-schema output format and parses
    the final text block itself, so usage is recorded before any parsing can
    fail, refusals and truncation are handled explicitly, and narration text
    blocks around server-tool calls (web search) do not break parsing.
    """

    FALLBACK_BETA = "server-side-fallback-2026-07-01"

    def __init__(self, settings: Settings):
        super().__init__(settings)
        import anthropic  # imported lazily so mock mode needs no SDK at import time

        self._anthropic = anthropic
        self._client = anthropic.Anthropic()
        if not (getattr(self._client, "api_key", None) or getattr(self._client, "auth_token", None)):
            raise RuntimeError("no Anthropic credential: set ANTHROPIC_API_KEY (or run `ant auth login`)")
        from anthropic.lib._parse._transform import transform_schema

        self._transform_schema = transform_schema

    def _complete(self, system: str, user: str, schema: type[T], *, effort: str, web_search: bool, label: str, max_tokens: int | None, model: str | None = None, venture_id: str | None = None) -> T:
        anthropic = self._anthropic
        from .config import FALLBACK_MODELS

        model = model or self.settings.model
        kwargs: dict[str, Any] = dict(
            model=model,
            max_tokens=max_tokens or self.settings.max_tokens,
            # Stable prefix first so the cache hits across calls sharing a system prompt.
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
            thinking={"type": "adaptive"},
            output_config={"effort": effort, "format": {"type": "json_schema", "schema": self._transform_schema(schema.model_json_schema())}},
        )
        if model in FALLBACK_MODELS:
            kwargs["fallbacks"] = "default"
            kwargs["betas"] = [self.FALLBACK_BETA]
        if web_search and self.settings.web_search:
            kwargs["tools"] = [{"type": "web_search_20260209", "name": "web_search", "max_uses": 6}]
        try:
            response = self._client.beta.messages.create(**kwargs)
        except anthropic.RateLimitError as e:  # SDK already retried
            raise RuntimeError(f"rate limited: {e}") from e
        except anthropic.APIStatusError as e:
            raise RuntimeError(f"api error {e.status_code}: {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise RuntimeError(f"connection error: {e}") from e
        except TypeError as e:  # missing credential surfaces here in the SDK
            raise RuntimeError(f"credential error: {e}") from e

        # usage first: money was spent whatever the content looks like
        usage = getattr(response, "usage", None)
        inp = int(getattr(usage, "input_tokens", 0) or 0)
        out = int(getattr(usage, "output_tokens", 0) or 0)
        cache_read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
        cache_write = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
        stu = getattr(usage, "server_tool_use", None)
        searches = int(getattr(stu, "web_search_requests", 0) or 0) if stu is not None else 0
        served_model = str(getattr(response, "model", None) or model)
        cost = estimate_cost_usd(served_model, inp, out, cache_read, cache_write, searches)
        self.usage.add(label, inp, out, cache_read, cost, venture_id=venture_id)

        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            raise LLMRefusal(f"refused ({getattr(details, 'category', None)}): {getattr(details, 'explanation', '')}")
        if response.stop_reason == "max_tokens":
            raise RuntimeError("output truncated at max_tokens; raise WORKHOUSE_MAX_TOKENS or shorten the task")
        texts = [getattr(b, "text", "") for b in response.content if getattr(b, "type", "") == "text" and getattr(b, "text", "").strip()]
        if not texts:
            raise RuntimeError("model returned no text block")
        # The JSON answer is the final text block; earlier ones are narration around tool use.
        for text in reversed(texts):
            try:
                return schema.model_validate(_extract_json(text))
            except (ValueError, ValidationError):
                continue
        raise RuntimeError("model output did not match the schema")


def _extract_json(text: str) -> Any:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if m:
        return json.loads(m.group(1))
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError("no JSON object found in model output")


# --------------------------------------------------------------------------- #
# Offline mock
# --------------------------------------------------------------------------- #


class MockLLM(LLM):
    """Deterministic stand-in for tests, demos and dry runs.

    Values are derived from a hash of (schema name, prompt), so the same
    prompt always yields the same output, and different prompts yield varied
    but plausible outputs (both approvals and rejections, both passes and
    fails at the originality gate, and so on).

    `examples` maps a schema class to a callable that receives (seed, user
    prompt) and returns an instance; rooms use this to make the demo richer.
    """

    def __init__(self, settings: Settings, examples: dict[type[BaseModel], Callable[[int, str], BaseModel]] | None = None):
        super().__init__(settings)
        self.examples: dict[type[BaseModel], Callable[[int, str], BaseModel]] = dict(examples or {})
        self.calls: list[dict[str, Any]] = []

    def _complete(self, system: str, user: str, schema: type[T], *, effort: str, web_search: bool, label: str, max_tokens: int | None, model: str | None = None, venture_id: str | None = None) -> T:
        seed = int(hashlib.sha256((schema.__name__ + "\n" + user).encode()).hexdigest()[:12], 16)
        self.calls.append({"label": label, "schema": schema.__name__, "effort": effort, "web_search": web_search, "user": user[:400]})
        # Pretend each call costs a little so budget logic is exercised.
        self.usage.add(label, 1200, 400, 0, 0.001, venture_id=venture_id)
        maker = self.examples.get(schema)
        if maker is not None:
            result = maker(seed, user)
            if isinstance(result, schema):
                return result
        return fake_instance(schema, seed, user)


def _bounds(field: Any) -> tuple[float | None, float | None]:
    lo = hi = None
    for m in getattr(field, "metadata", []) or []:
        if hasattr(m, "ge"):
            lo = float(m.ge)
        if hasattr(m, "gt"):
            lo = float(m.gt) + 1e-9
        if hasattr(m, "le"):
            hi = float(m.le)
        if hasattr(m, "lt"):
            hi = float(m.lt) - 1e-9
    return lo, hi


def _clamp_numeric(value: Any, field: Any) -> Any:
    lo, hi = _bounds(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    if lo is not None and value < lo:
        value = type(value)(lo) if isinstance(value, int) and float(lo).is_integer() else lo
    if hi is not None and value > hi:
        value = type(value)(hi) if isinstance(value, int) and float(hi).is_integer() else hi
    return value


def fake_instance(schema: type[T], seed: int, context: str = "") -> T:
    """Construct a plausible instance of any pydantic model deterministically."""
    rng = _Rng(seed)
    data = {name: _clamp_numeric(_fake_value(f.annotation, name, rng, context), f) for name, f in schema.model_fields.items() if f.is_required() or rng.chance(0.8)}
    # Required fields must always be present.
    for name, f in schema.model_fields.items():
        if f.is_required() and name not in data:
            data[name] = _clamp_numeric(_fake_value(f.annotation, name, rng, context), f)
    return schema.model_validate(data)


class _Rng:
    def __init__(self, seed: int):
        self.state = seed & 0xFFFFFFFF or 1

    def next(self) -> int:
        # xorshift32
        x = self.state
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        self.state = x & 0xFFFFFFFF
        return self.state

    def unit(self) -> float:
        return (self.next() % 10_000) / 10_000.0

    def chance(self, p: float) -> bool:
        return self.unit() < p

    def choice(self, seq: list[Any]) -> Any:
        return seq[self.next() % len(seq)]

    def randint(self, lo: int, hi: int) -> int:
        return lo + self.next() % (hi - lo + 1)


_WORDS = ["orbital", "trend", "niche", "signal", "market", "asset", "launch", "review", "ledger", "novel", "early", "edge", "operator", "channel", "proof"]


def _fake_str(name: str, rng: _Rng, context: str) -> str:
    n = name.lower()
    if n in {"id", "agent_id", "task_id", "venture_id"}:
        return f"{name}_{rng.next() % 9999}"
    if n.endswith("url") or n.endswith("urls"):
        return f"https://example.com/{rng.choice(_WORDS)}-{rng.next() % 999}"
    if n in {"verdict"}:
        return rng.choice(["approve", "revise", "reject"])
    if n in {"first_seen", "date"}:
        return f"2026-0{rng.randint(6, 9)}-{rng.randint(10, 28)}"
    words = " ".join(rng.choice(_WORDS) for _ in range(rng.randint(3, 7)))
    return f"[mock {name}] {words}"


def _fake_value(annotation: Any, name: str, rng: _Rng, context: str) -> Any:
    origin = get_origin(annotation)
    args = get_args(annotation)
    # Optional[X] / X | None
    if origin in (typing.Union, types.UnionType):
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) < len(args) and rng.chance(0.3):
            return None
        return _fake_value(non_none[0], name, rng, context)
    if origin is Literal:
        return rng.choice(list(args))
    if origin in (list, typing.List):
        inner = args[0] if args else str
        return [_fake_value(inner, name.rstrip("s"), rng, context) for _ in range(rng.randint(1, 3))]
    if origin in (dict, typing.Dict):
        key_t, val_t = (args + (str, str))[:2]
        return {str(_fake_value(key_t, name, rng, context)): _fake_value(val_t, name, rng, context) for _ in range(rng.randint(1, 3))}
    if isinstance(annotation, type):
        if issubclass(annotation, bool):
            return rng.chance(0.65)
        if issubclass(annotation, Enum):
            return rng.choice(list(annotation)).value
        if issubclass(annotation, int):
            return rng.randint(0, 9)
        if issubclass(annotation, float):
            return round(0.15 + rng.unit() * 0.8, 2)
        if issubclass(annotation, str):
            return _fake_str(name, rng, context)
        if issubclass(annotation, BaseModel):
            return fake_instance(annotation, rng.next(), context).model_dump()
    if annotation is Any:
        return _fake_str(name, rng, context)
    return _fake_str(name, rng, context)


def build_llm(settings: Settings, examples: dict[type[BaseModel], Callable[[int, str], BaseModel]] | None = None) -> LLM:
    if settings.mode == "live":
        try:
            return AnthropicLLM(settings)
        except Exception as e:  # missing SDK or credential problem
            log.warning("falling back to MockLLM: %s", e)
    return MockLLM(settings, examples)
