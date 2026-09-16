"""Numeric saturation and recency probes against free, unauthenticated feeds.

Any idea whose keywords return thousands of new repositories a month, or a
category that went from zero to forty products in nine months, is already a
commodity. These probes turn that into numbers the gate can use. They are
best-effort: no network, a proxy block or a rate limit simply yields
"unknown", never an exception.
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any

from .store import Store

GITHUB_LIMIT = 50  # repos created in the last 30 days matching the keyword
HN_LIMIT = 10  # stories in the last 90 days
CROWDED_GITHUB = 200  # beyond this the space is a commodity whatever the pitch says
TIMEOUT = 8.0
CACHE_HOURS = 24


def _get(url: str) -> dict[str, Any] | None:
    req = urllib.request.Request(url, headers={"User-Agent": "orbital-workhouse/0.1", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


def github_repo_count(keyword: str, days: int = 30) -> int | None:
    since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    q = urllib.parse.quote(f'"{keyword}" created:>{since}')
    data = _get(f"https://api.github.com/search/repositories?q={q}&per_page=1")
    if not data or "total_count" not in data:
        return None
    return int(data["total_count"])


def hn_story_count(keyword: str, days: int = 90) -> int | None:
    ts = int(time.time() - days * 86400)
    q = urllib.parse.quote(keyword)
    data = _get(f"https://hn.algolia.com/api/v1/search?query={q}&tags=story&numericFilters=created_at_i>{ts}&hitsPerPage=1")
    if not data or "nbHits" not in data:
        return None
    return int(data["nbHits"])


class Saturation:
    def __init__(self, store: Store, enabled: bool = True):
        self.store = store
        self.enabled = enabled

    def measure(self, keywords: list[str]) -> dict[str, dict[str, int | None]]:
        """Return {keyword: {github_30d, hn_90d}} using a daily cache."""
        out: dict[str, dict[str, int | None]] = {}
        if not self.enabled:
            return {k: {"github_30d": None, "hn_90d": None} for k in keywords[:3]}
        cache = self.store.get_kv("saturation_cache", {}) or {}
        now = time.time()
        for kw in [k.strip() for k in keywords[:3] if k.strip()]:
            entry = cache.get(kw.lower())
            if entry and now - float(entry.get("at", 0)) < CACHE_HOURS * 3600:
                out[kw] = {"github_30d": entry.get("github_30d"), "hn_90d": entry.get("hn_90d")}
                continue
            gh, hn = github_repo_count(kw), hn_story_count(kw)
            cache[kw.lower()] = {"at": now, "github_30d": gh, "hn_90d": hn}
            out[kw] = {"github_30d": gh, "hn_90d": hn}
        self.store.set_kv("saturation_cache", cache)
        return out

    @staticmethod
    def verdict(measures: dict[str, dict[str, int | None]]) -> tuple[str, float | None]:
        """Human-readable summary and a crowding floor (None if unknown)."""
        if not measures or all(v.get("github_30d") is None and v.get("hn_90d") is None for v in measures.values()):
            return "saturation probes unavailable (offline or blocked); judge crowding from what you know", None
        lines = []
        floor = 0.0
        for kw, v in measures.items():
            gh, hn = v.get("github_30d"), v.get("hn_90d")
            flag = ""
            if gh is not None and gh > CROWDED_GITHUB:
                flag = " COMMODITY"
                floor = max(floor, 0.85)
            elif (gh is not None and gh > GITHUB_LIMIT) or (hn is not None and hn > HN_LIMIT):
                flag = " crowded"
                floor = max(floor, 0.6)
            lines.append(f"'{kw}': {gh if gh is not None else '?'} new GitHub repos in 30 days, {hn if hn is not None else '?'} HN stories in 90 days{flag}")
        return "; ".join(lines), floor
