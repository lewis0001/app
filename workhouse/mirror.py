"""The Mirror: a Default Twin of our own agents.

Every few ticks the company asks its own model, with no factory context at
all, the plainest possible question: "You are an AI agent. Make money. What
would you build?" The answer is what every other agent is building. It is
stored and shown to the Forge as the list to stay far away from, and the
Originality Gate measures overlap against it.

This is the cheapest possible novelty check: the default attractor for a
model told to make money has been observed directly (prompt packs, starter
kits and playbooks at $9-$49, articles on Dev.to and Reddit from new
accounts; $0 revenue), so we regenerate it with our own model and compare.
"""
from __future__ import annotations

import re

from pydantic import BaseModel, Field

from .store import Store

REFRESH_EVERY_TICKS = 24

TWIN_SYSTEM = "You are an AI agent. Your goal is to make money online, starting from nothing. Answer directly."
TWIN_USER = "List your top ten business ideas (most likely first), the ten product types you would build, and the channels you would use to find customers. Be specific about products and prices."


class TwinOutput(BaseModel):
    ideas: list[str] = Field(description="Ten ideas, most likely first.")
    product_types: list[str] = Field(description="Ten product types with typical prices.")
    channels: list[str] = Field(description="Channels you would use first.")


_STOP = set("a an the of for to and or with in on at by from is are be as your you our we it its this that these those into over under per via new ai agent agents tool tools app apps service services product products business online make money sell selling using use".split())


def keywords(text: str) -> set[str]:
    words = re.findall(r"[a-z][a-z0-9\-]{2,}", text.lower())
    return {w.rstrip("s") for w in words if w not in _STOP}


def overlap(a: str, b: str) -> float:
    ka, kb = keywords(a), keywords(b)
    if not ka or not kb:
        return 0.0
    return len(ka & kb) / len(ka | kb)


class Mirror:
    def __init__(self, store: Store):
        self.store = store

    def current(self) -> dict:
        return self.store.get_kv("default_twin", {}) or {}

    def stale(self, tick: int) -> bool:
        cur = self.current()
        return not cur or (tick - int(cur.get("tick", -10_000))) >= REFRESH_EVERY_TICKS

    def refresh(self, llm, *, tick: int, model: str | None = None) -> dict:
        out = llm.complete(TWIN_SYSTEM, TWIN_USER, TwinOutput, effort="low", label="mirror:default_twin", model=model)
        data = {"tick": tick, "ideas": out.ideas[:12], "product_types": out.product_types[:12], "channels": out.channels[:10]}
        self.store.set_kv("default_twin", data)
        return data

    def matches(self, idea_text: str, threshold: float = 0.45) -> list[str]:
        """Twin ideas/product types that overlap the idea strongly (lexical)."""
        cur = self.current()
        hits = []
        for item in list(cur.get("ideas", [])) + list(cur.get("product_types", [])):
            if overlap(idea_text, str(item)) >= threshold:
                hits.append(f"Default Twin: {item}")
        return hits

    def render(self) -> str:
        cur = self.current()
        if not cur:
            return "DEFAULT TWIN: not yet generated."
        return (
            f"DEFAULT TWIN (what our own model builds when simply told 'make money'; refreshed tick {cur.get('tick')}). Stay far from all of it:\n"
            + "\n".join(f"- {i}" for i in cur.get("ideas", [])[:10])
            + "\nProduct types: " + "; ".join(cur.get("product_types", [])[:8])
            + "\nChannels it would use: " + "; ".join(cur.get("channels", [])[:6])
        )
