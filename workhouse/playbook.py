"""Outcome-grounded memory: the channel through which signals actually change
behaviour.

The affective-agents literature is clear that praise and blame as prompt
tone barely move output quality. What moves it is a curated record of what
was tried, what the evidence showed, and what to do instead (Reflexion,
ReasoningBank). So every strong negative signal writes a *pitfall* entry and
every strong positive signal writes a *strategy* entry, to the agent's own
playbook and to its room's playbook. Both are rendered into every prompt.
"""
from __future__ import annotations

from typing import Any

from .store import Store

MAX_ENTRIES = 12


class Playbook:
    def __init__(self, store: Store):
        self.store = store

    def _key(self, scope: str, ident: str) -> str:
        return f"playbook:{scope}:{ident}"

    def entries(self, scope: str, ident: str) -> list[dict[str, Any]]:
        return list(self.store.get_kv(self._key(scope, ident), []) or [])

    def add(self, scope: str, ident: str, kind: str, what: str, evidence: str, instead: str, *, tick: int) -> None:
        if not what.strip():
            return
        items = self.entries(scope, ident)
        entry = {"kind": kind, "what": what.strip()[:220], "evidence": evidence.strip()[:220], "instead": instead.strip()[:220], "tick": tick}
        # Collapse near-duplicates (same kind and same first 60 chars of `what`).
        items = [i for i in items if not (i.get("kind") == kind and i.get("what", "")[:60] == entry["what"][:60])]
        items.append(entry)
        self.store.set_kv(self._key(scope, ident), items[-MAX_ENTRIES:])

    def pitfall(self, agent_id: str, room: str, what: str, evidence: str, instead: str, *, tick: int) -> None:
        self.add("agent", agent_id, "pitfall", what, evidence, instead, tick=tick)
        self.add("room", room, "pitfall", what, evidence, instead, tick=tick)

    def strategy(self, agent_id: str, room: str, what: str, evidence: str, instead: str = "", *, tick: int) -> None:
        self.add("agent", agent_id, "strategy", what, evidence, instead, tick=tick)
        self.add("room", room, "strategy", what, evidence, instead, tick=tick)

    def render(self, agent_id: str, room: str) -> str:
        mine = self.entries("agent", agent_id)
        ours = [e for e in self.entries("room", room) if e not in mine]
        if not mine and not ours:
            return "PLAYBOOK: empty so far. Your first entries will come from reviews, the gate, revenue and kills."

        def fmt(e: dict[str, Any]) -> str:
            tail = f" -> instead: {e['instead']}" if e.get("instead") else ""
            return f"- [{e['kind']} t{e['tick']}] {e['what']} | evidence: {e['evidence']}{tail}"

        parts = []
        if mine:
            parts.append("YOUR PLAYBOOK (what actually happened to your own work):\n" + "\n".join(fmt(e) for e in mine[-8:]))
        if ours:
            parts.append(f"ROOM PLAYBOOK (#{room}):\n" + "\n".join(fmt(e) for e in ours[-6:]))
        return "\n".join(parts)
