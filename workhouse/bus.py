"""Inter-agent communication: direct messages, room channels, broadcasts and a
shared bulletin board of pinned facts.

Every agent's prompt includes (a) its unread inbox, (b) the last few posts in
its room channel, and (c) the bulletin board. The board is how a discovery in
the Observatory reaches the Forge in the same tick, and how the Ledger's kill
decision reaches everyone.
"""
from __future__ import annotations

from .models import Message, Pin
from .store import Store


class Bus:
    def __init__(self, store: Store):
        self.store = store

    # -- posting -------------------------------------------------------------
    def send(self, sender: str, to: str, content: str, *, channel: str = "general", tick: int = 0) -> Message:
        msg = Message(sender=sender, to=to, channel=channel, content=content.strip(), tick=tick)
        return self.store.messages.put(msg)

    def post_room(self, sender: str, room: str, content: str, *, tick: int = 0) -> Message:
        return self.send(sender, f"room:{room}", content, channel=room, tick=tick)

    def broadcast(self, sender: str, content: str, *, tick: int = 0) -> Message:
        return self.send(sender, "all", content, channel="all", tick=tick)

    def pin(self, key: str, content: str, *, pinned_by: str = "system", tick: int = 0) -> Pin:
        return self.store.pins.put(Pin(key=key, content=content.strip(), pinned_by=pinned_by, tick=tick))

    def unpin(self, key: str) -> None:
        self.store.pins.delete(key)

    # -- reading -------------------------------------------------------------
    def inbox(self, agent_id: str, *, since_tick: int = 0, limit: int = 8) -> list[Message]:
        msgs = [m for m in self.store.messages.since(since_tick) if m.to == agent_id]
        return msgs[-limit:]

    def room_feed(self, room: str, *, since_tick: int = 0, limit: int = 8) -> list[Message]:
        msgs = [m for m in self.store.messages.since(since_tick) if m.to in (f"room:{room}", "all")]
        return msgs[-limit:]

    def board(self, limit: int = 12) -> list[Pin]:
        pins = self.store.pins.all()
        pins.sort(key=lambda p: p.tick)
        return pins[-limit:]

    # -- rendering for prompts ------------------------------------------------
    def render_for(self, agent_id: str, room: str, *, current_tick: int, lookback: int = 3) -> str:
        since = max(0, current_tick - lookback)
        parts: list[str] = []
        board = self.board()
        if board:
            parts.append("BULLETIN BOARD (shared by every room):\n" + "\n".join(f"- [{p.key}] {p.content}" for p in board))
        inbox = self.inbox(agent_id, since_tick=since)
        if inbox:
            parts.append("MESSAGES TO YOU:\n" + "\n".join(f"- from {m.sender} (tick {m.tick}): {m.content}" for m in inbox))
        feed = self.room_feed(room, since_tick=since)
        if feed:
            parts.append(f"ROOM CHANNEL #{room}:\n" + "\n".join(f"- {m.sender} (tick {m.tick}): {m.content}" for m in feed))
        return "\n\n".join(parts) if parts else "No messages yet."
