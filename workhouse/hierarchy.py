"""Organisation chart, review routing, promotion and demotion.

Rank 0: the Director (CEO) on the Bridge.
Rank 1: one Head per room, reporting to the Director.
Rank 2: workers, reporting to their room's Head.

Review routing rules:
* A worker's work is reviewed by its Head.
* A Head's own work is reviewed by the Director.
* High-stakes work (money leaves, something is published, legal exposure)
  additionally needs the Director after the Head.
* Every Nth review (and every originality-sensitive review) is routed to a
  Head of a *different* room so a manager cannot simply wave through the
  people it depends on (anti-sycophancy, anti-collusion).
* Reviewers whose approvals keep being overturned lose review privileges for
  a while (calibration).
"""
from __future__ import annotations

from .models import Agent, Rank, Room, Task
from .store import Store

CROSS_REVIEW_EVERY = 3  # every third review of an agent is cross-room
PROMOTE_PERFORMANCE = 0.82
DEMOTE_PERFORMANCE = 0.25
PROBATION_PERFORMANCE = 0.35
MIN_TASKS_FOR_PROMOTION = 6


class Org:
    def __init__(self, store: Store):
        self.store = store

    # -- lookups -------------------------------------------------------------
    def director(self) -> Agent | None:
        return self.store.agents.first(rank=Rank.ceo)

    def head_of(self, room: str) -> Agent | None:
        r = self.store.rooms.get(room)
        if r and r.head_id:
            return self.store.agents.get(r.head_id)
        return self.store.agents.first(room=room, rank=Rank.head)

    def manager_of(self, agent: Agent) -> Agent | None:
        if agent.manager_id:
            m = self.store.agents.get(agent.manager_id)
            if m:
                return m
        if agent.rank == Rank.worker:
            return self.head_of(agent.room)
        if agent.rank == Rank.head:
            return self.director()
        return None

    def members(self, room: str) -> list[Agent]:
        return [a for a in self.store.agents.where(room=room) if a.status != "suspended"]

    def workers(self, room: str) -> list[Agent]:
        return [a for a in self.members(room) if a.rank == Rank.worker]

    # -- review routing -------------------------------------------------------
    def reviewers_for(self, author: Agent, task: Task) -> list[Agent]:
        """Ordered list of reviewers. The first reviews now; later ones only if
        the earlier verdict is approve."""
        chain: list[Agent] = []
        n_reviews = author.stats.approvals + author.stats.revisions + author.stats.rejections
        cross = (n_reviews + 1) % CROSS_REVIEW_EVERY == 0 or task.type in {"venture_pitch", "originality_review"}
        primary = self.manager_of(author)
        if cross:
            other = self._other_room_head(author.room, exclude={author.id})
            if other is not None:
                primary = other
        if primary is not None and primary.id != author.id and self._can_review(primary):
            chain.append(primary)
        if task.high_stakes:
            d = self.director()
            if d and d.id != author.id and all(d.id != r.id for r in chain):
                chain.append(d)
        if not chain:
            # Nobody suitable (e.g. the Director reviewing itself): pick any calibrated head.
            for cand in self.store.agents.where(rank=Rank.head):
                if cand.id != author.id and self._can_review(cand):
                    chain.append(cand)
                    break
        return chain

    def _other_room_head(self, room: str, exclude: set[str]) -> Agent | None:
        heads = [a for a in self.store.agents.where(rank=Rank.head) if a.room != room and a.id not in exclude and self._can_review(a)]
        if not heads:
            return None
        # Rotate deterministically by tick so different rooms cross-review over time.
        return heads[self.store.tick % len(heads)]

    @staticmethod
    def _can_review(agent: Agent) -> bool:
        return agent.status == "active" and agent.stats.reviewer_error_rate < 0.5

    # -- promotion / demotion ---------------------------------------------------
    def evaluate_standing(self, agent: Agent) -> str | None:
        """Adjust status/rank from performance. Returns an event string or None."""
        p = agent.stats.performance
        done = agent.stats.tasks_completed
        if agent.rank == Rank.worker and agent.status == "on_probation" and p >= PROBATION_PERFORMANCE + 0.15:
            agent.status = "active"
            return f"{agent.name} is off probation (performance {p:.2f})."
        if agent.status == "active" and p <= PROBATION_PERFORMANCE and done >= 3:
            agent.status = "on_probation"
            return f"{agent.name} placed on probation (performance {p:.2f})."
        if agent.status == "on_probation" and p <= DEMOTE_PERFORMANCE and done >= 5:
            if agent.rank == Rank.head:
                agent.rank = Rank.worker
                room = self.store.rooms.get(agent.room)
                if room and room.head_id == agent.id:
                    room.head_id = None
                    self.store.rooms.put(room)
                return f"{agent.name} demoted from Head of {agent.room} to worker."
            agent.status = "suspended"
            return f"{agent.name} suspended for sustained poor performance."
        if agent.rank == Rank.worker and agent.status == "active" and p >= PROMOTE_PERFORMANCE and done >= MIN_TASKS_FOR_PROMOTION:
            room = self.store.rooms.get(agent.room)
            if room and not room.head_id:
                agent.rank = Rank.head
                agent.manager_id = (self.director() or agent).id
                room.head_id = agent.id
                self.store.rooms.put(room)
                return f"{agent.name} promoted to Head of {agent.room}."
        return None

    # -- rendering ----------------------------------------------------------------
    def render_chart(self) -> str:
        lines = []
        d = self.director()
        if d:
            lines.append(f"Director: {d.name} ({d.role})")
        for room in self.store.rooms.all():
            head = self.head_of(room.key)
            lines.append(f"Room {room.name} [{room.key}] - Head: {head.name if head else 'vacant'}")
            for w in self.workers(room.key):
                lines.append(f"    - {w.name}, {w.role}{' (probation)' if w.status == 'on_probation' else ''}")
        return "\n".join(lines)
