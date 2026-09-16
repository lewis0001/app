"""Build the company: rooms, the Director, Heads and workers with distinct
personas. Personas are deliberately different from one another; an org of
identical minds converges on identical ideas.
"""
from __future__ import annotations

from .config import Settings
from .engine import Engine
from .llm import LLM
from .models import Agent, Rank, Room as RoomRecord
from .rooms.bridge import all_rooms
from .store import Store

NAMES = {
    "bridge": ["Halcyon Vance"],
    "observatory": ["Ines Marlowe", "Tobias Achterberg", "Yuki Sandoval", "Priya Castellanos"],
    "forge": ["Marcus Adebayo", "Lena Sorensen", "Rafael Quint"],
    "market_bay": ["Dana Whitfield", "Kofi Brennan"],
    "ledger": ["Aurelio Nakamura", "Sigrid Bell"],
}


def build_company(settings: Settings, store: Store | None = None, llm: LLM | None = None) -> Engine:
    store = store or Store(settings.db_path)
    rooms = all_rooms()
    mock_examples = None
    if settings.mode != "live":
        from .mock_examples import EXAMPLES

        mock_examples = EXAMPLES
    engine = Engine(settings, store, rooms, llm=llm, mock_examples=mock_examples)

    bridge = next(r for r in rooms if r.key == "bridge")
    director = Agent(name=NAMES["bridge"][0], role=bridge.spec.head.title, room="bridge", rank=Rank.ceo, persona=bridge.spec.head.persona, skills=bridge.spec.head.skills)
    records: list[RoomRecord] = []
    agents: list[Agent] = []
    for room in rooms:
        spec = room.spec
        records.append(RoomRecord(key=spec.key, name=spec.name, purpose=spec.purpose, task_types=spec.task_types))
        if spec.key == "bridge":
            continue
        names = NAMES.get(spec.key, [])
        head = Agent(name=names[0] if names else f"Head of {spec.name}", role=spec.head.title, room=spec.key, rank=Rank.head, manager_id=director.id, persona=spec.head.persona, skills=spec.head.skills)
        agents.append(head)
        for i, w in enumerate(spec.workers):
            agents.append(Agent(name=names[i + 1] if len(names) > i + 1 else f"{w.title} {i+1}", role=w.title, room=spec.key, rank=Rank.worker, manager_id=head.id, persona=w.persona, skills=w.skills))
    engine.seed(director, records, agents)
    return engine
