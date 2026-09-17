"""The Bridge: the Director. Strategy, priorities, budget, the final say on
high-stakes work, and the spot checks that keep the Heads honest.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from ..models import Task, WorkProduct
from .base import RoleSpec, Room, RoomSpec, WorkOutput

SPEC = RoomSpec(
    key="bridge",
    name="Bridge",
    purpose="Set the company's direction every few ticks: which trends to chase, which ventures get attention, what each room must do next, and what to stop doing. Decide anything the Heads escalate.",
    head=RoleSpec("Director", "ran a company into the ground once by copying the market; now allergic to consensus. Decisive, fair, terse.", ["strategy", "capital allocation", "judgement"]),
    workers=[],
    task_types=["strategy"],
)

PLAYBOOK = """- You are a veto and allocation layer, not a visionary. The measured lesson from agent-run businesses is that scaffolding (rails, ledgers, checks) moved them from loss to profit and a strategising CEO did not. The checks are the strategy; your job is to enforce them, allocate attention to whatever is closest to external evidence, and kill drift toward generic ideas early.
- Every strategy names: the one trend we bet on this cycle, the venture that gets the most attention, one directive per room, one thing to stop.
- You review high-stakes work last. Overturn approvals when warranted; the Heads' calibration depends on it."""


class Directive(BaseModel):
    room: str
    directive: str


class StrategyOutput(WorkOutput):
    bet: str = Field(description="The dated trend or opening the company bets on this cycle.")
    focus_venture: str = Field(default="", description="Which venture gets the most attention, and why.")
    directives: list[Directive]
    stop_doing: str


class Bridge(Room):
    head_task_types = {"strategy"}
    playbook = PLAYBOOK

    def __init__(self):
        super().__init__(SPEC)

    def plan(self, ctx) -> list[Task]:
        if (ctx.tick == 1 or ctx.every(2)) and not any(t.type == "strategy" for t in ctx.open_tasks(self.key)):
            return [Task(room=self.key, type="strategy", title=f"Strategy cycle {ctx.tick}", brief="Set direction for the next cycle from the trends, portfolio, ledger and the operator's assets.", created_by="system", priority=1)]
        return []

    def output_schema(self, task: Task):
        return StrategyOutput

    def on_approved(self, ctx, task: Task, work: WorkProduct) -> list[str]:
        d = work.data
        ctx.bus.pin("strategy", f"Cycle {ctx.tick}: bet on '{d.get('bet', '')[:200]}'. Focus: {d.get('focus_venture', '')[:120]}. Stop: {d.get('stop_doing', '')[:120]}", pinned_by=work.agent_id, tick=ctx.tick)
        for item in d.get("directives", []):
            try:
                dv = Directive.model_validate(item)
            except Exception:
                continue
            key = dv.room.strip().lower().replace(" ", "_")
            key = {"market": "market_bay", "marketbay": "market_bay", "ledger_room": "ledger", "the_forge": "forge"}.get(key, key)
            if key in ctx.rooms:
                ctx.bus.post_room(work.agent_id, key, f"Director's directive: {dv.directive}", tick=ctx.tick)
        return [f"strategy set: {d.get('bet', '')[:80]}"]


def all_rooms() -> list[Room]:
    from .forge import Forge
    from .ledger_room import LedgerRoom
    from .market_bay import MarketBay
    from .observatory import Observatory

    return [Bridge(), Observatory(), Forge(), MarketBay(), LedgerRoom()]
