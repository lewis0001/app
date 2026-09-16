"""Venture lifecycle and the rules that keep the portfolio honest.

idea -> gated (passed originality gate) -> validating (cheapest proof of
demand) -> building -> launched -> earning -> scaling, or killed at any point.

Kill rules are mechanical so no agent can talk its way out of them:
* Not launched within LAUNCH_DEADLINE_TICKS of being gated: killed.
* Launched but no revenue within REVENUE_DEADLINE_TICKS: killed.
* Earning but revenue falls below MIN_TICK_REVENUE for STALE_TICKS: killed.
* Cost exceeds budget: frozen until the Ledger head or the human acts.
Killing a venture sends a negative signal to its owner scaled by how much
was spent, and a smaller one to the Head that approved it.
"""
from __future__ import annotations

from .models import Signal, SignalKind, Venture, VentureStage
from .store import Store

LAUNCH_DEADLINE_TICKS = 12
REVENUE_DEADLINE_TICKS = 20
STALE_TICKS = 15
MAX_ACTIVE_VENTURES = 4
STAGE_ORDER = [VentureStage.idea, VentureStage.gated, VentureStage.validating, VentureStage.building, VentureStage.launched, VentureStage.earning, VentureStage.scaling]


class Portfolio:
    def __init__(self, store: Store):
        self.store = store

    def active(self) -> list[Venture]:
        return [v for v in self.store.ventures.all() if v.stage not in (VentureStage.killed, VentureStage.idea)]

    def by_stage(self, stage: VentureStage) -> list[Venture]:
        return self.store.ventures.where(stage=stage)

    def has_capacity(self) -> bool:
        return len(self.active()) < MAX_ACTIVE_VENTURES

    def advance(self, v: Venture, to: VentureStage, *, tick: int, note: str = "") -> Venture:
        if to == VentureStage.killed:
            return self.kill(v, note or "advanced to killed", tick=tick)
        v.stage = to
        v.tick_updated = tick
        if note:
            v.milestones.append(f"t{tick}: {note}")
        return self.store.ventures.put(v)

    def kill(self, v: Venture, reason: str, *, tick: int) -> Venture:
        v.stage = VentureStage.killed
        v.kill_reason = reason
        v.tick_updated = tick
        v.milestones.append(f"t{tick}: killed - {reason}")
        return self.store.ventures.put(v)

    def enforce_rules(self, *, tick: int) -> list[tuple[Venture, str]]:
        """Apply mechanical kill rules. Returns (venture, reason) for each kill."""
        killed: list[tuple[Venture, str]] = []
        for v in self.active():
            age = tick - v.tick_created
            reason = ""
            if v.kill_by_tick and tick > v.kill_by_tick and v.stage not in (VentureStage.earning, VentureStage.scaling):
                reason = f"pre-registered kill date (tick {v.kill_by_tick}) passed without revenue"
            elif v.stage in (VentureStage.gated, VentureStage.validating, VentureStage.building) and age > LAUNCH_DEADLINE_TICKS:
                reason = f"not launched within {LAUNCH_DEADLINE_TICKS} ticks"
            elif v.stage == VentureStage.launched and v.revenue_cents == 0 and (tick - v.tick_updated) > REVENUE_DEADLINE_TICKS:
                reason = f"no revenue within {REVENUE_DEADLINE_TICKS} ticks of launch"
            elif v.stage in (VentureStage.earning, VentureStage.scaling) and v.ticks_without_revenue > STALE_TICKS:
                reason = f"no revenue for {STALE_TICKS} ticks"
            if reason:
                self.kill(v, reason, tick=tick)
                killed.append((v, reason))
        return killed

    def tick_counters(self, *, tick: int) -> None:
        for v in self.active():
            if v.stage in (VentureStage.launched, VentureStage.earning, VentureStage.scaling):
                v.ticks_without_revenue += 1
                self.store.ventures.put(v)

    def kill_signals(self, v: Venture, reason: str, *, tick: int) -> list[Signal]:
        sigs: list[Signal] = []
        spent = min(1.0, v.cost_cents / 5000)  # $50 of spend = full-magnitude pain
        if v.owner_agent_id:
            sigs.append(Signal(agent_id=v.owner_agent_id, kind=SignalKind.negative, magnitude=round(0.4 + 0.5 * spent, 3), source="venture_killed", reason=f"{v.name}: {reason}", tick=tick))
        return sigs

    def revenue_signals(self, v: Venture, amount_cents: int, *, tick: int) -> list[Signal]:
        sigs: list[Signal] = []
        mag = min(1.0, 0.3 + amount_cents / 10000)  # $100 = full magnitude
        if v.owner_agent_id:
            sigs.append(Signal(agent_id=v.owner_agent_id, kind=SignalKind.positive, magnitude=round(mag, 3), source="revenue", reason=f"{v.name} earned ${amount_cents/100:.2f}", tick=tick))
        return sigs

    def render(self, limit: int = 8) -> str:
        vs = sorted(self.store.ventures.all(), key=lambda v: (v.stage == VentureStage.killed, -v.revenue_cents, v.tick_created))[:limit]
        if not vs:
            return "Portfolio: no ventures yet."
        lines = ["Portfolio:"]
        for v in vs:
            lines.append(f"- {v.name} [{v.stage.value}] rail={v.revenue_rail or '?'} rev=${v.revenue_cents/100:.2f} cost=${v.cost_cents/100:.2f} originality={(v.originality.score if v.originality else 0):.2f}: {v.thesis[:120]}")
        return "\n".join(lines)
