"""The Ledger Room: finance. Money in, money out, LLM spend per venture,
budget allocation, kill recommendations and rail setup.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from ..models import AirlockField, Task, VentureStage, WorkProduct
from .base import RoleSpec, Room, RoomSpec, WorkOutput

SPEC = RoomSpec(
    key="ledger",
    name="Ledger Room",
    purpose="Keep the company solvent: track every cent including LLM spend, allocate budget to ventures with evidence, recommend kills, and get revenue rails connected.",
    head=RoleSpec("Bursar", "counts LLM tokens as money because they are; the only person in the building who enjoys saying no", ["unit economics", "budgeting"]),
    workers=[
        RoleSpec("Rails Analyst", "knows which payment rail fits which product and what the human must do once", ["payments", "compliance"]),
    ],
    task_types=["pnl_review", "rail_setup", "cost_audit"],
)

PLAYBOOK = """- LLM spend is a cost of goods. A venture that burns more than it can plausibly earn is killed, however clever.
- Recommend kills with numbers. Recommend budget with a payback estimate.
- Get a real rail connected early: no venture launches on 'manual' if a connector fits.
- Never approve spending; only the operator can. Prepare the request so the operator can decide in one minute."""


class Decision(BaseModel):
    venture: str
    action: str = Field(description="keep | kill | scale | freeze")
    reason: str
    budget_cents: int = 0


class PnlOutput(WorkOutput):
    decisions: list[Decision]
    company_runway_note: str = Field(description="How long LLM spend can continue under the cap at the current burn.")


class RailOutput(WorkOutput):
    rail: str = Field(description="stripe | lemonsqueezy | manual")
    why: str
    human_steps: list[str] = Field(default_factory=list)


class LedgerRoom(Room):
    head_task_types = {"pnl_review"}
    playbook = PLAYBOOK

    def __init__(self):
        super().__init__(SPEC)

    def plan(self, ctx) -> list[Task]:
        tasks: list[Task] = []
        active = ctx.portfolio.active()
        if active and ctx.tick % 5 == 2 and not any(t.type == "pnl_review" for t in ctx.open_tasks(self.key)):
            tasks.append(Task(room=self.key, type="pnl_review", title=f"P&L review (tick {ctx.tick})", brief="Review every active venture: revenue, cost, LLM spend, age, stage. Decide keep/kill/scale/freeze with numbers. Kills are recommendations to the Director.", created_by="system", priority=2, high_stakes=True))
        if not ctx.rails.configured():
            for v in active:
                if v.stage in (VentureStage.building, VentureStage.launched) and not ctx.task_exists(f"Rail setup: {v.name}"):
                    tasks.append(Task(room=self.key, type="rail_setup", title=f"Rail setup: {v.name}", brief=f"Venture {v.name} sells on '{v.revenue_rail}'. Pick the rail that fits and prepare the operator's one-time steps. Available connectors:\n{ctx.rails.render()}", created_by="system", priority=2, venture_id=v.id))
                    break
        if ctx.tick % 9 == 4 and ctx.store.ledger.count() > 0 and not any(t.type == "cost_audit" for t in ctx.open_tasks(self.key)):
            tasks.append(Task(room=self.key, type="cost_audit", title=f"Cost audit (tick {ctx.tick})", brief="Where is LLM spend going by room and venture? Which work produced nothing sellable? Recommend cuts.", created_by="system", priority=6))
        return tasks

    def output_schema(self, task: Task):
        return {"pnl_review": PnlOutput, "rail_setup": RailOutput}.get(task.type, WorkOutput)

    def task_prompt(self, ctx, agent, task: Task) -> str:
        ledger = "\n".join(f"- t{e.tick} {e.kind.value} ${e.amount_cents/100:.2f} {e.source} {e.memo[:60]} venture={e.venture_id or '-'}" for e in ctx.ledger.recent(25))
        return f"TASK ({task.type}): {task.title}\n{task.brief}\n\nRECENT LEDGER:\n{ledger or '(empty)'}\nLLM spend so far ${ctx.llm.usage.cost_usd:.2f} of cap ${ctx.settings.max_total_cost_usd:.2f}."

    def on_approved(self, ctx, task: Task, work: WorkProduct) -> list[str]:
        events: list[str] = []
        d = work.data
        if task.type == "pnl_review":
            for item in d.get("decisions", []):
                try:
                    dec = Decision.model_validate(item)
                except Exception:
                    continue
                v = next((x for x in ctx.store.ventures.all() if x.name.strip().lower() == dec.venture.strip().lower()), None)
                if not v or v.stage == VentureStage.killed:
                    continue
                if dec.action == "kill":
                    ctx.portfolio.kill(v, f"Ledger Room, approved by the Director: {dec.reason[:120]}", tick=ctx.tick)
                    for s in ctx.portfolio.kill_signals(v, dec.reason, tick=ctx.tick):
                        ctx.signal(s.agent_id, s.kind, s.magnitude, s.source, s.reason)
                    events.append(f"venture killed by P&L review: {v.name}")
                elif dec.action == "scale" and v.stage == VentureStage.earning:
                    ctx.portfolio.advance(v, VentureStage.scaling, tick=ctx.tick, note=dec.reason[:120])
                    events.append(f"venture scaling: {v.name}")
                if dec.budget_cents and dec.budget_cents != v.budget_cents:
                    if dec.budget_cents > ctx.settings.autonomous_spend_limit_cents:
                        ctx.airlock.request("approve_spend", f"Budget ${dec.budget_cents/100:.2f} for {v.name}", dec.reason, fields=[AirlockField(name="approved", label="Approve?", type="choice", choices=["yes", "no"]), AirlockField(name="note", label="Note", type="textarea", required=False)], venture_id=v.id, requested_by=work.agent_id, tick=ctx.tick, why_it_matters="Money leaves the company only with your approval.")
                    else:
                        v.budget_cents = dec.budget_cents
                        ctx.store.ventures.put(v)
            ctx.bus.pin("runway", str(d.get("company_runway_note") or "")[:300], pinned_by=work.agent_id, tick=ctx.tick)
        elif task.type == "rail_setup" and task.venture_id:
            rail = str(d.get("rail") or "manual").lower()
            conn = ctx.rails.get(rail)
            v = ctx.store.ventures.get(task.venture_id)
            if v:
                v.revenue_rail = rail
                ctx.store.ventures.put(v)
            if conn and not conn.configured() and conn.name != "manual":
                ctx.airlock.request("connect_rail", f"Connect {conn.name}", f"{conn.human_setup_once}\n\nWhy this rail: {d.get('why', '')}\n\nAfter that: {conn.automated_after}", fields=conn.setup_fields, why_it_matters="Without a rail no venture can be paid.", venture_id=task.venture_id, requested_by=work.agent_id, tick=ctx.tick, dedupe_key=f"connect_rail:{conn.name}", meta={"rail": conn.name})
                events.append(f"airlock: asked the operator to connect {conn.name}")
        elif task.type == "cost_audit":
            ctx.bus.pin("cost_audit", work.summary[:400], pinned_by=work.agent_id, tick=ctx.tick)
        return events
