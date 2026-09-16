"""The Forge: builders. Pitch ventures that pass the Originality Gate, prove
demand cheaply, then build the sellable thing.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from ..models import SignalKind, Task, Venture, VentureStage, WorkProduct
from .base import RoleSpec, Room, RoomSpec, WorkOutput

SPEC = RoomSpec(
    key="forge",
    name="Forge",
    purpose="Turn dated openings into ventures: pitch one specific idea that passes the Originality Gate, prove demand for under $20, then build the sellable thing.",
    head=RoleSpec("Forge Master", "has shipped things that failed and remembers why; hates features, loves first customers", ["product", "pricing", "scoping"]),
    workers=[
        RoleSpec("Builder", "would rather hand-assemble a dataset than write a landing page; distrusts anything that scales before it sells", ["data work", "tooling", "verification"]),
        RoleSpec("Maker", "thinks in offers and packaging; asks 'who pays, how much, by when' before writing a line", ["offers", "services", "packaging"]),
    ],
    task_types=["venture_pitch", "validate_demand", "build_asset"],
)

PLAYBOOK = """- One venture per pitch. Name the customer, the dated trigger, the rail money arrives on, the cheapest demand test, the first-dollar path and the kill condition.
- The Originality Gate will predict what a default AI agent would pitch and compare. Write that prediction yourself first and pitch something else.
- Prefer ventures that use an operator asset, involve unglamorous work, or depend on timing. If none of those apply, you are probably pitching slop.
- Demand tests cost under $20 and produce a number. Building starts only after a number.
- Build the smallest thing someone will pay for this week, priced, with the exact rail. Anything that needs an account, money or publishing goes to the Airlock via needs_human."""


class PitchOutput(WorkOutput):
    name: str
    thesis: str = Field(description="One paragraph: who pays, for what, why now.")
    customer: str = Field(description="Specific, findable buyer and where they gather.")
    trigger: str = Field(description="The dated thing that changed and makes this possible now.")
    default_agent_would_pitch: str = Field(description="What a generic AI agent would pitch from the same trends.")
    why_other_agents_wont: str = Field(description="Why 1000 other agents cannot do this next week.")
    operator_asset_used: str = Field(default="", description="Which operator asset this relies on, if any.")
    revenue_rail: str = Field(description="stripe | lemonsqueezy | manual | other (name it).")
    cheapest_demand_test: str = Field(description="Under $20, produces a number, what number kills it.")
    first_dollar_path: str = Field(description="Who is contacted where with what message.")
    kill_condition: str


class ValidationOutput(WorkOutput):
    test_performed: str = Field(description="What was actually done this tick (be honest if only prepared).")
    evidence: str = Field(description="Numbers, quotes, links.")
    verdict: str = Field(description="proceed | pivot | kill")
    pivot: str = Field(default="", description="If pivot: the new shape.")


class BuildOutput(WorkOutput):
    deliverable: str = Field(description="The sellable thing itself or its complete spec (code, dataset, service scope, listing copy).")
    price_cents: int = Field(ge=0)
    revenue_rail: str
    launch_checklist: list[str] = Field(default_factory=list, description="Everything left before a buyer can pay, marked [agent] or [human].")


class Forge(Room):
    head_task_types: set[str] = set()
    playbook = PLAYBOOK

    def __init__(self):
        super().__init__(SPEC)

    def plan(self, ctx) -> list[Task]:
        tasks: list[Task] = []
        pf = ctx.portfolio
        if pf.has_capacity() and not any(t.type == "venture_pitch" for t in ctx.open_tasks(self.key)):
            tasks.append(Task(room=self.key, type="venture_pitch", title=f"Venture pitch (tick {ctx.tick})", brief="Pitch one venture that passes the Originality Gate. Use the top trends, the deep dives on the board and the operator's assets.", created_by="system", priority=3))
        for v in pf.by_stage(VentureStage.gated):
            if not ctx.task_exists(f"Validate demand: {v.name}"):
                tasks.append(Task(room=self.key, type="validate_demand", title=f"Validate demand: {v.name}", brief=f"Venture: {v.name}\nThesis: {v.thesis}\nRun the cheapest demand test: {v.next_steps[0] if v.next_steps else 'design it'}. Report a number and a verdict.", created_by="system", priority=2, venture_id=v.id, assigned_to=v.owner_agent_id))
        for v in pf.by_stage(VentureStage.building):
            if not ctx.task_exists(f"Build: {v.name}") and not ctx.task_done(f"Build: {v.name}"):
                tasks.append(Task(room=self.key, type="build_asset", title=f"Build: {v.name}", brief=f"Venture: {v.name}\nThesis: {v.thesis}\nBuild the smallest sellable thing, priced, on rail '{v.revenue_rail}'. Include the complete deliverable.", created_by="system", priority=2, venture_id=v.id, assigned_to=v.owner_agent_id))
        return tasks

    def output_schema(self, task: Task):
        return {"venture_pitch": PitchOutput, "validate_demand": ValidationOutput, "build_asset": BuildOutput}.get(task.type, WorkOutput)

    def wants_web_search(self, task: Task) -> bool:
        return task.type in {"venture_pitch", "validate_demand"}

    def task_prompt(self, ctx, agent, task: Task) -> str:
        extra = ""
        if task.type == "venture_pitch":
            from ..originality import render_checks, render_registry

            killed = [v for v in ctx.store.ventures.all() if v.stage == VentureStage.killed]
            extra = "\n\nTHE GATE WILL CHECK:\n" + render_checks() + "\n\nDO NOT PITCH (slop registry):\n" + render_registry()
            if killed:
                extra += "\n\nALREADY KILLED (do not repeat): " + "; ".join(f"{v.name} ({v.kill_reason})" for v in killed[-6:])
        return f"TASK ({task.type}): {task.title}\n{task.brief}{extra}"

    def on_approved(self, ctx, task: Task, work: WorkProduct) -> list[str]:
        events: list[str] = []
        d = work.data
        if task.type == "venture_pitch":
            text = "\n".join(f"{k}: {v}" for k, v in d.items() if isinstance(v, str))
            name = str(d.get("name") or work.title).strip()
            duplicate = next((x for x in ctx.store.ventures.all() if x.name.strip().lower() == name.lower()), None)
            if duplicate is not None:
                ctx.signal(work.agent_id, SignalKind.negative, 0.4, "gate", f"'{name}' already exists in the portfolio ({duplicate.stage.value}); pitch something new")
                ctx.bus.send("system", work.agent_id, f"'{name}' is already a venture ({duplicate.stage.value}). The portfolio is on the board; pitch something that is not on it.", tick=ctx.tick)
                return [f"duplicate pitch ignored: {name}"]
            report = ctx.gate.evaluate(name, text + "\n\n" + work.content, label=f"gate:{work.agent_id}")
            if report.passed and ctx.portfolio.has_capacity():
                v = Venture(name=name[:120], thesis=str(d.get("thesis") or work.summary), why_other_agents_wont=str(d.get("why_other_agents_wont") or ""), stage=VentureStage.gated, owner_agent_id=work.agent_id, room=self.key, revenue_rail=str(d.get("revenue_rail") or "manual").lower(), originality=report, next_steps=[str(d.get("cheapest_demand_test") or ""), str(d.get("first_dollar_path") or "")], tick_created=ctx.tick, tick_updated=ctx.tick)
                v.milestones.append(f"t{ctx.tick}: passed the Originality Gate ({report.score:.2f})")
                ctx.store.ventures.put(v)
                ctx.signal(work.agent_id, SignalKind.positive, 0.6, "gate", f"'{v.name}' passed the Originality Gate ({report.score:.2f})")
                ctx.bus.pin(f"venture:{v.id}", f"New venture '{v.name}' (gate {report.score:.2f}): {v.thesis[:200]}", pinned_by=work.agent_id, tick=ctx.tick)
                ctx.bus.broadcast(work.agent_id, f"'{v.name}' passed the gate. Why other agents won't: {v.why_other_agents_wont[:200]}", tick=ctx.tick)
                events.append(f"venture created: {v.name} (gate {report.score:.2f})")
            else:
                reason = report.verdict_reason if not report.passed else "portfolio is full"
                ctx.signal(work.agent_id, SignalKind.negative, 0.5, "gate", f"pitch '{work.title}' failed the Originality Gate: {reason[:100]}")
                ctx.bus.send("system", work.agent_id, f"Originality Gate on '{work.title}': {reason}\nDefault AI would build: {report.default_ai_would_build}", tick=ctx.tick)
                events.append(f"pitch failed the gate: {work.title} ({report.score:.2f})")
        elif task.type == "validate_demand" and task.venture_id:
            v = ctx.store.ventures.get(task.venture_id)
            if v:
                verdict = str(d.get("verdict") or "").lower()
                v.milestones.append(f"t{ctx.tick}: validation '{verdict}': {str(d.get('evidence') or '')[:160]}")
                if verdict.startswith("kill"):
                    ctx.portfolio.kill(v, "demand test failed: " + str(d.get("evidence") or "")[:120], tick=ctx.tick)
                    events.append(f"venture killed after validation: {v.name}")
                elif verdict.startswith("pivot"):
                    v.thesis = str(d.get("pivot") or v.thesis)
                    ctx.portfolio.advance(v, VentureStage.gated, tick=ctx.tick, note="pivoted")
                    events.append(f"venture pivoted: {v.name}")
                else:
                    ctx.portfolio.advance(v, VentureStage.building, tick=ctx.tick, note="demand validated")
                    events.append(f"venture validated: {v.name}")
        elif task.type == "build_asset" and task.venture_id:
            v = ctx.store.ventures.get(task.venture_id)
            if v:
                v.revenue_rail = str(d.get("revenue_rail") or v.revenue_rail).lower()
                v.milestones.append(f"t{ctx.tick}: built '{work.title}' priced ${int(d.get('price_cents') or 0)/100:.2f}")
                v.next_steps = [str(x) for x in d.get("launch_checklist", [])][:8]
                v.tick_updated = ctx.tick
                ctx.store.ventures.put(v)
                ctx.bus.send(work.agent_id, (ctx.org.head_of("market_bay") or ctx.org.director()).id, f"Built and ready for launch planning: {v.name}. Price ${int(d.get('price_cents') or 0)/100:.2f} on {v.revenue_rail}. Checklist: {'; '.join(v.next_steps)}", tick=ctx.tick)
                ctx.queue(Task(room="market_bay", type="launch_plan", title=f"Launch plan: {v.name}", brief=f"Venture: {v.name}\nThesis: {v.thesis}\nDeliverable: {str(d.get('deliverable') or '')[:1500]}\nPrice: ${int(d.get('price_cents') or 0)/100:.2f} on {v.revenue_rail}.\nPlan the launch: channels other agents ignore, first ten buyers, exact messages, what the human must approve.", created_by=work.agent_id, priority=2, venture_id=v.id, high_stakes=True, inputs={"price_cents": int(d.get("price_cents") or 0)}))
                events.append(f"built: {v.name}; launch plan queued")
        return events

    def on_rejected(self, ctx, task: Task, work: WorkProduct) -> list[str]:
        if task.type == "validate_demand" and task.venture_id:
            v = ctx.store.ventures.get(task.venture_id)
            if v and v.stage == VentureStage.gated:
                v.milestones.append(f"t{ctx.tick}: validation work rejected; retry")
                ctx.store.ventures.put(v)
        return []
