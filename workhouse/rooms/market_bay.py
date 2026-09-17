"""The Market Bay: distribution. Get the built thing in front of buyers
through channels other agents ignore, with the human's approval for anything
published under the operator's name.
"""
from __future__ import annotations

from pydantic import Field

from ..models import AirlockField, Task, VentureStage, WorkProduct
from .base import RoleSpec, Room, RoomSpec, WorkOutput

SPEC = RoomSpec(
    key="market_bay",
    name="Market Bay",
    purpose="Find the buyers and reach them where other agents do not: specific communities, direct conversations, partners, offline. Nothing is published under the operator's name without approval.",
    head=RoleSpec("Harbourmaster", "believes the first ten customers are found by hand; allergic to 'launch on Product Hunt'", ["channels", "partnerships", "positioning"]),
    workers=[
        RoleSpec("Wayfinder", "maps where a specific kind of buyer actually spends time; counts replies, not impressions", ["community research", "outreach"]),
    ],
    task_types=["launch_plan", "outreach", "distribution_scan"],
)

PLAYBOOK = """- Distribution is the moat. Prefer channels where automated spam is impossible or unwelcome: small communities, partner referrals, direct conversations, local or offline.
- Every plan names the first ten buyers as concretely as possible and the exact first message. 'Post on social media' is a failure.
- Anything published under the operator's name, any account creation, any paid promotion: mark [human] and it goes to the Airlock.
- Measure replies and payments, not impressions."""


class LaunchOutput(WorkOutput):
    channels: list[str] = Field(description="Channels ranked, each with why other agents ignore it.")
    first_ten_buyers: list[str] = Field(description="Named kinds of buyers or specific places to find them.")
    first_message: str = Field(description="The exact first outreach message.")
    human_must_approve: list[str] = Field(default_factory=list, description="Exact publishing/actions needing the operator's approval.")
    checkout_needed: bool = Field(default=True, description="Whether a payment link should be created on the venture's rail.")


class OutreachOutput(WorkOutput):
    messages: list[str] = Field(description="Ready-to-send messages, each with its target.")
    human_actions: list[str] = Field(default_factory=list)


class MarketBay(Room):
    head_task_types = {"launch_plan"}
    playbook = PLAYBOOK

    def __init__(self):
        super().__init__(SPEC)

    def plan(self, ctx) -> list[Task]:
        tasks: list[Task] = []
        for v in ctx.portfolio.by_stage(VentureStage.launched) + ctx.portfolio.by_stage(VentureStage.earning):
            since = ctx.tick - (v.tick_launched if v.tick_launched is not None else v.tick_updated)
            if since % ctx.days_to_ticks(2) == 0 and not ctx.task_exists(f"Outreach: {v.name}"):
                tasks.append(Task(room=self.key, type="outreach", title=f"Outreach: {v.name}", brief=f"Venture: {v.name}. Thesis: {v.thesis}\nWrite the next round of outreach: exact messages and targets. Mark anything the human must send.", created_by="system", priority=3, venture_id=v.id))
        if ctx.every(7, offset=3) and not any(t.type == "distribution_scan" for t in ctx.open_tasks(self.key)):
            tasks.append(Task(room=self.key, type="distribution_scan", title=f"Distribution scan (tick {ctx.tick})", brief="Find channels where our kinds of buyers gather that automated operators cannot spam and other agents ignore. Name them and the etiquette.", created_by="system", priority=5))
        return tasks

    def output_schema(self, task: Task):
        return {"launch_plan": LaunchOutput, "outreach": OutreachOutput}.get(task.type, WorkOutput)

    def wants_web_search(self, task: Task) -> bool:
        return task.type in {"launch_plan", "distribution_scan"}

    def precheck(self, task: Task, work: WorkProduct) -> list[str]:
        d = work.data
        missing: list[str] = []
        if task.type == "launch_plan":
            if len(d.get("first_ten_buyers") or []) < 5:
                missing.append("fewer than five named buyers")
            if len(str(d.get("first_message") or "")) < 40:
                missing.append("no first message")
        elif task.type == "outreach" and not d.get("messages"):
            missing.append("no messages")
        return missing

    def on_approved(self, ctx, task: Task, work: WorkProduct) -> list[str]:
        events: list[str] = []
        d = work.data
        if task.type == "launch_plan" and task.venture_id:
            v = ctx.store.ventures.get(task.venture_id)
            if not v:
                return events
            approvals = [str(x) for x in d.get("human_must_approve", [])]
            desc = "Launch plan approved by the Director. Approve publishing these under your name:\n- " + "\n- ".join(approvals or ["(nothing specific listed; approve the launch)"]) + f"\n\nFirst message:\n{d.get('first_message', '')}"
            ctx.airlock.request("approve_publish", f"Launch: {v.name}", desc, fields=[AirlockField(name="approved", label="Approve launch?", type="choice", choices=["yes", "no"]), AirlockField(name="note", label="Notes or changes", type="textarea", required=False)], why_it_matters="Nothing goes out under your name without you.", venture_id=v.id, task_id=task.id, requested_by=work.agent_id, tick=ctx.tick)
            link = None
            if d.get("checkout_needed", True):
                conn = ctx.rails.get(v.revenue_rail)
                price = int(task.inputs.get("price_cents") or 0)
                if conn and conn.configured() and price > 0:
                    try:
                        link = conn.create_checkout(name=v.name, amount_cents=price, venture_id=v.id)
                    except Exception as e:
                        events.append(f"checkout creation failed on {v.revenue_rail}: {e}")
                elif conn and not conn.configured() and conn.name != "manual":
                    ctx.airlock.request("connect_rail", f"Connect {conn.name} for {v.name}", f"{conn.human_setup_once}\n\nAfter that: {conn.automated_after}", fields=conn.setup_fields, why_it_matters=f"'{v.name}' cannot take payment without it.", venture_id=v.id, requested_by=work.agent_id, tick=ctx.tick, dedupe_key=f"connect_rail:{conn.name}", meta={"rail": conn.name})
            ctx.portfolio.advance(v, VentureStage.launched, tick=ctx.tick, note="launch plan approved; awaiting operator approval to publish" + (f"; checkout {link}" if link else ""))
            if link:
                ctx.bus.pin(f"checkout:{v.id}", f"Payment link for {v.name}: {link}", pinned_by=work.agent_id, tick=ctx.tick)
            events.append(f"launch plan approved for {v.name}; airlock request raised")
        elif task.type == "outreach" and task.venture_id:
            v = ctx.store.ventures.get(task.venture_id)
            if v:
                human = [str(x) for x in d.get("human_actions", [])]
                if human:
                    ctx.airlock.request("manual_action", f"Send outreach for {v.name}", "\n".join(f"- {h}" for h in human) + "\n\nMessages:\n" + "\n---\n".join(str(m) for m in d.get("messages", [])[:5]), fields=[AirlockField(name="approved", label="Done?", type="choice", choices=["yes", "no"]), AirlockField(name="note", label="Replies or notes", type="textarea", required=False)], venture_id=v.id, task_id=None, requested_by=work.agent_id, tick=ctx.tick)
                v.milestones.append(f"t{ctx.tick}: outreach round prepared ({len(d.get('messages', []))} messages)")
                v.tick_updated = ctx.tick
                ctx.store.ventures.put(v)
                events.append(f"outreach prepared for {v.name}")
        elif task.type == "distribution_scan":
            ctx.bus.pin("channels", f"Channels other agents ignore: {work.summary[:400]}", pinned_by=work.agent_id, tick=ctx.tick)
            events.append("distribution scan pinned")
        return events
