"""The Observatory: trend scouts. Their job is to find what is new before it
is obvious, and to keep the company honest about what other agents are doing.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from ..models import Task, TrendSignal, WorkProduct
from .base import RoleSpec, Room, RoomSpec, WorkOutput

BEATS = [
    ("protocols", "agent-to-agent commerce, payment protocols (x402, AP2, ACP), MCP/skill marketplaces, new APIs and platform launches"),
    ("culture", "consumer behaviour shifts, niche communities forming, cultural moments, new hobbies and formats, what people suddenly pay for"),
    ("rules", "regulation, compliance deadlines, tax and platform policy changes that create urgent, dated needs"),
    ("b2b_ai_pain", "new pains created by AI adoption inside companies: evals, agent observability, data cleaning for agents, AI compliance, vendor vetting"),
    ("physical", "physical-world and local openings: shortages, local services, hardware, robotics data collection, maker communities, events"),
]

SPEC = RoomSpec(
    key="observatory",
    name="Observatory",
    purpose="Find what is genuinely new (weeks old, not months), estimate how crowded it already is with other AI agents, and hand the Forge dated, evidenced openings.",
    head=RoleSpec("Chief Scout", "sceptical of anything trending on the front page; believes the money is one layer below the headline", ["trend analysis", "source triangulation"]),
    workers=[
        RoleSpec("Protocol Scout", "reads changelogs and RFCs for fun; excited by plumbing nobody else finds glamorous", ["developer ecosystems", "protocols"]),
        RoleSpec("Culture Scout", "lurks in small communities; notices what people complain about and what they brag about", ["communities", "consumer behaviour"]),
        RoleSpec("Rules Scout", "tracks regulation and platform policy; knows a deadline is a business model", ["regulation", "compliance"]),
    ],
    task_types=["scan_trends", "deep_dive", "slop_watch"],
)

PLAYBOOK = """- A trend is only useful to us if we can date it (when did it appear?), size the crowd (how many agents/people are already on it?), and name a buyer.
- Prefer sources one layer below the front page: changelogs, RFCs, issue trackers, niche forums, job boards, regulatory dockets, local notices.
- Report freshness (1 = weeks old), crowding (1 = every AI agent is already on it) and exploitability (1 = we could earn from it this week) honestly. A high-freshness, low-crowding, high-exploitability find is worth more than ten headlines.
- Every scan must include at least one finding a default AI agent would not surface.
- Never invent sources. If web search is unavailable, say so and reason from what you know, dated."""


class TrendItem(BaseModel):
    title: str
    summary: str = Field(description="What it is, who it affects, why it just happened.")
    source_urls: list[str] = Field(default_factory=list)
    first_seen: str = Field(default="", description="Approximate date it appeared (YYYY-MM or YYYY-MM-DD).")
    freshness: float = Field(ge=0, le=1)
    crowding: float = Field(ge=0, le=1)
    exploitability: float = Field(ge=0, le=1)
    buyer: str = Field(default="", description="Who would pay, specifically.")


class ScanOutput(WorkOutput):
    trends: list[TrendItem] = Field(description="Two to five dated findings.")


class DeepDiveOutput(WorkOutput):
    opportunity: str = Field(description="The specific opening for a one-human-plus-agents company.")
    buyers: list[str] = Field(default_factory=list, description="Named kinds of buyers and where they gather.")
    what_default_agents_do_here: str = Field(description="What the crowd of AI agents is already doing in this space.")
    unclaimed_angle: str = Field(description="The angle nobody is taking and why.")


class SlopWatchOutput(WorkOutput):
    patterns: list[str] = Field(description="Things AI agents are visibly building right now, phrased as patterns to avoid.")


class Observatory(Room):
    head_task_types = {"slop_watch"}
    playbook = PLAYBOOK

    def __init__(self):
        super().__init__(SPEC)

    def plan(self, ctx) -> list[Task]:
        tasks: list[Task] = []
        scouts = ctx.org.workers(self.key)
        open_scans = [t for t in ctx.open_tasks(self.key) if t.type == "scan_trends"]
        for i, scout in enumerate(scouts):
            if len(open_scans) + len(tasks) >= max(1, len(scouts)):
                break
            beat, desc = BEATS[(ctx.tick + i) % len(BEATS)]
            tasks.append(Task(room=self.key, type="scan_trends", title=f"Scan: {beat} (tick {ctx.tick})", brief=f"Beat: {beat}. Look for: {desc}. Return dated findings with freshness, crowding, exploitability and a buyer.", created_by="system", priority=4, inputs={"beat": beat}))
        if ctx.tick % 6 == 1 and not any(t.type == "slop_watch" for t in ctx.open_tasks(self.key)):
            tasks.append(Task(room=self.key, type="slop_watch", title=f"Slop watch (tick {ctx.tick})", brief="What are AI agents and 'AI side hustle' operators visibly building right now? List concrete patterns so the company can avoid them, and note which platforms are pushing back.", created_by="system", priority=5))
        for t in ctx.top_trends(3):
            if t.opportunity >= 0.3 and t.status == "new" and not ctx.task_exists(f"Deep dive: {t.title}"):
                tasks.append(Task(room=self.key, type="deep_dive", title=f"Deep dive: {t.title}", brief=f"Trend: {t.title}. {t.summary}\nSources: {', '.join(t.source_urls) or 'none'}\nFind the unclaimed angle for a one-human-plus-agents company and name the buyers.", created_by="system", priority=3, inputs={"trend_id": t.id}))
        return tasks

    def output_schema(self, task: Task):
        return {"scan_trends": ScanOutput, "deep_dive": DeepDiveOutput, "slop_watch": SlopWatchOutput}.get(task.type, WorkOutput)

    def wants_web_search(self, task: Task) -> bool:
        return True

    def task_prompt(self, ctx, agent, task: Task) -> str:
        return f"TASK ({task.type}): {task.title}\n{task.brief}\n\nExisting trends on record (do not duplicate; add new or update with evidence):\n" + ("\n".join(f"- {t.title}" for t in ctx.top_trends(12)) or "(none)")

    def on_approved(self, ctx, task: Task, work: WorkProduct) -> list[str]:
        events: list[str] = []
        if task.type == "scan_trends":
            existing = {t.title.strip().lower() for t in ctx.store.trends.all()}
            added = 0
            for item in work.data.get("trends", []):
                try:
                    ti = TrendItem.model_validate(item)
                except Exception:
                    continue
                if ti.title.strip().lower() in existing:
                    continue
                ts = TrendSignal(title=ti.title[:160], summary=ti.summary, source_urls=ti.source_urls, first_seen=ti.first_seen, freshness=ti.freshness, crowding=ti.crowding, exploitability=ti.exploitability, found_by=work.agent_id, tick_found=ctx.tick)
                ctx.store.trends.put(ts)
                existing.add(ti.title.strip().lower())
                added += 1
            events.append(f"observatory recorded {added} new trends from {work.title}")
            top = ctx.top_trends(1)
            if top:
                ctx.bus.pin("top_trend", f"{top[0].title} (opportunity {top[0].opportunity:.2f}): {top[0].summary[:200]}", pinned_by=work.agent_id, tick=ctx.tick)
        elif task.type == "deep_dive":
            tid = task.inputs.get("trend_id")
            t = ctx.store.trends.get(tid) if tid else None
            if t:
                t.status = "watching"
                ctx.store.trends.put(t)
            ctx.bus.pin(f"angle:{(tid or work.id)[-6:]}", f"Unclaimed angle on '{task.title[11:]}': {work.data.get('unclaimed_angle', '')[:240]}", pinned_by=work.agent_id, tick=ctx.tick)
            head = ctx.org.head_of("forge")
            if head:
                ctx.bus.send(work.agent_id, head.id, f"Deep dive ready: {work.title}. Opportunity: {work.data.get('opportunity', '')[:300]}", tick=ctx.tick)
            events.append(f"deep dive pinned: {work.title}")
        elif task.type == "slop_watch":
            pats = [str(p) for p in work.data.get("patterns", [])][:12]
            ctx.store.set_kv("slop_extra", pats)
            ctx.bus.pin("slop_watch", "What other agents are building right now (avoid): " + "; ".join(pats)[:600], pinned_by=work.agent_id, tick=ctx.tick)
            events.append(f"slop watch updated with {len(pats)} patterns")
        return events
