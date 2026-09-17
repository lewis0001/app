"""The tick loop that runs the company.

One tick:
 1. consume Airlock answers (connector keys, approvals, operator assets, caps)
 2. poll revenue rails -> ledger -> revenue signals
 3. venture bookkeeping and mechanical kill rules
 4. emotional decay
 5. planning: rooms queue tasks
 6. work: agents claim and perform tasks (parallel, budget-capped)
 7. review: managers judge submitted work; verdicts become signals
 8. standing: probation / promotion / demotion
 9. charge LLM spend to the ledger; pause if the total cap is hit
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from . import emotions
from .airlock import Airlock
from .bus import Bus
from .config import Settings
from .hierarchy import Org
from .llm import LLM, build_llm
from .models import (
    Agent,
    LedgerKind,
    Rank,
    Review,
    ReviewVerdict,
    Room as RoomRecord,
    Signal,
    SignalKind,
    Task,
    TaskStatus,
    TickLog,
    TrendSignal,
    Venture,
    VentureStage,
    WorkProduct,
)
from .mirror import Mirror
from .money import Ledger, Rails
from .operator import OperatorProfile
from .originality import Gate
from .playbook import Playbook
from .saturation import Saturation
from .prompts import system_prompt
from .review import REVIEW_SYSTEM, ReviewOutput, apply_review_outcome, build_review_prompt, overturn, reviewer_stats_note
from .rooms.base import Room
from .store import Store
from .ventures import Portfolio

log = logging.getLogger("workhouse.engine")

MAX_ATTEMPTS = 3
VERDICT_VERB = {ReviewVerdict.approve: "approved", ReviewVerdict.revise: "sent back", ReviewVerdict.reject: "rejected", ReviewVerdict.escalate: "escalated"}
DIRECTOR_PLANS_EVERY = 4  # ticks


class Context:
    """Everything a room or agent needs, bundled."""

    def __init__(self, settings: Settings, store: Store, llm: LLM, rooms: dict[str, Room]):
        self.settings = settings
        self.store = store
        self.llm = llm
        self.rooms = rooms
        self.bus = Bus(store)
        self.org = Org(store)
        self.ledger = Ledger(store)
        self.rails = Rails(store, self.ledger)
        self.airlock = Airlock(store)
        self.portfolio = Portfolio(store, settings.ticks_per_day)
        self.playbook = Playbook(store)
        self.mirror = Mirror(store)
        self.saturation = Saturation(store, enabled=(settings.mode == "live" and settings.saturation_probes))
        self.operator = OperatorProfile.load(settings.operator_profile_path)
        self.airlock_new_this_tick = 0
        self.gate = Gate(llm, lambda: self.operator.render(), lambda: self.top_trends(), model=settings.review_model or None, mirror=self.mirror, saturation=self.saturation)
        self.events: list[str] = []
        self.tick_cost_start = 0.0

    # -- tick state ---------------------------------------------------------
    @property
    def tick(self) -> int:
        return self.store.tick

    def log(self, event: str) -> None:
        self.events.append(event)
        log.info("t%d %s", self.tick, event)

    def tick_cost(self) -> float:
        return self.llm.usage.cost_usd - self.tick_cost_start

    def days_to_ticks(self, days: float) -> int:
        return self.settings.days_to_ticks(days)

    def every(self, days: float, offset: int = 0) -> bool:
        """True on the ticks where a job with this cadence (in days) should run."""
        period = self.days_to_ticks(days)
        return (self.tick + offset) % period == 0

    def budget_ok(self) -> bool:
        return self.tick_cost() < self.settings.max_tick_cost_usd and self.llm.usage.cost_usd < self.settings.max_total_cost_usd

    # -- lookups --------------------------------------------------------------
    def top_trends(self, n: int = 8) -> list[TrendSignal]:
        ts = [t for t in self.store.trends.all() if t.status in ("new", "watching", "exploiting")]
        ts.sort(key=lambda t: -t.opportunity)
        return ts[:n]

    def open_tasks(self, room: str | None = None) -> list[Task]:
        return self.store.tasks.where(lambda t: t.status in (TaskStatus.queued, TaskStatus.in_progress, TaskStatus.submitted, TaskStatus.in_review, TaskStatus.blocked) and (room is None or t.room == room))

    def task_exists(self, title: str, room: str | None = None) -> bool:
        key = title.strip().lower()
        return any(t.title.strip().lower() == key for t in self.open_tasks(room))

    def task_done(self, title: str) -> bool:
        key = title.strip().lower()
        return any(t.title.strip().lower() == key and t.status == TaskStatus.done for t in self.store.tasks.all())

    # -- prompts ----------------------------------------------------------------
    def effort_for(self, agent: Agent) -> str:
        return emotions.behaviour(agent.emotion, self.settings.base_effort).effort

    def system_prompt_for(self, agent: Agent) -> str:
        room = self.rooms.get(agent.room)
        name = room.spec.name if room else agent.room
        purpose = room.spec.purpose if room else ""
        playbook = getattr(room, "playbook", "") if room else ""
        return system_prompt(agent, name, purpose, playbook)

    def company_state(self) -> str:
        return "\n\n".join([
            self.operator.render(),
            "ORG CHART\n" + self.org.render_chart(),
            "LEDGER: " + self.ledger.render(),
            self.portfolio.render(),
            "REVENUE RAILS\n" + self.rails.render(),
            self.airlock.render(),
            "TOP TRENDS\n" + ("\n".join(f"- {t.title} (opportunity {t.opportunity:.2f}): {t.summary[:140]}" for t in self.top_trends(6)) or "(none yet)"),
        ])

    def user_prompt_for(self, agent: Agent, task: Task, room_instructions: str) -> str:
        return "\n\n".join([
            f"TICK {self.tick}.",
            emotions.render_state(agent, self.settings.base_effort),
            self.playbook.render(agent.id, agent.room),
            self.bus.render_for(agent.id, agent.room, current_tick=self.tick),
            self.company_state(),
            room_instructions,
            ("REVISION REQUESTED. Previous reviewer notes: " + " | ".join(task.revision_notes)) if task.revision_notes else "",
            (f"THE OPERATOR ANSWERED THIS TASK'S ESCALATION: {'approved' if task.inputs['human_response'].get('approved') else 'declined'}. Note: {task.inputs['human_response'].get('note') or '(none)'}") if isinstance(task.inputs.get("human_response"), dict) else "",
        ]).strip()

    # -- helpers used by rooms -------------------------------------------------
    def queue(self, task: Task) -> Task | None:
        if self.task_exists(task.title, task.room):
            return None
        task.tick_created = self.tick
        task.tick_updated = self.tick
        self.store.tasks.put(task)
        return task

    def signal(self, agent_id: str, kind: SignalKind, magnitude: float, source: str, reason: str) -> Signal | None:
        agent = self.store.agents.get(agent_id)
        if not agent:
            return None
        sig = Signal(agent_id=agent_id, kind=kind, magnitude=round(max(0.0, min(1.0, magnitude)), 3), source=source, reason=reason, tick=self.tick)
        emotions.receive(agent, sig)
        self.store.signals.put(sig)
        self.store.agents.put(agent)
        return sig


class Engine:
    def __init__(self, settings: Settings, store: Store, rooms: list[Room], llm: LLM | None = None, mock_examples: dict | None = None):
        self.settings = settings
        self.store = store
        self.llm = llm or build_llm(settings, mock_examples)
        self.rooms = {r.key: r for r in rooms}
        self.ctx = Context(settings, store, self.llm, self.rooms)
        # spend and pause state survive restarts: seed from the ledger and kv
        spent = self.ctx.ledger.totals()["llm_cost_cents"] / 100 + float(store.get_kv("llm_cost_carry_usd", 0.0) or 0.0)
        self.llm.usage.cost_usd = max(self.llm.usage.cost_usd, spent)
        self.paused_reason: str = str(store.get_kv("paused_reason", "") or "")
        self.workers = 4

    # ------------------------------------------------------------------ setup
    def seed(self, director: Agent, room_records: list[RoomRecord], agents: list[Agent]) -> None:
        """Create the org once. Safe to call again: it is a no-op if seeded."""
        if self.store.get_kv("seeded"):
            return
        with self.store.transaction():
            self.store.agents.put(director)
            for r in room_records:
                self.store.rooms.put(r)
            for a in agents:
                self.store.agents.put(a)
            for r in room_records:
                head = next((a for a in agents if a.room == r.key and a.rank == Rank.head), None)
                if head:
                    r.head_id = head.id
                r.member_ids = [a.id for a in agents if a.room == r.key]
                self.store.rooms.put(r)
            self.store.set_kv("seeded", True)
        self.ctx.bus.pin("mission", "Make money by doing what other AI agents would not think to do. Timing, operator assets, specificity, unglamorous work, contrarian evidence, distribution.", tick=0)

    # ------------------------------------------------------------------- tick
    def run_tick(self) -> TickLog:
        ctx = self.ctx
        ctx.events = []
        ctx.airlock_new_this_tick = 0
        ctx.tick_cost_start = self.llm.usage.cost_usd
        self.store.tick = self.store.tick + 1
        tl = TickLog(tick=self.store.tick)
        t0 = time.time()
        try:
            self._recover_stale()
            self._consume_airlock()
            self._poll_money()
            if not self.paused_reason:
                self._ventures()  # kill clocks do not run while the factory cannot work
            self._decay()
            if self.paused_reason:
                ctx.log(f"paused: {self.paused_reason}")
            else:
                self._mirror()
                self._plan()
                self._work()
                self._review()
            self._standing()
            self._charge_llm()
        except Exception as e:  # a tick must never kill the daemon
            log.exception("tick failed")
            ctx.log(f"ERROR: {type(e).__name__}: {e}")
        tl.events = list(ctx.events)
        tl.llm_calls = self.llm.usage.calls
        tl.llm_cost_usd = round(ctx.tick_cost(), 4)
        tl.finished_at = f"{time.time() - t0:.1f}s"
        self.store.ticks.put(tl)
        return tl

    def run(self, ticks: int | None = None, *, on_tick: Callable[[TickLog], None] | None = None) -> None:
        n = 0
        while ticks is None or n < ticks:
            tl = self.run_tick()
            n += 1
            if on_tick:
                on_tick(tl)
            if ticks is None or n < ticks:
                time.sleep(self.settings.tick_seconds)

    # --------------------------------------------------------- tick sections
    def _recover_stale(self) -> None:
        """A crash or an exception mid-tick can leave tasks in_progress or
        in_review; anything from an earlier tick goes back to the queue."""
        ctx = self.ctx
        for t in self.store.tasks.where(lambda t: t.status in (TaskStatus.in_progress, TaskStatus.in_review) and t.tick_updated < ctx.tick):
            if t.status == TaskStatus.in_progress:
                t.status = TaskStatus.queued
                t.assigned_to = None
                t.attempts = max(0, t.attempts - 1)
            else:
                t.status = TaskStatus.submitted
            t.tick_updated = ctx.tick
            self.store.tasks.put(t)
            ctx.log(f"recovered stale task: {t.title}")

    def _set_paused(self, reason: str) -> None:
        self.paused_reason = reason
        self.store.set_kv("paused_reason", reason)

    def _consume_airlock(self) -> None:
        ctx = self.ctx
        for req in ctx.airlock.unconsumed():
            resp = {k: v for k, v in req.response.items() if not k.startswith("_")}
            dismissed = req.status.value == "dismissed"
            try:
                if req.type == "connect_rail" or req.type == "provide_credential":
                    rail = resp.pop("rail", None) or req.response.get("_rail")
                    conn = ctx.rails.get(rail) if rail else None
                    if conn:
                        conn.configure(resp)
                        ctx.log(f"airlock: connector {conn.name} configured")
                        ctx.bus.pin(f"rail:{conn.name}", f"{conn.name} is connected. {conn.automated_after}", tick=ctx.tick)
                    # never keep a pasted secret in the request record once it has been applied
                    for f in req.fields:
                        if f.type == "secret" and f.name in req.response:
                            req.response[f.name] = "***"
                elif req.type == "raise_cap":
                    try:
                        new_cap = float(str(resp.get("new_cap_usd") or "").replace("$", "").replace(",", "").split()[0])
                    except (ValueError, IndexError):
                        new_cap = 0.0
                    if new_cap > self.settings.max_total_cost_usd:
                        self.settings.max_total_cost_usd = new_cap
                        self._set_paused("")
                        ctx.log(f"airlock: LLM cap raised to ${new_cap:.2f}")
                    elif self.paused_reason and not dismissed:
                        from .models import AirlockField

                        ctx.airlock.request("raise_cap", "Raise the LLM spend cap", f"'{resp.get('new_cap_usd')}' is not above the current cap ${self.settings.max_total_cost_usd:.2f}. Enter a larger number.", fields=[AirlockField(name="new_cap_usd", label="New total cap (USD)", type="number")], tick=ctx.tick, why_it_matters="No work happens while paused.", dedupe_key=f"raise_cap:{ctx.tick}")
                elif req.type == "operator_asset":
                    ctx.operator.merge(resp)
                    ctx.operator.save(self.settings.operator_profile_path)
                    ctx.log("airlock: operator profile updated")
                elif req.type == "enter_revenue":
                    cents = int(float(resp.get("amount_usd") or 0) * 100)
                    if cents > 0:
                        e = ctx.ledger.revenue(cents, venture_id=req.venture_id or resp.get("venture_id") or None, source="manual", memo=str(resp.get("memo") or "manual entry"), tick=ctx.tick)
                        ctx.log(f"airlock: manual revenue ${cents/100:.2f}")
                        self._revenue_signals(e.venture_id, cents)
                elif req.type in ("approve_spend", "approve_publish", "decision", "legal_check", "create_account", "manual_action"):
                    note = str(resp.get("note") or resp.get("answer") or "")
                    explicit = resp.get("approved", resp.get("decision"))
                    if dismissed:
                        approved = False
                    elif explicit is not None:
                        approved = str(explicit).strip().lower() in {"yes", "true", "1", "approve", "approved", "done"}
                    else:  # free-text answer: a leading refusal counts as a decline
                        approved = not note.strip().lower().startswith(("no", "reject", "decline", "don't", "do not", "stop", "cancel"))
                    task = self.store.tasks.get(req.task_id) if req.task_id else None
                    if task and task.status == TaskStatus.blocked:
                        task.inputs["human_response"] = {"approved": approved, "note": note}
                        if approved:
                            # the human's yes is the final verdict; the work is accepted as submitted
                            works = self.store.work.where(task_id=task.id)
                            author = self.store.agents.get(task.assigned_to or "")
                            if works and author:
                                self._approve(task, works[-1], author, None)
                            else:
                                task.status = TaskStatus.queued
                                self.store.tasks.put(task)
                        else:
                            task.status = TaskStatus.cancelled
                            task.tick_updated = ctx.tick
                            self.store.tasks.put(task)
                    v = self.store.ventures.get(req.venture_id) if req.venture_id else None
                    if v:
                        v.milestones.append(f"t{ctx.tick}: human {'approved' if approved else 'declined'} '{req.title}' {note}".strip())
                        self.store.ventures.put(v)
                        if req.type == "approve_spend" and approved and req.response.get("_budget_cents"):
                            v.budget_cents = int(req.response["_budget_cents"])
                            self.store.ventures.put(v)
                            ctx.ledger.record(LedgerKind.adjustment, 0, venture_id=v.id, source="human", memo=f"budget approved ${v.budget_cents/100:.2f}", tick=ctx.tick)
                        if req.type == "approve_publish" and not approved:
                            ctx.portfolio.kill(v, f"operator declined the launch: {note or 'no reason given'}", tick=ctx.tick)
                            if v.owner_agent_id:
                                ctx.playbook.pitfall(v.owner_agent_id, v.room, f"launch of '{v.name}' declined by the operator", note or "declined", "ask what the operator will put their name to before building", tick=ctx.tick)
                            ctx.log(f"venture killed: {v.name} (operator declined launch)")
                    ctx.bus.pin(f"human:{req.id}", f"Operator {'approved' if approved else 'declined'}: {req.title}. {note}".strip(), pinned_by="human", tick=ctx.tick)
                    if req.type == "decision":
                        # escalating was the right call either way; the author, not the reviewer, learns from a decline
                        if req.requested_by and req.requested_by not in ("system", "human") and not approved:
                            ctx.signal(req.requested_by, SignalKind.positive, 0.15, "calibration", f"the operator agreed with your concern on '{req.title}'")
                        if task and task.assigned_to and not approved:
                            ctx.signal(task.assigned_to, SignalKind.negative, 0.3, "human", f"the operator declined '{req.title}'")
                    elif req.requested_by and req.requested_by not in ("system", "human"):
                        ctx.signal(req.requested_by, SignalKind.positive if approved else SignalKind.negative, 0.3, "human", f"operator {'approved' if approved else 'declined'} '{req.title}'")
                    ctx.log(f"airlock: {req.type} '{req.title}' {'approved' if approved else 'declined'}")
            except Exception as e:
                ctx.log(f"airlock: failed to apply {req.id}: {e}")
            ctx.airlock.mark_consumed(req)

    def _poll_money(self) -> None:
        ctx = self.ctx
        for entry in ctx.rails.poll_all(tick=ctx.tick):
            ctx.log(f"revenue ${entry.amount_cents/100:.2f} via {entry.source}: {entry.memo}")
            self._revenue_signals(entry.venture_id, entry.amount_cents)
            ctx.bus.broadcast("system", f"Revenue: ${entry.amount_cents/100:.2f} via {entry.source} ({entry.memo}).", tick=ctx.tick)

    def _revenue_signals(self, venture_id: str | None, cents: int) -> None:
        ctx = self.ctx
        v = self.store.ventures.get(venture_id) if venture_id else None
        if v:
            if v.stage in (VentureStage.launched, VentureStage.building, VentureStage.validating, VentureStage.gated):
                ctx.portfolio.advance(v, VentureStage.earning, tick=ctx.tick, note="first revenue")
            for s in ctx.portfolio.revenue_signals(v, cents, tick=ctx.tick):
                ctx.signal(s.agent_id, s.kind, s.magnitude, s.source, s.reason)
                owner = self.store.agents.get(s.agent_id)
                if owner:
                    owner.stats.revenue_attributed_cents += cents
                    self.store.agents.put(owner)
                    ctx.playbook.strategy(owner.id, v.room, f"venture '{v.name}' earned ${cents/100:.2f}", f"ledger entry via {v.revenue_rail or 'a connector'}", "do more of the thing that produced this buyer", tick=ctx.tick)
        # everyone gets a small lift from money arriving
        for a in self.store.agents.all():
            if not v or a.id != v.owner_agent_id:
                ctx.signal(a.id, SignalKind.positive, 0.1, "revenue", f"the company earned ${cents/100:.2f}")

    def _ventures(self) -> None:
        ctx = self.ctx
        ctx.portfolio.tick_counters(tick=ctx.tick)
        for v, reason in ctx.portfolio.enforce_rules(tick=ctx.tick):
            ctx.log(f"venture killed: {v.name} ({reason})")
            for s in ctx.portfolio.kill_signals(v, reason, tick=ctx.tick):
                ctx.signal(s.agent_id, s.kind, s.magnitude, s.source, s.reason)
            if v.owner_agent_id:
                ctx.playbook.pitfall(v.owner_agent_id, v.room, f"venture '{v.name}' was killed", reason, "prove demand faster and cheaper next time; pick a trigger that is newer", tick=ctx.tick)
            ctx.bus.pin(f"killed:{v.id}", f"Venture '{v.name}' was killed: {reason}. Do not propose it again in the same form.", tick=ctx.tick)
            for t in ctx.open_tasks():
                if t.venture_id == v.id:
                    t.status = TaskStatus.cancelled
                    self.store.tasks.put(t)

    def _mirror(self) -> None:
        ctx = self.ctx
        if ctx.mirror.stale(ctx.tick, self.settings.ticks_per_day) and ctx.budget_ok():
            try:
                data = ctx.mirror.refresh(self.llm, tick=ctx.tick, model=self.settings.review_model or None)
                ctx.bus.pin("default_twin", "What our own model builds when told 'make money' (stay far from it): " + "; ".join(data["ideas"][:6]), tick=ctx.tick)
                ctx.log("mirror: Default Twin refreshed")
            except Exception as e:
                ctx.log(f"mirror failed: {e}")

    def _decay(self) -> None:
        for a in self.store.agents.all():
            emotions.tick_decay(a)
            self.store.agents.put(a)

    def _plan(self) -> None:
        ctx = self.ctx
        for room in self.rooms.values():
            try:
                for t in room.plan(ctx):
                    if ctx.queue(t):
                        ctx.log(f"queued [{room.key}] {t.title}")
            except Exception as e:
                ctx.log(f"plan failed for {room.key}: {e}")

    def _claim(self, agent: Agent) -> Task | None:
        room_tasks = [t for t in self.store.tasks.where(status=TaskStatus.queued, room=agent.room) if not t.assigned_to or t.assigned_to == agent.id]
        # Heads prefer tasks assigned to them or head-level types; workers take the rest.
        head_types = getattr(self.rooms.get(agent.room), "head_task_types", set())
        if agent.rank == Rank.worker and self.ctx.org.head_of(agent.room) is not None:
            room_tasks = [t for t in room_tasks if t.type not in head_types]
        elif agent.rank == Rank.head:
            preferred = [t for t in room_tasks if t.type in head_types or t.assigned_to == agent.id]
            if preferred:
                room_tasks = preferred
            elif ctx_workers_idle(self, agent.room):
                room_tasks = []  # let workers do the work when they are free
        room_tasks.sort(key=lambda t: (t.priority, t.tick_created))
        return room_tasks[0] if room_tasks else None

    def _work(self) -> None:
        ctx = self.ctx
        jobs: list[tuple[Agent, Task]] = []
        agents = self.store.agents.all()
        usable = {a.id for a in agents if a.status != "suspended"}
        for t in self.store.tasks.where(status=TaskStatus.queued):
            if t.assigned_to and t.assigned_to not in usable:  # assignee left or was suspended
                t.assigned_to = None
                self.store.tasks.put(t)
        if agents:  # rotate who picks first so the same worker does not hog the interesting tasks
            k = ctx.tick % len(agents)
            agents = agents[k:] + agents[:k]
        for a in agents:
            if a.status == "suspended" or a.room not in self.rooms:
                continue
            if any(t.assigned_to == a.id and t.status == TaskStatus.in_progress for t in ctx.open_tasks()):
                continue
            t = self._claim(a)
            if not t:
                continue
            t.assigned_to = a.id
            t.status = TaskStatus.in_progress
            t.attempts += 1
            t.tick_updated = ctx.tick
            self.store.tasks.put(t)
            jobs.append((a, t))

        def do(job: tuple[Agent, Task]) -> tuple[Agent, Task, WorkProduct | None, str]:
            agent, task = job
            if not ctx.budget_ok():
                return agent, task, None, "budget"
            try:
                wp = self.rooms[agent.room].perform(ctx, agent, task)
                return agent, task, wp, ""
            except Exception as e:  # keep the tick alive
                log.exception("perform failed")
                return agent, task, None, f"{type(e).__name__}: {e}"

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            results = list(pool.map(do, jobs))
        for agent, task, wp, err in results:
            task = self.store.tasks.get(task.id) or task
            if wp is None:
                if err == "budget":
                    task.attempts = max(0, task.attempts - 1)  # not the agent's fault
                    task.assigned_to = None
                    task.status = TaskStatus.queued
                elif task.attempts >= MAX_ATTEMPTS:
                    task.status = TaskStatus.cancelled
                    task.inputs["error"] = err
                    ctx.log(f"cancelled after {task.attempts} failures [{task.room}] {task.title}: {err}")
                else:
                    task.status = TaskStatus.queued
                task.tick_updated = ctx.tick
                self.store.tasks.put(task)
                if task.status == TaskStatus.queued:
                    ctx.log(f"deferred [{task.room}] {task.title}: {err}")
                continue
            self.store.work.put(wp)
            task.status = TaskStatus.submitted
            task.tick_updated = ctx.tick
            self.store.tasks.put(task)
            ctx.log(f"{agent.name} submitted [{task.type}] {wp.title}")
            # continuous divergence pressure: any output that collapses toward the Default Twin is a receipt
            twin_hits = ctx.mirror.matches(f"{wp.title} {wp.summary} {wp.differentiation_claim}", threshold=0.6)
            if twin_hits:
                ctx.signal(agent.id, SignalKind.negative, 0.3, "twin", f"'{wp.title}' collapses toward the Default Twin ({twin_hits[0][14:60]})")

    def _review(self) -> None:
        ctx = self.ctx
        submitted = self.store.tasks.where(status=TaskStatus.submitted)
        jobs = []
        for task in submitted:
            author = self.store.agents.get(task.assigned_to or "")
            works = self.store.work.where(task_id=task.id)
            if not author or not works:
                continue
            work = works[-1]
            # code-only evidence gate: missing evidence never reaches an LLM reviewer
            missing = self.rooms[task.room].precheck(task, work) if task.room in self.rooms else []
            if missing:
                if task.attempts >= MAX_ATTEMPTS:
                    task.status = TaskStatus.rejected
                    ctx.log(f"rejected by the evidence check after {task.attempts} attempts: {task.title} ({missing[0]})")
                else:
                    task.status = TaskStatus.queued
                    task.revision_notes = (task.revision_notes + [f"evidence check: {m}" for m in missing])[-6:]
                    ctx.log(f"sent back by the evidence check: {task.title} ({'; '.join(missing)[:120]})")
                task.tick_updated = ctx.tick
                self.store.tasks.put(task)
                ctx.signal(author.id, SignalKind.negative, 0.15, "evidence", f"'{work.title}' lacked required evidence: {missing[0][:80]}")
                continue
            chain = ctx.org.reviewers_for(author, task)
            if not chain:
                # nobody can review (single-agent org); auto-approve with a note
                self._approve(task, work, author, None)
                continue
            task.status = TaskStatus.in_review
            self.store.tasks.put(task)
            jobs.append((task, work, author, chain))

        def do(job):
            task, work, author, chain = job
            reviews: list[Review] = []
            queue = list(chain)
            seen: set[str] = set()
            while queue:
                reviewer = queue.pop(0)
                if reviewer.id in seen:
                    continue
                if not ctx.budget_ok():
                    return task, work, author, []  # an incomplete chain is no verdict; retry next tick
                seen.add(reviewer.id)
                try:
                    rv = self._conduct_review(task, work, author, reviewer, cross_room=(reviewer.room != author.room and reviewer.rank != Rank.ceo))
                except Exception as e:  # the tick must survive one failed review call
                    log.exception("review failed")
                    ctx.log(f"review of '{work.title}' by {reviewer.name} failed: {type(e).__name__}: {e}")
                    return task, work, author, []
                if rv.verdict == ReviewVerdict.reject and author.rank == Rank.ceo:
                    rv.verdict = ReviewVerdict.revise  # the Director's work is challenged, never rejected outright
                if rv.verdict == ReviewVerdict.escalate and author.rank == Rank.ceo:
                    # Strategy is the Director's call; the challenger's concerns travel with it instead of blocking it.
                    rv.verdict = ReviewVerdict.approve
                    rv.feedback = "Approved with concerns on record: " + rv.feedback
                reviews.append(rv)
                if rv.verdict == ReviewVerdict.escalate and reviewer.rank != Rank.ceo:
                    director = ctx.org.director()
                    if director and director.id not in seen and director.id != author.id:
                        queue = [director]  # escalations go up the chain before they reach the human
                        continue
                if rv.verdict != ReviewVerdict.approve:
                    break
            return task, work, author, reviews

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            results = list(pool.map(do, jobs))
        for task, work, author, reviews in results:
            task = self.store.tasks.get(task.id) or task
            author = self.store.agents.get(author.id) or author
            if not reviews:
                task.status = TaskStatus.submitted  # try again next tick
                self.store.tasks.put(task)
                continue
            for i, rv in enumerate(reviews):
                reviewer = self.store.agents.get(rv.reviewer_id) or author
                apply_review_outcome(self.store, rv, self.store.agents.get(author.id) or author, reviewer)
                ctx.log(f"{reviewer.name} {VERDICT_VERB[rv.verdict]} '{work.title}' (q{rv.quality:.1f} o{rv.originality:.1f} i{rv.impact:.1f})")
                if i > 0 and rv.verdict != ReviewVerdict.approve:
                    if overturn(self.store, reviews[i - 1], rv, ctx.tick):
                        ctx.log(f"{self.store.agents.get(reviews[i-1].reviewer_id).name}'s approval was overturned by {reviewer.name}")
            final = reviews[-1]
            author = self.store.agents.get(author.id) or author
            ctx.bus.send(final.reviewer_id, author.id, f"Review of '{work.title}': {final.verdict.value}. {final.feedback}", tick=ctx.tick)
            if final.verdict == ReviewVerdict.reject:
                ctx.playbook.pitfall(author.id, author.room, f"'{work.title}' ({task.type}) was rejected", final.feedback, "; ".join(final.required_changes) or "start from the operator's assets and a dated trigger", tick=ctx.tick)
            elif final.verdict == ReviewVerdict.approve and final.originality >= 0.7:
                ctx.playbook.strategy(author.id, author.room, f"'{work.title}' ({task.type}) approved with originality {final.originality:.1f}", final.feedback, tick=ctx.tick)
            if final.verdict == ReviewVerdict.approve:
                self._approve(task, work, author, final)
            elif final.verdict == ReviewVerdict.revise and task.attempts < MAX_ATTEMPTS:
                task.status = TaskStatus.queued
                task.revision_notes = (task.revision_notes + final.required_changes + [final.feedback])[-6:]
                task.tick_updated = ctx.tick
                self.store.tasks.put(task)
            elif final.verdict == ReviewVerdict.escalate:
                task.status = TaskStatus.blocked
                self.store.tasks.put(task)
                from .models import AirlockField

                ctx.airlock.request("decision", f"Escalated: {task.title}", f"{final.feedback}\n\nWork summary: {work.summary}", fields=[AirlockField(name="approved", label="Let them proceed?", type="choice", choices=["yes", "no"]), AirlockField(name="note", label="Your guidance", type="textarea", required=False)], venture_id=task.venture_id, task_id=task.id, requested_by=final.reviewer_id, tick=ctx.tick, why_it_matters="The Director escalated this beyond what agents may decide alone.")
                ctx.log(f"escalated to the Airlock: {task.title}")
            else:
                task.status = TaskStatus.rejected
                task.tick_updated = ctx.tick
                self.store.tasks.put(task)
                for ev in self.rooms[task.room].on_rejected(ctx, task, work):
                    ctx.log(ev)

    def _conduct_review(self, task: Task, work: WorkProduct, author: Agent, reviewer: Agent, *, cross_room: bool) -> Review:
        ctx = self.ctx
        note = ""
        if cross_room:
            note += "\nThis is a cross-room review: you do not depend on this author. Judge only the work."
        if emotions.behaviour(author.emotion).complacent:
            note += "\nLook for recycled approaches."
        if task.type == "venture_pitch":
            twin = ctx.mirror.twin_for_brief(self.llm, task.id, task.brief, tick=ctx.tick, model=self.settings.review_model or None)
            if twin:
                note += "\n\nWHAT A DEFAULT AGENT PITCHED FROM THE SAME BRIEF (no company context):\n" + twin + "\nScore originality as distance from this."
        prompt = build_review_prompt(task, work, reviewer, ctx.company_state(), note)
        system = ctx.system_prompt_for(reviewer) + "\n\n" + REVIEW_SYSTEM
        out = ctx.llm.complete(system, prompt, ReviewOutput, effort=ctx.effort_for(reviewer), label=f"review:{reviewer.name}", model=self.settings.review_model or None, venture_id=task.venture_id)
        verdict = out.verdict
        if task.high_stakes and verdict == ReviewVerdict.approve and reviewer.rank != Rank.ceo:
            pass  # the chain adds the Director after this approval
        return Review(
            work_product_id=work.id,
            task_id=task.id,
            reviewer_id=reviewer.id,
            author_id=author.id,
            verdict=verdict,
            quality=out.quality,
            originality=out.originality if out.differentiation_claim_holds else min(out.originality, 0.3),
            impact=out.impact,
            feedback=out.feedback,
            required_changes=out.required_changes,
            cross_room=cross_room,
            tick=ctx.tick,
        )

    def _approve(self, task: Task, work: WorkProduct, author: Agent, review: Review | None) -> None:
        ctx = self.ctx
        task.status = TaskStatus.done
        task.tick_updated = ctx.tick
        self.store.tasks.put(task)
        author = self.store.agents.get(author.id) or author
        author.stats.tasks_completed += 1
        self.store.agents.put(author)
        try:
            for ev in self.rooms[task.room].on_approved(ctx, task, work):
                ctx.log(ev)
        except Exception as e:
            log.exception("on_approved failed")
            ctx.log(f"on_approved failed for {task.title}: {e}")

    def _standing(self) -> None:
        ctx = self.ctx
        for a in self.store.agents.all():
            ev = ctx.org.evaluate_standing(a)
            if ev:
                self.store.agents.put(a)
                ctx.log(ev)
                ctx.bus.broadcast("system", ev, tick=ctx.tick)
                good = any(w in ev for w in ("promoted", "off probation", "back from mentoring"))
                ctx.signal(a.id, SignalKind.positive if good else SignalKind.negative, 0.6 if good else 0.4, "standing", ev)

    def _charge_llm(self) -> None:
        ctx = self.ctx
        records = self.llm.usage.drain()
        if records:
            by_venture: dict[str, float] = {}
            for label, venture_id, cost in records:
                key = venture_id or ""
                by_venture[key] = by_venture.get(key, 0.0) + cost
            carries: dict[str, float] = dict(self.store.get_kv("llm_cost_carry", {}) or {})
            for key, cost in by_venture.items():
                total = cost + float(carries.get(key, 0.0))
                entry = ctx.ledger.llm_cost(total, venture_id=key or None, memo=f"tick {ctx.tick}: {sum(1 for _, v, _ in records if (v or '') == key)} calls", tick=ctx.tick)
                charged = (entry.amount_cents / 100) if entry else 0.0
                carries[key] = round(total - charged, 6)
            self.store.set_kv("llm_cost_carry", carries)
            self.store.set_kv("llm_cost_carry_usd", round(sum(carries.values()), 6))
        if self.llm.usage.cost_usd >= self.settings.max_total_cost_usd and not self.paused_reason:
            self._set_paused(f"LLM spend ${self.llm.usage.cost_usd:.2f} reached the cap ${self.settings.max_total_cost_usd:.2f}")
            from .models import AirlockField

            ctx.airlock.request("raise_cap", "Raise the LLM spend cap", self.paused_reason + ". The factory is paused until you raise it.", fields=[AirlockField(name="new_cap_usd", label="New total cap (USD)", type="number")], tick=ctx.tick, why_it_matters="No work happens while paused.")
            ctx.log("paused: " + self.paused_reason)

    # ---------------------------------------------------------------- reports
    def snapshot(self) -> dict[str, Any]:
        ctx = self.ctx
        return {
            "tick": self.store.tick,
            "mode": self.settings.mode,
            "model": self.settings.model,
            "paused": self.paused_reason,
            "ledger": ctx.ledger.totals(),
            "llm": {"calls": self.llm.usage.calls, "cost_usd": round(self.llm.usage.cost_usd, 4), "cap_usd": self.settings.max_total_cost_usd},
            "agents": [a.model_dump() | {"mood": emotions.mood_label(a.emotion), "behaviour": emotions.behaviour(a.emotion, self.settings.base_effort).__dict__} for a in self.store.agents.all()],
            "rooms": [r.model_dump() | {"purpose": self.rooms[r.key].spec.purpose if r.key in self.rooms else r.purpose} for r in self.store.rooms.all()],
            "tasks": [t.model_dump() for t in self.store.tasks.all()[-80:]],
            "ventures": [v.model_dump() | {"profit_cents": v.profit_cents} for v in self.store.ventures.all()],
            "trends": [t.model_dump() | {"opportunity": t.opportunity} for t in ctx.top_trends(12)],
            "airlock": [r.model_dump() for r in ctx.airlock.open()],
            "airlock_resolved": [_mask_secrets(r) for r in self.store.airlock.where(lambda r: r.status.value != "open")][-20:],
            "board": [p.model_dump() for p in ctx.bus.board()],
            "messages": [m.model_dump() for m in self.store.messages.all()[-40:]],
            "reviews": [r.model_dump() for r in self.store.reviews.all()[-40:]],
            "signals": [s.model_dump() for s in self.store.signals.all()[-60:]],
            "ledger_entries": [e.model_dump() for e in ctx.ledger.recent(30)],
            "ticks": [t.model_dump() for t in self.store.ticks.all()[-20:]],
            "operator": ctx.operator.model_dump(),
            "default_twin": ctx.mirror.current(),
            "ticks_per_day": self.settings.ticks_per_day,
            "rails": [{"name": c.name, "configured": c.configured(), "description": c.description, "human_setup_once": c.human_setup_once, "fields": [f.model_dump() for f in c.setup_fields]} for c in ctx.rails.connectors.values()],
        }


def _mask_secrets(req) -> dict[str, Any]:
    d = req.model_dump()
    secret_names = {f.name for f in req.fields if f.type == "secret"}
    d["response"] = {k: ("***" if k in secret_names and v else v) for k, v in d.get("response", {}).items()}
    return d


def ctx_workers_idle(engine: Engine, room: str) -> bool:
    """True if some worker in the room has no task in progress."""
    busy = {t.assigned_to for t in engine.ctx.open_tasks(room) if t.status == TaskStatus.in_progress}
    return any(w.id not in busy for w in engine.ctx.org.workers(room))
