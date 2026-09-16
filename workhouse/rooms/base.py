"""Room interface. Each room is a department with its own roles, task types,
prompts, structured output schemas and side effects.

The engine is room-agnostic: it asks every room what tasks to queue this
tick, hands claimed tasks to `perform`, routes the resulting WorkProduct
through review, and calls `on_approved` when a manager signs it off.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ..models import Agent, Task, WorkProduct

if TYPE_CHECKING:  # pragma: no cover
    from ..engine import Context


@dataclass
class RoleSpec:
    title: str
    persona: str
    skills: list[str] = field(default_factory=list)


@dataclass
class RoomSpec:
    key: str
    name: str
    purpose: str
    head: RoleSpec
    workers: list[RoleSpec]
    task_types: list[str]


class WorkOutput(BaseModel):
    """Default structured output for a piece of work. Rooms may subclass."""

    title: str = Field(description="Short title of the deliverable.")
    summary: str = Field(description="Two or three sentences a busy manager can act on.")
    content: str = Field(description="The deliverable itself in markdown. Complete, specific, usable.")
    differentiation_claim: str = Field(description="Exactly how this differs from what a default AI agent would produce from the same brief, and why that matters commercially.")
    self_assessment: float = Field(ge=0, le=1, description="Your honest estimate that a demanding manager approves this.")
    message_to_team: str = Field(default="", description="Optional: one useful message to post to your room channel (a finding, a request, a warning).")
    needs_human: str = Field(default="", description="If, and only if, this cannot progress without the human operator: what exactly they must do. Otherwise empty.")


class Room:
    spec: RoomSpec

    def __init__(self, spec: RoomSpec):
        self.spec = spec

    @property
    def key(self) -> str:
        return self.spec.key

    # -- hooks -----------------------------------------------------------------
    def plan(self, ctx: "Context") -> list[Task]:
        """Return new tasks to queue this tick (may be empty)."""
        return []

    def output_schema(self, task: Task) -> type[BaseModel]:
        return WorkOutput

    def task_prompt(self, ctx: "Context", agent: Agent, task: Task) -> str:
        """Room-specific instructions appended to the shared agent prompt."""
        return f"TASK ({task.type}): {task.title}\n{task.brief}"

    def perform(self, ctx: "Context", agent: Agent, task: Task) -> WorkProduct:
        schema = self.output_schema(task)
        system = ctx.system_prompt_for(agent)
        user = ctx.user_prompt_for(agent, task, self.task_prompt(ctx, agent, task))
        out = ctx.llm.complete(system, user, schema, effort=ctx.effort_for(agent), web_search=self.wants_web_search(task), label=f"{self.key}:{agent.name}:{task.type}")
        return self.to_work_product(ctx, agent, task, out)

    def wants_web_search(self, task: Task) -> bool:
        return False

    def to_work_product(self, ctx: "Context", agent: Agent, task: Task, out: BaseModel) -> WorkProduct:
        d: dict[str, Any] = out.model_dump()
        wp = WorkProduct(
            task_id=task.id,
            agent_id=agent.id,
            kind=task.type,
            title=str(d.get("title") or task.title)[:200],
            summary=str(d.get("summary") or "")[:2000],
            content=str(d.get("content") or ""),
            data={k: v for k, v in d.items() if k not in {"title", "summary", "content", "differentiation_claim", "self_assessment", "message_to_team", "needs_human"}},
            differentiation_claim=str(d.get("differentiation_claim") or ""),
            self_assessment=float(d.get("self_assessment") or 0.5),
            version=task.attempts,
            tick=ctx.tick,
        )
        msg = str(d.get("message_to_team") or "").strip()
        if msg:
            ctx.bus.post_room(agent.id, self.key, f"{agent.name}: {msg}", tick=ctx.tick)
        needs = str(d.get("needs_human") or "").strip()
        if needs:
            ctx.airlock.request("manual_action", f"{self.spec.name}: {task.title}", needs, venture_id=task.venture_id, task_id=task.id, requested_by=agent.id, tick=ctx.tick)
        return wp

    def on_approved(self, ctx: "Context", task: Task, work: WorkProduct) -> list[str]:
        """Side effects after approval. Returns event strings for the tick log."""
        return []

    def on_rejected(self, ctx: "Context", task: Task, work: WorkProduct) -> list[str]:
        return []
