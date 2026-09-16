"""Core data model of the workhouse.

Everything the factory knows is one of these records. They are plain pydantic
models so they serialise to JSON for SQLite and for the dashboard, and so the
LLM gateway can use several of them directly as structured-output schemas.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# Emotions
# --------------------------------------------------------------------------- #


class Emotion(BaseModel):
    """Four-dimensional affect state. See emotions.py for dynamics."""

    valence: float = 0.1  # -1 (miserable) .. +1 (elated)
    arousal: float = 0.4  # 0 (flat) .. 1 (wired)
    confidence: float = 0.6  # 0 (no faith in own judgement) .. 1 (certain)
    stress: float = 0.2  # 0 (relaxed) .. 1 (overwhelmed)


class SignalKind(str, Enum):
    positive = "positive"
    negative = "negative"


class Signal(BaseModel):
    """A performance signal delivered to one agent."""

    id: str = Field(default_factory=lambda: new_id("sig"))
    agent_id: str
    kind: SignalKind
    magnitude: float = Field(ge=0.0, le=1.0)
    source: str  # e.g. "review", "revenue", "venture_killed", "peer", "human"
    reason: str
    tick: int = 0
    deltas: dict[str, float] = Field(default_factory=dict)
    created_at: str = Field(default_factory=now_iso)


# --------------------------------------------------------------------------- #
# Organisation
# --------------------------------------------------------------------------- #


class Rank(int, Enum):
    ceo = 0
    head = 1
    worker = 2


class AgentStats(BaseModel):
    tasks_completed: int = 0
    approvals: int = 0
    revisions: int = 0
    rejections: int = 0
    reviews_given: int = 0
    revenue_attributed_cents: int = 0
    streak: int = 0  # consecutive approvals (negative = consecutive rejections)
    # Rolling performance score in [0, 1]; drives promotion/demotion.
    performance: float = 0.5
    # Reviewer calibration: how often this agent's approvals were later
    # contradicted by a higher-up or by reality (0 = never wrong).
    reviewer_error_rate: float = 0.0


class Agent(BaseModel):
    id: str = Field(default_factory=lambda: new_id("agt"))
    name: str
    role: str  # human-readable job title
    room: str  # room key
    rank: Rank = Rank.worker
    manager_id: str | None = None
    # A short personality seed. Diversity of seeds keeps the org from
    # converging on the same ideas (mode collapse).
    persona: str = ""
    skills: list[str] = Field(default_factory=list)
    emotion: Emotion = Field(default_factory=Emotion)
    stats: AgentStats = Field(default_factory=AgentStats)
    status: Literal["active", "on_probation", "suspended"] = "active"
    # Last few signals, newest last, for prompt rendering.
    recent_signals: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=now_iso)


class Room(BaseModel):
    key: str
    name: str
    purpose: str
    head_id: str | None = None
    member_ids: list[str] = Field(default_factory=list)
    # What kinds of tasks this room handles.
    task_types: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Work
# --------------------------------------------------------------------------- #


class TaskStatus(str, Enum):
    queued = "queued"
    in_progress = "in_progress"
    submitted = "submitted"
    in_review = "in_review"
    approved = "approved"
    revise = "revise"
    rejected = "rejected"
    blocked = "blocked"
    done = "done"
    cancelled = "cancelled"


class Task(BaseModel):
    id: str = Field(default_factory=lambda: new_id("tsk"))
    room: str
    type: str  # room-specific task type, e.g. "scan_trends", "build_asset"
    title: str
    brief: str
    created_by: str  # agent id or "system" / "human"
    assigned_to: str | None = None
    status: TaskStatus = TaskStatus.queued
    priority: int = 5  # 1 (urgent) .. 9 (whenever)
    venture_id: str | None = None
    parent_task_id: str | None = None
    inputs: dict[str, Any] = Field(default_factory=dict)
    # Set when a review sends the work back.
    revision_notes: list[str] = Field(default_factory=list)
    attempts: int = 0
    # High-stakes work (money leaves the company, something is published
    # externally, legal exposure) gets an extra review layer.
    high_stakes: bool = False
    tick_created: int = 0
    tick_updated: int = 0
    created_at: str = Field(default_factory=now_iso)


class WorkProduct(BaseModel):
    id: str = Field(default_factory=lambda: new_id("wp"))
    task_id: str
    agent_id: str
    kind: str  # mirrors task.type
    title: str
    summary: str
    content: str  # the deliverable itself (markdown)
    data: dict[str, Any] = Field(default_factory=dict)  # structured payload
    # The agent's own claim about how this differs from what a default AI
    # agent would produce. Reviewers check this claim.
    differentiation_claim: str = ""
    self_assessment: float = 0.5  # 0..1
    version: int = 1
    tick: int = 0
    created_at: str = Field(default_factory=now_iso)


class ReviewVerdict(str, Enum):
    approve = "approve"
    revise = "revise"
    reject = "reject"
    escalate = "escalate"


class Review(BaseModel):
    id: str = Field(default_factory=lambda: new_id("rev"))
    work_product_id: str
    task_id: str
    reviewer_id: str
    author_id: str
    verdict: ReviewVerdict
    quality: float = Field(ge=0.0, le=1.0)
    originality: float = Field(ge=0.0, le=1.0)
    impact: float = Field(ge=0.0, le=1.0)
    feedback: str
    # Concrete, checkable asks for a revision.
    required_changes: list[str] = Field(default_factory=list)
    # Whether this was a cross-room review (independent of the author's chain).
    cross_room: bool = False
    tick: int = 0
    created_at: str = Field(default_factory=now_iso)


# --------------------------------------------------------------------------- #
# Ventures and money
# --------------------------------------------------------------------------- #


class VentureStage(str, Enum):
    idea = "idea"
    gated = "gated"  # passed the originality gate
    validating = "validating"
    building = "building"
    launched = "launched"
    earning = "earning"
    scaling = "scaling"
    killed = "killed"


class OriginalityReport(BaseModel):
    """Output of the originality gate for one idea."""

    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    default_ai_would_build: str = ""
    divergence: str = ""
    slop_matches: list[str] = Field(default_factory=list)
    check_results: dict[str, bool] = Field(default_factory=dict)
    freshness: float = 0.5
    crowding: float = 0.5
    verdict_reason: str = ""


class Venture(BaseModel):
    id: str = Field(default_factory=lambda: new_id("ven"))
    name: str
    thesis: str
    why_other_agents_wont: str = ""
    stage: VentureStage = VentureStage.idea
    owner_agent_id: str | None = None
    room: str = "forge"
    revenue_rail: str = ""
    originality: OriginalityReport | None = None
    budget_cents: int = 0
    revenue_cents: int = 0
    cost_cents: int = 0
    # Ticks since the last cent of revenue; used by the kill rule.
    ticks_without_revenue: int = 0
    milestones: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
    kill_reason: str = ""
    # Pre-registered kill date (tick) and the novelty half-life the pitch claimed.
    kill_by_tick: int | None = None
    novelty_half_life_ticks: int = 0
    core_keywords: list[str] = Field(default_factory=list)
    trend_ids: list[str] = Field(default_factory=list)
    tick_created: int = 0
    tick_updated: int = 0
    created_at: str = Field(default_factory=now_iso)

    @property
    def profit_cents(self) -> int:
        return self.revenue_cents - self.cost_cents


class LedgerKind(str, Enum):
    revenue = "revenue"
    cost = "cost"
    llm_cost = "llm_cost"
    adjustment = "adjustment"


class LedgerEntry(BaseModel):
    id: str = Field(default_factory=lambda: new_id("led"))
    kind: LedgerKind
    amount_cents: int
    currency: str = "USD"
    venture_id: str | None = None
    agent_id: str | None = None
    source: str = ""  # connector name, "manual", "llm"
    memo: str = ""
    external_ref: str = ""  # e.g. Stripe charge id
    tick: int = 0
    created_at: str = Field(default_factory=now_iso)


class TrendSignal(BaseModel):
    """Something new in the world that might be exploitable."""

    id: str = Field(default_factory=lambda: new_id("trd"))
    title: str
    summary: str
    source_urls: list[str] = Field(default_factory=list)
    first_seen: str = ""  # approximate date the thing appeared
    freshness: float = 0.5  # 1 = brand new
    crowding: float = 0.5  # 1 = every AI agent is already on it
    exploitability: float = 0.5  # 1 = we could earn from it this week
    status: Literal["new", "watching", "exploiting", "dead"] = "new"
    found_by: str = ""
    tick_found: int = 0
    created_at: str = Field(default_factory=now_iso)

    @property
    def opportunity(self) -> float:
        return round(self.freshness * (1.0 - self.crowding) * self.exploitability, 4)


# --------------------------------------------------------------------------- #
# Human interface
# --------------------------------------------------------------------------- #


class AirlockField(BaseModel):
    name: str
    label: str
    type: Literal["text", "secret", "number", "boolean", "url", "textarea", "choice"] = "text"
    choices: list[str] = Field(default_factory=list)
    required: bool = True


class AirlockStatus(str, Enum):
    open = "open"
    resolved = "resolved"
    dismissed = "dismissed"


class AirlockRequest(BaseModel):
    """Something only the human can do. Shown on the dashboard."""

    id: str = Field(default_factory=lambda: new_id("air"))
    type: str  # see airlock.py REQUEST_TYPES
    title: str
    description: str
    why_it_matters: str = ""
    fields: list[AirlockField] = Field(default_factory=list)
    status: AirlockStatus = AirlockStatus.open
    response: dict[str, Any] = Field(default_factory=dict)
    venture_id: str | None = None
    task_id: str | None = None
    requested_by: str = ""
    priority: int = 5
    tick: int = 0
    resolved_tick: int | None = None
    created_at: str = Field(default_factory=now_iso)


# --------------------------------------------------------------------------- #
# Communication
# --------------------------------------------------------------------------- #


class Message(BaseModel):
    id: str = Field(default_factory=lambda: new_id("msg"))
    sender: str  # agent id, "system" or "human"
    # Recipient: an agent id, a room key ("room:forge"), or "all".
    to: str
    channel: str = "general"
    content: str
    tick: int = 0
    created_at: str = Field(default_factory=now_iso)


class Pin(BaseModel):
    """A pinned fact on the shared bulletin board (visible to every agent)."""

    key: str
    content: str
    pinned_by: str = "system"
    tick: int = 0
    created_at: str = Field(default_factory=now_iso)


class TickLog(BaseModel):
    tick: int
    started_at: str = Field(default_factory=now_iso)
    finished_at: str = ""
    events: list[str] = Field(default_factory=list)
    llm_calls: int = 0
    llm_cost_usd: float = 0.0
