"""Review protocol: managers judge work, verdicts become signals.

Anti-sycophancy measures:
* The reviewer sees the author's differentiation claim and must state what a
  default AI would have produced before scoring originality.
* The reviewer never sees the author's emotional state or name, only the work.
* Every Nth review is cross-room (see hierarchy.py) and the Director spot
  checks approvals; an overturned approval costs the reviewer.
* Verdict distribution is tracked; a reviewer that approves everything is
  told so in its prompt.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from . import emotions
from .models import Agent, Review, ReviewVerdict, Signal, SignalKind, Task, WorkProduct
from .store import Store


class ReviewOutput(BaseModel):
    """What the reviewer LLM must return."""

    default_ai_would_have_produced: str = Field(description="Your honest prediction of what a generic AI agent given the same brief would have made.")
    differentiation_claim_holds: bool = Field(description="Whether the author's differentiation claim survives that comparison.")
    quality: float = Field(ge=0, le=1, description="Craft, correctness, completeness.")
    originality: float = Field(ge=0, le=1, description="How far it diverges from the default-AI output and from the slop registry.")
    impact: float = Field(ge=0, le=1, description="Expected contribution to revenue or to finding revenue.")
    verdict: ReviewVerdict
    feedback: str = Field(description="Direct, specific, addressed to the author.")
    required_changes: list[str] = Field(default_factory=list, description="Checkable items for a revision. Empty if approved.")
    praise: str = Field(default="", description="One specific thing done well, if any.")


REVIEW_SYSTEM = """You are a manager in an autonomous company of AI agents whose goal is to make money by doing what other AI agents would not think to do.
You are reviewing a subordinate's work. Be the reviewer you would want: specific, demanding, fair, unsentimental.

Rules:
- First write what a generic AI agent would have produced from the same brief. Then judge the work against THAT. Work that matches the generic output is a failure even if polished.
- Approve only work you would stake your own standing on.
- 'revise' when the core is right but specific things must change; list them as checkable items.
- 'reject' when the approach itself is generic, unfounded, or unsellable; say what to do instead.
- 'escalate' only when the decision needs money, publishing, legal exposure, or the Director's strategy call.
- Do not soften feedback and do not pad it. Specific beats kind.
- Never approve because the work sounds confident. Never reject because it sounds unsure. Judge the evidence, not the tone.
- Anything the author reports as done must be backed by evidence they could not have written themselves (a ledger entry, a payment record, a reply, a log). A summary is not evidence.
- If you cannot judge this work (outside your competence, missing evidence), choose 'escalate' with the reason; that is always acceptable.
"""


def build_review_prompt(task: Task, work: WorkProduct, reviewer: Agent, context: str, stats_note: str) -> str:
    return f"""TASK ({task.type}, priority {task.priority}{', HIGH STAKES' if task.high_stakes else ''}): {task.title}
BRIEF: {task.brief}
{('PREVIOUS REVISION NOTES: ' + ' | '.join(task.revision_notes)) if task.revision_notes else ''}

WORK PRODUCT v{work.version}: {work.title}
SUMMARY: {work.summary}
AUTHOR'S DIFFERENTIATION CLAIM: {work.differentiation_claim or '(none given - treat as a defect)'}

CONTENT:
{work.content}

STRUCTURED DATA: {work.data if work.data else '(none)'}

COMPANY CONTEXT:
{context}

{stats_note}
Return your review."""


def reviewer_stats_note(reviewer: Agent, store: Store) -> str:
    """Kept for the dashboard; no longer rendered into prompts, because a
    reviewer that knows what its verdicts trigger mislabels (judge blinding)."""
    mine = store.reviews.where(reviewer_id=reviewer.id)
    if len(mine) < 4:
        return ""
    approvals = sum(1 for r in mine if r.verdict == ReviewVerdict.approve)
    rate = approvals / len(mine)
    if rate > 0.8:
        return f"Note on your own record: you approved {approvals}/{len(mine)} reviews. That is suspiciously lenient; the Director is spot-checking your approvals."
    if rate < 0.2:
        return f"Note on your own record: you approved {approvals}/{len(mine)} reviews. Make sure you are rejecting the work, not the effort."
    return ""


def review_to_signals(review: Review, author: Agent, reviewer: Agent) -> list[Signal]:
    """Translate a verdict into performance signals for the author (and a small
    one for the reviewer to reward doing the work of reviewing)."""
    out: list[Signal] = []
    score = (review.quality + review.originality * 1.5 + review.impact) / 3.5
    if review.verdict == ReviewVerdict.approve:
        out.append(Signal(agent_id=author.id, kind=SignalKind.positive, magnitude=round(0.35 + 0.6 * score, 3), source="review", reason=f"approved by {reviewer.name}: {review.feedback[:80]}", tick=review.tick))
    elif review.verdict == ReviewVerdict.revise:
        out.append(Signal(agent_id=author.id, kind=SignalKind.negative, magnitude=round(0.25 + 0.3 * (1 - score), 3), source="review", reason=f"sent back by {reviewer.name}: {review.feedback[:80]}", tick=review.tick))
    elif review.verdict == ReviewVerdict.reject:
        out.append(Signal(agent_id=author.id, kind=SignalKind.negative, magnitude=round(0.6 + 0.4 * (1 - score), 3), source="review", reason=f"rejected by {reviewer.name}: {review.feedback[:80]}", tick=review.tick))
    else:  # escalate: neutral for the author, nothing yet
        pass
    if review.originality <= 0.3 and review.verdict != ReviewVerdict.escalate:
        out.append(Signal(agent_id=author.id, kind=SignalKind.negative, magnitude=0.3, source="originality", reason="work judged indistinguishable from a generic AI agent's", tick=review.tick))
    return out


def apply_review_outcome(store: Store, review: Review, author: Agent, reviewer: Agent) -> list[Signal]:
    """Update stats and emotions for author and reviewer; persist; return signals."""
    signals = review_to_signals(review, author, reviewer)
    if review.verdict == ReviewVerdict.approve:
        author.stats.approvals += 1
        author.stats.streak = author.stats.streak + 1 if author.stats.streak >= 0 else 1
    elif review.verdict == ReviewVerdict.revise:
        author.stats.revisions += 1
        author.stats.streak = min(0, author.stats.streak) - 1
    elif review.verdict == ReviewVerdict.reject:
        author.stats.rejections += 1
        author.stats.streak = min(0, author.stats.streak) - 1
    reviewer.stats.reviews_given += 1
    # calibration: every review given lets an earlier overturn fade
    reviewer.stats.reviewer_error_rate = round(reviewer.stats.reviewer_error_rate * 0.85, 4)
    for s in signals:
        emotions.receive(author, s)
        store.signals.put(s)
    store.agents.put(author)
    store.agents.put(reviewer)
    store.reviews.put(review)
    return signals


def overturn(store: Store, earlier: Review, later: Review, tick: int) -> Signal | None:
    """The Director (or a cross-room head) contradicted an earlier approval."""
    if earlier.verdict != ReviewVerdict.approve or later.verdict == ReviewVerdict.approve:
        return None
    reviewer = store.agents.get(earlier.reviewer_id)
    if not reviewer:
        return None
    reviewer.stats.reviewer_error_rate = round(min(1.0, reviewer.stats.reviewer_error_rate + 0.3), 4)
    sig = Signal(agent_id=reviewer.id, kind=SignalKind.negative, magnitude=0.5, source="calibration", reason=f"your approval of '{earlier.work_product_id}' was overturned", tick=tick)
    emotions.receive(reviewer, sig)
    store.signals.put(sig)
    store.agents.put(reviewer)
    return sig
