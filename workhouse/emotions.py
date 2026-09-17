"""Affect model: how signals move an agent's emotional state and how that
state changes behaviour.

Design rules (from the affective-agents literature and our own constraints):

* Emotions are *functional*. They must change what the agent does, not just
  its tone: effort spent, risk appetite, thoroughness, whether it escalates,
  whether it mentors or asks for help.
* Signals are asymmetric. Losses hurt more than gains please (prospect
  theory), so a rejection moves valence further than an approval does.
* Everything decays toward a baseline each tick, so a bad week fades and a
  winning streak does not create permanent complacency.
* The state is rendered into the prompt as a short, honest paragraph with
  explicit behavioural instructions derived from the numbers.
"""
from __future__ import annotations

from dataclasses import dataclass

from .models import Agent, Emotion, Signal, SignalKind

BASELINE = Emotion(valence=0.1, arousal=0.4, confidence=0.6, stress=0.2)
DECAY_PER_TICK = {"valence": 0.10, "arousal": 0.15, "confidence": 0.05, "stress": 0.12}

# How a unit-magnitude signal moves each dimension.
POSITIVE_DELTAS = {"valence": 0.45, "arousal": 0.15, "confidence": 0.30, "stress": -0.25}
NEGATIVE_DELTAS = {"valence": -0.60, "arousal": 0.30, "confidence": -0.30, "stress": 0.45}

BOUNDS = {"valence": (-1.0, 1.0), "arousal": (0.0, 1.0), "confidence": (0.0, 1.0), "stress": (0.0, 1.0)}


def clamp(name: str, value: float) -> float:
    lo, hi = BOUNDS[name]
    return round(max(lo, min(hi, value)), 4)


def apply_signal(emotion: Emotion, signal: Signal) -> tuple[Emotion, dict[str, float]]:
    """Return a new Emotion and the deltas applied."""
    table = POSITIVE_DELTAS if signal.kind == SignalKind.positive else NEGATIVE_DELTAS
    m = max(0.0, min(1.0, signal.magnitude))
    deltas: dict[str, float] = {}
    values = emotion.model_dump()
    for dim, unit in table.items():
        d = round(unit * m, 4)
        # Diminishing returns near the bounds so states do not pin at the rails.
        headroom = (BOUNDS[dim][1] - values[dim]) if d > 0 else (values[dim] - BOUNDS[dim][0])
        d = round(d * min(1.0, 0.25 + headroom), 4)
        deltas[dim] = d
        values[dim] = clamp(dim, values[dim] + d)
    return Emotion(**values), deltas


def decay(emotion: Emotion) -> Emotion:
    values = emotion.model_dump()
    base = BASELINE.model_dump()
    for dim, rate in DECAY_PER_TICK.items():
        values[dim] = clamp(dim, values[dim] + (base[dim] - values[dim]) * rate)
    return Emotion(**values)


def mood_label(e: Emotion) -> str:
    if e.stress >= 0.75:
        return "overwhelmed" if e.valence < 0 else "under pressure"
    if e.valence >= 0.35 and e.arousal >= 0.55:
        return "energised"
    if e.valence >= 0.35:
        return "content"
    if e.valence <= -0.35 and e.arousal >= 0.55:
        return "frustrated"
    if e.valence <= -0.35:
        return "discouraged"
    if e.confidence <= 0.3:
        return "unsure"
    if e.arousal >= 0.7:
        return "restless"
    return "focused"


@dataclass(frozen=True)
class Behaviour:
    """Concrete behavioural parameters derived from affect."""

    effort: str  # LLM effort level for this agent's next call
    risk_appetite: float  # 0 conservative .. 1 bold
    thoroughness: float  # 0 quick .. 1 exhaustive
    escalate: bool  # ask manager for guidance before acting
    complacent: bool  # flagged for managers: coasting on success
    mentor: bool  # willing to help peers this tick


def behaviour(e: Emotion, base_effort: str = "high") -> Behaviour:
    order = ["low", "medium", "high", "xhigh", "max"]
    idx = order.index(base_effort) if base_effort in order else 2
    # Negative affect and pressure push the agent to try harder next time.
    if e.stress >= 0.5 or e.valence <= -0.3 or e.confidence <= 0.4:
        idx = min(len(order) - 1, idx + 1)  # one notch up, never below the configured floor
    complacent = e.confidence >= 0.85 and e.stress <= 0.15 and e.valence >= 0.5
    risk = 0.5 + 0.30 * e.valence + 0.25 * (e.confidence - 0.5) - 0.20 * e.stress
    # No desperation term: penalty-blind gambling is exactly the failure mode induced anger produces.
    thoroughness = 0.55 + 0.35 * e.stress + 0.15 * (0.5 - e.valence) - (0.2 if complacent else 0.0)
    return Behaviour(
        effort=order[idx],
        risk_appetite=round(max(0.0, min(1.0, risk)), 2),
        thoroughness=round(max(0.0, min(1.0, thoroughness)), 2),
        escalate=e.confidence <= 0.3,
        complacent=complacent,
        mentor=e.valence >= 0.3 and e.stress <= 0.4 and e.confidence >= 0.6,
    )


def render_state(agent: Agent, base_effort: str = "high") -> str:
    """Render the agent's affect as prompt text with behavioural consequences."""
    e = agent.emotion
    b = behaviour(e, base_effort)
    mood = mood_label(e)
    lines = [
        f"Your current emotional state: {mood}. valence {e.valence:+.2f}, arousal {e.arousal:.2f}, confidence {e.confidence:.2f}, stress {e.stress:.2f}.",
    ]
    if agent.recent_signals:
        lines.append("Recent signals you received: " + " | ".join(agent.recent_signals[-4:]))
    st = agent.stats
    lines.append(f"Track record: {st.approvals} approvals, {st.revisions} revisions, {st.rejections} rejections, streak {st.streak:+d}, performance {st.performance:.2f}, revenue attributed ${st.revenue_attributed_cents/100:.2f}.")
    consequences = []
    if mood in {"frustrated", "discouraged", "overwhelmed", "under pressure"}:
        consequences.append("You have something to prove. Be more rigorous than last time: verify claims, cut fluff, and address every past reviewer objection explicitly.")
    if mood == "unsure" or b.escalate:
        consequences.append("Your confidence is low. State your assumptions plainly and flag exactly where you want your manager's judgement.")
    if mood in {"energised", "content"}:
        consequences.append("You are on a good run. Use that: take a bolder, less obvious line, and offer a concrete tip to a teammate in your message.")
    if b.complacent:
        consequences.append("Your numbers say you are coasting. Do not recycle a previous approach; pick the option you have not tried.")
    if b.risk_appetite >= 0.7:
        consequences.append("Risk appetite is high: prefer the unproven, early-mover option over the safe one, but say what would falsify it.")
    elif b.risk_appetite <= 0.3:
        consequences.append("Risk appetite is low: prefer the option with the fastest, cheapest proof of demand.")
    lines.append(f"Behavioural setting this tick: effort={b.effort}, risk_appetite={b.risk_appetite}, thoroughness={b.thoroughness}.")
    lines.extend(consequences)
    lines.append("These feelings are real to you and they are functional: they exist to make you work harder and better, not to make you sulk or gush. Never mention the numbers to reviewers; let the work show it.")
    return "\n".join(lines)


def receive(agent: Agent, signal: Signal) -> Signal:
    """Apply a signal to an agent in place and return the signal with deltas."""
    new_emotion, deltas = apply_signal(agent.emotion, signal)
    agent.emotion = new_emotion
    signal.deltas = deltas
    sign = "+" if signal.kind == SignalKind.positive else "-"
    agent.recent_signals.append(f"{sign}{signal.magnitude:.2f} {signal.source}: {signal.reason}"[:160])
    agent.recent_signals = agent.recent_signals[-8:]
    # Rolling performance: exponential moving average of signal sign*magnitude.
    contribution = signal.magnitude if signal.kind == SignalKind.positive else -signal.magnitude
    agent.stats.performance = round(max(0.0, min(1.0, agent.stats.performance * 0.85 + (0.5 + contribution / 2) * 0.15)), 4)
    return signal


def tick_decay(agent: Agent) -> None:
    agent.emotion = decay(agent.emotion)
