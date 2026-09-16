from workhouse import emotions
from workhouse.models import Agent, Emotion, Signal, SignalKind


def test_negative_signal_hurts_more_than_positive_helps():
    a = Agent(name="A", role="r", room="forge")
    base = a.emotion.valence
    emotions.receive(a, Signal(agent_id=a.id, kind=SignalKind.negative, magnitude=0.5, source="review", reason="x"))
    drop = base - a.emotion.valence
    b = Agent(name="B", role="r", room="forge")
    emotions.receive(b, Signal(agent_id=b.id, kind=SignalKind.positive, magnitude=0.5, source="review", reason="x"))
    rise = b.emotion.valence - base
    assert drop > rise > 0


def test_bounds_and_decay_toward_baseline():
    a = Agent(name="A", role="r", room="forge")
    for _ in range(20):
        emotions.receive(a, Signal(agent_id=a.id, kind=SignalKind.negative, magnitude=1.0, source="review", reason="x"))
    assert -1.0 <= a.emotion.valence <= 1.0 and 0.0 <= a.emotion.stress <= 1.0
    assert emotions.mood_label(a.emotion) in {"overwhelmed", "frustrated", "discouraged", "under pressure"}
    for _ in range(200):
        emotions.tick_decay(a)
    assert abs(a.emotion.valence - emotions.BASELINE.valence) < 0.02
    assert abs(a.emotion.stress - emotions.BASELINE.stress) < 0.02


def test_stress_raises_effort_and_complacency_is_flagged():
    stressed = Emotion(valence=-0.5, arousal=0.7, confidence=0.4, stress=0.7)
    assert emotions.behaviour(stressed, "high").effort == "xhigh"
    coasting = Emotion(valence=0.8, arousal=0.4, confidence=0.95, stress=0.05)
    b = emotions.behaviour(coasting, "high")
    assert b.complacent and b.risk_appetite > 0.6
    assert emotions.behaviour(Emotion(), "high").effort == "high"


def test_render_state_mentions_consequences():
    a = Agent(name="A", role="r", room="forge", emotion=Emotion(valence=-0.6, arousal=0.7, confidence=0.3, stress=0.6))
    text = emotions.render_state(a)
    assert "something to prove" in text and "effort=xhigh" in text


def test_performance_tracks_signals():
    a = Agent(name="A", role="r", room="forge")
    for _ in range(10):
        emotions.receive(a, Signal(agent_id=a.id, kind=SignalKind.positive, magnitude=0.9, source="review", reason="x"))
    assert a.stats.performance > 0.7
    assert len(a.recent_signals) <= 8
