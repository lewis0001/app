import pytest

from workhouse import review as review_mod
from workhouse.airlock import Airlock
from workhouse.bus import Bus
from workhouse.config import Settings
from workhouse.hierarchy import Org
from workhouse.llm import MockLLM, fake_instance
from workhouse.models import Agent, AirlockField, Rank, Review, ReviewVerdict, Room, Task, Venture, WorkProduct
from workhouse.money import Ledger, Rails
from workhouse.originality import CHECKS, GateOutput, Gate, score, slop_matches
from workhouse.store import Store


def test_store_roundtrip_and_transactions(tmp_path):
    s = Store(tmp_path / "t.db")
    a = s.agents.put(Agent(name="Ada", role="Scout", room="observatory"))
    assert s.agents.get(a.id).name == "Ada"
    with pytest.raises(RuntimeError):
        with s.transaction():
            s.tasks.put(Task(room="forge", type="x", title="t", brief="b", created_by="system"))
            raise RuntimeError("boom")
    assert s.tasks.count() == 0
    s.tick = 3
    assert s.tick == 3


def test_mock_llm_is_deterministic_and_schema_valid():
    llm = MockLLM(Settings())
    a = llm.complete("s", "same prompt", Review)
    b = llm.complete("s", "same prompt", Review)
    assert a == b and 0 <= a.quality <= 1
    v = fake_instance(Venture, 5)
    assert isinstance(v, Venture)


def test_slop_registry_and_gate_scoring():
    assert "Prompt packs / prompt marketplace" in slop_matches("sell a prompt pack for dentists")
    assert slop_matches("hand-verified registry of x402 endpoints") == []
    good = GateOutput(default_ai_would_build="x", divergence="y", check_results={c.key: True for c in CHECKS}, freshness=0.8, crowding=0.2, verdict_reason="ok")
    r = score(good, [])
    assert r.passed and r.score > 0.8
    bad = GateOutput(default_ai_would_build="x", divergence="y", check_results={c.key: (c.key not in {"not_slop", "why_not_1000_agents"}) for c in CHECKS}, freshness=0.8, crowding=0.2, verdict_reason="no")
    r2 = score(bad, ["AI newsletter / curated digest"])
    assert not r2.passed and "slop" in r2.verdict_reason


def test_gate_runs_through_llm():
    g = Gate(MockLLM(Settings()), lambda: "operator", lambda: [])
    rep = g.evaluate("idea", "text")
    assert 0.0 <= rep.score <= 1.0 and isinstance(rep.passed, bool)


def test_ledger_idempotent_and_venture_totals(tmp_path):
    s = Store(tmp_path / "t.db")
    led = Ledger(s)
    v = s.ventures.put(Venture(name="V", thesis="t"))
    led.revenue(1000, venture_id=v.id, source="stripe", external_ref="stripe:1")
    led.revenue(1000, venture_id=v.id, source="stripe", external_ref="stripe:1")
    led.llm_cost(0.5, venture_id=v.id)
    t = led.totals()
    assert t["revenue_cents"] == 1000 and t["llm_cost_cents"] == 50 and t["profit_cents"] == 950
    assert s.ventures.get(v.id).revenue_cents == 1000 and s.ventures.get(v.id).cost_cents == 50
    rails = Rails(s, led)
    assert rails.get("manual").configured() and not rails.get("stripe").configured()
    rails.get("stripe").configure({"secret_key": "sk_test_1"})
    assert rails.get("stripe").configured()


def test_airlock_dedupe_validation_and_consumption(tmp_path):
    s = Store(tmp_path / "t.db")
    air = Airlock(s)
    r1 = air.request("connect_rail", "Connect Stripe", "d", fields=[AirlockField(name="secret_key", label="k", type="secret")])
    r2 = air.request("connect_rail", "Connect Stripe", "d")
    assert r1.id == r2.id and len(air.open()) == 1
    with pytest.raises(ValueError):
        air.resolve(r1.id, {})
    air.resolve(r1.id, {"secret_key": "x"})
    assert len(air.open()) == 0 and len(air.unconsumed()) == 1
    air.mark_consumed(air.unconsumed()[0])
    assert air.unconsumed() == []


def test_review_routing_cross_room_and_high_stakes(tmp_path):
    s = Store(tmp_path / "t.db")
    org = Org(s)
    d = s.agents.put(Agent(name="Dir", role="Director", room="bridge", rank=Rank.ceo))
    for key in ("observatory", "forge"):
        s.rooms.put(Room(key=key, name=key, purpose="p"))
    h1 = s.agents.put(Agent(name="H1", role="Head", room="observatory", rank=Rank.head, manager_id=d.id))
    h2 = s.agents.put(Agent(name="H2", role="Head", room="forge", rank=Rank.head, manager_id=d.id))
    s.rooms.put(Room(key="observatory", name="o", purpose="p", head_id=h1.id))
    s.rooms.put(Room(key="forge", name="f", purpose="p", head_id=h2.id))
    w = s.agents.put(Agent(name="W", role="w", room="observatory", rank=Rank.worker, manager_id=h1.id))
    normal = Task(room="observatory", type="scan_trends", title="t", brief="b", created_by="system")
    assert [r.name for r in org.reviewers_for(w, normal)] == ["H1"]
    hs = Task(room="observatory", type="scan_trends", title="t", brief="b", created_by="system", high_stakes=True)
    assert [r.name for r in org.reviewers_for(w, hs)] == ["H1", "Dir"]
    pitch = Task(room="observatory", type="venture_pitch", title="t", brief="b", created_by="system")
    assert [r.name for r in org.reviewers_for(w, pitch)] == ["H2"]  # cross-room
    # the Director's own work still gets a reviewer
    assert org.reviewers_for(d, normal)


def test_review_outcome_updates_stats_and_emotions(tmp_path):
    s = Store(tmp_path / "t.db")
    a = s.agents.put(Agent(name="A", role="w", room="forge"))
    m = s.agents.put(Agent(name="M", role="h", room="forge", rank=Rank.head))
    t = Task(room="forge", type="build", title="x", brief="b", created_by="system")
    w = WorkProduct(task_id=t.id, agent_id=a.id, kind="build", title="w", summary="s", content="c")
    rv = Review(work_product_id=w.id, task_id=t.id, reviewer_id=m.id, author_id=a.id, verdict=ReviewVerdict.reject, quality=0.3, originality=0.2, impact=0.3, feedback="generic")
    sigs = review_mod.apply_review_outcome(s, rv, a, m)
    assert any(x.source == "originality" for x in sigs)
    assert s.agents.get(a.id).stats.rejections == 1 and s.agents.get(a.id).emotion.valence < 0
    later = Review(work_product_id=w.id, task_id=t.id, reviewer_id="dir", author_id=a.id, verdict=ReviewVerdict.reject, quality=0.3, originality=0.2, impact=0.3, feedback="no")
    earlier = Review(work_product_id=w.id, task_id=t.id, reviewer_id=m.id, author_id=a.id, verdict=ReviewVerdict.approve, quality=0.8, originality=0.8, impact=0.8, feedback="yes")
    assert review_mod.overturn(s, earlier, later, tick=2) is not None
    assert s.agents.get(m.id).stats.reviewer_error_rate > 0


def test_bus_rendering(tmp_path):
    s = Store(tmp_path / "t.db")
    bus = Bus(s)
    bus.pin("k", "fact", tick=1)
    bus.send("system", "agt_1", "hello", tick=1)
    bus.post_room("agt_2", "forge", "posted", tick=1)
    text = bus.render_for("agt_1", "forge", current_tick=1)
    assert "fact" in text and "hello" in text and "posted" in text
