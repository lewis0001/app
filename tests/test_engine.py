from fastapi.testclient import TestClient

from workhouse.dashboard.app import create_app
from workhouse.models import AirlockStatus, LedgerKind, TaskStatus, VentureStage


def test_company_seeds_org(engine):
    org = engine.ctx.org
    assert org.director() is not None
    assert {r.key for r in engine.store.rooms.all()} == {"bridge", "observatory", "forge", "market_bay", "ledger"}
    for key in ("observatory", "forge", "market_bay", "ledger"):
        assert org.head_of(key) is not None
    engine.seed(org.director(), [], [])  # idempotent
    assert engine.store.agents.count() == 12


def test_multi_tick_run_produces_the_whole_story(engine):
    for _ in range(20):
        engine.run_tick()
    s = engine.store
    assert s.tick == 20
    assert s.trends.count() > 0, "the Observatory must record trends"
    assert s.reviews.count() > 0 and s.signals.count() > 0
    assert any(t.status == TaskStatus.done for t in s.tasks.all())
    kinds = {x.kind for x in s.signals.all()}
    assert len(kinds) == 2, "both positive and negative signals must flow"
    assert any(v.stage != VentureStage.idea for v in s.ventures.all()), "a venture must pass the gate within 20 mock ticks"
    assert any(e.kind == LedgerKind.llm_cost for e in s.ledger.all()), "LLM spend is charged to the ledger"
    moods = {a.emotion.valence for a in s.agents.all()}
    assert len(moods) > 1, "agents must diverge emotionally"
    # every tick was logged, and no tick raised
    assert all("ERROR" not in ev for tl in s.ticks.all() for ev in tl.events)


def test_airlock_roundtrip_configures_rail_and_revenue_signals(engine):
    engine.run_tick()
    air = engine.ctx.airlock
    req = air.request("connect_rail", "Connect stripe", "d", fields=engine.ctx.rails.get("stripe").setup_fields, tick=1)
    req.response["_rail"] = "stripe"
    engine.store.airlock.put(req)
    air.resolve(req.id, {"secret_key": "sk_test_abc"}, tick=1)
    engine.run_tick()
    assert engine.ctx.rails.get("stripe").configured()
    stored = engine.store.airlock.get(req.id)
    assert stored.response["secret_key"] == "***", "pasted secrets must not stay in the request record"
    assert "sk_test_abc" not in str(engine.snapshot())
    # manual revenue through the airlock
    v = engine.store.ventures.put(__import__("workhouse.models", fromlist=["Venture"]).Venture(name="V", thesis="t", stage=VentureStage.launched, owner_agent_id=engine.store.agents.all()[-1].id))
    r = air.request("enter_revenue", "Record", "d", venture_id=v.id, tick=2)
    air.resolve(r.id, {"amount_usd": "12.5", "memo": "test"}, tick=2)
    engine.run_tick()
    v2 = engine.store.ventures.get(v.id)
    assert v2.revenue_cents == 1250
    # it reached 'earning' on the first dollar (a later mock P&L review may kill it; that is allowed)
    assert any("first revenue" in m for m in v2.milestones) and v2.stage in (VentureStage.earning, VentureStage.killed)
    assert any(sg.source == "revenue" for sg in engine.store.signals.all())


def test_cost_cap_pauses_and_raise_cap_resumes(engine):
    engine.settings.max_total_cost_usd = 0.0001
    engine.run_tick()
    engine.run_tick()
    assert engine.paused_reason
    reqs = [r for r in engine.ctx.airlock.open() if r.type == "raise_cap"]
    assert reqs
    engine.ctx.airlock.resolve(reqs[0].id, {"new_cap_usd": "1000"}, tick=engine.store.tick)
    engine.run_tick()
    assert not engine.paused_reason and engine.settings.max_total_cost_usd == 1000


def test_dashboard_api(engine):
    client = TestClient(create_app(engine))
    assert client.get("/").status_code == 200
    assert client.post("/api/tick").status_code == 200
    state = client.get("/api/state").json()
    assert state["tick"] == 1 and len(state["agents"]) == 12 and state["rooms"]
    r = client.post("/api/revenue", json={"amount_usd": 5, "memo": "cash"})
    assert r.status_code == 200
    assert client.get("/api/state").json()["ledger"]["revenue_cents"] == 500
    agent_id = state["agents"][0]["id"]
    assert client.post("/api/signal", json={"agent_id": agent_id, "kind": "negative", "reason": "sloppy"}).status_code == 200
    assert client.post("/api/message", json={"content": "I have a Discord with 2k members"}).status_code == 200
    assert client.post("/api/operator", json={"skills": "welding; Portuguese", "hours_per_week_available": 3}).json()["skills"] == ["welding", "Portuguese"]
    open_reqs = client.get("/api/state").json()["airlock"]
    if open_reqs:
        rid = open_reqs[0]["id"]
        assert client.post(f"/api/airlock/{rid}/dismiss", json={"response": {"reason": "no"}}).json()["status"] == AirlockStatus.dismissed.value
