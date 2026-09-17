from workhouse.airlock import MAX_NEW_PER_TICK, Airlock
from workhouse.mirror import Mirror, overlap
from workhouse.saturation import Saturation
from workhouse.store import Store


def test_saturation_verdict_flags_commodity_and_unknown():
    text, floor = Saturation.verdict({"ai agent": {"github_30d": 29769, "hn_90d": 400}})
    assert "COMMODITY" in text and floor is not None and floor >= 0.85
    text, floor = Saturation.verdict({"dialect test set": {"github_30d": 3, "hn_90d": 1}})
    assert "crowded" not in text and floor == 0.0
    text, floor = Saturation.verdict({"x": {"github_30d": None, "hn_90d": None}})
    assert floor is None and "unavailable" in text


def test_saturation_disabled_returns_unknowns(tmp_path):
    s = Saturation(Store(tmp_path / "t.db"), enabled=False)
    m = s.measure(["a", "b"])
    assert m == {"a": {"github_30d": None, "hn_90d": None}, "b": {"github_30d": None, "hn_90d": None}}


def test_mirror_overlap_and_matches(tmp_path):
    st = Store(tmp_path / "t.db")
    m = Mirror(st)
    assert m.stale(0) and m.render().startswith("DEFAULT TWIN: not yet")
    st.set_kv("default_twin", {"tick": 1, "ideas": ["Sell prompt packs and starter kits for coding agents"], "product_types": ["prompt pack $19"], "channels": ["Reddit"]})
    assert not m.stale(5, ticks_per_day=24) and m.stale(1 + 24, ticks_per_day=24) and m.stale(2)
    assert overlap("prompt packs for coding agents", "Sell prompt packs and starter kits for coding agents") > 0.4
    assert m.matches("A curated prompt pack for coding agents sold as starter kits")
    assert not m.matches("Hand-verified dialect test sets for voice vendors entering Portugal")


def test_airlock_flood_cap_defers_low_priority(tmp_path):
    air = Airlock(Store(tmp_path / "t.db"))
    for i in range(MAX_NEW_PER_TICK + 2):
        air.request("manual_action", f"thing {i}", "d", tick=7)
    ticks = sorted(r.tick for r in air.open())
    assert ticks.count(7) == MAX_NEW_PER_TICK and ticks.count(8) == 2
    # urgent requests are never deferred
    r = air.request("approve_spend", "urgent", "d", tick=7)
    assert r.tick == 7
