"""Ledger and revenue rails.

The ledger is the single source of truth for money. Revenue arrives through
*connectors* (Stripe, Lemon Squeezy, Gumroad, ... or manual entry from the
dashboard). Each connector says which one-time human steps it needs; those
become Airlock requests. Once configured, polling is automatic.

Money never leaves the company without a human approval unless the operator
has set an autonomous spend limit in Settings.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from .models import AirlockField, LedgerEntry, LedgerKind
from .store import Store

log = logging.getLogger("workhouse.money")


class Ledger:
    def __init__(self, store: Store):
        self.store = store

    def record(self, kind: LedgerKind, amount_cents: int, *, venture_id: str | None = None, agent_id: str | None = None, source: str = "", memo: str = "", external_ref: str = "", tick: int = 0) -> LedgerEntry:
        if external_ref and self.store.ledger.first(external_ref=external_ref):
            # idempotent: a connector may report the same charge twice
            return self.store.ledger.first(external_ref=external_ref)  # type: ignore[return-value]
        entry = LedgerEntry(kind=kind, amount_cents=int(amount_cents), venture_id=venture_id, agent_id=agent_id, source=source, memo=memo, external_ref=external_ref, tick=tick)
        self.store.ledger.put(entry)
        if venture_id:
            v = self.store.ventures.get(venture_id)
            if v:
                if kind == LedgerKind.revenue:
                    v.revenue_cents += entry.amount_cents
                    v.ticks_without_revenue = 0
                elif kind in (LedgerKind.cost, LedgerKind.llm_cost):
                    v.cost_cents += entry.amount_cents
                v.tick_updated = tick
                self.store.ventures.put(v)
        return entry

    def revenue(self, cents: int, **kw: Any) -> LedgerEntry:
        return self.record(LedgerKind.revenue, cents, **kw)

    def cost(self, cents: int, **kw: Any) -> LedgerEntry:
        return self.record(LedgerKind.cost, cents, **kw)

    def llm_cost(self, usd: float, **kw: Any) -> LedgerEntry | None:
        cents = int(round(usd * 100))
        if cents <= 0:
            return None
        return self.record(LedgerKind.llm_cost, cents, source="llm", **kw)

    def totals(self) -> dict[str, int]:
        t = {"revenue_cents": 0, "cost_cents": 0, "llm_cost_cents": 0}
        for e in self.store.ledger.all():
            if e.kind == LedgerKind.revenue:
                t["revenue_cents"] += e.amount_cents
            elif e.kind == LedgerKind.cost:
                t["cost_cents"] += e.amount_cents
            elif e.kind == LedgerKind.llm_cost:
                t["llm_cost_cents"] += e.amount_cents
        t["profit_cents"] = t["revenue_cents"] - t["cost_cents"] - t["llm_cost_cents"]
        return t

    def recent(self, n: int = 20) -> list[LedgerEntry]:
        return self.store.ledger.all()[-n:]

    def render(self) -> str:
        t = self.totals()
        return (
            f"Revenue ${t['revenue_cents']/100:.2f}, costs ${t['cost_cents']/100:.2f}, LLM spend ${t['llm_cost_cents']/100:.2f}, "
            f"profit ${t['profit_cents']/100:.2f}."
        )


# --------------------------------------------------------------------------- #
# Connectors
# --------------------------------------------------------------------------- #


class RevenueConnector:
    """A way money can reach us. Subclasses set `name`, `setup_fields`."""

    name: str = "base"
    description: str = ""
    setup_fields: list[AirlockField] = []
    # Plain-language list of what the human does once.
    human_setup_once: str = ""
    automated_after: str = ""

    def __init__(self, store: Store):
        self.store = store

    def config(self) -> dict[str, Any]:
        return self.store.get_kv(f"connector:{self.name}", {}) or {}

    def configure(self, values: dict[str, Any]) -> None:
        cfg = self.config()
        cfg.update({k: v for k, v in values.items() if v not in (None, "")})
        self.store.set_kv(f"connector:{self.name}", cfg)

    def configured(self) -> bool:
        cfg = self.config()
        return all(cfg.get(f.name) for f in self.setup_fields if f.required)

    def poll(self, *, tick: int) -> list[dict[str, Any]]:
        """Return new revenue events as dicts: amount_cents, external_ref, memo, venture_id (optional)."""
        return []

    def create_checkout(self, *, name: str, amount_cents: int, currency: str = "usd", venture_id: str | None = None) -> str | None:
        """Create a sellable link for a product. None if unsupported."""
        return None


class ManualConnector(RevenueConnector):
    name = "manual"
    description = "The human records money received elsewhere on the dashboard."
    human_setup_once = "Nothing. Enter revenue on the dashboard when it arrives."
    automated_after = "Nothing is automated; use this only as a bridge while a real rail is set up."
    setup_fields: list[AirlockField] = []

    def configured(self) -> bool:
        return True


class StripeConnector(RevenueConnector):
    """Stripe via its REST API with the standard library (no SDK dependency).

    Polls balance transactions for new charges and can create Payment Links
    for products the Forge builds. Uses a *restricted* key so the agents can
    read balance and create links but never issue refunds or payouts.
    """

    name = "stripe"
    description = "Card payments, subscriptions and Payment Links through Stripe."
    human_setup_once = "Create a Stripe account (KYC), then create a restricted API key with Payment Links write + Balance/Charges read and paste it here."
    automated_after = "Agents create Payment Links for products and the ledger polls charges automatically."
    setup_fields = [
        AirlockField(name="secret_key", label="Stripe restricted secret key (rk_live_... or sk_test_...)", type="secret"),
    ]
    BASE = "https://api.stripe.com/v1"

    def _call(self, method: str, path: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
        key = self.config().get("secret_key")
        if not key:
            raise RuntimeError("stripe not configured")
        body = urllib.parse.urlencode(data or {}).encode() if data else None
        req = urllib.request.Request(f"{self.BASE}{path}", data=body, method=method)
        req.add_header("Authorization", f"Bearer {key}")
        req.add_header("Content-Type", "application/x-www-form-urlencoded")
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"stripe {e.code}: {e.read().decode()[:300]}") from e

    def poll(self, *, tick: int) -> list[dict[str, Any]]:
        cfg = self.config()
        params: dict[str, Any] = {"limit": 50, "type": "charge"}
        if cfg.get("last_created"):
            params["created[gt]"] = int(cfg["last_created"])
        data = self._call("GET", "/balance_transactions?" + urllib.parse.urlencode(params))
        events = []
        newest = int(cfg.get("last_created", 0) or 0)
        for tx in data.get("data", []):
            newest = max(newest, int(tx.get("created", 0)))
            events.append({
                "amount_cents": int(tx.get("net", tx.get("amount", 0))),
                "external_ref": f"stripe:{tx.get('id')}",
                "memo": tx.get("description") or "stripe charge",
                "venture_id": None,
            })
        if newest:
            self.configure({"last_created": newest})
        return events

    def create_checkout(self, *, name: str, amount_cents: int, currency: str = "usd", venture_id: str | None = None) -> str | None:
        price = self._call("POST", "/prices", {
            "currency": currency,
            "unit_amount": int(amount_cents),
            "product_data[name]": name[:250],
        })
        link = self._call("POST", "/payment_links", {
            "line_items[0][price]": price["id"],
            "line_items[0][quantity]": 1,
            "metadata[venture_id]": venture_id or "",
        })
        return link.get("url")


class LemonSqueezyConnector(RevenueConnector):
    name = "lemonsqueezy"
    description = "Merchant-of-record checkout for digital products (handles VAT/sales tax)."
    human_setup_once = "Create a Lemon Squeezy store (KYC), create an API key, paste the key and store id."
    automated_after = "Ledger polls orders; agents propose products the human activates once."
    setup_fields = [
        AirlockField(name="api_key", label="Lemon Squeezy API key", type="secret"),
        AirlockField(name="store_id", label="Store id", type="text"),
    ]
    BASE = "https://api.lemonsqueezy.com/v1"

    def poll(self, *, tick: int) -> list[dict[str, Any]]:
        cfg = self.config()
        req = urllib.request.Request(f"{self.BASE}/orders?filter[store_id]={urllib.parse.quote(str(cfg['store_id']))}&page[size]=50")
        req.add_header("Authorization", f"Bearer {cfg['api_key']}")
        req.add_header("Accept", "application/vnd.api+json")
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"lemonsqueezy {e.code}") from e
        events = []
        for order in data.get("data", []):
            attrs = order.get("attributes", {})
            if attrs.get("status") != "paid":
                continue
            events.append({
                "amount_cents": int(attrs.get("total", 0)),
                "external_ref": f"lemonsqueezy:{order.get('id')}",
                "memo": attrs.get("first_order_item", {}).get("product_name") or "order",
                "venture_id": None,
            })
        return events


CONNECTOR_CLASSES: list[type[RevenueConnector]] = [StripeConnector, LemonSqueezyConnector, ManualConnector]


class Rails:
    """Registry of connectors plus the polling loop that feeds the ledger."""

    def __init__(self, store: Store, ledger: Ledger):
        self.store = store
        self.ledger = ledger
        self.connectors: dict[str, RevenueConnector] = {c.name: c(store) for c in CONNECTOR_CLASSES}

    def get(self, name: str) -> RevenueConnector | None:
        return self.connectors.get(name)

    def configured(self) -> list[RevenueConnector]:
        return [c for c in self.connectors.values() if c.configured() and c.name != "manual"]

    def poll_all(self, *, tick: int) -> list[LedgerEntry]:
        entries: list[LedgerEntry] = []
        for c in self.configured():
            try:
                for ev in c.poll(tick=tick):
                    if ev["amount_cents"] <= 0:
                        continue
                    entries.append(self.ledger.revenue(ev["amount_cents"], venture_id=ev.get("venture_id"), source=c.name, memo=ev.get("memo", ""), external_ref=ev.get("external_ref", ""), tick=tick))
            except Exception as e:  # network or auth problems must not stop the factory
                log.warning("connector %s poll failed: %s", c.name, e)
        return [e for e in entries if e is not None]

    def render(self) -> str:
        parts = []
        for c in self.connectors.values():
            state = "configured" if c.configured() else "NOT configured"
            parts.append(f"- {c.name}: {state}. {c.description}")
        return "\n".join(parts)
