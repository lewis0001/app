"""The Airlock: the only place where the human is needed.

Agents cannot create accounts, pass KYC, sign contracts, move money out,
publish under the operator's name, or reveal the operator's unique assets.
When a task needs one of those, it raises an AirlockRequest. The dashboard
lists them with the fields to fill; the engine consumes the answers.
"""
from __future__ import annotations

from typing import Any

from .models import AirlockField, AirlockRequest, AirlockStatus, now_iso
from .store import Store

# Human oversight has a capacity: past this many *new* requests in one tick the
# rest are deferred (kept as tasks/messages) instead of flooding the queue.
MAX_NEW_PER_TICK = 4

REQUEST_TYPES: dict[str, dict[str, Any]] = {
    "connect_rail": {"label": "Connect a revenue rail", "priority": 1},
    "approve_spend": {"label": "Approve spending money", "priority": 1},
    "approve_publish": {"label": "Approve publishing externally", "priority": 2},
    "provide_credential": {"label": "Provide an API key or login", "priority": 2},
    "create_account": {"label": "Create an account (needs a human/KYC)", "priority": 2},
    "legal_check": {"label": "Legal or terms-of-service check", "priority": 2},
    "manual_action": {"label": "Do something in the physical or human world", "priority": 3},
    "operator_asset": {"label": "Tell us about an asset only you have", "priority": 3},
    "raise_cap": {"label": "Raise the LLM spend cap", "priority": 1},
    "enter_revenue": {"label": "Record revenue received outside a connector", "priority": 4},
    "decision": {"label": "Make a judgement call the agents cannot", "priority": 3},
}


class Airlock:
    def __init__(self, store: Store):
        self.store = store
        self._last_tick = -1
        self._new_this_tick = 0

    def request(self, type: str, title: str, description: str, *, fields: list[AirlockField] | None = None, why_it_matters: str = "", venture_id: str | None = None, task_id: str | None = None, requested_by: str = "system", tick: int = 0, priority: int | None = None, dedupe_key: str | None = None, meta: dict[str, Any] | None = None) -> AirlockRequest:
        if type not in REQUEST_TYPES:
            type = "decision"
        # Do not spam the human with the same open request.
        key = dedupe_key or f"{type}:{title.strip().lower()}"
        existing = self.store.airlock.first(lambda r: r.status == AirlockStatus.open and r.response.get("_dedupe") == key)
        if existing:
            return existing
        if tick != self._last_tick:
            self._last_tick, self._new_this_tick = tick, 0
        self._new_this_tick += 1
        if self._new_this_tick > MAX_NEW_PER_TICK and priority is None and REQUEST_TYPES[type]["priority"] >= 3:
            # Low-priority overflow: record it as a deferred request that surfaces next tick.
            tick = tick + 1
        req = AirlockRequest(
            type=type,
            title=title.strip(),
            description=description.strip(),
            why_it_matters=why_it_matters.strip(),
            fields=fields or [],
            venture_id=venture_id,
            task_id=task_id,
            requested_by=requested_by,
            priority=priority if priority is not None else REQUEST_TYPES[type]["priority"],
            tick=tick,
            response={"_dedupe": key, **{f"_{k}": v for k, v in (meta or {}).items()}},
        )
        return self.store.airlock.put(req)

    def open(self) -> list[AirlockRequest]:
        reqs = self.store.airlock.where(status=AirlockStatus.open)
        reqs.sort(key=lambda r: (r.priority, r.tick))
        return reqs

    def resolve(self, request_id: str, response: dict[str, Any], *, tick: int = 0) -> AirlockRequest | None:
        req = self.store.airlock.get(request_id)
        if not req or req.status != AirlockStatus.open:
            return req
        missing = [f.label for f in req.fields if f.required and response.get(f.name) in (None, "")]
        if missing:
            raise ValueError("missing required fields: " + ", ".join(missing))
        req.response = {**req.response, **response, "_resolved_at": now_iso()}
        req.status = AirlockStatus.resolved
        req.resolved_tick = tick
        return self.store.airlock.put(req)

    def dismiss(self, request_id: str, *, reason: str = "", tick: int = 0) -> AirlockRequest | None:
        req = self.store.airlock.get(request_id)
        if not req or req.status != AirlockStatus.open:
            return req
        req.status = AirlockStatus.dismissed
        req.response = {**req.response, "_reason": reason}
        req.resolved_tick = tick
        return self.store.airlock.put(req)

    def unconsumed(self) -> list[AirlockRequest]:
        """Resolved requests the engine has not yet acted on."""
        return self.store.airlock.where(lambda r: r.status == AirlockStatus.resolved and not r.response.get("_consumed"))

    def mark_consumed(self, req: AirlockRequest) -> None:
        req.response["_consumed"] = True
        self.store.airlock.put(req)

    def open_for_venture(self, venture_id: str) -> list[AirlockRequest]:
        return [r for r in self.open() if r.venture_id == venture_id]

    def render(self, limit: int = 8) -> str:
        reqs = self.open()[:limit]
        if not reqs:
            return "Airlock: nothing waiting on the human."
        return "Airlock (waiting on the human):\n" + "\n".join(f"- [{r.type}] {r.title} (priority {r.priority}, since tick {r.tick})" for r in reqs)
