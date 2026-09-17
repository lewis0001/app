"""FastAPI dashboard: the human's window into the factory and the Airlock.

    python -m workhouse dashboard

Serves a single static page and a JSON API. A background thread can tick the
engine on an interval so the whole thing runs unattended.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from ..engine import Engine
from ..models import AirlockField, SignalKind

STATIC = Path(__file__).parent / "static"


class ResolveBody(BaseModel):
    response: dict[str, Any] = {}


class RevenueBody(BaseModel):
    amount_usd: float
    memo: str = ""
    venture_id: str | None = None


class MessageBody(BaseModel):
    content: str
    to: str = "all"


class RunBody(BaseModel):
    interval_seconds: float | None = None


class Runner:
    def __init__(self, engine: Engine):
        self.engine = engine
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.stop_flag = threading.Event()
        self.interval = engine.settings.tick_seconds

    def tick(self) -> dict[str, Any]:
        with self.lock:
            tl = self.engine.run_tick()
        return tl.model_dump()

    def start(self, interval: float | None = None) -> None:
        if self.thread and self.thread.is_alive():
            return
        if interval:
            self.interval = interval
        self.stop_flag.clear()

        def loop() -> None:
            while not self.stop_flag.is_set():
                try:
                    self.tick()
                except Exception:  # never die
                    pass
                self.stop_flag.wait(self.interval)

        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_flag.set()

    @property
    def running(self) -> bool:
        return bool(self.thread and self.thread.is_alive() and not self.stop_flag.is_set())


def create_app(engine: Engine) -> FastAPI:
    app = FastAPI(title="Orbital Workhouse")
    runner = Runner(engine)
    app.state.engine = engine
    app.state.runner = runner

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (STATIC / "index.html").read_text()

    @app.get("/api/state")
    def state() -> JSONResponse:
        snap = engine.snapshot()
        snap["running"] = runner.running
        snap["interval_seconds"] = runner.interval
        return JSONResponse(snap)

    @app.post("/api/tick")
    def tick() -> dict[str, Any]:
        return runner.tick()

    @app.post("/api/run")
    def run(body: RunBody) -> dict[str, Any]:
        runner.start(body.interval_seconds)
        return {"running": runner.running, "interval_seconds": runner.interval}

    @app.post("/api/stop")
    def stop() -> dict[str, Any]:
        runner.stop()
        return {"running": False}

    @app.post("/api/airlock/{request_id}/resolve")
    def resolve(request_id: str, body: ResolveBody) -> dict[str, Any]:
        try:
            with runner.lock:
                req = engine.ctx.airlock.resolve(request_id, body.response, tick=engine.store.tick)
        except ValueError as e:
            raise HTTPException(400, str(e))
        if not req:
            raise HTTPException(404, "no such request")
        return req.model_dump()

    @app.post("/api/airlock/{request_id}/dismiss")
    def dismiss(request_id: str, body: ResolveBody) -> dict[str, Any]:
        with runner.lock:
            req = engine.ctx.airlock.dismiss(request_id, reason=str(body.response.get("reason", "")), tick=engine.store.tick)
        if not req:
            raise HTTPException(404, "no such request")
        return req.model_dump()

    @app.post("/api/revenue")
    def revenue(body: RevenueBody) -> dict[str, Any]:
        cents = int(round(body.amount_usd * 100))
        if cents <= 0:
            raise HTTPException(400, "amount must be positive")
        with runner.lock:
            entry = engine.ctx.ledger.revenue(cents, venture_id=body.venture_id, source="manual", memo=body.memo or "manual entry", tick=engine.store.tick)
            engine._revenue_signals(body.venture_id, cents)
        return entry.model_dump()

    @app.post("/api/operator")
    def operator(body: dict[str, Any]) -> dict[str, Any]:
        with runner.lock:
            engine.ctx.operator.merge(body)
            engine.ctx.operator.save(engine.settings.operator_profile_path)
        return engine.ctx.operator.model_dump()

    @app.post("/api/message")
    def message(body: MessageBody) -> dict[str, Any]:
        with runner.lock:
            msg = engine.ctx.bus.send("human", body.to, body.content, channel="human", tick=engine.store.tick)
            if body.to == "all":
                engine.ctx.bus.pin("operator_note", body.content[:400], pinned_by="human", tick=engine.store.tick)
        return msg.model_dump()

    @app.post("/api/signal")
    def signal(body: dict[str, Any]) -> dict[str, Any]:
        """The human can praise or criticise an agent directly."""
        kind = SignalKind.positive if str(body.get("kind", "positive")) == "positive" else SignalKind.negative
        with runner.lock:
            sig = engine.ctx.signal(str(body.get("agent_id")), kind, float(body.get("magnitude", 0.5)), "human", str(body.get("reason", "operator feedback")))
        if not sig:
            raise HTTPException(404, "no such agent")
        return sig.model_dump()

    @app.get("/api/task/{task_id}")
    def task(task_id: str) -> dict[str, Any]:
        t = engine.store.tasks.get(task_id)
        if not t:
            raise HTTPException(404, "no such task")
        return {"task": t.model_dump(), "work": [w.model_dump() for w in engine.store.work.where(task_id=task_id)], "reviews": [r.model_dump() for r in engine.store.reviews.where(task_id=task_id)]}

    @app.get("/api/venture/{venture_id}")
    def venture(venture_id: str) -> dict[str, Any]:
        v = engine.store.ventures.get(venture_id)
        if not v:
            raise HTTPException(404, "no such venture")
        tasks = engine.store.tasks.where(venture_id=venture_id)
        return {"venture": v.model_dump(), "tasks": [t.model_dump() for t in tasks], "work": [w.model_dump() for t in tasks for w in engine.store.work.where(task_id=t.id)]}

    return app
