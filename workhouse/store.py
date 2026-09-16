"""SQLite persistence.

One table per record type: (id TEXT PRIMARY KEY, tick INTEGER, json TEXT).
Records are small and few (hundreds to low thousands), so filtering happens in
Python after loading; simplicity and durability matter more than query speed.
A `kv` table holds scalars such as the current tick.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Callable, Generic, Iterable, TypeVar

from pydantic import BaseModel

from .models import (
    Agent,
    AirlockRequest,
    LedgerEntry,
    Message,
    Pin,
    Review,
    Room,
    Signal,
    Task,
    TickLog,
    TrendSignal,
    Venture,
    WorkProduct,
)

T = TypeVar("T", bound=BaseModel)


class Repo(Generic[T]):
    def __init__(self, store: "Store", table: str, model: type[T], id_attr: str = "id"):
        self.store = store
        self.table = table
        self.model = model
        self.id_attr = id_attr
        with store._lock:
            store._conn.execute(f"CREATE TABLE IF NOT EXISTS {table} (id TEXT PRIMARY KEY, tick INTEGER DEFAULT 0, json TEXT NOT NULL)")
            store._conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_tick ON {table}(tick)")

    def put(self, obj: T) -> T:
        oid = getattr(obj, self.id_attr)
        tick = int(getattr(obj, "tick", getattr(obj, "tick_created", 0)) or 0)
        with self.store._lock:
            self.store._conn.execute(
                f"INSERT INTO {self.table}(id, tick, json) VALUES (?, ?, ?) ON CONFLICT(id) DO UPDATE SET tick=excluded.tick, json=excluded.json",
                (oid, tick, obj.model_dump_json()),
            )
            self.store._maybe_commit()
        return obj

    def put_many(self, objs: Iterable[T]) -> None:
        for o in objs:
            self.put(o)

    def get(self, oid: str) -> T | None:
        with self.store._lock:
            row = self.store._conn.execute(f"SELECT json FROM {self.table} WHERE id=?", (oid,)).fetchone()
        return self.model.model_validate_json(row[0]) if row else None

    def delete(self, oid: str) -> None:
        with self.store._lock:
            self.store._conn.execute(f"DELETE FROM {self.table} WHERE id=?", (oid,))
            self.store._maybe_commit()

    def all(self) -> list[T]:
        with self.store._lock:
            rows = self.store._conn.execute(f"SELECT json FROM {self.table} ORDER BY tick, rowid").fetchall()
        return [self.model.model_validate_json(r[0]) for r in rows]

    def where(self, pred: Callable[[T], bool] | None = None, **eq: Any) -> list[T]:
        out = []
        for obj in self.all():
            if pred is not None and not pred(obj):
                continue
            if all(getattr(obj, k, None) == v for k, v in eq.items()):
                out.append(obj)
        return out

    def first(self, pred: Callable[[T], bool] | None = None, **eq: Any) -> T | None:
        res = self.where(pred, **eq)
        return res[0] if res else None

    def count(self) -> int:
        with self.store._lock:
            return int(self.store._conn.execute(f"SELECT COUNT(*) FROM {self.table}").fetchone()[0])

    def since(self, tick: int) -> list[T]:
        with self.store._lock:
            rows = self.store._conn.execute(f"SELECT json FROM {self.table} WHERE tick >= ? ORDER BY tick, rowid", (tick,)).fetchall()
        return [self.model.model_validate_json(r[0]) for r in rows]


class Store:
    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL") if self.path != ":memory:" else None
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._lock = threading.RLock()
        self._tx_depth = 0
        self._conn.execute("CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL)")

        self.agents: Repo[Agent] = Repo(self, "agents", Agent)
        self.rooms: Repo[Room] = Repo(self, "rooms", Room, id_attr="key")
        self.tasks: Repo[Task] = Repo(self, "tasks", Task)
        self.work: Repo[WorkProduct] = Repo(self, "work_products", WorkProduct)
        self.reviews: Repo[Review] = Repo(self, "reviews", Review)
        self.signals: Repo[Signal] = Repo(self, "signals", Signal)
        self.ventures: Repo[Venture] = Repo(self, "ventures", Venture)
        self.ledger: Repo[LedgerEntry] = Repo(self, "ledger", LedgerEntry)
        self.trends: Repo[TrendSignal] = Repo(self, "trends", TrendSignal)
        self.airlock: Repo[AirlockRequest] = Repo(self, "airlock", AirlockRequest)
        self.messages: Repo[Message] = Repo(self, "messages", Message)
        self.pins: Repo[Pin] = Repo(self, "pins", Pin, id_attr="key")
        self.ticks: Repo[TickLog] = Repo(self, "ticks", TickLog, id_attr="tick")

    # -- transactions -------------------------------------------------------
    def _maybe_commit(self) -> None:
        # autocommit mode (isolation_level=None) unless inside `transaction()`
        if self._tx_depth == 0 and self._conn.in_transaction:
            self._conn.commit()

    class _Tx:
        def __init__(self, store: "Store"):
            self.store = store

        def __enter__(self):
            self.store._lock.acquire()
            if self.store._tx_depth == 0:
                self.store._conn.execute("BEGIN")
            self.store._tx_depth += 1
            return self.store

        def __exit__(self, exc_type, exc, tb):
            self.store._tx_depth -= 1
            try:
                if self.store._tx_depth == 0:
                    if exc_type is None:
                        self.store._conn.execute("COMMIT")
                    else:
                        self.store._conn.execute("ROLLBACK")
            finally:
                self.store._lock.release()
            return False

    def transaction(self) -> "Store._Tx":
        return Store._Tx(self)

    # -- key/value ----------------------------------------------------------
    def get_kv(self, key: str, default: Any = None) -> Any:
        with self._lock:
            row = self._conn.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set_kv(self, key: str, value: Any) -> None:
        with self._lock:
            self._conn.execute("INSERT INTO kv(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, json.dumps(value)))
            self._maybe_commit()

    @property
    def tick(self) -> int:
        return int(self.get_kv("tick", 0))

    @tick.setter
    def tick(self, value: int) -> None:
        self.set_kv("tick", int(value))

    def close(self) -> None:
        with self._lock:
            self._conn.close()
