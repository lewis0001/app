"""Command line entry point.

    python -m workhouse seed            # create the org (idempotent)
    python -m workhouse tick [N]        # run N ticks (default 1)
    python -m workhouse run             # tick forever on the configured interval
    python -m workhouse dashboard       # serve the dashboard (and tick on demand)
    python -m workhouse status          # print a summary
    python -m workhouse reset           # delete the database

Environment: ANTHROPIC_API_KEY (live mode), WORKHOUSE_MODE=mock|live,
WORKHOUSE_MODEL, WORKHOUSE_DB, WORKHOUSE_TICK_SECONDS, WORKHOUSE_MAX_TOTAL_COST_USD,
WORKHOUSE_PORT. See config.py.
"""
from __future__ import annotations

import argparse
import logging
import sys

from . import emotions
from .company import build_company
from .config import Settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="workhouse", description="Orbital Workhouse: an autonomous, emotional, hierarchical agent company.")
    parser.add_argument("command", choices=["seed", "tick", "run", "dashboard", "status", "reset"])
    parser.add_argument("n", nargs="?", type=int, default=1, help="ticks to run (for `tick`)")
    parser.add_argument("--db", help="database path (overrides WORKHOUSE_DB)")
    parser.add_argument("--mode", choices=["mock", "live"], help="override mode")
    parser.add_argument("--port", type=int)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    settings = Settings.from_env()
    if args.db:
        from pathlib import Path

        settings.db_path = Path(args.db)
    if args.mode:
        settings.mode = args.mode
    if args.port:
        settings.port = args.port

    if args.command == "reset":
        for suffix in ("", "-wal", "-shm"):
            p = settings.db_path.with_name(settings.db_path.name + suffix)
            if p.exists():
                p.unlink()
        print(f"deleted {settings.db_path}")
        return 0

    engine = build_company(settings)
    if args.command == "seed":
        print(engine.ctx.org.render_chart())
        return 0
    if args.command == "tick":
        for _ in range(max(1, args.n)):
            tl = engine.run_tick()
            print(f"--- tick {tl.tick} ({tl.finished_at}, ${tl.llm_cost_usd:.3f}) ---")
            for ev in tl.events:
                print("  " + ev)
        print(engine.ctx.ledger.render())
        print(engine.ctx.airlock.render())
        return 0
    if args.command == "run":
        print(f"running in {settings.mode} mode every {settings.tick_seconds}s; Ctrl-C to stop")
        try:
            engine.run(on_tick=lambda tl: print(f"tick {tl.tick}: {len(tl.events)} events, ${tl.llm_cost_usd:.3f}"))
        except KeyboardInterrupt:
            pass
        return 0
    if args.command == "dashboard":
        import uvicorn

        from .dashboard.app import create_app

        app = create_app(engine)
        print(f"dashboard on http://{settings.host}:{settings.port}  (mode {settings.mode})")
        uvicorn.run(app, host=settings.host, port=settings.port, log_level="warning")
        return 0
    if args.command == "status":
        snap = engine.snapshot()
        print(f"tick {snap['tick']} | mode {snap['mode']} | {engine.ctx.ledger.render()}")
        print(engine.ctx.org.render_chart())
        for a in engine.store.agents.all():
            print(f"  {a.name:22s} {emotions.mood_label(a.emotion):14s} perf {a.stats.performance:.2f} streak {a.stats.streak:+d}")
        print(engine.ctx.portfolio.render())
        print(engine.ctx.airlock.render())
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
