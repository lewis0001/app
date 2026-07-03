"""Live trading bot: volatility-targeted momentum ensemble on BTC-USD 1h.

Runs the exact strategy that was walk-forward validated in research/:
an ensemble of 48/72/96-hour z-score momentum signals, sized by a 60% p.a.
volatility target. Long when the recent move is statistically large and up,
short when large and down, sized down when realized vol spikes.

Modes:
  paper (default) — computes the target position each hour and simulates
      fills against the live price, persisting equity/position to a state
      file. No keys, no risk.
  live — prints the orders that WOULD be sent. Wire up your exchange in
      ExchangeExecutor.execute() before trusting it with money.

Usage:
    python -m bot.live              # one evaluation (cron-friendly)
    python -m bot.live --loop       # evaluate every hour, forever
    python -m bot.live --mode live  # execution stub (prints orders)
"""
import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from . import strategies

log = logging.getLogger("btcbot")

API = "https://api.exchange.coinbase.com/products/{product}/candles"
GRANULARITY = 3600

# Final configuration chosen by walk-forward optimization (research/walkforward.py).
CONFIG = {
    "lookbacks": (48, 72, 96),
    "z_period": 168,
    "z_entry": 1.5,
    "target_vol": 0.5,
}
COST_PER_SIDE = 0.0008  # 8 bps fee+slippage assumption for paper fills
HISTORY_BARS = 600      # >= z_period + max lookback + vol_period, with margin


def fetch_candles(product: str = "BTC-USD", bars: int = HISTORY_BARS) -> pd.DataFrame:
    """Fetch the most recent `bars` hourly candles (paginated, newest last).
    Drops the still-forming current hour so signals only use closed candles."""
    session = requests.Session()
    rows = {}
    end = int(time.time() // GRANULARITY * GRANULARITY)
    start_needed = end - bars * GRANULARITY
    cursor = end
    while cursor > start_needed:
        chunk_start = max(cursor - 300 * GRANULARITY, start_needed)
        resp = session.get(
            API.format(product=product),
            params={
                "granularity": GRANULARITY,
                "start": datetime.fromtimestamp(chunk_start, tz=timezone.utc).isoformat(),
                "end": datetime.fromtimestamp(cursor, tz=timezone.utc).isoformat(),
            },
            timeout=30,
        )
        resp.raise_for_status()
        for ts, low, high, open_, close, vol in resp.json():
            rows[int(ts)] = (open_, high, low, close, vol)
        cursor = chunk_start
        time.sleep(0.15)

    df = pd.DataFrame.from_dict(rows, orient="index",
                                columns=["open", "high", "low", "close", "volume"])
    df.index = pd.to_datetime(df.index, unit="s", utc=True)
    df = df.sort_index().astype(float)
    now_bucket = pd.Timestamp.now(tz="UTC").floor("h")
    return df[df.index < now_bucket]


def target_position(df: pd.DataFrame) -> float:
    """Target position in [-1, 1] as a fraction of account equity."""
    pos = strategies.vol_momentum_ens_sized(df, **CONFIG)
    return float(pos.iloc[-1])


class PaperExecutor:
    """Simulates fills and tracks equity in a JSON state file."""

    def __init__(self, state_path: str = "bot_state.json"):
        self.path = Path(state_path)
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
        else:
            self.state = {"position": 0.0, "equity": 1.0, "last_price": None,
                          "last_ts": None, "trades": []}

    def execute(self, target: float, price: float, ts: str):
        st = self.state
        # mark to market since the last evaluation
        if st["last_price"] is not None and st["position"] != 0:
            st["equity"] *= 1 + st["position"] * (price / st["last_price"] - 1)
        turnover = abs(target - st["position"])
        if turnover > 1e-9:
            st["equity"] *= 1 - turnover * COST_PER_SIDE
            st["trades"].append({"ts": ts, "from": st["position"], "to": target,
                                 "price": price})
            st["trades"] = st["trades"][-500:]
            log.info("PAPER FILL %+0.2f -> %+0.2f @ %.2f", st["position"], target, price)
        st.update({"position": target, "last_price": price, "last_ts": ts})
        self.path.write_text(json.dumps(st, indent=2))
        log.info("equity=%.4f position=%+0.2f price=%.2f", st["equity"], target, price)


class ExchangeExecutor:
    """Stub for real execution. Prints the order it would place.

    To go live, implement `execute` against your venue (e.g. ccxt:
    create a market order for (target - current) * equity / price BTC on a
    perpetual, or spot with margin for shorts). Start with tiny size.
    """

    def execute(self, target: float, price: float, ts: str):
        log.warning("LIVE MODE STUB — would set position to %+0.2f of equity @ ~%.2f",
                    target, price)


def evaluate_once(executor, product: str = "BTC-USD"):
    df = fetch_candles(product)
    if len(df) < CONFIG["z_period"] + max(CONFIG["lookbacks"]) + 10:
        raise RuntimeError(f"not enough history: {len(df)} bars")
    target = target_position(df)
    price = float(df["close"].iloc[-1])
    ts = df.index[-1].isoformat()
    log.info("bar %s close=%.2f -> target position %+0.2f", ts, price, target)
    executor.execute(target, price, ts)
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["paper", "live"], default="paper")
    parser.add_argument("--loop", action="store_true", help="run every hour forever")
    parser.add_argument("--product", default="BTC-USD")
    parser.add_argument("--state", default="bot_state.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    executor = PaperExecutor(args.state) if args.mode == "paper" else ExchangeExecutor()

    while True:
        try:
            evaluate_once(executor, args.product)
        except Exception:
            log.exception("evaluation failed; will retry next cycle")
        if not args.loop:
            break
        # wake shortly after the top of the next hour so the candle is closed
        now = time.time()
        sleep_s = GRANULARITY - (now % GRANULARITY) + 30
        log.info("sleeping %.0f s until next hourly candle", sleep_s)
        time.sleep(sleep_s)


if __name__ == "__main__":
    main()
