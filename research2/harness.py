"""Shared evaluation harness for the 15-minute intraday strategy search.

Contract for strategy modules (research2/strategies/<name>.py):

    def signal(df: pd.DataFrame, **params) -> pd.Series
        # df: 15m OHLCV, UTC DatetimeIndex, columns open/high/low/close/volume
        # return: target position in [-1, 1], index-aligned with df,
        #         computed CAUSALLY (row t may only use data up to row t;
        #         the harness applies it to bar t+1, but any .shift(-1),
        #         centered rolling, or use of the current bar's high/low
        #         to decide an entry AT that bar's own price is lookahead)
    PARAM_GRID = {"param_name": [values...], ...}   # swept by evaluate_module

Periods:
    TRAIN  2021-01 .. 2024-12   (parameter selection ONLY here)
    RECENT 2025-01 .. present   (untouched validation; this decides survival —
                                 the user trades NOW, so RECENT is weighted most)

Selection rule used by evaluate_module: rank by train score
(Sharpe with drawdown/inactivity penalties); report recent + full metrics
for the top configs. A strategy 'passes' only if the TRAIN-chosen config
is also profitable on RECENT with >= 0.7 trades/day.
"""
import importlib
import itertools
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from bot import backtest  # noqa: E402

TRAIN_END = "2025-01-01"
BARS_PER_YEAR = 4 * 24 * 365
COST_PER_SIDE = 0.0005  # 5 bps taker+slippage on a liquid BTC perp
DATA = "data/btc_15m.csv"


def load_data(path: str = DATA) -> pd.DataFrame:
    return backtest.load_csv(path)


def metrics(df: pd.DataFrame, position: pd.Series,
            cost_per_side: float = COST_PER_SIDE) -> dict:
    pos = position.reindex(df.index).fillna(0.0).clip(-1, 1)
    held = pos.shift(1).fillna(0.0)
    asset_ret = df["close"].pct_change().fillna(0.0)
    turnover = pos.diff().abs().fillna(pos.abs())
    strat_ret = held * asset_ret - (turnover * cost_per_side).shift(1).fillna(0.0)
    equity = (1 + strat_ret).cumprod()
    years = len(df) / BARS_PER_YEAR
    vol = strat_ret.std()
    entries = ((pos != 0) & (pos.shift(1).fillna(0.0) == 0)).sum()
    days = max((df.index[-1] - df.index[0]).days, 1)
    yearly = (1 + strat_ret).groupby(df.index.year).prod() - 1
    return {
        "total_return": float(equity.iloc[-1] - 1),
        "cagr": float(equity.iloc[-1] ** (1 / years) - 1) if equity.iloc[-1] > 0 else -1.0,
        "sharpe": float(strat_ret.mean() / vol * np.sqrt(BARS_PER_YEAR)) if vol > 0 else 0.0,
        "max_drawdown": float((equity / equity.cummax() - 1).min()),
        "trades_per_day": float(entries / days),
        "exposure": float((held != 0).mean()),
        "yearly": {int(y): round(float(r), 4) for y, r in yearly.items()},
    }


def score(m: dict) -> float:
    s = m["sharpe"]
    if m["max_drawdown"] < -0.35:
        s -= 2.0
    if m["trades_per_day"] < 0.7:   # user wants daily trades
        s -= 2.0
    if m["trades_per_day"] > 15:    # fee suicide zone
        s -= 1.0
    return s


def evaluate_module(module_name: str, top: int = 5, data_path: str = DATA) -> dict:
    """Sweep the module's PARAM_GRID on TRAIN, validate top configs on RECENT."""
    mod = importlib.import_module(f"research2.strategies.{module_name}")
    df = load_data(data_path)
    train = df[df.index < TRAIN_END]
    recent = df[df.index >= TRAIN_END]

    keys = list(mod.PARAM_GRID)
    results = []
    for combo in itertools.product(*(mod.PARAM_GRID[k] for k in keys)):
        params = dict(zip(keys, combo))
        m_train = metrics(train, mod.signal(train, **params))
        results.append({"params": params, "train": m_train, "score": score(m_train)})
    results.sort(key=lambda r: r["score"], reverse=True)

    report = {"module": module_name, "n_configs": len(results), "top": []}
    for r in results[:top]:
        pos_recent = mod.signal(recent, **r["params"])
        pos_full = mod.signal(df, **r["params"])
        entry = {
            "params": r["params"],
            "train": r["train"],
            "recent": metrics(recent, pos_recent),
            "full": metrics(df, pos_full),
        }
        entry["passes"] = (
            entry["train"]["sharpe"] > 0.5
            and entry["recent"]["total_return"] > 0
            and entry["recent"]["sharpe"] > 0.3
            and entry["recent"]["trades_per_day"] >= 0.7
        )
        report["top"].append(entry)
    return report


if __name__ == "__main__":
    print(json.dumps(evaluate_module(sys.argv[1]), indent=2, default=str))
