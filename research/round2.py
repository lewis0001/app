"""Round 2: refine the winning family (volatility-gated momentum) with
hysteresis exits, vol targeting, and lookback ensembles.

Ranked on train (2021-2023), validated on test (2024-2026).
Usage: python -m research.round2
"""
import itertools
import sys

sys.path.insert(0, ".")
from bot import backtest, strategies  # noqa: E402

TRAIN_END = "2024-01-01"


def main():
    df = backtest.load_csv("data/btc_1h.csv")
    train = df[df.index < TRAIN_END]
    test = df[df.index >= TRAIN_END]

    grids = {
        "vol_momentum_v2": {
            "lookback": [48, 72, 120],
            "z_period": [168, 336],
            "z_entry": [1.25, 1.5, 2.0],
            "z_exit": [0.0, 0.25, 0.5],
            "target_vol": [0.4, 0.6, 0.8],
        },
        "vol_momentum_ensemble": {
            "lookbacks": [(48, 72, 120), (24, 72, 168), (48, 96, 168)],
            "z_period": [168, 336],
            "z_entry": [1.25, 1.5, 2.0],
            "z_exit": [0.0, 0.25, 0.5],
            "target_vol": [0.4, 0.6, 0.8],
        },
    }

    rows = []
    for family, grid in grids.items():
        fn = getattr(strategies, family)
        keys = list(grid)
        for combo in itertools.product(*(grid[k] for k in keys)):
            params = dict(zip(keys, combo))
            tr = backtest.run(train, fn(train, **params))
            # rank purely on train, with a drawdown guardrail
            s = tr.sharpe - (2.0 if tr.max_drawdown < -0.35 else 0.0)
            rows.append((s, family, params, tr))

    rows.sort(key=lambda r: r[0], reverse=True)
    print("TOP 8 ON TRAIN (with test + full validation)\n")
    for s, family, params, tr in rows[:8]:
        fn = getattr(strategies, family)
        te = backtest.run(test, fn(test, **params))
        fu = backtest.run(df, fn(df, **params))
        print(f"{family} {params}")
        print(f"  train: {tr.summary()}")
        print(f"  test : {te.summary()}")
        print(f"  full : {fu.summary()}\n")


if __name__ == "__main__":
    main()
