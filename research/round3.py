"""Round 3: keep the robust always-in-market momentum signal, fix drawdown
with volatility-targeted sizing and lookback ensembles.

Usage: python -m research.round3
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
        "vol_momentum_sized": {
            "lookback": [48, 72, 96],
            "z_period": [168],
            "z_entry": [1.0, 1.25, 1.5, 1.75],
            "target_vol": [0.4, 0.5, 0.6, 0.8],
        },
        "vol_momentum_ens_sized": {
            "lookbacks": [(48, 72, 120), (48, 72, 96), (24, 48, 72)],
            "z_period": [168],
            "z_entry": [1.0, 1.25, 1.5, 1.75],
            "target_vol": [0.4, 0.5, 0.6, 0.8],
        },
    }

    rows = []
    for family, grid in grids.items():
        fn = getattr(strategies, family)
        keys = list(grid)
        for combo in itertools.product(*(grid[k] for k in keys)):
            params = dict(zip(keys, combo))
            tr = backtest.run(train, fn(train, **params))
            s = tr.sharpe - (2.0 if tr.max_drawdown < -0.35 else 0.0)
            rows.append((s, family, params, tr))

    rows.sort(key=lambda r: r[0], reverse=True)
    print("TOP 8 ON TRAIN (validated on test + full)\n")
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
