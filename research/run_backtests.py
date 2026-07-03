"""Parameter sweep + out-of-sample validation for the strategy library.

Train period : 2021-01 .. 2023-12  (bull 2021, bear 2022, recovery 2023)
Test period  : 2024-01 .. present  (bull 2024, decline 2025-26)

Configs are ranked on the TRAIN period only (Sharpe with a drawdown floor),
then the best of each family is re-run on the untouched TEST period.

Usage: python -m research.run_backtests [--top 3]
"""
import argparse
import itertools
import sys

import pandas as pd

sys.path.insert(0, ".")
from bot import backtest, strategies  # noqa: E402

TRAIN_END = "2024-01-01"


def sweep_space():
    """(family, param-dict) pairs for every config in the sweep."""
    space = {
        "ema_cross": {
            "fast": [12, 24, 48],
            "slow": [72, 96, 168, 336],
        },
        "donchian_breakout": {
            "entry": [24, 48, 96, 168],
            "exit_": [12, 24, 48],
        },
        "donchian_atr": {
            "entry": [24, 48, 96, 168],
            "atr_period": [24, 48],
            "atr_mult": [2.0, 3.0, 4.0],
        },
        "rsi_dip": {
            "rsi_period": [2, 3, 4],
            "buy_below": [10, 20, 30],
            "sell_above": [70, 80, 90],
            "trend_ema": [96, 168, 336],
        },
        "vol_momentum": {
            "lookback": [24, 48, 72, 120],
            "z_period": [168, 336],
            "z_entry": [0.5, 1.0, 1.5],
        },
        "regime_hybrid": {
            "er_period": [48, 72, 120],
            "er_threshold": [0.2, 0.25, 0.3],
            "slow": [96, 168],
        },
    }
    for family, grid in space.items():
        keys = list(grid)
        for combo in itertools.product(*(grid[k] for k in keys)):
            yield family, dict(zip(keys, combo))


def score(res: backtest.Result) -> float:
    """Rank key: Sharpe, hard-penalized if drawdown is worse than -40%
    or the strategy barely trades."""
    s = res.sharpe
    if res.max_drawdown < -0.40:
        s -= 2.0
    if res.n_trades < 30:
        s -= 2.0
    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--data", default="data/btc_1h.csv")
    args = parser.parse_args()

    df = backtest.load_csv(args.data)
    train = df[df.index < TRAIN_END]
    test = df[df.index >= TRAIN_END]
    print(f"train: {train.index[0]:%Y-%m-%d} .. {train.index[-1]:%Y-%m-%d} ({len(train)} bars)")
    print(f"test : {test.index[0]:%Y-%m-%d} .. {test.index[-1]:%Y-%m-%d} ({len(test)} bars)\n")

    bh_train = backtest.run(train, strategies.buy_and_hold(train))
    bh_test = backtest.run(test, strategies.buy_and_hold(test))
    print(f"benchmark buy&hold train: {bh_train.summary()}")
    print(f"benchmark buy&hold test : {bh_test.summary()}\n")

    rows = []
    for family, params in sweep_space():
        fn = getattr(strategies, family)
        res = backtest.run(train, fn(train, **params))
        rows.append({"family": family, "params": params, "train": res, "score": score(res)})
    ranked = pd.DataFrame(rows).sort_values("score", ascending=False)

    print("=" * 100)
    print(f"TOP {args.top} PER FAMILY (ranked on train, validated on test)")
    print("=" * 100)
    for family in ranked["family"].unique():
        fam = ranked[ranked["family"] == family].head(args.top)
        print(f"\n### {family}")
        for _, row in fam.iterrows():
            fn = getattr(strategies, family)
            test_res = backtest.run(test, fn(test, **row["params"]))
            full_res = backtest.run(df, fn(df, **row["params"]))
            print(f"  params: {row['params']}")
            print(f"    train: {row['train'].summary()}")
            print(f"    test : {test_res.summary()}")
            print(f"    full : {full_res.summary()}")


if __name__ == "__main__":
    main()
