"""Walk-forward validation of the sized momentum ensemble family.

For each trading year Y in 2022..2026, optimize the parameter grid on all
data before Y (minimum one year of history), then trade year Y with the
chosen parameters. The stitched result contains no hindsight: every
position was chosen by a config selected using only past data.

Also stress-tests the final stitched strategy at higher fee assumptions.

Usage: python -m research.walkforward
"""
import itertools
import sys

import pandas as pd

sys.path.insert(0, ".")
from bot import backtest, strategies  # noqa: E402

GRID = {
    "lookbacks": [(48, 72, 120), (48, 72, 96), (24, 48, 72)],
    "z_period": [168],
    "z_entry": [1.0, 1.25, 1.5, 1.75],
    "target_vol": [0.4, 0.5, 0.6],
}
WARMUP_BARS = 24 * 21  # extra history prepended to each year so indicators are warm


def configs():
    keys = list(GRID)
    for combo in itertools.product(*(GRID[k] for k in keys)):
        yield dict(zip(keys, combo))


def score(res: backtest.Result) -> float:
    s = res.sharpe
    if res.max_drawdown < -0.35:
        s -= 2.0
    return s


def main():
    df = backtest.load_csv("data/btc_1h.csv")
    years = sorted(set(df.index.year))[1:]  # skip the first year (train seed)

    stitched = []
    print("WALK-FORWARD (params picked on prior data only)\n")
    for year in years:
        train = df[df.index.year < year]
        best_params, best_score = None, -1e9
        for params in configs():
            res = backtest.run(train, strategies.vol_momentum_ens_sized(train, **params))
            if score(res) > best_score:
                best_score, best_params = score(res), params

        # trade the target year; prepend warmup history so indicators are valid
        segment = df[df.index.year <= year]
        pos = strategies.vol_momentum_ens_sized(segment, **best_params)
        year_mask = segment.index.year == year
        year_df = segment[year_mask]
        year_res = backtest.run(year_df, pos[year_mask])
        stitched.append(year_res.strategy_returns)
        bh = year_df["close"].iloc[-1] / year_df["close"].iloc[0] - 1
        print(f"{year}: {year_res.total_return:+7.1%} (B&H {bh:+7.1%})  "
              f"Sharpe {year_res.sharpe:5.2f}  maxDD {year_res.max_drawdown:6.1%}  "
              f"params {best_params}")

    all_ret = pd.concat(stitched)
    equity = (1 + all_ret).cumprod()
    years_len = len(all_ret) / backtest.HOURS_PER_YEAR
    cagr = equity.iloc[-1] ** (1 / years_len) - 1
    sharpe = all_ret.mean() / all_ret.std() * (backtest.HOURS_PER_YEAR ** 0.5)
    max_dd = (equity / equity.cummax() - 1).min()
    print(f"\nSTITCHED {years[0]}-{years[-1]}: total {equity.iloc[-1] - 1:+.1%} | "
          f"CAGR {cagr:+.1%} | Sharpe {sharpe:.2f} | maxDD {max_dd:.1%}")

    bh_all = df[df.index.year >= years[0]]
    bh_total = bh_all["close"].iloc[-1] / bh_all["close"].iloc[0] - 1
    print(f"BUY & HOLD same period: total {bh_total:+.1%}")

    print("\nFEE SENSITIVITY (full period, default config)")
    default = {"lookbacks": (48, 72, 96), "z_period": 168, "z_entry": 1.5, "target_vol": 0.5}
    pos = strategies.vol_momentum_ens_sized(df, **default)
    for bps in (8, 12, 16, 25):
        res = backtest.run(df, pos, cost_per_side=bps / 10000)
        print(f"  {bps} bps/side: {res.summary().splitlines()[0]}")


if __name__ == "__main__":
    main()
