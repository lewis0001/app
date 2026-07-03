"""The 500x question, answered with data instead of opinions.

1. Survival analysis: entering at a random bar, what fraction of positions
   escape liquidation for 1h/4h/24h at each leverage tier — giving the
   trade the BENEFIT of picking the luckier direction each time.
2. Strategy simulation: the validated hourly momentum ensemble run through
   the isolated-margin liquidation engine at each leverage tier.

Usage: python -m research2.leverage_reality
"""
import sys

sys.path.insert(0, ".")
from bot import backtest, leverage, strategies  # noqa: E402
from bot.live import CONFIG  # noqa: E402


def main():
    df15 = backtest.load_csv("data/btc_15m.csv")
    recent15 = df15[df15.index >= "2025-01-01"]

    print("SURVIVAL: fraction of random entries NOT liquidated within horizon")
    print("(15m bars, 2025-2026 data, taking the LUCKIER direction each entry)\n")
    print(f"{'leverage':>9} {'liq. distance':>14} {'1h':>7} {'4h':>7} {'24h':>7}")
    for lev in (10, 25, 50, 100, 200, 500):
        s = leverage.liquidation_survival(recent15, lev)
        print(f"{lev:>8}x {0.9 / lev:>13.2%} {s['1h']:>7.1%} {s['4h']:>7.1%} {s['24h']:>7.1%}")

    print("\nSTRATEGY AT LEVERAGE: hourly momentum ensemble, full margin per trade,")
    print("5 bps taker fees on notional, intrabar liquidation (2021-2026, 1h bars)\n")
    df1h = backtest.load_csv("data/btc_1h.csv")
    pos = strategies.vol_momentum_ens_sized(df1h, **CONFIG)
    for lev in (1, 2, 3, 5, 10, 25, 50, 100, 500):
        res = leverage.run_leveraged(df1h, pos, leverage=lev,
                                     bars_per_year=24 * 365)
        print(f"{lev:>4}x: {res.summary()}")


if __name__ == "__main__":
    main()
