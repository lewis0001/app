"""Render the equity-curve comparison chart for the README.

Usage: python -m research.make_chart
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, ".")
from bot import backtest, strategies  # noqa: E402
from bot.live import CONFIG  # noqa: E402

STRATEGY_COLOR = "#2a78d6"
BENCH_COLOR = "#1baf7a"
TEXT = "#3d3d3a"
MUTED = "#6f6e64"
GRID = "#e8e7df"


def main():
    df = backtest.load_csv("data/btc_1h.csv")
    strat = backtest.run(df, strategies.vol_momentum_ens_sized(df, **CONFIG))
    bench = backtest.run(df, strategies.buy_and_hold(df))

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(strat.equity.index, strat.equity, color=STRATEGY_COLOR, lw=2,
            label="Momentum ensemble (this bot)")
    ax.plot(bench.equity.index, bench.equity, color=BENCH_COLOR, lw=2,
            label="Buy & hold BTC")
    ax.set_yscale("log")

    # direct end labels (relief for the low-contrast aqua)
    for res, color, name in [(strat, STRATEGY_COLOR, f"{strat.equity.iloc[-1]:.1f}x"),
                             (bench, BENCH_COLOR, f"{bench.equity.iloc[-1]:.1f}x")]:
        ax.annotate(name, (res.equity.index[-1], res.equity.iloc[-1]),
                    xytext=(8, 0), textcoords="offset points",
                    color=color, fontsize=10, fontweight="bold", va="center")

    ax.axvline(mdates.datestr2num("2024-01-01"), color=MUTED, lw=1, ls="--", alpha=0.6)
    ax.text(mdates.datestr2num("2024-01-15"), ax.get_ylim()[0] * 1.15,
            "out-of-sample →", color=MUTED, fontsize=9)

    ax.set_title("BTC-USD 1h — vol-targeted momentum ensemble vs buy & hold\n"
                 "8 bps/side costs · long/short · 2021–2026",
                 color=TEXT, fontsize=11, loc="left")
    ax.set_ylabel("growth of $1 (log scale)", color=MUTED, fontsize=9)
    ticks = [0.5, 1, 2, 4, 8]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}x" for t in ticks])
    ax.minorticks_off()
    ax.grid(True, which="major", color=GRID, lw=0.8)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT, loc="upper left")
    ax.margins(x=0.06)

    fig.tight_layout()
    fig.savefig("research/equity_curve.png", bbox_inches="tight")
    print("wrote research/equity_curve.png")
    print("strategy:", strat.summary())
    print("bench   :", bench.summary())


if __name__ == "__main__":
    main()
