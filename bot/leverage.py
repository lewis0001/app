"""Leveraged backtest with intrabar liquidation simulation.

Models isolated-margin futures the way retail venues (e.g. PrimeXBT)
liquidate: a fixed fraction of equity is posted as margin per trade, the
position's notional is margin * leverage, and if the adverse intrabar
excursion from the entry price reaches ~90% of 1/leverage (the remaining
10% approximates maintenance margin + liquidation penalty), the position
is force-closed and the entire margin is lost.

Fees are charged on NOTIONAL, so at high leverage they consume a large
share of margin: at 500x and 5 bps taker, entry + exit costs 50% of the
margin before the market moves at all.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

BARS_PER_YEAR_15M = 4 * 24 * 365


@dataclass
class LevResult:
    equity: pd.Series
    n_trades: int
    n_liquidations: int
    total_return: float
    cagr: float
    max_drawdown: float
    win_rate: float
    ruined: bool                 # equity effectively wiped (< 1% of start)
    yearly_returns: dict = field(default_factory=dict)

    def summary(self) -> str:
        flag = "  ** ACCOUNT WIPED **" if self.ruined else ""
        years = ", ".join(f"{y}: {r:+.1%}" for y, r in self.yearly_returns.items())
        return (f"total {self.total_return:+.1%} | CAGR {self.cagr:+.1%} | "
                f"maxDD {self.max_drawdown:.1%} | trades {self.n_trades} | "
                f"liquidations {self.n_liquidations} | win {self.win_rate:.0%}{flag}\n"
                f"    by year: {years}")


def run_leveraged(df: pd.DataFrame, target_position: pd.Series, leverage: float,
                  margin_fraction: float = 1.0, cost_per_side: float = 0.0005,
                  liq_buffer: float = 0.9, bars_per_year: int = BARS_PER_YEAR_15M) -> LevResult:
    """target_position: direction in [-1, 1], decided at each bar's close,
    executed at the next bar's open. Liquidation is checked against each
    bar's high/low relative to the volume-weighted average entry price.
    """
    pos = target_position.reindex(df.index).fillna(0.0).values
    open_, high, low, close = (df[c].values for c in ("open", "high", "low", "close"))
    liq_dist = (1.0 / leverage) * liq_buffer

    equity = np.empty(len(df))
    eq = 1.0
    direction = 0.0          # current held direction (sign + size in [-1,1])
    entry = 0.0              # average entry price
    margin = 0.0             # equity posted for the open trade
    notional_per_eq = 0.0    # leverage * margin_fraction * |direction|
    trade_pnls = []
    n_liq = 0

    for i in range(len(df)):
        if i > 0 and direction != 0.0 and eq > 0:
            # check liquidation first (intrabar), using this bar's extremes
            adverse = (low[i] / entry - 1) if direction > 0 else (1 - high[i] / entry)
            if adverse <= -liq_dist:
                eq -= margin                      # margin gone
                trade_pnls.append(-margin)
                n_liq += 1
                direction, margin, notional_per_eq = 0.0, 0.0, 0.0

        target = pos[i - 1] if i > 0 else 0.0     # decided last bar, filled this bar
        if direction != 0.0 and (np.sign(target) != np.sign(direction) or target == 0.0):
            # close at this bar's open
            ret = (open_[i] / entry - 1) * np.sign(direction)
            pnl = margin * leverage * abs(direction) * ret - margin * leverage * abs(direction) * cost_per_side
            eq += pnl
            trade_pnls.append(pnl)
            direction, margin, notional_per_eq = 0.0, 0.0, 0.0
        if direction == 0.0 and target != 0.0 and eq > 0.01:
            direction = target
            entry = open_[i]
            margin = eq * margin_fraction * abs(target)
            eq -= margin * leverage * cost_per_side  # entry fee on notional
        equity[i] = max(eq, 0.0)
        if eq <= 0.01:
            equity[i:] = max(eq, 0.0)
            break

    eq_series = pd.Series(equity, index=df.index)
    total = eq_series.iloc[-1] - 1
    years = len(df) / bars_per_year
    cagr = (eq_series.iloc[-1] ** (1 / years) - 1) if eq_series.iloc[-1] > 0 else -1.0
    max_dd = (eq_series / eq_series.cummax() - 1).min()
    wins = sum(1 for p in trade_pnls if p > 0)
    yearly = ((eq_series.groupby(df.index.year).last()
               / eq_series.groupby(df.index.year).first()) - 1).to_dict()

    return LevResult(
        equity=eq_series,
        n_trades=len(trade_pnls),
        n_liquidations=n_liq,
        total_return=float(total),
        cagr=float(cagr),
        max_drawdown=float(max_dd),
        win_rate=wins / len(trade_pnls) if trade_pnls else 0.0,
        ruined=bool(eq_series.iloc[-1] < 0.01),
        yearly_returns={int(k): float(v) for k, v in yearly.items()},
    )


def liquidation_survival(df: pd.DataFrame, leverage: float, liq_buffer: float = 0.9) -> dict:
    """For every bar as a hypothetical entry (at close), how long until the
    adverse excursion hits the liquidation distance, for the LUCKIER of the
    two directions. Returns survival fractions at several horizons."""
    liq_dist = (1.0 / leverage) * liq_buffer
    close = df["close"].values
    high, low = df["high"].values, df["low"].values
    horizons = {"1h": 4, "4h": 16, "24h": 96}
    out = {}
    n = len(df) - max(horizons.values()) - 1
    step = max(1, n // 20000)  # sample entries for speed
    idxs = range(1, n, step)
    for name, h in horizons.items():
        survived = 0
        total = 0
        for i in idxs:
            entry = close[i]
            lo_min = low[i + 1:i + 1 + h].min()
            hi_max = high[i + 1:i + 1 + h].max()
            long_ok = (lo_min / entry - 1) > -liq_dist
            short_ok = (1 - hi_max / entry) > -liq_dist
            survived += 1 if (long_ok or short_ok) else 0
            total += 1
        out[name] = survived / total
    return out
