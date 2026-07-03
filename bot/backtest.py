"""Vectorized backtester for long/short position strategies.

Contract: a strategy produces a target position series in {-1, 0, +1}
(or fractional) where the position at bar t is decided using data up to
and including the close of bar t. The backtester applies that position
to the close-to-close return of bar t+1, so there is no lookahead.

Costs are charged on turnover: |pos_t - pos_{t-1}| * cost_per_side.
Default cost is 8 bps per side (taker fee + slippage on a liquid
BTC perpetual/spot book), i.e. a full long->short flip costs 16 bps.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

HOURS_PER_YEAR = 24 * 365


@dataclass
class Result:
    equity: pd.Series          # compounded equity curve, starts at 1.0
    position: pd.Series        # position actually held during each bar's return
    strategy_returns: pd.Series
    n_trades: int
    total_return: float
    cagr: float
    sharpe: float
    max_drawdown: float
    win_rate: float            # fraction of round-trip trades with positive PnL
    exposure: float            # fraction of bars with a nonzero position
    yearly_returns: pd.Series

    def summary(self) -> str:
        years = ", ".join(f"{y}: {r:+.1%}" for y, r in self.yearly_returns.items())
        return (
            f"total {self.total_return:+.1%} | CAGR {self.cagr:+.1%} | "
            f"Sharpe {self.sharpe:.2f} | maxDD {self.max_drawdown:.1%} | "
            f"trades {self.n_trades} | win {self.win_rate:.0%} | "
            f"exposure {self.exposure:.0%}\n    by year: {years}"
        )


def run(df: pd.DataFrame, target_position: pd.Series, cost_per_side: float = 0.0008) -> Result:
    """df needs a 'close' column and a DatetimeIndex; target_position is aligned to df."""
    pos = target_position.reindex(df.index).fillna(0.0)
    # Position decided at close of t earns the return of bar t+1.
    held = pos.shift(1).fillna(0.0)
    asset_ret = df["close"].pct_change().fillna(0.0)
    turnover = pos.diff().abs().fillna(pos.abs())
    # Turnover happens at the close of t; charge it against bar t+1's PnL.
    costs = (turnover * cost_per_side).shift(1).fillna(0.0)
    strat_ret = held * asset_ret - costs

    equity = (1 + strat_ret).cumprod()
    total_return = equity.iloc[-1] - 1
    n_hours = len(df)
    years = n_hours / HOURS_PER_YEAR
    cagr = equity.iloc[-1] ** (1 / years) - 1 if years > 0 and equity.iloc[-1] > 0 else -1.0
    vol = strat_ret.std()
    sharpe = (strat_ret.mean() / vol * np.sqrt(HOURS_PER_YEAR)) if vol > 0 else 0.0
    max_dd = (equity / equity.cummax() - 1).min()

    # Round-trip trade accounting: a trade is a maximal run of constant nonzero position.
    trade_ids = (pos != pos.shift(1)).cumsum()
    pnl_by_run = strat_ret.groupby(trade_ids.shift(1)).sum()
    active_runs = pos.groupby(trade_ids).first()
    trade_pnls = pnl_by_run.reindex(active_runs[active_runs != 0].index).dropna()
    n_trades = len(trade_pnls)
    win_rate = float((trade_pnls > 0).mean()) if n_trades else 0.0
    exposure = float((held != 0).mean())

    yearly = (1 + strat_ret).groupby(df.index.year).prod() - 1

    return Result(
        equity=equity,
        position=held,
        strategy_returns=strat_ret,
        n_trades=n_trades,
        total_return=float(total_return),
        cagr=float(cagr),
        sharpe=float(sharpe),
        max_drawdown=float(max_dd),
        win_rate=win_rate,
        exposure=exposure,
        yearly_returns=yearly,
    )


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = df.set_index("timestamp").sort_index()
    return df[["open", "high", "low", "close", "volume"]].astype(float)
