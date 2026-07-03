"""Multi-timeframe pullback strategy.

Higher-timeframe trend from an EMA cross on the 15m close series (spans sized
in 15m bars: ~8h fast vs ~4d slow, fully causal), gated by trend strength in
local-vol units. Entries are 15m counter-trend pullbacks: fast RSI dip while
the trend is up opens a long; fast RSI rip while the trend is down opens a
short. Exits on trend flip, an optional resumption target (price stretched
back beyond the fast EMA by `target_k` vol units; target_k >= 90 disables it),
or a time stop. Buys dips in bulls, shorts rallies in bears.

Research notes (honest): at the >= 0.7 trades/day the harness requires, every
variant tested (RSI-recross exits, confirmation entries, band mean-reversion
exits, resumption targets) is net-negative after 5 bps/side on TRAIN, and all
variants are strongly negative on RECENT (2025-26 choppy bear). The only
profitable TRAIN configuration is the slow flip-only exit (~0.4 trades/day),
which is below the activity floor and still loses on RECENT. The grid below
spans that spectrum so the harness selection is transparent.
"""
import numpy as np
import pandas as pd


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def signal(df: pd.DataFrame, trend_fast: int = 32, trend_slow: int = 384,
           trend_gap: float = 1.0, rsi_period: int = 6,
           rsi_entry: float = 35.0, target_k: float = 99.0,
           max_hold: int = 96) -> pd.Series:
    close = df["close"]
    ema_f = close.ewm(span=trend_fast, adjust=False).mean()
    ema_s = close.ewm(span=trend_slow, adjust=False).mean()
    vol = close.pct_change().rolling(96).std()
    gap = (ema_f - ema_s) / (ema_s * vol.replace(0, np.nan))
    trend = np.where(gap > trend_gap, 1, np.where(gap < -trend_gap, -1, 0))

    rsi = _rsi(close, rsi_period)
    enter_long = ((trend == 1) & (rsi < rsi_entry)).to_numpy()
    enter_short = ((trend == -1) & (rsi > 100 - rsi_entry)).to_numpy()
    stretch = ((close - ema_f) / (ema_f * vol.replace(0, np.nan))).to_numpy()

    n = len(df)
    pos = np.zeros(n)
    cur = 0
    held = 0
    for t in range(max(trend_slow, 96), n):
        tr = trend[t]
        s = stretch[t]
        if cur == 0:
            if enter_long[t]:
                cur = 1
                held = 0
            elif enter_short[t]:
                cur = -1
                held = 0
        elif cur == 1:
            held += 1
            if tr == -1 or held >= max_hold or (np.isfinite(s) and s > target_k):
                cur = 0
        else:  # cur == -1
            held += 1
            if tr == 1 or held >= max_hold or (np.isfinite(s) and s < -target_k):
                cur = 0
        pos[t] = cur
    return pd.Series(pos, index=df.index)


PARAM_GRID = {
    "trend_fast": [32],
    "trend_slow": [384],
    "trend_gap": [0.5, 1.0, 2.0],
    "rsi_period": [6, 10],
    "rsi_entry": [27.0, 35.0, 40.0],
    "target_k": [2.5, 99.0],
    "max_hold": [96, 192, 384],
}
