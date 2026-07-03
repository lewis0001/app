"""Multi-timeframe pullback strategy.

Higher-timeframe trend from an EMA cross on the 15m close series (spans sized
in 15m bars, ~8h vs ~4d horizons, fully causal), optionally gated by trend
strength measured in local-vol units. Entries are 15m pullbacks WITH a
resumption confirmation: in an uptrend, the fast RSI must first dip below the
entry level within the last `conf_window` bars and then cross back up through
50 (we buy the turn of the dip, not the falling knife); mirrored for shorts
in downtrends. Exits on a resumption target (price stretched back above/below
the fast EMA by `target_k` vol units), trend flip, or a time stop.
Buys dips in bulls, shorts rallies in bears — two-sided by construction.
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
           trend_gap: float = 0.5, rsi_period: int = 6,
           rsi_entry: float = 25.0, conf_window: int = 12,
           target_k: float = 1.5, max_hold: int = 192) -> pd.Series:
    close = df["close"]
    ema_f = close.ewm(span=trend_fast, adjust=False).mean()
    ema_s = close.ewm(span=trend_slow, adjust=False).mean()
    vol = close.pct_change().rolling(96).std()
    gap = (ema_f - ema_s) / (ema_s * vol.replace(0, np.nan))
    trend = np.where(gap > trend_gap, 1, np.where(gap < -trend_gap, -1, 0))

    rsi = _rsi(close, rsi_period)
    dipped = (rsi < rsi_entry).rolling(conf_window).max().fillna(0).astype(bool)
    ripped = (rsi > 100 - rsi_entry).rolling(conf_window).max().fillna(0).astype(bool)
    cross_up = ((rsi > 50) & (rsi.shift(1) <= 50)).to_numpy()
    cross_dn = ((rsi < 50) & (rsi.shift(1) >= 50)).to_numpy()
    dipped_prev = dipped.shift(1).fillna(False).to_numpy()
    ripped_prev = ripped.shift(1).fillna(False).to_numpy()

    # resumption target: price stretched beyond fast EMA by target_k vol units
    stretch = ((close - ema_f) / (ema_f * vol.replace(0, np.nan))).to_numpy()

    n = len(df)
    pos = np.zeros(n)
    cur = 0
    bars_held = 0
    warmup = max(trend_slow, 96)
    for t in range(warmup, n):
        tr = trend[t]
        if cur == 0:
            if tr == 1 and dipped_prev[t] and cross_up[t]:
                cur = 1
                bars_held = 0
            elif tr == -1 and ripped_prev[t] and cross_dn[t]:
                cur = -1
                bars_held = 0
        elif cur == 1:
            bars_held += 1
            if stretch[t] > target_k or tr == -1 or bars_held >= max_hold:
                cur = 0
        else:  # cur == -1
            bars_held += 1
            if stretch[t] < -target_k or tr == 1 or bars_held >= max_hold:
                cur = 0
        pos[t] = cur
    return pd.Series(pos, index=df.index)


PARAM_GRID = {
    "trend_fast": [32],
    "trend_slow": [384],
    "trend_gap": [0.5, 1.0, 2.0],
    "rsi_period": [6],
    "rsi_entry": [25.0, 32.0],
    "conf_window": [12, 24],
    "target_k": [1.5, 2.5],
    "max_hold": [96, 192],
}
