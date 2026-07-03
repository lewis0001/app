"""Volatility squeeze breakout on 15m BTC.

Idea: detect volatility compression (normalized Bollinger band width at a
low percentile of its multi-day history), then trade the direction of the
range break with volume confirmation. Manage the trade with an ATR trailing
stop and a time stop. Squeezes resolve violently both ways -> long/short.

Causality: all indicator inputs at row t use only rows <= t; the breakout
reference range and the volume baseline are shifted by one bar so the
current bar's own extremes never define the level it must break.
"""
import numpy as np
import pandas as pd


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def signal(df: pd.DataFrame,
           bb_period: int = 80,
           squeeze_lookback: int = 960,
           squeeze_pct: float = 0.25,
           arm_window: int = 16,
           breakout_len: int = 24,
           vol_mult: float = 1.2,
           atr_period: int = 56,
           trail_mult: float = 3.0,
           max_hold: int = 192) -> pd.Series:
    close = df["close"]

    # --- squeeze detection: normalized band width vs its rolling quantile
    bw = close.rolling(bb_period).std() / close
    thresh = bw.rolling(squeeze_lookback).quantile(squeeze_pct)
    squeeze = (bw <= thresh).astype(float)
    # armed if a squeeze was active on any of the previous `arm_window` bars
    armed = squeeze.shift(1).rolling(arm_window, min_periods=1).max().fillna(0.0) > 0

    # --- breakout levels from the PRIOR `breakout_len` bars (exclude bar t)
    hi = df["high"].rolling(breakout_len).max().shift(1)
    lo = df["low"].rolling(breakout_len).min().shift(1)
    brk_up = close > hi
    brk_dn = close < lo

    # --- volume confirmation vs prior baseline
    vol_base = df["volume"].rolling(bb_period).mean().shift(1)
    vol_ok = df["volume"] > vol_mult * vol_base

    atr = _atr(df, atr_period)

    long_sig = (armed & brk_up & vol_ok).to_numpy()
    short_sig = (armed & brk_dn & vol_ok).to_numpy()
    c = close.to_numpy()
    a = atr.to_numpy()

    n = len(df)
    pos = np.zeros(n)
    state = 0
    stop = 0.0
    held = 0
    for t in range(n):
        if state == 1:
            held += 1
            stop = max(stop, c[t] - trail_mult * a[t])
            if c[t] < stop or held >= max_hold:
                state = 0
        elif state == -1:
            held += 1
            stop = min(stop, c[t] + trail_mult * a[t])
            if c[t] > stop or held >= max_hold:
                state = 0
        if state == 0 and not np.isnan(a[t]):
            if long_sig[t]:
                state = 1
                stop = c[t] - trail_mult * a[t]
                held = 0
            elif short_sig[t]:
                state = -1
                stop = c[t] + trail_mult * a[t]
                held = 0
        pos[t] = state
    return pd.Series(pos, index=df.index)


PARAM_GRID = {
    "bb_period": [80],
    "squeeze_lookback": [960],
    "squeeze_pct": [0.2, 0.3],
    "arm_window": [8, 24],
    "breakout_len": [16, 32],
    "vol_mult": [1.0, 1.4],
    "trail_mult": [2.0, 3.0],
    "max_hold": [96, 192],
}
