"""Volatility squeeze breakout on 15m BTC.

Detect volatility compression (normalized Bollinger band width in a low
percentile of its multi-day history), then trade the direction of the range
break, optionally waiting for a pullback/retest of the broken level instead
of chasing the breakout bar. Optional higher-timeframe EMA trend alignment
and volume confirmation. Trades are managed with an ATR trailing stop,
breakeven floor after 1R, optional R-multiple target, and a time stop.

Causality: every input at row t uses only rows <= t; breakout reference
ranges and the volume baseline are shifted one bar so the current bar's own
extremes never define the level being broken.
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
           squeeze_pct: float = 0.4,
           arm_window: int = 24,
           breakout_len: int = 16,
           vol_mult: float = 1.0,
           atr_period: int = 56,
           trail_mult: float = 3.0,
           target_r: float = 0.0,
           trend_len: int = 2688,
           retest_atr: float = 0.0,   # >0: wait for pullback within this many ATR of the broken level
           retest_wait: int = 64,     # max bars to wait for the retest
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

    # --- higher-timeframe trend alignment (0 disables the filter)
    if trend_len > 0:
        ema = close.ewm(span=trend_len, adjust=False).mean()
        up_ok = close > ema
        dn_ok = close < ema
    else:
        up_ok = pd.Series(True, index=df.index)
        dn_ok = pd.Series(True, index=df.index)

    atr = _atr(df, atr_period)

    long_sig = (armed & brk_up & vol_ok & up_ok).to_numpy()
    short_sig = (armed & brk_dn & vol_ok & dn_ok).to_numpy()
    c = close.to_numpy()
    a = atr.to_numpy()
    hi_np = hi.to_numpy()
    lo_np = lo.to_numpy()

    n = len(df)
    pos = np.zeros(n)
    state = 0          # -1 short, 0 flat, +1 long
    pend = 0           # pending retest direction
    pend_level = 0.0
    pend_age = 0
    stop = 0.0
    entry = 0.0
    risk = 0.0
    held = 0
    for t in range(n):
        if state == 1:
            held += 1
            stop = max(stop, c[t] - trail_mult * a[t])
            if c[t] >= entry + risk:          # 1R in favor -> breakeven floor
                stop = max(stop, entry)
            if (c[t] < stop or held >= max_hold
                    or (target_r > 0 and c[t] >= entry + target_r * risk)):
                state = 0
        elif state == -1:
            held += 1
            stop = min(stop, c[t] + trail_mult * a[t])
            if c[t] <= entry - risk:
                stop = min(stop, entry)
            if (c[t] > stop or held >= max_hold
                    or (target_r > 0 and c[t] <= entry - target_r * risk)):
                state = 0
        if state == 0 and not np.isnan(a[t]):
            enter = 0
            if retest_atr <= 0:
                if long_sig[t]:
                    enter = 1
                elif short_sig[t]:
                    enter = -1
            else:
                # register a fresh pending breakout (newest wins)
                if long_sig[t]:
                    pend, pend_level, pend_age = 1, hi_np[t], 0
                elif short_sig[t]:
                    pend, pend_level, pend_age = -1, lo_np[t], 0
                elif pend != 0:
                    pend_age += 1
                    if pend_age > retest_wait:
                        pend = 0
                if pend == 1:
                    if c[t] < pend_level - 0.5 * a[t]:   # failed breakout
                        pend = 0
                    elif c[t] <= pend_level + retest_atr * a[t]:
                        enter = 1
                        pend = 0
                elif pend == -1:
                    if c[t] > pend_level + 0.5 * a[t]:
                        pend = 0
                    elif c[t] >= pend_level - retest_atr * a[t]:
                        enter = -1
                        pend = 0
            if enter != 0:
                state = enter
                entry = c[t]
                risk = trail_mult * a[t]
                stop = entry - state * risk
                held = 0
        pos[t] = state
    return pd.Series(pos, index=df.index)


PARAM_GRID = {
    "bb_period": [80],
    "squeeze_lookback": [960],
    "squeeze_pct": [0.4, 0.5],
    "arm_window": [24, 48],
    "breakout_len": [16, 32],
    "vol_mult": [1.0],
    "trail_mult": [2.0, 3.0],
    "target_r": [0.0],
    "trend_len": [2688],
    "retest_atr": [0.0, 0.5, 1.0],
    "retest_wait": [64],
    "max_hold": [192, 384],
}
