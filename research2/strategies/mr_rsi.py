"""Fast mean reversion on 15m BTC with a two-tier (probe / full) sizing
scheme and a realized-vol regime gate.

Signal anatomy (all TRAIN-derived design choices):
  * Deviation z = (close/EMA(ema_span) - 1) / (bar_vol * sqrt(ema_span)).
    Panic dip = z < -th, euphoric rip = z > th (mirror of a fast RSI).
  * Regime gate that provably earns its keep on TRAIN: the 4-week
    realized-vol percentile. Dips bought in the TOP of the vol
    distribution earn 15-50 bps over the next few hours; the same dips
    in quiet tape lose money. Longs therefore require volp > vol_gate.
  * Two-tier sizing: a small probe position (probe_size) on moderate
    dips (z < -probe_th, volp > vol_gate_probe) keeps the daily trade
    cadence with tiny fee outlay; full size only on the deep dips where
    the gross edge clearly exceeds the 10 bps round-trip cost. A probe
    is upgraded to full size if the dip deepens.
  * Shorts are structurally handicapped on this asset (upside momentum
    continues), so the short leg needs BOTH a stiffer threshold
    (short_mult * th) and a bear higher-timeframe trend (trend_z < 0);
    the grid may effectively disable it.
  * Exits: z normalization (z >= -exit_z for longs) or a time stop of
    max_hold bars, so positions never fight a fresh trend for long.

Causality: every input at row t uses closes up to and including row t;
the harness applies the position to bar t+1 close-to-close returns.
"""
import numpy as np
import pandas as pd


def signal(df: pd.DataFrame, ema_span: int = 20, th: float = 1.0,
           probe_th: float = 0.6, probe_size: float = 0.15,
           vol_gate: float = 0.6, vol_gate_probe: float = 0.4,
           exit_z: float = 0.25, max_hold: int = 8,
           short_mult: float = 1.25, shorts_on: int = 0,
           vol_period: int = 96, pctile_window: int = 2688,
           trend_period: int = 384) -> pd.Series:
    close = df["close"]
    ret1 = close.pct_change()

    vol = ret1.rolling(vol_period).std()
    ema = close.ewm(span=ema_span, adjust=False).mean()
    z = (close / ema - 1) / (vol * np.sqrt(ema_span)).replace(0, np.nan)
    z_v = z.to_numpy()

    volp = vol.rolling(pctile_window).rank(pct=True)
    trend_z = (close.pct_change(trend_period)
               / (ret1.rolling(trend_period).std() * np.sqrt(trend_period)
                  ).replace(0, np.nan)).fillna(0.0)

    hot_full = (volp > vol_gate).fillna(False)
    hot_probe = (volp > vol_gate_probe).fillna(False)
    bear = trend_z < 0

    full_long = ((z < -th) & hot_full).fillna(False).to_numpy()
    probe_long = ((z < -probe_th) & hot_probe).fillna(False).to_numpy()
    if shorts_on:
        full_short = ((z > th * short_mult) & hot_full
                      & bear).fillna(False).to_numpy()
        probe_short = ((z > probe_th * short_mult) & hot_probe
                       & bear).fillna(False).to_numpy()
    else:
        full_short = np.zeros(len(df), dtype=bool)
        probe_short = np.zeros(len(df), dtype=bool)

    n = len(df)
    pos = np.zeros(n)
    p = 0.0
    hold = 0
    for i in range(n):
        zi = z_v[i]
        if p > 0:
            hold += 1
            if full_long[i]:
                p = 1.0          # upgrade probe to full size
            if (not np.isnan(zi) and zi >= -exit_z) or hold >= max_hold:
                p = 0.0
        elif p < 0:
            hold += 1
            if full_short[i]:
                p = -1.0
            if (not np.isnan(zi) and zi <= exit_z) or hold >= max_hold:
                p = 0.0
        if p == 0:
            if full_long[i]:
                p = 1.0
                hold = 0
            elif full_short[i]:
                p = -1.0
                hold = 0
            elif probe_long[i]:
                p = probe_size
                hold = 0
            elif probe_short[i]:
                p = -probe_size
                hold = 0
        pos[i] = p
    return pd.Series(pos, index=df.index)


PARAM_GRID = {
    "ema_span": [12, 20],
    "th": [1.0, 1.5],
    "probe_th": [0.6, 0.75, 1.0],
    "probe_size": [0.15],
    "vol_gate": [0.55, 0.65],
    "exit_z": [0.0, 0.25],
    "max_hold": [8],
    "shorts_on": [0, 1],
}
