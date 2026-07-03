"""Anchored VWAP reversion on 15m BTC bars, regime-gated with confirmation.

Daily-anchored VWAP (cumulative typical-price*volume / cumulative volume,
reset each UTC day). Deviation of close from VWAP is standardized by its
own rolling std (z). TRAIN diagnostics (forward-return buckets of z by
multi-day trend regime) showed the fade edge lives almost entirely in
non-uptrend regimes: deep dips below VWAP revert and pops above VWAP fail
when the multi-day trend is down; in uptrends fading either tail loses.

Rules:
  - Downtrend (trend_z < -gate): long when z < -k_dn AND z ticks up
    (confirmation: don't catch the falling knife); short when z > k_s
    AND z ticks down.
  - Neutral (|trend_z| <= gate): long only, deeper threshold neutral_k,
    same confirmation. (neutral_k = 99 disables the leg.)
  - Uptrend: flat.
  - Exit at VWAP touch (z crosses z_exit), after max_hold bars (the edge
    decays after ~2-4 h), or at UTC day end (no carry across the anchor
    reset). Opposite-band entries flip the position.

Causality: rolling/cumulative transforms of data up to bar t only; the
day-end flatten uses the bar's own clock time (23:45 UTC).
"""
import numpy as np
import pandas as pd


def signal(df: pd.DataFrame, k_dn: float = 2.0, k_s: float = 2.5,
           gate: float = 0.5, neutral_k: float = 2.5,
           max_hold: int = 16, z_exit: float = 0.0,
           sd_window: int = 384, trend_lb: int = 288,
           min_day_bar: int = 2) -> pd.Series:
    idx = df.index
    close = df["close"]

    # --- daily-anchored VWAP ---------------------------------------------
    tp = (df["high"] + df["low"] + close) / 3.0
    day = pd.Series(idx.normalize(), index=idx)
    pv_cum = (tp * df["volume"]).groupby(day).cumsum()
    v_cum = df["volume"].groupby(day).cumsum()
    vwap = (pv_cum / v_cum.replace(0, np.nan)).ffill()

    # --- standardized deviation from VWAP --------------------------------
    dev = (close - vwap) / vwap
    sd = dev.rolling(sd_window, min_periods=sd_window // 2).std()
    z = dev / sd.replace(0, np.nan)

    # --- multi-day trend regime -------------------------------------------
    tr = close.pct_change(trend_lb)
    tr_sd = tr.rolling(672, min_periods=336).std()
    tz = (tr / tr_sd.replace(0, np.nan)).fillna(0.0)

    bar_of_day = day.groupby(day).cumcount().values
    day_end = (idx.hour == 23) & (idx.minute == 45)
    day_start = bar_of_day == 0

    zv = z.values
    tzv = tz.values
    dz = np.diff(zv, prepend=np.nan)  # z tick direction (causal)
    ok_bar = bar_of_day >= min_day_bar
    down = tzv < -gate
    neutral = np.abs(tzv) <= gate

    long_entry = ok_bar & (dz > 0) & (
        (down & (zv < -k_dn)) | (neutral & (zv < -neutral_k))
    )
    short_entry = ok_bar & (dz < 0) & down & (zv > k_s)
    flatten = day_end | day_start | np.isnan(zv)

    n = len(zv)
    out = np.zeros(n)
    pos = 0.0
    hold = 0
    for t in range(n):
        if pos > 0 and (zv[t] >= z_exit or hold >= max_hold):
            pos = 0.0
        elif pos < 0 and (zv[t] <= -z_exit or hold >= max_hold):
            pos = 0.0
        if long_entry[t]:
            pos, hold = 1.0, 0
        elif short_entry[t]:
            pos, hold = -1.0, 0
        if flatten[t]:
            pos = 0.0
        if pos != 0.0:
            hold += 1
        out[t] = pos

    return pd.Series(out, index=idx)


PARAM_GRID = {
    "k_dn": [2.0, 2.5],
    "k_s": [2.0, 2.5],
    "gate": [0.25, 0.5],
    "neutral_k": [2.5, 99.0],
    "max_hold": [16, 32],
    "z_exit": [0.0],
    "sd_window": [384],
    "trend_lb": [288],
    "min_day_bar": [2],
}
