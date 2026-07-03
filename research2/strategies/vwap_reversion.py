"""Anchored VWAP reversion on 15m BTC bars, gated by trend regime.

Daily-anchored VWAP (cumulative typical-price*volume / cumulative volume,
reset each UTC day). Deviation of close from VWAP is standardized by its
own rolling std. Diagnostics on TRAIN showed the fade only has an edge
when the multi-day trend is NOT up: deep dips below VWAP revert and pops
above VWAP fail in downtrends, while in uptrends both tails continue
(fading is toxic). So entries are gated on a normalized multi-day trend:

  long  when z < -k_long  and trend_z < long_gate   (fade dip)
  short when z >  k_short and trend_z < short_gate  (fade pop, bear only)

Exit at VWAP touch (z crossing z_exit), opposite band (position flips),
or UTC day end (no overnight carry across the VWAP anchor reset).

Causality: rolling/cumulative transforms of data up to bar t only; the
day-end flatten uses the bar's own clock time (23:45 UTC).
"""
import numpy as np
import pandas as pd


def signal(df: pd.DataFrame, k_long: float = 2.5, k_short: float = 2.0,
           long_gate: float = 0.5, short_gate: float = -0.3,
           z_exit: float = 0.0, sd_window: int = 384,
           trend_lb: int = 288, min_day_bar: int = 2) -> pd.Series:
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
    ok_bar = bar_of_day >= min_day_bar

    long_entry = (zv < -k_long) & (tzv < long_gate) & ok_bar
    short_entry = (zv > k_short) & (tzv < short_gate) & ok_bar

    # two one-sided state machines (entry / exit / hold), ffill-based
    long_state = np.where(long_entry, 1.0,
                          np.where(zv >= z_exit, 0.0, np.nan))
    short_state = np.where(short_entry, -1.0,
                           np.where(zv <= -z_exit, 0.0, np.nan))

    long_state[day_end | day_start] = 0.0
    short_state[day_end | day_start] = 0.0

    long_pos = pd.Series(long_state, index=idx).ffill().fillna(0.0)
    short_pos = pd.Series(short_state, index=idx).ffill().fillna(0.0)

    pos = (long_pos + short_pos).clip(-1, 1)
    pos[np.isnan(zv)] = 0.0
    return pos


PARAM_GRID = {
    "k_long": [2.0, 2.5, 3.0],
    "k_short": [1.5, 2.0, 2.5],
    "long_gate": [0.0, 0.5],
    "short_gate": [-0.5, -0.3],
    "z_exit": [0.0],
    "sd_window": [192, 384],
    "trend_lb": [288],
    "min_day_bar": [2],
}
