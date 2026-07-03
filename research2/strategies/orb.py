"""Opening-range breakout (ORB) on 15m BTC bars, US-session variant.

Mechanics
---------
* A session opens at `open_hour` UTC (13.5 = 13:30, the US cash open).
* The opening range is the high/low of the first `range_bars` 15m bars.
* After the range completes, a close above the range high targets +1 and a
  close below the range low targets -1, but only when aligned with a slow
  EMA trend filter (close vs EMA, optionally also the EMA's slope).
  Stop-and-reverse is allowed while the trade window is open.
* Hard stop: once the close crosses back through the OPPOSITE side of the
  opening range, the session is done (flat for the rest of the window).
* Time stop: flat `max_hold` bars after range formation (<= session end).
* Optional volume confirmation (breakout bar volume > vol_mult * 7d median)
  and optional vol-targeted sizing (quantized to 0.25 steps like baseline).

Causality: the opening range is referenced only at bars strictly after the
formation window; every entry/exit is decided on bar t's close and filled by
the harness at bar t+1. The opposite-range stop uses the previous bar's
cummax, so the stop takes effect one bar after it is observed.

What did NOT work (kept for the record): the naive ORB (no trend filter,
00:00 session, stop-and-reverse, no range stop) loses -0.4..-0.9 Sharpe on
TRAIN; fading it also loses once 5 bps/side fees are charged. All of the
edge here comes from the 13:30 session + trend alignment + cutting losers
at the far side of the range.
"""
import numpy as np
import pandas as pd


def signal(df: pd.DataFrame, open_hour: float = 13.5, range_bars: int = 2,
           max_hold: int = 92, trend_span: int = 672, ema_slope: bool = False,
           vol_mult: float = 0.0, target_vol: float = 0.0,
           atr_period: int = 96) -> pd.Series:
    idx = df.index
    session = pd.Series(
        (idx - pd.Timedelta(minutes=int(open_hour * 60))).floor("1D"), index=idx)
    barno = session.groupby(session).cumcount()

    in_range = barno < range_bars
    or_high = df["high"].where(in_range).groupby(session).transform("max")
    or_low = df["low"].where(in_range).groupby(session).transform("min")
    window = (barno >= range_bars) & (barno < range_bars + max_hold)

    long_break = window & (df["close"] > or_high)
    short_break = window & (df["close"] < or_low)

    ema = df["close"].ewm(span=trend_span, adjust=False).mean()
    long_break &= df["close"] > ema
    short_break &= df["close"] < ema
    if ema_slope:
        rising = ema.diff(16) > 0
        long_break &= rising
        short_break &= ~rising

    if vol_mult > 0:
        vmed = df["volume"].rolling(672).median()
        vok = df["volume"] > vol_mult * vmed
        long_break &= vok
        short_break &= vok

    raw = pd.Series(
        np.where(long_break, 1.0, np.where(short_break, -1.0, np.nan)),
        index=idx,
    )
    pos = raw.groupby(session).ffill().where(window, 0.0).fillna(0.0)

    # Opposite-range stop: once close crosses the far side of the opening
    # range, stay flat for the remainder of the session window.
    stop_hit = (((pos > 0) & (df["close"] < or_low)) |
                ((pos < 0) & (df["close"] > or_high))).astype(float)
    stopped = stop_hit.groupby(session).cummax().groupby(session).shift(1)
    pos = pos.where(stopped.fillna(0.0) == 0.0, 0.0)

    if target_vol > 0:
        bar_vol = df["close"].pct_change().rolling(288).std()
        annual_vol = bar_vol * np.sqrt(4 * 24 * 365)
        scale = (target_vol / annual_vol).clip(upper=1.0)
        scale = ((scale / 0.25).round() * 0.25).fillna(0.0)
        pos = pos * scale
    return pos


PARAM_GRID = {
    "open_hour": [13.5],
    "range_bars": [2, 4],
    "max_hold": [64, 92],
    "trend_span": [384, 672, 2688],
    "ema_slope": [False, True],
    "vol_mult": [0.0, 1.5],
    "target_vol": [0.0, 0.5],
}
