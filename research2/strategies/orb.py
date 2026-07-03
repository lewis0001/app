"""Opening-range breakout (ORB) on 15m BTC bars.

Session opens at a configurable UTC time (00:00 daily open or 13:30 US open).
The opening range is the high/low of the first `range_bars` 15m bars.
After the range completes, a close above the range high targets +1, a close
below the range low targets -1 (stop-and-reverse if the opposite side breaks
later in the window). The position is flattened `max_hold` bars after range
formation or at session end, whichever comes first.

Optional compression filter: only trade sessions whose opening range is
narrow relative to ATR (range < range_atr_max * ATR * sqrt(range_bars)).

Causality: the opening range is only referenced at bars strictly after the
range-formation window; entry is decided on bar t's close and the harness
fills it on bar t+1.
"""
import numpy as np
import pandas as pd


def signal(df: pd.DataFrame, open_hour: float = 0.0, range_bars: int = 4,
           max_hold: int = 32, range_atr_max: float = 99.0,
           entry_buf: float = 0.0, trend_span: int = 0,
           atr_period: int = 96) -> pd.Series:
    idx = df.index
    session = (idx - pd.Timedelta(minutes=int(open_hour * 60))).floor("1D")
    session = pd.Series(session, index=idx)
    barno = session.groupby(session).cumcount()

    in_range = barno < range_bars
    or_high = df["high"].where(in_range).groupby(session).transform("max")
    or_low = df["low"].where(in_range).groupby(session).transform("min")

    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(atr_period).mean()
    # ATR as of the last bar of the opening range, broadcast over the session
    atr_form = atr.where(barno == range_bars - 1).groupby(session).transform("max")

    compressed = (or_high - or_low) <= range_atr_max * atr_form * np.sqrt(range_bars)
    window = (barno >= range_bars) & (barno < range_bars + max_hold)
    tradable = window & compressed.fillna(False)

    buf = entry_buf * atr_form
    long_break = tradable & (df["close"] > or_high + buf)
    short_break = tradable & (df["close"] < or_low - buf)
    if trend_span > 0:
        ema = df["close"].ewm(span=trend_span, adjust=False).mean()
        long_break &= df["close"] > ema
        short_break &= df["close"] < ema

    raw = pd.Series(
        np.where(long_break, 1.0, np.where(short_break, -1.0, np.nan)),
        index=idx,
    )
    direction = raw.groupby(session).ffill()
    return direction.where(tradable, 0.0).fillna(0.0)


PARAM_GRID = {
    "open_hour": [0.0, 13.5],
    "range_bars": [4, 8, 16],
    "max_hold": [32, 92],
    "range_atr_max": [1.0, 99.0],
    "entry_buf": [0.0, 0.5],
    "trend_span": [0, 384, 1344],
}
