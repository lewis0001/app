"""Fast mean reversion on 15m BTC: buy vol-normalized panic dips below a
~3h EMA, gated by a realized-vol-percentile regime filter; optional
bear-gated short leg for euphoric rips.

TRAIN study findings baked into the design:
  * Dip-buying only pays in HIGH realized-vol regimes: z < -1 dips in the
    top half of the 4-week vol distribution earn ~10-20 bps over the next
    2-4 hours, while the same dips in quiet tape LOSE money.
    -> longs require the vol percentile above vol_gate (the regime filter
       flips the sign of the edge, i.e. it earns its keep).
  * Shorting rips is structurally toxic (upside momentum continues)
    unless the higher-timeframe trend is already down; even then the edge
    is thin, so the short leg uses a stiffer threshold short_mult * th
    and can be effectively disabled by a large short_mult.
  * Exit on z normalization (z back above -exit_z for longs) or a time
    stop of max_hold bars, so a position never fights a fresh trend long.

Causality: all inputs at row t use closes up to and including t; the
harness applies the position to bar t+1 returns.
"""
import numpy as np
import pandas as pd


def signal(df: pd.DataFrame, ema_span: int = 12, th: float = 1.0,
           short_mult: float = 99.0, vol_gate: float = 0.5,
           exit_z: float = 0.0, max_hold: int = 32,
           vol_period: int = 96, pctile_window: int = 2688,
           trend_period: int = 384) -> pd.Series:
    close = df["close"]
    ret1 = close.pct_change()

    vol = ret1.rolling(vol_period).std()
    ema = close.ewm(span=ema_span, adjust=False).mean()
    z = (close / ema - 1) / (vol * np.sqrt(ema_span)).replace(0, np.nan)
    z_v = z.to_numpy()

    # regime gates
    volp = vol.rolling(pctile_window).rank(pct=True)
    trend_z = (close.pct_change(trend_period)
               / (ret1.rolling(trend_period).std() * np.sqrt(trend_period)
                  ).replace(0, np.nan)).fillna(0.0)

    hot = (volp > vol_gate).fillna(False)
    bear = trend_z < 0

    entry_long = ((z < -th) & hot).fillna(False).to_numpy()
    entry_short = ((z > th * short_mult) & hot & bear).fillna(False).to_numpy()

    n = len(df)
    pos = np.zeros(n)
    p = 0
    hold = 0
    for i in range(n):
        zi = z_v[i]
        if p == 1:
            hold += 1
            if (not np.isnan(zi) and zi >= -exit_z) or hold >= max_hold:
                p = 0
        elif p == -1:
            hold += 1
            if (not np.isnan(zi) and zi <= exit_z) or hold >= max_hold:
                p = 0
        if p == 0:
            if entry_long[i]:
                p = 1
                hold = 0
            elif entry_short[i]:
                p = -1
                hold = 0
        pos[i] = p
    return pd.Series(pos, index=df.index)


PARAM_GRID = {
    "ema_span": [12, 20],
    "th": [0.9, 1.0, 1.15],
    "short_mult": [1.25, 99.0],   # 99 = short leg off
    "vol_gate": [0.4, 0.5, 0.6],
    "exit_z": [0.0, 0.25],
    "max_hold": [16, 32],
}
