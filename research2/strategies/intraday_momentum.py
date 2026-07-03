"""Intraday momentum continuation on 15m BTC bars.

Core idea (b) from the family brief: short-lookback (2-12h) z-scored momentum
with fast hysteresis exits -- enter when the z-scored short-horizon return
spikes past a threshold, exit as soon as it decays back toward zero, so
holds last hours rather than days. Position is vol-targeted.
"""
import numpy as np
import pandas as pd

BARS_PER_YEAR = 4 * 24 * 365


def signal(df: pd.DataFrame, lookback: int = 16, z_period: int = 384,
           z_entry: float = 1.5, z_exit: float = 0.25,
           target_vol: float = 0.6, vol_period: int = 288) -> pd.Series:
    close = df["close"]
    ret = close.pct_change(lookback)
    mean = ret.rolling(z_period).mean()
    std = ret.rolling(z_period).std()
    z = (ret - mean) / std.replace(0, np.nan)

    # Hysteresis state machine, vectorised:
    # +1 when z > z_entry, -1 when z < -z_entry, 0 once |z| < z_exit,
    # otherwise carry the previous state.
    raw = np.where(z > z_entry, 1.0,
                   np.where(z < -z_entry, -1.0,
                            np.where(z.abs() < z_exit, 0.0, np.nan)))
    direction = pd.Series(raw, index=df.index).ffill().fillna(0.0)

    bar_vol = close.pct_change().rolling(vol_period).std()
    annual_vol = bar_vol * np.sqrt(BARS_PER_YEAR)
    scale = (target_vol / annual_vol).clip(upper=1.0)
    scale = ((scale / 0.25).round() * 0.25).fillna(0.0)
    return direction * scale


PARAM_GRID = {
    "lookback": [8, 16, 24, 48],       # 2h, 4h, 6h, 12h
    "z_period": [192, 384],            # 2d, 4d
    "z_entry": [1.25, 1.5, 2.0],
    "z_exit": [0.25, 0.5],
    "target_vol": [0.6],
}
