"""Baseline: the validated hourly z-momentum idea transplanted to 15m bars.
Serves as the reference the specialist strategies must beat, and as a
template for the strategy-module contract (signal + PARAM_GRID)."""
import numpy as np
import pandas as pd


def signal(df: pd.DataFrame, lookback: int = 96, z_period: int = 672,
           z_entry: float = 1.5, target_vol: float = 0.5,
           vol_period: int = 288) -> pd.Series:
    ret = df["close"].pct_change(lookback)
    mean = ret.rolling(z_period).mean()
    std = ret.rolling(z_period).std()
    z = (ret - mean) / std.replace(0, np.nan)
    direction = pd.Series(
        np.where(z > z_entry, 1.0, np.where(z < -z_entry, -1.0, np.nan)),
        index=df.index,
    ).ffill().fillna(0.0)
    hourly_vol = df["close"].pct_change().rolling(vol_period).std()
    annual_vol = hourly_vol * np.sqrt(4 * 24 * 365)
    scale = (target_vol / annual_vol).clip(upper=1.0)
    scale = ((scale / 0.25).round() * 0.25).fillna(0.0)
    return direction * scale


PARAM_GRID = {
    "lookback": [48, 96, 192],
    "z_period": [672],
    "z_entry": [1.25, 1.5],
    "target_vol": [0.5, 0.7],
}
