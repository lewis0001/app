"""Candidate strategies. Each returns a target position series in {-1, 0, +1}
aligned to the input dataframe, using only information available at each bar's
close (the backtester applies positions to the following bar).

The library covers the three families that research says survive both bull
and bear markets:
  * long/short trend following (works in bear markets because it shorts)
  * short-term mean reversion filtered by the higher-timeframe trend
  * regime-switched hybrids of the two
"""
import numpy as np
import pandas as pd

from . import indicators as ta


def buy_and_hold(df: pd.DataFrame) -> pd.Series:
    return pd.Series(1.0, index=df.index)


def ema_cross(df: pd.DataFrame, fast: int = 24, slow: int = 96, long_short: bool = True) -> pd.Series:
    f, s = ta.ema(df["close"], fast), ta.ema(df["close"], slow)
    pos = np.where(f > s, 1.0, -1.0 if long_short else 0.0)
    return pd.Series(pos, index=df.index)


def donchian_breakout(df: pd.DataFrame, entry: int = 48, exit_: int = 24) -> pd.Series:
    """Classic turtle-style channel breakout, long and short, exit on the
    opposite side of a shorter channel."""
    upper_e, lower_e, _ = ta.donchian(df, entry)
    upper_x, lower_x, _ = ta.donchian(df, exit_)
    close = df["close"]
    pos = np.zeros(len(df))
    state = 0.0
    up_e, lo_e = upper_e.values, lower_e.values
    up_x, lo_x = upper_x.values, lower_x.values
    c = close.values
    for i in range(len(df)):
        if state == 0:
            if c[i] > up_e[i]:
                state = 1.0
            elif c[i] < lo_e[i]:
                state = -1.0
        elif state == 1 and c[i] < lo_x[i]:
            state = -1.0 if c[i] < lo_e[i] else 0.0
        elif state == -1 and c[i] > up_x[i]:
            state = 1.0 if c[i] > up_e[i] else 0.0
        pos[i] = state
    return pd.Series(pos, index=df.index)


def rsi_dip(df: pd.DataFrame, rsi_period: int = 3, buy_below: float = 20,
            sell_above: float = 80, trend_ema: int = 168) -> pd.Series:
    """Mean reversion in the direction of the higher-timeframe trend:
    buy oversold dips in an uptrend, short overbought rips in a downtrend.
    Exit when RSI normalizes past 50."""
    r = ta.rsi(df["close"], rsi_period)
    trend_up = df["close"] > ta.ema(df["close"], trend_ema)
    pos = np.zeros(len(df))
    state = 0.0
    rv, tu = r.values, trend_up.values
    for i in range(len(df)):
        if state == 0:
            if tu[i] and rv[i] < buy_below:
                state = 1.0
            elif not tu[i] and rv[i] > sell_above:
                state = -1.0
        elif state == 1 and (rv[i] > 50 or not tu[i]):
            state = 0.0
        elif state == -1 and (rv[i] < 50 or tu[i]):
            state = 0.0
        pos[i] = state
    return pd.Series(pos, index=df.index)


def vol_momentum(df: pd.DataFrame, lookback: int = 72, z_period: int = 168,
                 z_entry: float = 1.0) -> pd.Series:
    """Volatility-gated momentum: take the sign of the past `lookback`-hour
    move, but only when the move is statistically large (z-score of the
    lookback return vs its own history). Stays flat in quiet chop."""
    ret = df["close"].pct_change(lookback)
    z = ta.zscore(ret, z_period)
    pos = np.where(z > z_entry, 1.0, np.where(z < -z_entry, -1.0, np.nan))
    return pd.Series(pos, index=df.index).ffill().fillna(0.0)


def _z_hysteresis(z: pd.Series, z_entry: float, z_exit: float) -> pd.Series:
    """State machine: enter long when z > z_entry, exit to flat when z < z_exit;
    symmetric for shorts. Unlike ffill-forever, a faded move returns to cash."""
    zv = z.values
    pos = np.zeros(len(z))
    state = 0.0
    for i in range(len(z)):
        if state == 0:
            if zv[i] > z_entry:
                state = 1.0
            elif zv[i] < -z_entry:
                state = -1.0
        elif state == 1 and zv[i] < z_exit:
            state = -1.0 if zv[i] < -z_entry else 0.0
        elif state == -1 and zv[i] > -z_exit:
            state = 1.0 if zv[i] > z_entry else 0.0
        pos[i] = state
    return pd.Series(pos, index=z.index)


def vol_scale(df: pd.DataFrame, target_annual_vol: float = 0.60,
              vol_period: int = 72, step: float = 0.25) -> pd.Series:
    """Position size multiplier in (0, 1]: shrink when realized vol exceeds the
    target. Quantized to `step` so the size doesn't churn (and pay fees) every bar."""
    hourly_vol = df["close"].pct_change().rolling(vol_period).std()
    annual_vol = hourly_vol * np.sqrt(24 * 365)
    scale = (target_annual_vol / annual_vol).clip(upper=1.0)
    # warmup bars have no vol estimate -> no position
    return ((scale / step).round() * step).fillna(0.0)


def vol_momentum_v2(df: pd.DataFrame, lookback: int = 72, z_period: int = 168,
                    z_entry: float = 1.5, z_exit: float = 0.25,
                    target_vol: float = 0.60) -> pd.Series:
    """vol_momentum with two fixes: (1) hysteresis exit to flat when the move
    fades instead of holding a stale signal forever; (2) volatility-targeted
    position sizing to cap drawdowns when the market goes wild."""
    ret = df["close"].pct_change(lookback)
    z = ta.zscore(ret, z_period)
    direction = _z_hysteresis(z, z_entry, z_exit)
    return direction * vol_scale(df, target_vol)


def vol_momentum_ensemble(df: pd.DataFrame, lookbacks=(48, 72, 120),
                          z_period: int = 168, z_entry: float = 1.5,
                          z_exit: float = 0.25, target_vol: float = 0.60) -> pd.Series:
    """Average of vol_momentum_v2 direction across several lookbacks — less
    sensitive to any single parameter being lucky. Position is fractional."""
    dirs = []
    for lb in lookbacks:
        ret = df["close"].pct_change(lb)
        z = ta.zscore(ret, z_period)
        dirs.append(_z_hysteresis(z, z_entry, z_exit))
    combined = sum(dirs) / len(dirs)
    # quantize so partial agreement doesn't produce fee-churning micro-adjustments
    combined = (combined * 3).round() / 3
    return combined * vol_scale(df, target_vol)


def vol_momentum_sized(df: pd.DataFrame, lookback: int = 72, z_period: int = 168,
                       z_entry: float = 1.5, target_vol: float = 0.60,
                       vol_period: int = 72) -> pd.Series:
    """The round-1 winner (always-in-market z-momentum direction) with
    volatility-targeted sizing layered on top. The direction signal stays
    untouched — sizing alone is what caps the drawdown."""
    ret = df["close"].pct_change(lookback)
    z = ta.zscore(ret, z_period)
    direction = pd.Series(
        np.where(z > z_entry, 1.0, np.where(z < -z_entry, -1.0, np.nan)),
        index=df.index,
    ).ffill().fillna(0.0)
    return direction * vol_scale(df, target_vol, vol_period)


def vol_momentum_ens_sized(df: pd.DataFrame, lookbacks=(48, 72, 120),
                           z_period: int = 168, z_entry: float = 1.5,
                           target_vol: float = 0.60, vol_period: int = 72) -> pd.Series:
    """Ensemble of always-in-market z-momentum directions (one per lookback),
    vol-targeted. Fractional positions when the lookbacks disagree."""
    dirs = []
    for lb in lookbacks:
        ret = df["close"].pct_change(lb)
        z = ta.zscore(ret, z_period)
        d = pd.Series(
            np.where(z > z_entry, 1.0, np.where(z < -z_entry, -1.0, np.nan)),
            index=df.index,
        ).ffill().fillna(0.0)
        dirs.append(d)
    combined = sum(dirs) / len(dirs)
    combined = (combined * 3).round() / 3
    return combined * vol_scale(df, target_vol, vol_period)


def regime_hybrid(df: pd.DataFrame, er_period: int = 72, er_threshold: float = 0.25,
                  fast: int = 24, slow: int = 96,
                  rsi_period: int = 3, buy_below: float = 20, sell_above: float = 80,
                  trend_ema: int = 168) -> pd.Series:
    """Regime switch: efficiency ratio above threshold -> trend regime,
    follow the EMA cross long/short. Below threshold -> chop regime,
    trade RSI mean reversion with the trend filter."""
    er = ta.efficiency_ratio(df["close"], er_period)
    trend_pos = ema_cross(df, fast, slow)
    mr_pos = rsi_dip(df, rsi_period, buy_below, sell_above, trend_ema)
    return pd.Series(np.where(er > er_threshold, trend_pos, mr_pos), index=df.index)


def donchian_atr(df: pd.DataFrame, entry: int = 48, atr_period: int = 24,
                 atr_mult: float = 3.0) -> pd.Series:
    """Donchian breakout entries with a chandelier (ATR trailing) stop.
    After a stop-out it waits for the next breakout instead of reversing."""
    upper, lower, _ = ta.donchian(df, entry)
    a = ta.atr(df, atr_period)
    c = df["close"].values
    up, lo, av = upper.values, lower.values, a.values
    pos = np.zeros(len(df))
    state, stop = 0.0, 0.0
    for i in range(len(df)):
        if state == 1:
            stop = max(stop, c[i] - atr_mult * av[i])
            if c[i] < stop:
                state = 0.0
        elif state == -1:
            stop = min(stop, c[i] + atr_mult * av[i])
            if c[i] > stop:
                state = 0.0
        if state == 0:
            if c[i] > up[i]:
                state, stop = 1.0, c[i] - atr_mult * av[i]
            elif c[i] < lo[i]:
                state, stop = -1.0, c[i] + atr_mult * av[i]
        pos[i] = state
    return pd.Series(pos, index=df.index)
