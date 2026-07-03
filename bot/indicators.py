"""Technical indicators used by the strategies. All functions are causal:
the value at index t uses only data up to and including t."""
import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    tr = atr(df, period)
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / tr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / tr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean().fillna(0.0)


def donchian(df: pd.DataFrame, period: int):
    """Highest high / lowest low over the *previous* `period` bars
    (shifted by 1 so a breakout compares the current bar against past bars only)."""
    upper = df["high"].rolling(period).max().shift(1)
    lower = df["low"].rolling(period).min().shift(1)
    mid = (upper + lower) / 2
    return upper, lower, mid


def zscore(series: pd.Series, period: int) -> pd.Series:
    mean = series.rolling(period).mean()
    std = series.rolling(period).std()
    return (series - mean) / std.replace(0, np.nan)


def efficiency_ratio(close: pd.Series, period: int = 24) -> pd.Series:
    """Kaufman efficiency ratio: |net change| / sum of |bar changes|.
    Near 1 = clean trend, near 0 = chop."""
    change = close.diff(period).abs()
    volatility = close.diff().abs().rolling(period).sum()
    return (change / volatility.replace(0, np.nan)).fillna(0.0)
