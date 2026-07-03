"""Fast mean reversion on 15m BTC: short-period RSI panic/euphoria fades,
gated by a regime filter so we do not fade genuine trends.

Idea:
  * RSI(2-4) on 15m closes. RSI < entry -> buy the panic dip;
    RSI > 100-entry -> short the euphoric rip.
  * Exit when RSI normalizes past 50 (long: rsi >= exit_level,
    short: rsi <= 100-exit_level) or after a time stop of max_hold bars.
  * Regime gates (both must allow an entry):
      - Kaufman efficiency ratio over er_period bars must be below er_max
        (only fade choppy, non-trending tape).
      - Trend z-score (return over trend_period normalized by per-bar vol
        * sqrt(trend_period)): block longs in a strong downtrend
        (trend_z < -trend_gate) and shorts in a strong uptrend
        (trend_z > trend_gate). trend_gate = inf disables the gate.

Causality: every input at row t uses closes up to and including t; the
position decided at t is applied by the harness to bar t+1 returns.
"""
import numpy as np
import pandas as pd


def _rsi(close: pd.Series, n: int) -> pd.Series:
    diff = close.diff()
    up = diff.clip(lower=0.0).rolling(n).mean()
    dn = (-diff.clip(upper=0.0)).rolling(n).mean()
    rs = up / dn.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    # all-gain windows -> RSI 100, all-loss -> 0
    rsi = rsi.where(dn > 0, 100.0)
    rsi = rsi.where(up > 0, 0.0)
    rsi[up.isna() | dn.isna()] = np.nan
    return rsi


def signal(df: pd.DataFrame, rsi_n: int = 3, entry: float = 10.0,
           exit_level: float = 55.0, max_hold: int = 16,
           er_period: int = 96, er_max: float = 0.35,
           trend_period: int = 384, trend_gate: float = 1.5) -> pd.Series:
    close = df["close"]
    rsi = _rsi(close, rsi_n)

    # Kaufman efficiency ratio: |net move| / path length over er_period
    net = (close - close.shift(er_period)).abs()
    path = close.diff().abs().rolling(er_period).sum()
    er = (net / path.replace(0, np.nan)).fillna(1.0)

    # Trend z-score over trend_period, normalized by realized per-bar vol
    bar_vol = close.pct_change().rolling(trend_period).std()
    trend_ret = close.pct_change(trend_period)
    trend_z = trend_ret / (bar_vol * np.sqrt(trend_period)).replace(0, np.nan)
    trend_z = trend_z.fillna(0.0)

    calm = er < er_max
    entry_long = ((rsi < entry) & calm & (trend_z > -trend_gate)).to_numpy()
    entry_short = ((rsi > 100 - entry) & calm & (trend_z < trend_gate)).to_numpy()

    rsi_v = rsi.to_numpy()
    exit_up = exit_level          # long exits when rsi >= this
    exit_dn = 100.0 - exit_level  # short exits when rsi <= this

    n = len(df)
    pos = np.zeros(n)
    p = 0
    hold = 0
    for i in range(n):
        if p == 1:
            hold += 1
            if (not np.isnan(rsi_v[i]) and rsi_v[i] >= exit_up) or hold >= max_hold:
                p = 0
        elif p == -1:
            hold += 1
            if (not np.isnan(rsi_v[i]) and rsi_v[i] <= exit_dn) or hold >= max_hold:
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
    "rsi_n": [2, 3, 4],
    "entry": [10.0, 15.0],
    "exit_level": [50.0, 60.0],
    "max_hold": [12, 24],
    "er_max": [0.30, 1.1],           # 1.1 = ER gate off
    "trend_gate": [1.25, float("inf")],  # inf = trend gate off
}
