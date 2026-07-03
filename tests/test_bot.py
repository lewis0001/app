"""Sanity tests: no lookahead, correct cost accounting, valid strategy output."""
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, ".")
from bot import backtest, strategies  # noqa: E402


def make_df(n=2000, seed=7):
    rng = np.random.default_rng(seed)
    ret = rng.normal(0, 0.01, n)
    close = 30000 * np.exp(np.cumsum(ret))
    idx = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
    high = close * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.003, n)))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    vol = np.abs(rng.normal(100, 20, n))
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": vol}, index=idx)


def test_no_lookahead():
    """Perfect foresight of the NEXT bar must not be exploitable: a position
    equal to the sign of the current bar's return, shifted as the backtester
    does, must differ from an unshifted (cheating) application."""
    df = make_df()
    future_sign = pd.Series(np.sign(df["close"].pct_change().shift(-1)), index=df.index)
    res = backtest.run(df, future_sign, cost_per_side=0.0)
    # with correct 1-bar delay, foresight of bar t+1 IS exploitable —
    # sanity-check the delay by confirming a same-bar signal is NOT.
    same_bar = pd.Series(np.sign(df["close"].pct_change()), index=df.index)
    res_same = backtest.run(df, same_bar, cost_per_side=0.0)
    assert res.total_return > 10 * max(res_same.total_return, 0.01)


def test_costs_charged_on_turnover():
    df = make_df()
    df["close"] = 100.0  # flat price: all PnL must come from costs
    df["open"] = df["high"] = df["low"] = 100.0
    pos = pd.Series(0.0, index=df.index)
    pos.iloc[10] = 1.0  # enter (1 unit turnover) ...
    pos.iloc[11:] = 0.0  # ... and exit (1 unit turnover)
    res = backtest.run(df, pos, cost_per_side=0.001)
    assert res.total_return == pytest.approx((1 - 0.001) ** 2 - 1, rel=1e-9)


def test_strategy_positions_bounded():
    df = make_df()
    for name in ["vol_momentum_ens_sized", "vol_momentum_sized", "vol_momentum",
                 "ema_cross", "donchian_atr", "rsi_dip", "regime_hybrid"]:
        pos = getattr(strategies, name)(df)
        assert pos.index.equals(df.index), name
        assert pos.abs().max() <= 1.0 + 1e-9, name
        assert not pos.isna().any(), name


def test_vol_scale_caps_at_one():
    df = make_df()
    scale = strategies.vol_scale(df, target_annual_vol=0.6)
    valid = scale.dropna()
    assert (valid <= 1.0 + 1e-9).all()
    assert (valid >= 0).all()
