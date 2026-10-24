import numpy as np
import pandas as pd
import pytest
from bars import returns_and_volatility, synthetic_ticks, to_bars


def test_ohlc_matches_manual_values():
    idx = pd.date_range("2026-01-05 09:30", periods=6, freq="1min")
    prices = pd.Series([10.0, 12.0, 9.0, 11.0, 15.0, 14.0], index=idx)
    bars = to_bars(prices, "3min")
    assert bars.iloc[0].tolist() == [10.0, 12.0, 9.0, 9.0]
    assert bars.iloc[1].tolist() == [11.0, 15.0, 11.0, 14.0]


def test_bar_invariants_hold_on_random_data():
    bars = to_bars(synthetic_ticks(1000, seed=4), "5min")
    assert (bars["high"] >= bars[["open", "close"]].max(axis=1)).all()
    assert (bars["low"] <= bars[["open", "close"]].min(axis=1)).all()


def test_gaps_do_not_create_fake_bars():
    idx = pd.to_datetime(["2026-01-05 09:30", "2026-01-05 09:31", "2026-01-05 11:00"])
    bars = to_bars(pd.Series([1.0, 2.0, 3.0], index=idx), "5min")
    assert len(bars) == 2  # the empty bars between 09:35 and 11:00 are dropped


def test_returns_and_vol_warmup():
    bars = to_bars(synthetic_ticks(600, seed=1), "5min")
    out = returns_and_volatility(bars, window=6)
    assert np.isnan(out["log_return"].iloc[0])
    assert out["rolling_vol"].iloc[:6].isna().all()
    assert out["rolling_vol"].iloc[6:].notna().all()
    assert out["log_return"].iloc[1] == pytest.approx(np.log(bars["close"].iloc[1] / bars["close"].iloc[0]))
