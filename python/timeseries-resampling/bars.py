"""Turn tick-level prices into OHLC bars and rolling statistics with pandas resampling."""

import numpy as np
import pandas as pd


def synthetic_ticks(n: int = 600, seed: int = 0, start: str = "2026-01-05 09:30") -> pd.Series:
    """Random-walk prices, one tick per 20 seconds."""
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, periods=n, freq="20s")
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.0005, n))), index=index, name="price")


def to_bars(prices: pd.Series, rule: str = "5min") -> pd.DataFrame:
    """Open, high, low, close per bar. Empty bars are dropped rather than invented."""
    bars = prices.resample(rule).ohlc()
    return bars.dropna(how="all")


def returns_and_volatility(bars: pd.DataFrame, window: int = 6) -> pd.DataFrame:
    """Log returns of the close and a rolling standard deviation of them."""
    out = pd.DataFrame(index=bars.index)
    out["log_return"] = np.log(bars["close"]).diff()
    out["rolling_vol"] = out["log_return"].rolling(window, min_periods=window).std()
    return out


if __name__ == "__main__":
    bars = to_bars(synthetic_ticks())
    print(bars.head())
    print(returns_and_volatility(bars).tail())
