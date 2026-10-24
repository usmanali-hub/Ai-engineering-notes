# Time-series resampling

**Goal:** convert high-frequency prices into OHLC bars and rolling volatility, the first step of most
market-data and sensor-data work.

## What's here

- `bars.py`: `synthetic_ticks` (random-walk prices), `to_bars` (OHLC via `resample`), `returns_and_volatility`
- `test_bars.py`: hand-computed bars, bar invariants on random data, gap handling, rolling warm-up

## Key ideas

- **Resample needs a DatetimeIndex.** `prices.resample("5min").ohlc()` gives open, high, low and close in one call.
- **Empty bars appear when there is a gap.** Resampling creates a row for every interval, so overnight or lunch gaps become NaN rows; `dropna(how="all")` removes them instead of inventing prices.
- **Check invariants, not just examples.** High must be at least max(open, close) and low at most min(open, close) on any data.
- **Use log returns for volatility.** They add across time and are symmetric; `min_periods=window` keeps the warm-up rows as NaN instead of reporting a volatility from two points.
- **The data is synthetic.** Prices are a random walk, so the numbers demonstrate the mechanics, not market behaviour.

## Run

```bash
pytest python/timeseries-resampling
python python/timeseries-resampling/bars.py
```
