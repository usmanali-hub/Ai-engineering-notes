# ai-engineering-notes

Small, runnable experiments and short write-ups from my AI engineering practice. Each folder has
code, tests and a README with what I tried, what I learned, and how to run it.

## Index

| Area | Note | What it covers |
|---|---|---|
| Python | [Config and logging](python/config-and-logging) | Typed settings from environment variables, safe secret handling, logging setup |
| Python | [Decorators: timing and retry](python/decorators-timing-retry) | A timing decorator and a retry decorator with exponential backoff |
| Python | [Streaming chunks](python/streaming-chunks) | Generators and context managers for processing large inputs lazily |
| Python | [Strict typing](python/strict-typing) | Common mypy strict errors and how to fix them |
| NumPy | [Vectorization benchmark](python/vectorization-benchmark) | Python loops vs NumPy, with measured speedups |
| SQL | [Window function cheat sheet](sql/window-functions) | Eight runnable window-function queries with hand-checked results |
| SQL | [Index vs no index](sql/index-explain) | EXPLAIN QUERY PLAN before and after an index: scan vs search, ~200x faster lookups, when an index cannot help, and the write cost |
| SQL | [Recursive CTEs](sql/recursive-cte) | Series, org chart, bill of materials, cycle-safe graph reachability and calendar gap filling, each checked against plain Python recursion |
| SQL | [Parameterized queries](sql/parameterized-queries) | Placeholders vs string formatting: injection and UNION-leak demos, IN lists, LIKE wildcard escaping, whitelisted sort columns, rollback on failure |
| pandas | [Clean a messy CSV](python/pandas-clean-csv) | Header, text, currency, date and duplicate cleanup in one short chain |
| pandas | [Time-series resampling](python/timeseries-resampling) | Tick prices to OHLC bars, log returns and rolling volatility |
| Visualization | [Matplotlib chart set](python/matplotlib-chart-set) | Line, bar and histogram helpers with one consistent, honest style |
| NumPy | [Outlier detection](python/outlier-detection) | z-score vs IQR vs median/MAD, and why z-scores can mask outliers |

## Run everything

```bash
pip install -r requirements-dev.txt
pytest
```
