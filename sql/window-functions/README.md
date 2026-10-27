# Window function cheat sheet

**Goal:** a runnable reference for the window functions used most in analytics SQL. Each query below runs
against one small table (`sales`: rep, region, day, amount) in an in-memory SQLite database, and
`test_windows.py` checks every result against a hand calculation.

## What's here

- `windows.py`: the sample data, all queries in a `QUERIES` dict, and `run(name)`
- `test_windows.py`: expected results, and a check that this README contains every query exactly as it runs

## The queries

### row_number

Numbers rows inside each partition. Unique even when values tie.

```sql
SELECT rep, day, amount,
       ROW_NUMBER() OVER (PARTITION BY rep ORDER BY day) AS n
FROM sales ORDER BY rep, day
```

### rank_vs_dense_rank

For ana's amounts 150, 150, 100: RANK gives 1, 1, 3 (skips after a tie), DENSE_RANK gives 1, 1, 2.

```sql
SELECT rep, day, amount,
       RANK()       OVER (PARTITION BY rep ORDER BY amount DESC) AS rnk,
       DENSE_RANK() OVER (PARTITION BY rep ORDER BY amount DESC) AS dense
FROM sales WHERE rep = 'ana' ORDER BY amount DESC, day
```

### lag_change

`LAG` reads the previous row in the partition. The first row has no previous row, so the result is NULL, not 0.

```sql
SELECT rep, day, amount,
       amount - LAG(amount) OVER (PARTITION BY rep ORDER BY day) AS change
FROM sales ORDER BY rep, day
```

### running_total

`SUM(...) OVER (... ORDER BY day)` is a cumulative sum per rep.

```sql
SELECT rep, day, amount,
       SUM(amount) OVER (PARTITION BY rep ORDER BY day) AS running
FROM sales ORDER BY rep, day
```

### moving_average_2

An explicit frame: this row and the one before. At the start there is only one row, so it averages just itself.

```sql
SELECT rep, day, amount,
       AVG(amount) OVER (PARTITION BY rep ORDER BY day ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS avg2
FROM sales ORDER BY rep, day
```

### share_of_region

A window aggregate without ORDER BY covers the whole partition, so each row can be divided by its region total.

```sql
SELECT rep, region, amount,
       ROUND(100.0 * amount / SUM(amount) OVER (PARTITION BY region), 1) AS pct_of_region
FROM sales ORDER BY region, rep, day
```

### top_day_per_rep

The standard top-1-per-group pattern: number rows inside a subquery, then filter. The extra `day` in ORDER BY makes ties deterministic.

```sql
SELECT rep, day, amount FROM (
  SELECT rep, day, amount,
         ROW_NUMBER() OVER (PARTITION BY rep ORDER BY amount DESC, day) AS rn
  FROM sales)
WHERE rn = 1 ORDER BY rep
```

### quartile

`NTILE(4)` splits the ordered rows into 4 equal-sized buckets.

```sql
SELECT rep, day, amount, NTILE(4) OVER (ORDER BY amount) AS quartile
FROM sales ORDER BY amount, rep, day
```

## Key ideas

- **PARTITION BY restarts the calculation; ORDER BY inside OVER sets the order within it.** Both are optional.
- **Window functions keep every row**, unlike GROUP BY, which collapses them.
- **You cannot filter on a window result in WHERE** (it is evaluated first). Wrap the query in a subquery or CTE, as in `top_day_per_rep`.
- **The default frame with ORDER BY runs from the start of the partition to the current row**, which is why `SUM() OVER (ORDER BY ...)` is a running total. Use `ROWS BETWEEN` to be explicit.

## Run

```bash
pytest sql/window-functions
python sql/window-functions/windows.py
```
