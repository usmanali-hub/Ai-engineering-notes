# Index vs no index

A runnable demo of what an index changes in SQLite, read from the query planner rather than guessed.
The table is 200,000 synthetic orders across 5,000 customers, built in memory with a fixed seed.

```
python index_demo.py
```

Typical output (timings vary by machine):

```
before: SCAN orders
after:  SEARCH orders USING INDEX idx_orders_customer (customer_id=?)
6.6 ms -> 0.033 ms per lookup (202x faster)
same answer: True
```

## What it shows

- **Scan vs search.** `EXPLAIN QUERY PLAN` reports `SCAN orders` (every row read) without an index and `SEARCH ... USING INDEX` with one. The answer is identical; only the work changes.
- **When an index cannot help.** With a composite index on `(customer_id, amount)`:
  - `WHERE customer_id = 42` uses it.
  - `WHERE customer_id + 0 = 42` scans: wrapping the indexed column in an expression hides it from the index.
  - `WHERE amount > 99` scans: the leftmost-prefix rule means the second column alone is not searchable.
  - `WHERE customer_id = 42 AND amount > 50` uses both columns.
- **Indexes cost writes.** Inserting 50,000 rows took about 1.5x longer with the index in place, since every insert also updates the index.

## Limits

Speedup depends on table size, selectivity and cache state; the ~200x figure is for this synthetic data, not a general constant. Tests assert only a loose `> 10x` bound and the exact plan text.

## Tests

`pytest sql/index-explain` checks the plan strings, that the answer is unchanged, the unusable-index cases, determinism, and the insert timing helper.
