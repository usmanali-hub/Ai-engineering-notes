# Recursive CTEs

Five problems that plain `SELECT` cannot express because the depth is not known in advance.
Everything runs in memory on SQLite; `python recursive_cte.py` prints each result.

| Problem | Function | What it shows |
|---|---|---|
| Generate 1..n | `series` | Anchor row plus a recursive step with a stop condition |
| Org chart | `reporting_chain` | Depth and full path (`Ada > Ben > Di > Flo`) for every employee |
| Bill of materials | `parts_explosion` | Quantities multiply down the tree: 2 wheels x 36 spokes = 72 spokes |
| Graph reachability | `reachable` | `UNION` drops rows already seen, so a cycle terminates |
| Calendar gaps | `daily_totals_with_gaps_filled` | Generate every date, then LEFT JOIN sales so quiet days read 0 |

## The cycle trap

`walks_without_guard` runs the same graph with `UNION ALL` and only a depth limit. On the cycle
1 -> 2 -> 3 -> 1 it returns 7, 14 and 27 rows at depth 5, 10 and 20: the count keeps growing because
nothing remembers which nodes were visited. `reachable` uses `UNION` and returns the four distinct
nodes. Without a limit, the `UNION ALL` version would never finish.

## How it is checked

`pytest sql/recursive-cte` compares each query with ordinary Python recursion: depth of every node
on 30 random trees, reachability against a breadth-first search on 40 random graphs, and the bill of
materials against a recursive function. Edge cases covered: unknown start node, a cycle, and `n = 0`.

## Limits

- `UNION` deduplicates whole rows, so it is only a valid cycle guard when the row carries nothing
  that changes on each step (no depth or path column). To keep a path, check it for repeats instead.
- SQLite has no recursion depth setting by default; other engines cap it (SQL Server defaults to 100).
- The data is tiny and invented to make results checkable by hand.
