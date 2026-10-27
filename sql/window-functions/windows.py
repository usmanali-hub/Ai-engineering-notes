"""Window function cheat sheet: every query runs against a small in-memory SQLite table."""

import sqlite3

SALES = [
    # (rep, region, day, amount)
    ("ana", "east", 1, 100),
    ("ana", "east", 2, 150),
    ("ana", "east", 3, 150),
    ("ben", "east", 1, 90),
    ("ben", "east", 2, 200),
    ("cy", "west", 1, 120),
    ("cy", "west", 2, 60),
    ("cy", "west", 3, 180),
]

QUERIES = {
    "row_number": """
SELECT rep, day, amount,
       ROW_NUMBER() OVER (PARTITION BY rep ORDER BY day) AS n
FROM sales ORDER BY rep, day""",
    "rank_vs_dense_rank": """
SELECT rep, day, amount,
       RANK()       OVER (PARTITION BY rep ORDER BY amount DESC) AS rnk,
       DENSE_RANK() OVER (PARTITION BY rep ORDER BY amount DESC) AS dense
FROM sales WHERE rep = 'ana' ORDER BY amount DESC, day""",
    "lag_change": """
SELECT rep, day, amount,
       amount - LAG(amount) OVER (PARTITION BY rep ORDER BY day) AS change
FROM sales ORDER BY rep, day""",
    "running_total": """
SELECT rep, day, amount,
       SUM(amount) OVER (PARTITION BY rep ORDER BY day) AS running
FROM sales ORDER BY rep, day""",
    "moving_average_2": """
SELECT rep, day, amount,
       AVG(amount) OVER (PARTITION BY rep ORDER BY day ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS avg2
FROM sales ORDER BY rep, day""",
    "share_of_region": """
SELECT rep, region, amount,
       ROUND(100.0 * amount / SUM(amount) OVER (PARTITION BY region), 1) AS pct_of_region
FROM sales ORDER BY region, rep, day""",
    "top_day_per_rep": """
SELECT rep, day, amount FROM (
  SELECT rep, day, amount,
         ROW_NUMBER() OVER (PARTITION BY rep ORDER BY amount DESC, day) AS rn
  FROM sales)
WHERE rn = 1 ORDER BY rep""",
    "quartile": """
SELECT rep, day, amount, NTILE(4) OVER (ORDER BY amount) AS quartile
FROM sales ORDER BY amount, rep, day""",
}


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE sales (rep TEXT, region TEXT, day INTEGER, amount INTEGER)")
    conn.executemany("INSERT INTO sales VALUES (?, ?, ?, ?)", SALES)
    return conn


def run(name: str) -> list[tuple]:
    return connect().execute(QUERIES[name]).fetchall()


if __name__ == "__main__":
    for name in QUERIES:
        print(f"-- {name}")
        for row in run(name):
            print(row)
