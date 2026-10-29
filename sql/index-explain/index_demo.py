"""Show what an index changes: the query plan, the work done and the runtime, in SQLite."""

import random
import sqlite3
import time

N_ROWS = 200_000
QUERY = "SELECT COUNT(*), SUM(amount) FROM orders WHERE customer_id = ?"


def build(n_rows: int = N_ROWS, n_customers: int = 5_000, seed: int = 0) -> sqlite3.Connection:
    rng = random.Random(seed)
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, amount REAL)")
    conn.executemany(
        "INSERT INTO orders (customer_id, amount) VALUES (?, ?)",
        ((rng.randrange(n_customers), round(rng.uniform(1, 100), 2)) for _ in range(n_rows)),
    )
    conn.commit()
    return conn


def plan(conn: sqlite3.Connection, sql: str = QUERY) -> str:
    """The last column of each EXPLAIN QUERY PLAN row, joined into one string."""
    return " | ".join(row[-1] for row in conn.execute("EXPLAIN QUERY PLAN " + sql, (42,)))


def time_queries(conn: sqlite3.Connection, repeats: int = 200) -> float:
    """Average milliseconds per lookup over many different customers."""
    start = time.perf_counter()
    for i in range(repeats):
        conn.execute(QUERY, (i * 7 % 5_000,)).fetchone()
    return (time.perf_counter() - start) / repeats * 1000


def compare() -> dict[str, object]:
    conn = build()
    before_plan, before_ms = plan(conn), time_queries(conn)
    answer_before = conn.execute(QUERY, (42,)).fetchone()
    conn.execute("CREATE INDEX idx_orders_customer ON orders (customer_id)")
    after_plan, after_ms = plan(conn), time_queries(conn)
    answer_after = conn.execute(QUERY, (42,)).fetchone()
    return {
        "before_plan": before_plan,
        "after_plan": after_plan,
        "before_ms": before_ms,
        "after_ms": after_ms,
        "speedup": before_ms / after_ms,
        "same_answer": answer_before == answer_after,
    }


def explain_cases(n_rows: int = 20_000) -> dict[str, str]:
    """Query plans for situations where an index is, and is not, usable."""
    conn = build(n_rows)
    conn.execute("CREATE INDEX idx_customer_amount ON orders (customer_id, amount)")

    def p(sql: str) -> str:
        return " | ".join(row[-1] for row in conn.execute("EXPLAIN QUERY PLAN " + sql))

    return {
        "equality on the leading column": p("SELECT COUNT(*) FROM orders WHERE customer_id = 42"),
        "function applied to the indexed column": p("SELECT COUNT(*) FROM orders WHERE customer_id + 0 = 42"),
        "filter on a non-leading column only": p("SELECT COUNT(*) FROM orders WHERE amount > 99"),
        "leading column plus range on the second": p(
            "SELECT amount FROM orders WHERE customer_id = 42 AND amount > 50"
        ),
    }


def insert_seconds(with_index: bool, n_rows: int = 50_000) -> float:
    """Time to insert ``n_rows`` more rows; an index must be updated on every write."""
    conn = build(1_000)
    if with_index:
        conn.execute("CREATE INDEX idx_customer ON orders (customer_id)")
    start = time.perf_counter()
    conn.executemany("INSERT INTO orders (customer_id, amount) VALUES (?, ?)", ((i % 5_000, 1.0) for i in range(n_rows)))
    return time.perf_counter() - start


if __name__ == "__main__":
    r = compare()
    print("before:", r["before_plan"])
    print("after: ", r["after_plan"])
    print(f"{r['before_ms']:.3f} ms -> {r['after_ms']:.4f} ms per lookup ({r['speedup']:.0f}x faster)")
    print("same answer:", r["same_answer"])
    for name, text in explain_cases().items():
        print(f"{name}: {text}")
    plain, indexed = insert_seconds(False), insert_seconds(True)
    print(f"50,000 inserts: {plain:.3f}s without the index, {indexed:.3f}s with it ({indexed / plain:.1f}x)")
