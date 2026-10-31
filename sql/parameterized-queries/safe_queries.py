"""Parameterized SQL from Python: how to pass values safely, and what goes wrong when you do not.

A placeholder (``?`` or ``:name``) sends the value to the database separately from the SQL text, so
the value can never be read as SQL. Placeholders only work for *values*. Table and column names,
sort directions and LIMIT-style structure must be validated against a whitelist instead.
"""

import sqlite3
from collections.abc import Sequence

SORTABLE = {"name": "name", "price": "price", "stock": "stock"}


def demo_database() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE product (id INTEGER PRIMARY KEY, name TEXT, category TEXT, price REAL, stock INTEGER);
        INSERT INTO product VALUES
            (1, 'Keyboard', 'office', 49.5, 10), (2, 'Mouse', 'office', 19.0, 0),
            (3, 'Monitor', 'office', 189.0, 4), (4, 'Desk lamp', 'home', 25.0, 12),
            (5, '100% cotton towel', 'home', 12.0, 30), (6, 'O''Brien mug', 'home', 8.5, 7);
        CREATE TABLE secret (id INTEGER PRIMARY KEY, token TEXT);
        INSERT INTO secret VALUES (1, 'sk-live-abc123');
        """
    )
    return conn


def unsafe_by_category(conn: sqlite3.Connection, category: str) -> list[tuple[int, str]]:
    """DO NOT COPY: builds SQL by string formatting, so the value becomes part of the statement."""
    sql = f"SELECT id, name FROM product WHERE category = '{category}'"
    return [(r[0], r[1]) for r in conn.execute(sql)]


def by_category(conn: sqlite3.Connection, category: str) -> list[tuple[int, str]]:
    """Same query with a placeholder: the input is data, whatever it contains."""
    return [(r[0], r[1]) for r in conn.execute("SELECT id, name FROM product WHERE category = ?", (category,))]


def in_stock_between(conn: sqlite3.Connection, low: float, high: float) -> list[str]:
    """Named placeholders read better when there are several values."""
    sql = "SELECT name FROM product WHERE stock > 0 AND price BETWEEN :low AND :high ORDER BY name"
    return [r[0] for r in conn.execute(sql, {"low": low, "high": high})]


def by_ids(conn: sqlite3.Connection, ids: Sequence[int]) -> list[str]:
    """IN (...) needs one placeholder per value; the *placeholders* are generated, the values are not."""
    if not ids:
        return []
    marks = ", ".join("?" for _ in ids)
    return [r[0] for r in conn.execute(f"SELECT name FROM product WHERE id IN ({marks}) ORDER BY id", tuple(ids))]


def _escape_like(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def search_name(conn: sqlite3.Connection, text: str) -> list[str]:
    """Substring search. A placeholder stops injection but not LIKE wildcards, so escape % and _ too."""
    sql = "SELECT name FROM product WHERE name LIKE ? ESCAPE '\\' ORDER BY id"
    return [r[0] for r in conn.execute(sql, (f"%{_escape_like(text)}%",))]


def list_products(conn: sqlite3.Connection, sort_by: str = "name", descending: bool = False) -> list[str]:
    """Identifiers cannot be bound, so the column comes from a whitelist and anything else is rejected."""
    if sort_by not in SORTABLE:
        raise ValueError(f"cannot sort by {sort_by!r}; choose one of {sorted(SORTABLE)}")
    direction = "DESC" if descending else "ASC"
    return [r[0] for r in conn.execute(f"SELECT name FROM product ORDER BY {SORTABLE[sort_by]} {direction}, id")]


def restock(conn: sqlite3.Connection, updates: Sequence[tuple[int, int]]) -> int:
    """Many writes with executemany inside one transaction; all succeed or none do."""
    with conn:  # commits on success, rolls back if anything raises
        cur = conn.executemany("UPDATE product SET stock = stock + ? WHERE id = ?", [(qty, pid) for pid, qty in updates])
    return cur.rowcount


if __name__ == "__main__":
    db = demo_database()
    attack = "x' OR '1'='1"
    print("unsafe  :", unsafe_by_category(db, attack))
    print("safe    :", by_category(db, attack))
    leak = "x' UNION SELECT id, token FROM secret --"
    print("unsafe leak:", unsafe_by_category(db, leak))
    print("safe leak  :", by_category(db, leak))
    print("LIKE '100%':", search_name(db, "100%"), "| LIKE 'O''Brien':", search_name(db, "O'Brien"))
