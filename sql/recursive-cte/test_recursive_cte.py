import random
import sqlite3

import pytest
import recursive_cte as r


@pytest.fixture
def db():
    return r.demo_database()


def test_series(db):
    assert r.series(db, 5) == [1, 2, 3, 4, 5]
    assert r.series(db, 1) == [1]
    assert r.series(db, 0) == [1]  # the anchor row always exists; the WHERE only limits recursion


def test_reporting_chain_depth_and_path(db):
    got = {name: (depth, path) for _, name, depth, path in r.reporting_chain(db)}
    assert got["Ada"] == (0, "Ada")
    assert got["Flo"] == (3, "Ada > Ben > Di > Flo")
    assert len(got) == 6


def test_reporting_chain_matches_python_on_random_trees():
    rng = random.Random(0)
    for _ in range(30):
        n = rng.randint(1, 40)
        parent = {1: None, **{i: rng.randint(1, i - 1) for i in range(2, n + 1)}}
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE emp (id INTEGER PRIMARY KEY, name TEXT, manager_id INTEGER)")
        conn.executemany("INSERT INTO emp VALUES (?, ?, ?)", [(i, f"e{i}", parent[i]) for i in parent])

        depths = {1: 0}
        for i in range(2, n + 1):  # a parent always has a smaller id, so it is already known
            depths[i] = depths[parent[i]] + 1  # type: ignore[index]

        got = {int(name[1:]): d for _, name, d, _ in r.reporting_chain(conn)}
        assert got == depths


def test_parts_explosion_multiplies_quantities_down_the_tree(db):
    # bolt: 1 frame x 8, plus 2 wheels x 4 = 16; spoke: 2 wheels x 36 = 72
    assert r.parts_explosion(db, "bike") == {"bolt": 16, "rim": 2, "spoke": 72, "tube": 3}
    assert r.parts_explosion(db, "wheel") == {"bolt": 4, "rim": 1, "spoke": 36}


def test_parts_explosion_matches_python_recursion(db):
    bom: dict[str, list[tuple[str, int]]] = {}
    for parent, child, qty in db.execute("SELECT parent, child, qty FROM bom"):
        bom.setdefault(parent, []).append((child, qty))

    def explode(part: str, mult: int, out: dict[str, int]) -> None:
        if part not in bom:
            out[part] = out.get(part, 0) + mult
            return
        for child, qty in bom[part]:
            explode(child, mult * qty, out)

    expected: dict[str, int] = {}
    explode("bike", 1, expected)
    assert r.parts_explosion(db, "bike") == dict(sorted(expected.items()))


def test_reachable_terminates_on_a_cycle(db):
    assert r.reachable(db, 1) == [1, 2, 3, 4]  # 1 -> 2 -> 3 -> 1 is a cycle
    assert r.reachable(db, 5) == [5, 6]
    assert r.reachable(db, 99) == [99]


def test_reachable_matches_breadth_first_search_on_random_graphs():
    rng = random.Random(1)
    for _ in range(40):
        nodes = rng.randint(2, 12)
        edges = {(rng.randint(1, nodes), rng.randint(1, nodes)) for _ in range(rng.randint(1, 25))}
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE edge (src INTEGER, dst INTEGER)")
        conn.executemany("INSERT INTO edge VALUES (?, ?)", sorted(edges))
        start = rng.randint(1, nodes)
        seen, frontier = {start}, [start]
        while frontier:
            cur = frontier.pop()
            for a, b in edges:
                if a == cur and b not in seen:
                    seen.add(b)
                    frontier.append(b)
        assert r.reachable(conn, start) == sorted(seen)


def test_union_all_without_a_visited_check_keeps_growing_on_a_cycle(db):
    counts = [r.walks_without_guard(db, 1, d) for d in (5, 10, 20)]
    assert counts == sorted(counts) and len(set(counts)) == 3  # only the depth limit stops it
    assert counts[-1] > len(r.reachable(db, 1))  # far more rows than distinct nodes


def test_calendar_gaps_are_filled_with_zero(db):
    got = r.daily_totals_with_gaps_filled(db)
    assert [d for d, _ in got] == [f"2026-01-0{i}" for i in range(1, 7)]
    assert [v for _, v in got] == [15.0, 0.0, 0.0, 7.5, 0.0, 2.0]
    assert sum(v for _, v in got) == sum(row[0] for row in db.execute("SELECT amount FROM sales"))
