from pathlib import Path

import pytest
import windows


def test_row_number_restarts_per_partition():
    rows = windows.run("row_number")
    assert [r[3] for r in rows] == [1, 2, 3, 1, 2, 1, 2, 3]


def test_rank_skips_after_ties_dense_rank_does_not():
    rows = windows.run("rank_vs_dense_rank")  # ana: 150, 150, 100
    assert [(r[3], r[4]) for r in rows] == [(1, 1), (1, 1), (3, 2)]


def test_lag_is_null_on_first_row_of_each_partition():
    changes = [r[3] for r in windows.run("lag_change")]
    assert changes == [None, 50, 0, None, 110, None, -60, 120]


def test_running_total_matches_python_cumsum():
    expected = []
    totals: dict[str, int] = {}
    for rep, _region, _day, amount in sorted(windows.SALES):
        totals[rep] = totals.get(rep, 0) + amount
        expected.append(totals[rep])
    assert [r[3] for r in windows.run("running_total")] == expected


def test_moving_average_uses_only_available_rows_at_the_start():
    avgs = [r[3] for r in windows.run("moving_average_2")]
    assert avgs[:3] == [100.0, 125.0, 150.0]  # first row averages just itself


def test_share_of_region_sums_to_100_per_region():
    by_region: dict[str, float] = {}
    for _rep, region, _amount, pct in windows.run("share_of_region"):
        by_region[region] = by_region.get(region, 0) + pct
    assert by_region["east"] == pytest.approx(100, abs=0.2) and by_region["west"] == pytest.approx(100, abs=0.2)


def test_top_day_per_rep_breaks_ties_by_day():
    assert windows.run("top_day_per_rep") == [("ana", 2, 150), ("ben", 2, 200), ("cy", 3, 180)]


def test_quartiles_are_balanced():
    counts = {}
    for row in windows.run("quartile"):
        counts[row[3]] = counts.get(row[3], 0) + 1
    assert counts == {1: 2, 2: 2, 3: 2, 4: 2}


def test_readme_contains_every_query_so_docs_match_code():
    readme = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")
    for name, sql in windows.QUERIES.items():
        assert sql.strip() in readme, name
