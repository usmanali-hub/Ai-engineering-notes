import index_demo as d
import pytest


@pytest.fixture(scope="module")
def result():
    return d.compare()


def test_plan_changes_from_scan_to_index_search(result):
    assert result["before_plan"] == "SCAN orders"
    assert result["after_plan"].startswith("SEARCH orders USING INDEX idx_orders_customer")
    assert "customer_id=?" in result["after_plan"]


def test_index_changes_speed_but_never_the_answer(result):
    assert result["same_answer"] is True
    assert result["speedup"] > 10  # measured around 200x; the margin keeps the test stable on slow machines


def test_explain_cases_show_when_an_index_cannot_help():
    cases = d.explain_cases()
    assert cases["equality on the leading column"].startswith("SEARCH") and "customer_id=?" in cases["equality on the leading column"]
    assert "SEARCH" not in cases["function applied to the indexed column"]  # wrapping the column defeats the index
    assert cases["filter on a non-leading column only"] == "SCAN orders"  # leftmost-prefix rule
    assert "customer_id=? AND amount>?" in cases["leading column plus range on the second"]


def test_build_is_deterministic_and_sized():
    a, b = d.build(500), d.build(500)
    assert a.execute("SELECT COUNT(*), SUM(amount) FROM orders").fetchone() == b.execute("SELECT COUNT(*), SUM(amount) FROM orders").fetchone()
    assert a.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 500


def test_insert_timing_returns_positive_numbers():
    assert d.insert_seconds(False, 2_000) > 0 and d.insert_seconds(True, 2_000) > 0
