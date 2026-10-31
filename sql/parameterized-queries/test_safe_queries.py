import sqlite3

import pytest
import safe_queries as q


@pytest.fixture
def db():
    return q.demo_database()


ATTACKS = ["x' OR '1'='1", "' OR 1=1 --", "x'; DROP TABLE product; --", "\" OR \"\"=\""]


@pytest.mark.parametrize("attack", ATTACKS)
def test_placeholder_treats_attack_text_as_plain_data(db, attack):
    assert q.by_category(db, attack) == []
    assert db.execute("SELECT COUNT(*) FROM product").fetchone()[0] == 6  # nothing was dropped or changed


def test_string_formatting_is_injectable(db):
    assert len(q.unsafe_by_category(db, "x' OR '1'='1")) == 6  # the whole table, not zero rows


def test_string_formatting_can_leak_another_table(db):
    assert q.unsafe_by_category(db, "x' UNION SELECT id, token FROM secret --") == [(1, "sk-live-abc123")]
    assert q.by_category(db, "x' UNION SELECT id, token FROM secret --") == []


def test_legitimate_values_work_the_same_both_ways(db):
    assert q.by_category(db, "office") == q.unsafe_by_category(db, "office")
    assert [n for _, n in q.by_category(db, "home")] == ["Desk lamp", "100% cotton towel", "O'Brien mug"]


def test_apostrophes_need_no_manual_escaping(db):
    assert q.search_name(db, "O'Brien") == ["O'Brien mug"]
    with pytest.raises(sqlite3.OperationalError):  # the unsafe version breaks on an honest apostrophe
        q.unsafe_by_category(db, "O'Brien")


def test_named_placeholders_and_stock_filter(db):
    assert q.in_stock_between(db, 10, 50) == ["100% cotton towel", "Desk lamp", "Keyboard"]
    assert "Mouse" not in q.in_stock_between(db, 0, 1000)  # stock 0 is excluded


def test_in_list_generates_one_placeholder_per_value(db):
    assert q.by_ids(db, [3, 1]) == ["Keyboard", "Monitor"]
    assert q.by_ids(db, []) == []
    assert q.by_ids(db, [99]) == []
    assert q.by_ids(db, list(range(1, 2000))) is not None  # many values still bound individually


def test_like_wildcards_are_escaped(db):
    assert q.search_name(db, "100%") == ["100% cotton towel"]
    assert q.search_name(db, "%") == ["100% cotton towel"]  # a literal percent sign, not "match everything"
    assert q.search_name(db, "_") == []  # a literal underscore, not "any one character"
    assert q.search_name(db, "mon") == ["Monitor"]  # LIKE is case-insensitive for ASCII in SQLite
    assert len(q.search_name(db, "")) == 6


def test_sort_column_is_whitelisted(db):
    assert q.list_products(db, "price")[0] == "O'Brien mug"
    assert q.list_products(db, "price", descending=True)[0] == "Monitor"
    for bad in ["name; DROP TABLE product", "id", "price DESC", "", "NAME"]:
        with pytest.raises(ValueError):
            q.list_products(db, bad)
    assert db.execute("SELECT COUNT(*) FROM product").fetchone()[0] == 6


def test_restock_commits_all_updates(db):
    assert q.restock(db, [(1, 5), (2, 3)]) == 2  # executemany reports the total rows changed
    stock = dict(db.execute("SELECT id, stock FROM product WHERE id IN (1, 2)"))
    assert stock == {1: 15, 2: 3}


def test_restock_rolls_back_everything_when_one_update_fails(db):
    with pytest.raises(sqlite3.Error):
        q.restock(db, [(1, 5), (2, object())])  # an unsupported value type raises part-way through
    assert db.execute("SELECT stock FROM product WHERE id = 1").fetchone()[0] == 10  # first update undone

