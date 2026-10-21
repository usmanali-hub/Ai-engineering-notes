import pytest
from typed_examples import avg_age, call_all, first_word, load_user


def test_load_user_with_and_without_age():
    assert load_user('{"name": "Ana", "age": 30}') == {"name": "Ana", "age": 30}
    assert load_user('{"name": "Bo"}') == {"name": "Bo", "age": None}


def test_avg_age_skips_missing():
    users = [{"name": "a", "age": 10}, {"name": "b", "age": None}, {"name": "c", "age": 20}]
    assert avg_age(users) == 15.0


def test_avg_age_no_data():
    with pytest.raises(ValueError):
        avg_age([{"name": "a", "age": None}])


@pytest.mark.parametrize("text,expected", [("hello world", "hello"), (None, ""), ("   ", ""), ("", "")])
def test_first_word(text, expected):
    assert first_word(text) == expected


def test_protocol_accepts_any_matching_object():
    class Upper:
        def run(self, x: int) -> str:
            return f"item-{x}".upper()

    assert call_all([Upper(), Upper()], 3) == ["ITEM-3", "ITEM-3"]
