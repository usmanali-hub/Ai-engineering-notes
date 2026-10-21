import logging

import pytest
from retry_timing import retry, timed


class Flaky:
    def __init__(self, failures: int, exc: type[Exception] = ConnectionError) -> None:
        self.failures, self.exc, self.calls = failures, exc, 0

    def __call__(self) -> str:
        self.calls += 1
        if self.calls <= self.failures:
            raise self.exc("boom")
        return "ok"


def test_succeeds_after_transient_failures():
    waits: list[float] = []
    f = retry(times=4, delay=1, backoff=2, sleep=waits.append)(Flaky(2))
    assert f() == "ok"
    assert waits == [1, 2]  # exponential backoff


def test_raises_after_exhausting_attempts():
    flaky = Flaky(10)
    with pytest.raises(ConnectionError):
        retry(times=3, sleep=lambda _: None)(flaky)()
    assert flaky.calls == 3


def test_does_not_retry_unlisted_exceptions():
    flaky = Flaky(5, exc=KeyError)
    with pytest.raises(KeyError):
        retry(times=3, exceptions=(ConnectionError,), sleep=lambda _: None)(flaky)()
    assert flaky.calls == 1


def test_invalid_times():
    with pytest.raises(ValueError):
        retry(times=0)


def test_timed_preserves_name_and_result(caplog):
    @timed
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    with caplog.at_level(logging.INFO):
        assert add(2, 3) == 5
    assert add.__name__ == "add" and add.__doc__ == "Add two numbers."
    assert "add took" in caplog.text


def test_timed_logs_even_when_function_raises(caplog):
    @timed
    def bad() -> None:
        raise RuntimeError("x")

    with caplog.at_level(logging.INFO), pytest.raises(RuntimeError):
        bad()
    assert "bad took" in caplog.text
