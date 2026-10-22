import numpy as np
import pytest
from bench import (
    best_time,
    rolling_mean_loop,
    rolling_mean_vec,
    run,
    sum_of_squares_loop,
    sum_of_squares_vec,
    zscore_loop,
    zscore_vec,
)


@pytest.fixture
def data():
    arr = np.random.default_rng(1).normal(size=500)
    return arr, arr.tolist()


def test_sum_of_squares_matches(data):
    arr, lst = data
    assert sum_of_squares_vec(arr) == pytest.approx(sum_of_squares_loop(lst))


def test_zscore_matches_and_is_standardized(data):
    arr, lst = data
    assert np.allclose(zscore_vec(arr), zscore_loop(lst))
    assert zscore_vec(arr).mean() == pytest.approx(0, abs=1e-12)
    assert zscore_vec(arr).std() == pytest.approx(1)


def test_rolling_mean_matches(data):
    arr, lst = data
    assert np.allclose(rolling_mean_vec(arr, 7), rolling_mean_loop(lst, 7))
    assert len(rolling_mean_vec(arr, 7)) == len(arr) - 6


def test_rolling_mean_known_values():
    out = rolling_mean_vec(np.array([1.0, 2.0, 3.0, 4.0]), 2)
    assert out.tolist() == [1.5, 2.5, 3.5]


def test_best_time_is_positive():
    assert best_time(lambda: sum(range(1000)), repeat=2) > 0


def test_vectorized_is_faster_on_large_input():
    # Python loops are typically 10-100x slower here, so a plain "faster" check is not flaky.
    for name, loop_s, vec_s in run(n=200_000):
        assert vec_s < loop_s, name
