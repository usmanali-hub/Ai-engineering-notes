import numpy as np
import pytest
from bias_variance import NOISE, plot, simulate


@pytest.fixture(scope="module")
def res():
    return simulate(n_datasets=200, seed=1)


def at(res, degree):
    return res.degrees.index(degree)


def test_decomposition_matches_directly_measured_test_error(res):
    # bias^2 + variance + noise^2 is computed from fitted curves; test_mse from fresh noisy targets.
    assert np.allclose(res.expected_error, res.test_mse, rtol=0.05)


def test_noise_term_is_the_irreducible_floor(res):
    assert res.noise2 == pytest.approx(NOISE**2)
    assert (res.test_mse > res.noise2 * 0.98).all()  # no model beats the noise level


def test_bias_falls_once_the_model_can_bend(res):
    assert res.bias2[at(res, 1)] > 20 * res.bias2[at(res, 5)]
    # a sine over one full period has no quadratic component: degree 2 is no better than degree 1
    assert res.bias2[at(res, 2)] == pytest.approx(res.bias2[at(res, 1)], rel=0.1)


def test_variance_grows_with_flexibility_and_matches_theory(res):
    assert (np.diff(res.variance) > 0).all()
    n = 20
    for d in (1, 3, 5, 8):  # for least squares with a fixed design the average variance is about sigma^2 (d+1) / n
        assert res.variance[at(res, d)] == pytest.approx(NOISE**2 * (d + 1) / n, rel=0.3)
    assert res.variance[at(res, 15)] > 20 * res.variance[at(res, 3)]


def test_training_error_always_falls_but_test_error_does_not(res):
    assert (np.diff(res.train_mse) <= 1e-12).all()  # nested models can only fit the training data better
    assert res.test_mse[-1] > 5 * res.test_mse.min()  # ...yet the most flexible model generalizes far worse
    assert res.train_mse[-1] < res.test_mse[-1]


def test_best_degree_is_in_the_middle(res):
    best = res.degrees[int(np.argmin(res.expected_error))]
    assert 3 <= best <= 6


def test_simulation_is_reproducible_and_seed_dependent():
    a = simulate(degrees=[1, 3], n_datasets=20, seed=5)
    b = simulate(degrees=[1, 3], n_datasets=20, seed=5)
    c = simulate(degrees=[1, 3], n_datasets=20, seed=6)
    assert np.array_equal(a.variance, b.variance) and not np.array_equal(a.variance, c.variance)


def test_plot_writes_a_png(tmp_path, res):
    out = tmp_path / "plot.png"
    plot(res, str(out))
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" and out.stat().st_size > 10_000
