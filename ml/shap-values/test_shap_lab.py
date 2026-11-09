import numpy as np
import pytest
import shap_lab as lab


def rows(n=40, d=4, seed=0):
    return np.random.default_rng(seed).normal(size=(n, d))


def test_efficiency_base_plus_contributions_equals_the_prediction():
    f = lambda r: np.sin(r[:, 0]) + r[:, 1] * r[:, 2] + (r[:, 3] > 0)
    bg, x = rows(), rows(1, seed=5)[0]
    base, phi = lab.exact_shapley(f, x, bg)
    assert base + phi.sum() == pytest.approx(float(f(x[None, :])[0]))


def test_linear_model_gives_weight_times_distance_from_background_mean():
    w = np.array([2.0, -1.0, 0.5, 0.0])
    f = lambda r: r @ w
    bg, x = rows(), rows(1, seed=7)[0]
    _, phi = lab.exact_shapley(f, x, bg)
    assert phi == pytest.approx(w * (x - bg.mean(axis=0)))


def test_unused_feature_gets_zero_and_interchangeable_features_get_equal_credit():
    f = lambda r: r[:, 0] * r[:, 1] + r[:, 0] + r[:, 1]
    bg = rows()
    bg[:, 1] = bg[:, 0]  # symmetry needs the two features to be exchangeable in the background too
    x = np.array([0.7, 0.7, 1.3, -2.0])
    _, phi = lab.exact_shapley(f, x, bg)
    assert phi[2] == pytest.approx(0.0, abs=1e-12) and phi[3] == pytest.approx(0.0, abs=1e-12)
    assert phi[0] == pytest.approx(phi[1])


def test_interaction_credit_is_shared_between_the_two_features():
    f = lambda r: r[:, 0] * r[:, 1]
    bg = np.zeros((1, 2))  # background at zero: the whole output is the interaction
    base, phi = lab.exact_shapley(f, np.array([2.0, 3.0]), bg)
    assert base == 0.0 and phi == pytest.approx([3.0, 3.0])


def test_treeshap_contributions_sum_to_the_raw_score():
    X, y, _ = lab.make_data(n=600)
    model = lab.fit(X, y)
    base, contrib = lab.treeshap(model, X[:50])
    assert base + contrib.sum(axis=1) == pytest.approx(model.predict(X[:50], raw_score=True))


def test_exact_values_for_a_fitted_model_satisfy_efficiency_and_track_treeshap():
    r = lab.compare(correlated_copy=False, n_explain=15, n_background=40)
    assert r["efficiency_error"] < 1e-9
    assert r["row_correlation"] > 0.95
    assert r["mean_abs_exact"][0] > r["mean_abs_exact"][3]  # x0 drives the outcome; x3 is noise


def test_a_near_copy_takes_part_of_the_credit():
    ind = lab.compare(correlated_copy=False, n_explain=15, n_background=40)
    cor = lab.compare(correlated_copy=True, n_explain=15, n_background=40)
    assert cor["mean_abs_exact"][0] < ind["mean_abs_exact"][0]
    assert cor["mean_abs_exact"][5] > ind["mean_abs_exact"][5]
