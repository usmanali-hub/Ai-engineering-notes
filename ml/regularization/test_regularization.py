import numpy as np
import pytest
import regularization as rg
from sklearn.linear_model import Lasso, Ridge


@pytest.fixture(scope="module")
def sparse():
    return rg.experiment("sparse", repeats=12)


@pytest.fixture(scope="module")
def dense():
    return rg.experiment("dense", repeats=12)


def test_worlds_have_the_stated_truth():
    s, d = rg.make_world("sparse"), rg.make_world("dense")
    assert np.count_nonzero(s.coef) == 5 and np.count_nonzero(d.coef) == 50
    assert s.X.shape == (100, 50)
    assert np.array_equal(rg.make_world("sparse", seed=3).y, rg.make_world("sparse", seed=3).y)
    with pytest.raises(ValueError):
        rg.make_world("moderate")


def test_lasso_wins_when_few_features_matter(sparse):
    assert sparse["lasso"]["test_error"] < sparse["ridge"]["test_error"] < sparse["ols"]["test_error"]
    assert sparse["lasso"]["coef_error"] < sparse["ridge"]["coef_error"]
    assert sparse["lasso"]["test_error"] < sparse["ols"]["test_error"] * 0.75


def test_lasso_finds_every_real_feature_but_also_keeps_some_noise(sparse):
    la = sparse["lasso"]
    assert la["true_features_found"] == pytest.approx(1.0, abs=0.02)
    assert 2 < la["false_features_kept"] < 25  # selection is good, not clean
    assert la["coefficients_zeroed"] > 20


def test_ridge_wins_when_many_small_effects_matter(dense):
    assert dense["ridge"]["test_error"] < dense["lasso"]["test_error"] < dense["ols"]["test_error"]


def test_every_model_pays_for_estimating_coefficients_from_100_rows(sparse, dense):
    for r in (sparse, dense):
        assert all(r[m]["test_error"] > 1.0 for m in ("ols", "ridge", "lasso"))  # 1.0 is the irreducible noise


def test_regularization_does_not_help_on_a_small_real_dataset():
    r = rg.real_data(repeats=2)
    assert max(r.values()) / min(r.values()) < 1.03  # 442 rows and 10 features: little to shrink
    assert 2500 < r["ols"] < 3500


def test_unscaled_lasso_drops_the_feature_measured_in_small_units():
    r = rg.scale_dependence()
    assert r["unscaled_kept"] == 1 and r["unscaled_small_unit_feature_kept"] == 0
    assert r["scaled_kept"] == 2


def test_ridge_shrinks_without_zeroing_and_lasso_zeroes():
    w = rg.make_world("sparse")
    from sklearn.preprocessing import StandardScaler

    Z = StandardScaler().fit_transform(w.X)
    y = w.y - w.y.mean()
    ridge, lasso = Ridge(alpha=50).fit(Z, y), Lasso(alpha=0.5).fit(Z, y)
    assert np.count_nonzero(ridge.coef_) == 50 and np.count_nonzero(lasso.coef_) < 50
    ols = np.linalg.lstsq(Z, y, rcond=None)[0]
    assert np.linalg.norm(ridge.coef_) < np.linalg.norm(ols)


def test_coefficients_map_back_to_original_units():
    w = rg.make_world("sparse")
    pipe = rg.models()["ridge"].fit(w.X * 7 + 3, w.y)
    scaler = pipe.named_steps["standardscaler"]
    coef = rg.original_scale_coef(pipe)
    intercept = pipe[-1].intercept_ - float(np.sum(pipe[-1].coef_ * scaler.mean_ / scaler.scale_))
    assert np.allclose((w.X * 7 + 3) @ coef + intercept, pipe.predict(w.X * 7 + 3))


def test_path_figure_is_written(tmp_path):
    out = tmp_path / "path.png"
    rg.path_figure(str(out))
    assert out.stat().st_size > 10_000


def test_summary_covers_both_worlds_and_the_real_data():
    text = rg.summary()
    assert "sparse truth" in text and "dense truth" in text and "diabetes" in text and "scale trap" in text
