import numpy as np
import pytest
import scaling as sc
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


@pytest.fixture(scope="module")
def cancer():
    return sc.load("breast_cancer")


@pytest.fixture(scope="module")
def wine():
    return sc.load("wine")


@pytest.fixture(scope="module")
def cancer_results(cancer):
    return sc.compare(*cancer, repeats=1, only=("knn", "svm_rbf", "forest"))


@pytest.fixture(scope="module")
def wine_results(wine):
    return sc.compare(*wine, repeats=1, only=("knn", "svm_rbf", "forest"))


def test_one_large_scale_feature_supplies_most_of_the_distance(cancer, wine):
    _, share_cancer = sc.distance_share(cancer[0])
    _, share_wine = sc.distance_share(wine[0])
    assert share_cancer > 0.5 and share_wine > 0.95  # equal influence would be 3% and 8%
    assert sc.distance_share(StandardScaler().fit_transform(wine[0]))[1] < 0.2  # after scaling, no feature dominates


def test_scaling_rescues_knn_and_svm_on_wine(wine_results):
    for model in ("knn", "svm_rbf"):
        row = wine_results[model]
        assert row["none"] < 0.75
        assert min(row["standard"], row["minmax"], row["robust"]) > 0.94


def test_scaling_helps_knn_and_svm_on_breast_cancer_too(cancer_results):
    for model in ("knn", "svm_rbf"):
        assert cancer_results[model]["standard"] > cancer_results[model]["none"] + 0.02


def test_the_choice_of_scaler_matters_far_less_than_scaling_at_all(cancer_results, wine_results):
    for results in (cancer_results, wine_results):
        for model in ("knn", "svm_rbf"):
            scaled = [results[model][s] for s in ("standard", "minmax", "robust")]
            assert max(scaled) - min(scaled) < 0.03


def test_forest_predictions_are_identical_with_or_without_scaling(cancer):
    X, y = cancer
    fit, test = slice(0, 400), slice(400, None)
    raw = RandomForestClassifier(n_estimators=50, random_state=0).fit(X[fit], y[fit]).predict(X[test])
    scaler = StandardScaler().fit(X[fit])
    scaled = RandomForestClassifier(n_estimators=50, random_state=0).fit(scaler.transform(X[fit]), y[fit]).predict(scaler.transform(X[test]))
    assert np.array_equal(raw, scaled)  # a monotonic rescale never changes where a tree can split


def test_forest_scores_do_not_depend_on_the_scaler(cancer_results, wine_results):
    for results in (cancer_results, wine_results):
        assert len({round(v, 9) for v in results["forest"].values()}) == 1


def test_knn_predictions_do_change_with_scaling(cancer):
    X, y = cancer
    raw = KNeighborsClassifier().fit(X[:400], y[:400]).predict(X[400:])
    s = StandardScaler().fit(X[:400])
    scaled = KNeighborsClassifier().fit(s.transform(X[:400]), y[:400]).predict(s.transform(X[400:]))
    assert not np.array_equal(raw, scaled)


def test_a_noise_feature_in_large_units_reduces_knn_to_a_coin_flip():
    r = sc.scale_trap()
    assert r["noise_share_of_distance"] > 0.99
    assert r["unscaled"] < 0.6 and r["standard"] > 0.85


def test_scaling_is_fitted_inside_each_fold_not_on_all_rows(cancer):
    """The pipeline's scaler must learn its mean from the training fold, never from the held-out rows."""
    X, y = cancer
    pipe = sc.with_scaler(KNeighborsClassifier(), "standard")
    pipe.fit(X[:300], y[:300])
    assert pipe[0].mean_[0] == pytest.approx(X[:300, 0].mean())
    assert pipe[0].mean_[0] != pytest.approx(X[:, 0].mean(), abs=1e-6)


def test_scaling_does_not_remove_outliers(cancer):
    """Corrupt 5% of values: every scaler leaves KNN worse off than the unscaled model on this data."""
    X, y = cancer
    results = sc.compare(sc.with_outliers(X), y, repeats=1, only=("knn", "forest"))
    for s in ("standard", "minmax", "robust"):
        assert results["knn"][s] < results["knn"]["none"] - 0.05
    assert results["forest"]["none"] > 0.9  # the tree model barely notices


def test_outlier_injection_is_reproducible_and_leaves_the_input_alone(cancer):
    X, _ = cancer
    before = X.copy()
    a, b = sc.with_outliers(X, seed=1), sc.with_outliers(X, seed=1)
    assert np.array_equal(a, b) and np.array_equal(X, before)
    assert (a != X).mean() == pytest.approx(0.05, abs=0.01)


def test_summary_mentions_every_dataset_and_the_trap():
    text = sc.summary(repeats=1, only=("knn",))
    assert "breast_cancer" in text and "wine" in text and "scale trap" in text and "outliers" in text


@pytest.fixture(scope="module")
def sweep():
    return sc.outlier_sweep()


def test_widespread_corruption_hurts_every_scaler(sweep):
    for pattern in ("5% of all values x20", "1% of all values x20"):
        for model in ("knn", "svm_rbf"):
            row = sweep[pattern][model]
            assert all(row[s] < row["none"] for s in ("standard", "minmax", "robust"))
    assert all(sweep["5% of all values x20"]["knn"][s] < 0.8 for s in ("standard", "minmax", "robust"))


def test_concentrated_corruption_lets_minmax_and_standard_help_svm_but_robust_collapses(sweep):
    for pattern in ("3% of rows in 10 columns x50", "3% of rows in 5 columns x100"):
        row = sweep[pattern]["svm_rbf"]
        assert row["minmax"] > row["none"] + 0.05 and row["standard"] > row["none"] + 0.05
        assert row["robust"] < row["minmax"] - 0.1  # a robust scale estimate does not shrink the outliers themselves


def test_no_scaler_wins_every_pattern(sweep):
    winners = {max(sweep[p]["knn"], key=sweep[p]["knn"].get) for p in sweep}  # type: ignore[arg-type]
    assert len(winners) >= 2


def test_corrupt_touches_only_the_chosen_columns_and_is_reproducible(cancer):
    X, _ = cancer
    out = sc.corrupt(X, (0, 1), 0.1, 10.0, seed=3)
    assert np.array_equal(out[:, 2:], X[:, 2:])
    assert np.array_equal(out, sc.corrupt(X, (0, 1), 0.1, 10.0, seed=3))
    assert 0.05 < (out[:, 0] != X[:, 0]).mean() < 0.15
