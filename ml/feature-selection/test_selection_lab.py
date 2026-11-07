import numpy as np
import pytest
import selection_lab as lab
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


def test_synthetic_layout_matches_the_stated_truth():
    X, y = lab.synthetic()
    assert X.shape == (300, 100) and set(np.unique(y)) == {0, 1}
    f, _ = f_classif(X, y)
    assert f[:5].mean() > 3 * f[10:].mean()  # informative columns carry far more signal than noise columns


@pytest.mark.parametrize("method", [m for m in lab.METHODS if m != "none"])
def test_each_selector_returns_exactly_k_columns(method):
    X, y = lab.synthetic()
    chosen = lab.selected(method, X, y, 10)
    assert len(chosen) == 10 and len(set(chosen.tolist())) == 10
    assert chosen.min() >= 0 and chosen.max() < 100


def test_none_keeps_every_column_and_unknown_method_raises():
    X, y = lab.synthetic()
    assert len(lab.selected("none", X, y, 10)) == 100
    with pytest.raises(ValueError):
        lab.selector("magic", 5)


def test_recovery_counts_partition_the_chosen_set():
    r = lab.recovery(np.array([0, 1, 7, 50, 99]))
    assert r == {"informative_found": 2, "redundant_found": 1, "noise_picked": 2}


def test_selection_inside_the_pipeline_is_honest_and_outside_is_a_leak():
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(100, 500)), rng.integers(0, 2, 100)  # pure noise: true AUC is 0.5
    honest, _ = lab.cv_auc("anova", X, y, 10, repeats=1)
    leaked = cross_val_score(
        LogisticRegression(max_iter=2000), SelectKBest(f_classif, k=10).fit_transform(X, y), y, cv=5, scoring="roc_auc"
    ).mean()
    assert abs(honest - 0.5) < 0.12
    assert leaked > honest + 0.2


def test_selection_beats_using_every_column_on_the_synthetic_data():
    for seed in (3, 4):
        X, y = lab.synthetic(seed)
        everything, _ = lab.cv_auc("none", X, y, 10, repeats=1)
        for m in ("anova", "rfe", "lasso", "forest"):
            score, _ = lab.cv_auc(m, X, y, 10, repeats=1)
            assert score > everything + 0.03, (seed, m)


def test_forest_finds_the_true_columns_better_than_it_picks_noise():
    X, y = lab.synthetic()
    r = lab.recovery(lab.selected("forest", X, y, 10))
    assert r["informative_found"] + r["redundant_found"] >= 8 and r["noise_picked"] <= 2


def test_stability_is_a_similarity_between_zero_and_one():
    X, y = lab.synthetic()
    s = lab.stability("anova", X, y, 10, n=6)
    assert 0.0 < s < 1.0


def test_real_data_does_not_need_selection():
    t = lab.compare_real(ks=(5,))
    assert t.loc["none (all 30)"].iloc[0] >= t[5].drop("none (all 30)").max() - 0.01


def test_paired_comparison_uses_the_same_datasets_and_signs_correctly():
    import pandas as pd

    auc = pd.DataFrame(
        {
            "none": [0.8, 0.8, 0.8, 0.8],
            "anova": [0.90, 0.91, 0.89, 0.90],
            "mutual_info": [0.91, 0.90, 0.90, 0.91],
            "rfe": [0.88, 0.89, 0.88, 0.88],
            "lasso": [0.90, 0.91, 0.89, 0.90],
            "forest": [0.92, 0.93, 0.91, 0.92],
        }
    )
    p = lab.paired_vs(auc)
    assert "anova" not in p.index and "none" not in p.index
    assert p.loc["forest", "wins"] == 4 and p.loc["rfe", "wins"] == 0
    assert p.loc["forest", "mean_diff"] == pytest.approx(0.02)
    assert p.loc["lasso", "wins"] == 0 and p.loc["lasso", "mean_diff"] == pytest.approx(0.0)
    assert (p["datasets"] == 4).all()
