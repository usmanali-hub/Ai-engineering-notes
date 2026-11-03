import leakage as lk
import numpy as np
import pytest
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline


@pytest.fixture(scope="module")
def selection():
    return lk.selection_leakage()


@pytest.fixture(scope="module")
def target():
    return lk.target_leakage()


@pytest.fixture(scope="module")
def group():
    return lk.group_leakage()


def test_selecting_features_before_cv_invents_skill_on_pure_noise(selection):
    assert selection["honest"] == pytest.approx(0.5, abs=0.08)  # the labels are coin flips
    assert selection["leaky"] > 0.75  # yet the leaky pipeline "predicts" them
    assert selection["leaky"] - selection["honest"] > 0.25


def test_the_pipeline_fix_is_exactly_per_fold_selection():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 5000))
    y = rng.integers(0, 2, 100)
    pipe = make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression(max_iter=1000))
    folds = StratifiedKFold(5, shuffle=True, random_state=0)
    assert cross_val_score(pipe, X, y, cv=folds).mean() == pytest.approx(lk.manual_honest_selection(), abs=1e-12)


def test_selection_leakage_shrinks_when_there_are_many_rows_per_feature():
    big = lk.selection_leakage(n=2000, p=50, k=20, repeats=3)
    assert big["leaky"] - big["honest"] < 0.03  # leakage is worst when features outnumber rows


def test_a_column_that_follows_the_outcome_inflates_auc_and_survives_a_holdout(target):
    assert target["leaky"] > 0.95
    assert 0.6 < target["honest"] < 0.75  # the real, modest signal in tenure, price and contract


def test_group_leakage_memorises_customers(group):
    assert group["leaky"] > 0.9
    assert group["honest"] == pytest.approx(0.5, abs=0.1)


def test_group_k_fold_never_splits_a_customer_but_random_folds_do():
    X, y, groups = lk.group_data()
    for train, test in GroupKFold(5).split(X, y, groups):
        assert set(groups[train]).isdisjoint(groups[test])
    shared = [len(set(groups[tr]) & set(groups[te])) / len(set(groups[te])) for tr, te in KFold(5, shuffle=True, random_state=0).split(X)]
    assert min(shared) > 0.9  # almost every test customer also appears in training


def test_scaler_leakage_is_real_but_tiny_here():
    r = lk.scaler_leakage()
    assert abs(r["leaky"] - r["honest"]) < 0.02
    assert r["honest"] > 0.95  # the model is genuinely good; the point is the leak barely moves it


def test_results_are_reproducible_for_a_seed():
    assert lk.group_leakage(seed=3) == lk.group_leakage(seed=3)
    assert lk.target_leakage(seed=3) == lk.target_leakage(seed=3)


def test_summary_reports_all_four_cases():
    assert len(lk.summary().splitlines()) == 4
